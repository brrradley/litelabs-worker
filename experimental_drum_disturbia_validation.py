from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

import multitrack_ground_truth_campaign as mt
import synthetic_ground_truth_campaign as sgt
import experimental_drum_refinement_campaign as base
from ground_truth_benchmark import _score

MODE = "experimental_drum_disturbia_validation_v2"
CHILDREN = base.CHILDREN


def log(message: str) -> None:
    print(f"[LiteLABS Disturbia drums] {message}", flush=True)


def progress(message: str, percent: int) -> None:
    print(f"[LiteLABS Disturbia drums] {percent:3d}% {message}", flush=True)


def child_category(name: str):
    n = base._safe_key(name)

    # Disturbia uses abbreviated studio-track names (kik_01, sn_01) rather
    # than the long-form names used by the first validator.  Treat claps as
    # snare-family material because Experimental has no separate clap child;
    # this mirrors the eventual user-facing snare/percussion grouping better.
    if (
        "kick" in n
        or "bass_drum" in n
        or re.search(r"(^|_)bd(?:_|$)", n)
        or re.search(r"(^|_)kik\d*(?:_|$)", n)
    ):
        return "kick", None
    if (
        "snare" in n
        or re.search(r"(^|_)snr(?:_|$)", n)
        or re.search(r"(^|_)sn(?:_|$)", n)
        or "clap" in n
    ):
        note = "Clap track mapped to snare-family because Experimental has no separate clap child." if "clap" in n else None
        return "snare", note
    if "tom" in n or "floor_tom" in n:
        return "toms", None
    if any(x in n for x in ["hihat", "hi_hat", "hat", "hh", "open_hat", "closed_hat"]):
        return "hh", None
    if any(x in n for x in ["cymbal", "ride", "crash", "china", "splash"]):
        return "cymbals", None
    return None, None


def build_child_refs(tracks: dict[str, np.ndarray], work: Path, sr: int):
    grouped = defaultdict(list)
    members = defaultdict(list)
    mapping_notes = []
    for name, audio in tracks.items():
        child, note = child_category(name)
        if child:
            grouped[child].append(audio)
            members[child].append(name)
            if note:
                mapping_notes.append({"track": name, "mapped_child": child, "note": note})
    refs, stats = {}, {}
    for child in CHILDREN:
        if not grouped.get(child):
            stats[child] = {"available": False, "members": []}
            continue
        audio = mt._sum_tracks(grouped[child])
        path = work / "child_refs" / f"{child}.wav"
        base._write(path, audio, sr)
        refs[child] = path
        stats[child] = {
            "available": True,
            "members": members[child],
            "rms_dbfs": round(20.0 * np.log10(max(base._rms(audio), 1e-12)), 3),
        }
    return refs, stats, mapping_notes


def group_scores(outputs: dict[str, np.ndarray], refs: dict[str, Path], sr: int) -> dict:
    refs_audio = {k: mt._load(v)[0] for k, v in refs.items()}
    groups = {
        "snare_toms": ("snare", "toms"),
        "hats_cymbals": ("hh", "cymbals"),
    }
    result = {}
    for label, members in groups.items():
        usable = [m for m in members if m in outputs and m in refs_audio]
        if not usable:
            result[label] = {"available": False, "members": []}
            continue
        out = mt._sum_tracks([outputs[m] for m in usable])
        ref = mt._sum_tracks([refs_audio[m] for m in usable])
        result[label] = {"available": True, "members": usable, **_score(out, ref, sr)}
    return result


def run() -> dict:
    zip_url = str(os.getenv("LITELABS_BENCHMARK_ZIP_URL", "https://literecords.com/tmp/disturbia_test.zip")).strip()
    output = Path(os.getenv("LITELABS_BENCHMARK_OUTPUT", "/workspace/litelabs-research/disturbia_experimental_drums_v2.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    timeout = max(300, min(3300, int(os.getenv("LITELABS_BENCHMARK_MODEL_TIMEOUT", "1800"))))
    model_dir = Path(os.getenv("LITELABS_AUDIO_SEPARATOR_MODEL_DIR", "/models/audio_separator"))
    started = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="litelabs_disturbia_drum_") as temp:
        root = Path(temp)
        archive = root / (Path(urlparse(zip_url).path).name or "multitracks.zip")
        extracted = root / "extracted"
        work = root / "work"

        progress("Downloading and preparing Disturbia multitracks", 2)
        mt._download(zip_url, archive)
        mt._safe_extract(archive, extracted)
        sgt._remove_macos_metadata(extracted)
        mt._track_category = sgt.track_category
        tracks, sr, inventory = mt._build_tracks(extracted, work)
        source_path, parent_refs, scale = mt._build_references(tracks, sr, work)
        integrity = sgt._reference_integrity(source_path, parent_refs, sr)
        if not integrity["exact_partition"]:
            raise RuntimeError(f"Synthetic reference integrity failed: {integrity}")
        child_refs, child_stats, mapping_notes = build_child_refs(tracks, work, sr)
        if not child_refs:
            raise RuntimeError("No drum child references found in Disturbia pack")
        log("child refs: " + ", ".join(f"{k}={child_stats[k]['members']}" for k in child_refs))

        progress("Running fixed BS-RoFormer drums parent", 15)
        sw_out = root / "sw"
        sw_out.mkdir()
        cmd = [
            "audio-separator", str(source_path), "--model_filename", "BS-Roformer-SW.ckpt",
            "--model_file_dir", str(model_dir), "--output_dir", str(sw_out), "--output_format", "FLAC",
            "--mdxc_segment_size", "256", "--mdxc_overlap", "8", "--mdxc_batch_size", "1", "--use_autocast",
        ]
        t0 = time.monotonic()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        sw_elapsed = time.monotonic() - t0
        if proc.returncode != 0:
            raise RuntimeError("BS-RoFormer failed: " + " | ".join((proc.stdout or "").splitlines()[-30:]))
        drums_path = mt._find_generated(sw_out, "drums")
        if drums_path is None:
            raise RuntimeError("BS-RoFormer drums parent not found")
        parent, _ = mt._load(drums_path)

        progress("Running current Experimental DrumSep once", 38)
        baseline, drumsep_elapsed, drumsep_tail = base._run_current_drumsep(parent, sr, root / "drumsep", timeout)
        progress("Applying only the previously successful mild Wiener candidate", 70)
        refined = base._wiener_refine(parent, baseline, power=1.0, n_fft=2048, hop=512)

        baseline_score = base._score_candidate("baseline_current_5stem", baseline, child_refs, parent, sr)
        refined_score = base._score_candidate("wiener_p1_fft2048", refined, child_refs, parent, sr)
        baseline_score["combined_group_scores"] = group_scores(baseline, child_refs, sr)
        refined_score["combined_group_scores"] = group_scores(refined, child_refs, sr)

        per_stem_decision = {}
        for stem in child_refs:
            b = baseline_score.get("scores", {}).get(stem, {})
            r = refined_score.get("scores", {}).get(stem, {})
            bq, rq = b.get("quality_score"), r.get("quality_score")
            if bq is None or rq is None:
                continue
            per_stem_decision[stem] = {
                "baseline_quality": bq,
                "refined_quality": rq,
                "delta": round(rq - bq, 3),
                "preferred": "wiener_p1_fft2048" if rq > bq else "baseline_current_5stem",
            }

        per_group_decision = {}
        for group in ("snare_toms", "hats_cymbals"):
            b = baseline_score["combined_group_scores"].get(group, {})
            r = refined_score["combined_group_scores"].get(group, {})
            bq, rq = b.get("quality_score"), r.get("quality_score")
            if bq is None or rq is None:
                continue
            per_group_decision[group] = {
                "baseline_quality": bq,
                "refined_quality": rq,
                "delta": round(rq - bq, 3),
                "preferred": "wiener_p1_fft2048" if rq > bq else "baseline_current_5stem",
            }

        result = {
            "ok": True,
            "mode": MODE,
            "source_zip_url": zip_url,
            "sample_rate": sr,
            "global_scale": scale,
            "reference_integrity": integrity,
            "inventory": inventory,
            "child_reference_stats": child_stats,
            "mapping_notes": mapping_notes,
            "bs_parent_elapsed_seconds": round(sw_elapsed, 3),
            "drumsep_elapsed_seconds": round(drumsep_elapsed, 3),
            "drumsep_runtime_tail": drumsep_tail,
            "candidates": [baseline_score, refined_score],
            "per_stem_decision": per_stem_decision,
            "per_group_decision": per_group_decision,
            "future_grouping_note": {
                "snare_toms": "Tracked because these may be combined later.",
                "hats_cymbals": "Tracked because hi-hat and cymbals may be combined later.",
            },
            "complete": True,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        progress("Disturbia Experimental drum validation complete", 100)
        log(f"retrieve {output}")
        return result


if __name__ == "__main__":
    result = run()
    while True:
        time.sleep(3600)
