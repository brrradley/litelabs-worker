from __future__ import annotations

import json
import os
import re
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

import multitrack_ground_truth_campaign as mt
from ground_truth_benchmark import _score


TARGETS = ["vocals", "bass", "drums", "piano", "guitar", "other"]


def log(message: str) -> None:
    print(f"[LiteLABS synthetic GT] {message}", flush=True)


def progress(message: str, percent: int) -> None:
    print(f"[LiteLABS synthetic GT] {percent:3d}% {message}", flush=True)


def track_category(name: str) -> str:
    """Map arbitrary studio track names into the six production parent stems.

    This is intentionally broader than the historical Disturbia mapper so packs
    such as Shout (Vocals, Perc_Toms, Synth, Agogo_HiHat...) are classified into
    the same parent taxonomy emitted by BS-Roformer-SW.
    """
    n = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    tokens = set(n.split())

    if any(x in n for x in ["vocal", "vox", "voice", "lead voc", "backing voc", "background voc", "bgv"]):
        return "vocals"
    if "bass" in tokens or "bass" in n:
        return "bass"
    if any(x in n for x in [
        "drum", "kick", "kik", "snare", " hi hat", "hihat", "hat", "tom",
        "perc", "clap", "cymbal", "ride", "crash", "shaker", "agogo",
    ]):
        return "drums"
    if any(x in n for x in [
        "piano", "pno", "keyboard", "keys", "organ", " org", "arp", "synth",
        "rhodes", "wurli", "clav",
    ]):
        return "piano"
    if any(x in n for x in ["guitar", "gtr"]):
        return "guitar"
    return "other"


def _reference_integrity(source_path: Path, refs: dict[str, Path], sr: int) -> dict:
    source, _ = mt._load(source_path)
    parts = []
    for target in TARGETS:
        audio, _ = mt._load(refs[target])
        parts.append(audio)
    rebuilt = mt._sum_tracks(parts)
    n = max(len(source), len(rebuilt))
    a = np.zeros((n, 2), dtype=np.float32)
    b = np.zeros((n, 2), dtype=np.float32)
    a[:len(source)] = source
    b[:len(rebuilt)] = rebuilt
    diff = a - b
    peak_error = float(np.max(np.abs(diff))) if diff.size else 0.0
    rms_error = float(np.sqrt(np.mean(diff.astype(np.float64) ** 2))) if diff.size else 0.0
    return {
        "exact_partition": bool(peak_error <= 2e-6),
        "peak_absolute_error": peak_error,
        "rms_error": rms_error,
        "source_peak": float(np.max(np.abs(a))) if a.size else 0.0,
    }


def _score_outputs(out: Path, refs: dict[str, Path], sr: int, work: Path) -> dict:
    scores = {}
    for target in TARGETS:
        generated = mt._find_generated(out, target)
        if generated is None:
            scores[target] = {"ok": False, "error": "generated stem not found"}
            continue
        norm = work / "scoring" / out.name / f"{target}.wav"
        mt._normalise_file(generated, norm, sr)
        estimate, _ = mt._load(norm)
        reference, _ = mt._load(refs[target])
        try:
            score = _score(estimate, reference, sr)
            scores[target] = {"ok": True, "file": generated.name, **score}
        except Exception as exc:
            scores[target] = {"ok": False, "file": generated.name, "error": str(exc), "error_type": exc.__class__.__name__}
    return scores


def run() -> dict:
    zip_url = str(os.getenv("LITELABS_BENCHMARK_ZIP_URL", "")).strip()
    if not zip_url:
        raise RuntimeError("LITELABS_BENCHMARK_ZIP_URL is required")

    output = Path(os.getenv("LITELABS_BENCHMARK_OUTPUT", "/workspace/litelabs-research/synthetic_ground_truth.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    timeout = max(300, min(3300, int(os.getenv("LITELABS_BENCHMARK_MODEL_TIMEOUT", "1800"))))
    model_dir = Path(os.getenv("LITELABS_AUDIO_SEPARATOR_MODEL_DIR", "/models/audio_separator"))
    model_dir.mkdir(parents=True, exist_ok=True)
    experiments = mt._normalise_experiments(None)
    started = time.monotonic()

    # Use the broader six-parent mapper for both inventory and references.
    mt._track_category = track_category

    with tempfile.TemporaryDirectory(prefix="litelabs_synthetic_gt_") as temp:
        root = Path(temp)
        archive = root / (Path(urlparse(zip_url).path).name or "multitracks.zip")
        extracted = root / "extracted"

        progress("Downloading multitrack pack", 2)
        mt._download(zip_url, archive)
        mt._safe_extract(archive, extracted)

        progress("Building exact studio mixture and references", 5)
        tracks, sr, inventory = mt._build_tracks(extracted, root / "work")
        source_path, refs, scale = mt._build_references(tracks, sr, root / "work")
        integrity = _reference_integrity(source_path, refs, sr)
        log(f"reference integrity exact_partition={integrity['exact_partition']} peak_error={integrity['peak_absolute_error']:.3g}")
        if not integrity["exact_partition"]:
            raise RuntimeError(f"Synthetic source/reference partition failed integrity check: {integrity}")

        categories = defaultdict(list)
        for item in inventory:
            categories[item["category"]].append(item["name"])
        for target in TARGETS:
            log(f"reference {target}: {', '.join(categories.get(target, [])) or '[silence]'}")

        rows = []
        for i, exp in enumerate(experiments):
            pct = int(8 + 86 * i / max(1, len(experiments)))
            progress(f"experiment {i + 1}/{len(experiments)}: {exp['id']}", pct)
            out = root / "outputs" / f"{i:02d}_{exp['id']}"
            out.mkdir(parents=True, exist_ok=True)

            cmd = [
                "audio-separator", str(source_path),
                "--model_filename", exp["model"],
                "--model_file_dir", str(model_dir),
                "--output_dir", str(out),
                "--output_format", "FLAC",
                "--mdxc_segment_size", str(exp["segment"]),
                "--mdxc_overlap", str(exp["overlap"]),
                "--mdxc_batch_size", str(exp["batch_size"]),
            ]
            if exp["autocast"]:
                cmd.append("--use_autocast")

            code, stdout, elapsed, peak_gpu = mt._run(cmd, timeout)
            row = {
                "experiment": exp,
                "return_code": code,
                "elapsed_seconds": round(float(elapsed), 3),
                "peak_gpu_mib": peak_gpu,
                "scores": {},
            }
            if code == 0:
                row["scores"] = _score_outputs(out, refs, sr, root / "work")
            else:
                row["error_tail"] = "\n".join(stdout.splitlines()[-40:])
            rows.append(row)

            snapshot = {
                "ok": all(r["return_code"] == 0 for r in rows),
                "mode": "synthetic_multitrack_ground_truth",
                "source_zip_url": zip_url,
                "sample_rate": sr,
                "global_scale": scale,
                "reference_integrity": integrity,
                "inventory": inventory,
                "category_members": dict(categories),
                "experiments_complete": len(rows),
                "experiments_total": len(experiments),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "results": rows,
            }
            output.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
            log(f"checkpoint written: {output}")

        progress("Synthetic ground-truth benchmark complete", 100)
        result = json.loads(output.read_text(encoding="utf-8"))
        result["complete"] = True
        result["ok"] = all(r["return_code"] == 0 for r in rows)
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result


if __name__ == "__main__":
    result = run()
    output = Path(os.getenv("LITELABS_BENCHMARK_OUTPUT", "/workspace/litelabs-research/synthetic_ground_truth.json"))
    log(f"complete: {result.get('ok')} in {result.get('elapsed_seconds')}s")
    log(f"retrieve {output} before stopping the Pod")
    while True:
        time.sleep(3600)
