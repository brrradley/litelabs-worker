from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import multitrack_ground_truth_campaign as mt
import synthetic_ground_truth_campaign as sgt
import experimental_ground_truth_campaign as exp
from drum_decomposition_v1 import _ensure_drumsep
from ground_truth_benchmark import _score

MODE = "experimental_child_ground_truth_v2"


def log(message: str) -> None:
    print(f"[LiteLABS experimental GT v2] {message}", flush=True)


def progress(message: str, percent: int) -> None:
    print(f"[LiteLABS experimental GT v2] {percent:3d}% {message}", flush=True)


def _run_bs_parent(source_path: Path, root: Path, timeout: int) -> tuple[dict[str, Path], float, str]:
    """Run the same BS-Roformer-SW path that the synthetic GT campaign already proved works.

    The research campaign image carries BS-Roformer-SW through audio-separator's model directory,
    not the production /models/bs_roformer_sw YAML+checkpoint layout. Using audio-separator here
    keeps the parent stage identical to the successful Shout/Disturbia supervised tests.
    """
    model_dir = Path(os.getenv("LITELABS_AUDIO_SEPARATOR_MODEL_DIR", "/models/audio_separator"))
    model = str(os.getenv("LITELABS_BS_MODEL", "BS-Roformer-SW.ckpt"))
    out = root / "bs_parent"
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        "audio-separator", str(source_path),
        "--model_filename", model,
        "--model_file_dir", str(model_dir),
        "--output_dir", str(out),
        "--output_format", "FLAC",
        "--mdxc_segment_size", "256",
        "--mdxc_overlap", "8",
        "--mdxc_batch_size", "1",
        "--use_autocast",
    ]
    started = time.monotonic()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    elapsed = time.monotonic() - started
    tail = "\n".join((proc.stdout or "").splitlines()[-40:])
    if proc.returncode != 0:
        raise RuntimeError("BS-Roformer-SW audio-separator parent stage failed: " + tail.replace("\n", " | "))

    stems: dict[str, Path] = {}
    for target in ("vocals", "bass", "drums", "piano", "guitar", "other"):
        found = mt._find_generated(out, target)
        if found is not None:
            stems[target] = found
    if "drums" not in stems:
        raise RuntimeError(f"BS-Roformer-SW drums parent not found; files={[p.name for p in out.rglob('*') if p.is_file()]}")
    return stems, elapsed, tail


def run() -> dict:
    zip_url = str(os.getenv("LITELABS_BENCHMARK_ZIP_URL", "")).strip()
    if not zip_url:
        raise RuntimeError("LITELABS_BENCHMARK_ZIP_URL is required")

    output = Path(os.getenv("LITELABS_BENCHMARK_OUTPUT", "/workspace/litelabs-research/experimental_ground_truth_v2.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    timeout = max(300, min(3300, int(os.getenv("LITELABS_BENCHMARK_MODEL_TIMEOUT", "1800"))))
    started = time.monotonic()

    if not exp.CURRENT_5_CONFIG.is_file() or not exp.CURRENT_5_CHECKPOINT.is_file():
        raise RuntimeError("Current experimental DrumSep 5-stem model is not installed in this image")
    six_config, six_checkpoint, _ = _ensure_drumsep(Path("/models/drumsep_mdx23c"))

    with tempfile.TemporaryDirectory(prefix="litelabs_exp_gt_v2_") as temp:
        root = Path(temp)
        archive = root / (Path(urlparse(zip_url).path).name or "multitracks.zip")
        extracted = root / "extracted"
        work = root / "work"

        progress("Downloading studio multitrack pack", 2)
        mt._download(zip_url, archive)
        mt._safe_extract(archive, extracted)
        sgt._remove_macos_metadata(extracted)

        progress("Building exact synthetic source and child references", 6)
        mt._track_category = sgt.track_category
        tracks, sr, inventory = mt._build_tracks(extracted, work)
        source_path, parent_refs, scale = mt._build_references(tracks, sr, work)
        integrity = sgt._reference_integrity(source_path, parent_refs, sr)
        if not integrity["exact_partition"]:
            raise RuntimeError(f"Synthetic reference integrity failed: {integrity}")

        child_refs, child_ref_stats, mapping_warnings = exp._build_child_references(tracks, work, sr)
        if not child_refs:
            raise RuntimeError("No usable experimental drum child references were found in this pack")
        log("available child references: " + ", ".join(child_refs.keys()))

        progress("Running production BS-Roformer parent separation", 12)
        source_audio, _ = mt._load(source_path)
        stems, sw_elapsed, sw_tail = _run_bs_parent(source_path, root, timeout)
        parent_drums, _ = mt._load(stems["drums"])
        true_drums, _ = mt._load(parent_refs["drums"])
        parent_score = _score(parent_drums, true_drums, sr)
        log(f"BS drums parent quality={parent_score.get('quality_score')} corr={parent_score.get('correlation')}")

        candidate_defs = [
            ("current_5stem_on_bs_parent", parent_drums, exp.CURRENT_5_CONFIG, exp.CURRENT_5_CHECKPOINT),
            ("current_5stem_on_full_mix", source_audio, exp.CURRENT_5_CONFIG, exp.CURRENT_5_CHECKPOINT),
            ("candidate_6stem_on_bs_parent", parent_drums, six_config, six_checkpoint),
            ("candidate_6stem_on_full_mix", source_audio, six_config, six_checkpoint),
        ]
        candidates = []

        for idx, (label, audio_in, config, checkpoint) in enumerate(candidate_defs):
            progress(f"Testing {label}", 30 + idx * 15)
            child_audio, elapsed, runtime_tail = exp._run_mdx23c(
                audio_in, sr, root / "candidates", label, config, checkpoint, timeout
            )
            if not child_audio:
                raise RuntimeError(f"{label} produced no recognised drum children")
            scored = exp._score_candidate(label, child_audio, child_refs, parent_drums, sr, work)
            scored["elapsed_seconds"] = round(elapsed, 3)
            scored["runtime_tail"] = runtime_tail
            scored["model_config"] = str(config)
            scored["model_checkpoint"] = str(checkpoint)
            scored["outputs_found"] = sorted(child_audio)
            candidates.append(scored)

            snapshot = {
                "ok": True,
                "mode": MODE,
                "source_zip_url": zip_url,
                "sample_rate": sr,
                "global_scale": scale,
                "reference_integrity": integrity,
                "inventory": inventory,
                "child_reference_stats": child_ref_stats,
                "mapping_warnings": mapping_warnings,
                "bs_parent_drums_score": parent_score,
                "bs_parent_elapsed_seconds": round(sw_elapsed, 3),
                "bs_parent_runtime_tail": sw_tail,
                "candidates_complete": len(candidates),
                "candidates_total": len(candidate_defs),
                "candidates": candidates,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            output.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
            log(f"checkpoint written: {output}")

        ranked = sorted(
            [
                {
                    "candidate": c["candidate"],
                    "energy_weighted_direct_correlation": c["energy_weighted_direct_correlation"],
                    "macro_direct_correlation": c["macro_direct_correlation"],
                }
                for c in candidates
            ],
            key=lambda x: x["energy_weighted_direct_correlation"] if x["energy_weighted_direct_correlation"] is not None else -999,
            reverse=True,
        )
        result = json.loads(output.read_text(encoding="utf-8"))
        result["ranking_by_energy_weighted_correlation"] = ranked
        result["complete"] = True
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        progress("Experimental child benchmark complete", 100)
        return result


if __name__ == "__main__":
    result = run()
    output = Path(os.getenv("LITELABS_BENCHMARK_OUTPUT", "/workspace/litelabs-research/experimental_ground_truth_v2.json"))
    log(f"complete in {result.get('elapsed_seconds')}s")
    log(f"retrieve {output} before stopping the Pod")
    while True:
        time.sleep(3600)
