from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
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
    """Map arbitrary studio track names into the six production parent stems."""
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


def _remove_macos_metadata(root: Path) -> None:
    """Strip Finder/AppleDouble metadata before the legacy track loader sees it."""
    removed_files = 0
    removed_dirs = 0
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        try:
            if path.is_dir() and path.name == "__MACOSX":
                shutil.rmtree(path, ignore_errors=True)
                removed_dirs += 1
            elif path.is_file() and (path.name.startswith("._") or path.name == ".DS_Store"):
                path.unlink(missing_ok=True)
                removed_files += 1
        except FileNotFoundError:
            pass
    log(f"removed macOS metadata: {removed_dirs} dirs, {removed_files} files")


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


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)) + 1e-15)


def _reference_stats(source_path: Path, refs: dict[str, Path]) -> dict:
    source, _ = mt._load(source_path)
    source_rms = _rms(source)
    result = {}
    for target in TARGETS:
        audio, _ = mt._load(refs[target])
        rms = _rms(audio)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        result[target] = {
            "rms": round(rms, 9),
            "rms_dbfs": round(20.0 * np.log10(max(rms, 1e-15)), 3),
            "peak": round(peak, 9),
            "energy_ratio_vs_source": round((rms / source_rms) ** 2, 6),
        }
    return result


def _mapping_warnings(inventory: list[dict]) -> list[dict]:
    warnings = []
    for item in inventory:
        name = str(item.get("name") or "")
        category = str(item.get("category") or "")
        low = name.lower()
        reason = None
        if category == "piano" and any(x in low for x in ["synth", "arp", "organ", "keyboard", "keys"]):
            reason = "Broad keys taxonomy: separator may route this source to piano or other."
        elif category == "other":
            reason = "Fallback taxonomy: track name does not identify a production parent stem unambiguously."
        if reason:
            warnings.append({"track": name, "mapped_category": category, "warning": reason})
    return warnings


def _audio_hash(audio: np.ndarray) -> str:
    packed = np.asarray(audio, dtype="<f4", order="C")
    return hashlib.sha256(packed.tobytes()).hexdigest()


def _routing_correlation(estimate: np.ndarray, reference: np.ndarray, max_points: int = 240000) -> float:
    n = min(len(estimate), len(reference))
    if n <= 0:
        return 0.0
    step = max(1, n // max_points)
    a = estimate[:n:step].astype(np.float64).reshape(-1)
    b = reference[:n:step].astype(np.float64).reshape(-1)
    a -= np.mean(a)
    b -= np.mean(b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-15:
        return 0.0
    return float(np.dot(a, b) / denom)


def _normalise_generated(out: Path, sr: int, work: Path) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, str]]:
    generated_audio: dict[str, np.ndarray] = {}
    generated_files: dict[str, str] = {}
    fingerprints: dict[str, str] = {}
    for target in TARGETS:
        generated = mt._find_generated(out, target)
        if generated is None:
            continue
        norm = work / "scoring" / out.name / f"{target}.wav"
        mt._normalise_file(generated, norm, sr)
        audio, _ = mt._load(norm)
        generated_audio[target] = audio
        generated_files[target] = generated.name
        fingerprints[target] = _audio_hash(audio)
    return generated_audio, generated_files, fingerprints


def _score_outputs(
    generated_audio: dict[str, np.ndarray],
    generated_files: dict[str, str],
    refs: dict[str, Path],
    sr: int,
) -> tuple[dict, dict, dict]:
    references = {target: mt._load(refs[target])[0] for target in TARGETS}
    scores: dict[str, dict] = {}
    matrix: dict[str, dict] = {}

    for output_target in TARGETS:
        estimate = generated_audio.get(output_target)
        if estimate is None:
            scores[output_target] = {"ok": False, "error": "generated stem not found"}
            matrix[output_target] = {target: None for target in TARGETS}
            continue

        reference = references[output_target]
        try:
            score = _score(estimate, reference, sr)
            scores[output_target] = {
                "ok": True,
                "file": generated_files[output_target],
                **score,
            }
        except Exception as exc:
            scores[output_target] = {
                "ok": False,
                "file": generated_files[output_target],
                "error": str(exc),
                "error_type": exc.__class__.__name__,
            }

        matrix[output_target] = {
            reference_target: round(_routing_correlation(estimate, references[reference_target]), 6)
            for reference_target in TARGETS
        }

    routing_by_reference = {}
    for reference_target in TARGETS:
        candidates = [
            (output_target, matrix.get(output_target, {}).get(reference_target))
            for output_target in TARGETS
        ]
        candidates = [(name, corr) for name, corr in candidates if corr is not None]
        if not candidates:
            routing_by_reference[reference_target] = {"best_output": None, "correlation": None}
            continue
        best_output, best_corr = max(candidates, key=lambda x: abs(float(x[1])))
        routing_by_reference[reference_target] = {
            "best_output": best_output,
            "correlation": round(float(best_corr), 6),
            "expected_output": reference_target,
            "landed_as_expected": best_output == reference_target,
        }

    return scores, matrix, routing_by_reference


def _parameter_effects(rows: list[dict]) -> dict:
    good = [row for row in rows if row.get("return_code") == 0 and row.get("output_fingerprints")]
    if not good:
        return {"baseline": None, "comparisons": [], "duplicate_output_groups": []}
    baseline = good[0]
    baseline_id = baseline["experiment"]["id"]
    comparisons = []
    for row in good[1:]:
        identical = []
        changed = []
        for target in TARGETS:
            if row["output_fingerprints"].get(target) == baseline["output_fingerprints"].get(target):
                identical.append(target)
            else:
                changed.append(target)
        comparisons.append({
            "experiment": row["experiment"]["id"],
            "vs_baseline": baseline_id,
            "identical_outputs": identical,
            "changed_outputs": changed,
            "all_outputs_identical": len(changed) == 0,
        })

    groups = defaultdict(list)
    for row in good:
        signature = tuple(row["output_fingerprints"].get(target, "missing") for target in TARGETS)
        groups[signature].append(row["experiment"]["id"])
    duplicate_groups = [members for members in groups.values() if len(members) > 1]
    return {
        "baseline": baseline_id,
        "comparisons": comparisons,
        "duplicate_output_groups": duplicate_groups,
    }


def _load_experiments() -> list[dict]:
    raw = str(os.getenv("LITELABS_BENCHMARK_EXPERIMENTS_JSON", "")).strip()
    if not raw:
        return mt._normalise_experiments(None)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid LITELABS_BENCHMARK_EXPERIMENTS_JSON: {exc}") from exc
    return mt._normalise_experiments(parsed)


def run() -> dict:
    zip_url = str(os.getenv("LITELABS_BENCHMARK_ZIP_URL", "")).strip()
    if not zip_url:
        raise RuntimeError("LITELABS_BENCHMARK_ZIP_URL is required")

    output = Path(os.getenv("LITELABS_BENCHMARK_OUTPUT", "/workspace/litelabs-research/synthetic_ground_truth.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    timeout = max(300, min(3300, int(os.getenv("LITELABS_BENCHMARK_MODEL_TIMEOUT", "1800"))))
    model_dir = Path(os.getenv("LITELABS_AUDIO_SEPARATOR_MODEL_DIR", "/models/audio_separator"))
    model_dir.mkdir(parents=True, exist_ok=True)
    experiments = _load_experiments()
    started = time.monotonic()

    mt._track_category = track_category

    with tempfile.TemporaryDirectory(prefix="litelabs_synthetic_gt_") as temp:
        root = Path(temp)
        archive = root / (Path(urlparse(zip_url).path).name or "multitracks.zip")
        extracted = root / "extracted"

        progress("Downloading multitrack pack", 2)
        mt._download(zip_url, archive)
        mt._safe_extract(archive, extracted)
        _remove_macos_metadata(extracted)

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

        reference_stats = _reference_stats(source_path, refs)
        mapping_warnings = _mapping_warnings(inventory)
        for warning in mapping_warnings:
            log(f"taxonomy warning {warning['track']} -> {warning['mapped_category']}: {warning['warning']}")

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
                "routing_correlation_matrix": {},
                "routing_by_reference": {},
                "output_fingerprints": {},
            }
            if code == 0:
                generated_audio, generated_files, fingerprints = _normalise_generated(out, sr, root / "work")
                scores, matrix, routing = _score_outputs(generated_audio, generated_files, refs, sr)
                row["scores"] = scores
                row["routing_correlation_matrix"] = matrix
                row["routing_by_reference"] = routing
                row["output_fingerprints"] = fingerprints
            else:
                row["error_tail"] = "\n".join(stdout.splitlines()[-40:])
            rows.append(row)

            snapshot = {
                "ok": all(r["return_code"] == 0 for r in rows),
                "mode": "synthetic_multitrack_ground_truth_v2",
                "source_zip_url": zip_url,
                "sample_rate": sr,
                "global_scale": scale,
                "reference_integrity": integrity,
                "reference_stats": reference_stats,
                "inventory": inventory,
                "category_members": dict(categories),
                "mapping_warnings": mapping_warnings,
                "experiments_complete": len(rows),
                "experiments_total": len(experiments),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "parameter_effects": _parameter_effects(rows),
                "results": rows,
            }
            output.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
            log(f"checkpoint written: {output}")

        progress("Synthetic ground-truth benchmark complete", 100)
        result = json.loads(output.read_text(encoding="utf-8"))
        result["complete"] = True
        result["ok"] = all(r["return_code"] == 0 for r in rows)
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["parameter_effects"] = _parameter_effects(rows)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result


if __name__ == "__main__":
    result = run()
    output = Path(os.getenv("LITELABS_BENCHMARK_OUTPUT", "/workspace/litelabs-research/synthetic_ground_truth.json"))
    log(f"complete: {result.get('ok')} in {result.get('elapsed_seconds')}s")
    log(f"retrieve {output} before stopping the Pod")
    while True:
        time.sleep(3600)
