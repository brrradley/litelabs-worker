from __future__ import annotations

import hashlib
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
import soundfile as sf

import multitrack_ground_truth_campaign as mt
import synthetic_ground_truth_campaign as sgt
from drum_decomposition_v1 import _ensure_drumsep
from ground_truth_benchmark import _score

MODE = "experimental_child_ground_truth_v1"
CHILDREN = ("kick", "snare", "toms", "hh", "cymbals")
CURRENT_5_CONFIG = Path("/models/drumsep_5stem/mdx23c_drumsep_5stem_aufr33_jarredou_config.yaml")
CURRENT_5_CHECKPOINT = Path("/models/drumsep_5stem/mdx23c_drumsep_5stem_aufr33_jarredou.ckpt")
MSS_REPO = Path(os.getenv("LITELABS_MSS_REPO_DIR", "/opt/music-source-separation-training"))


def log(message: str) -> None:
    print(f"[LiteLABS experimental GT] {message}", flush=True)


def progress(message: str, percent: int) -> None:
    print(f"[LiteLABS experimental GT] {percent:3d}% {message}", flush=True)


def _safe_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def child_category(name: str) -> tuple[str | None, str | None]:
    """Map studio track names to drum children, returning an optional warning."""
    n = _safe_key(name)
    if "kick" in n or re.search(r"(^|_)kik($|_)", n):
        return "kick", None
    if "snare" in n or re.search(r"(^|_)sn($|_)", n):
        return "snare", None
    if "tom" in n:
        warning = "Track also contains percussion; toms reference is not perfectly isolated." if "perc" in n else None
        return "toms", warning
    if any(x in n for x in ["hihat", "hi_hat", "hi_hat", "hat", "hh"]):
        warning = "Track also contains agogo/percussion; hi-hat reference is not perfectly isolated." if "agogo" in n else None
        return "hh", warning
    if any(x in n for x in ["cymbal", "cymbals", "ride", "crash"]):
        return "cymbals", None
    return None, None


def _sum(items: list[np.ndarray]) -> np.ndarray:
    return mt._sum_tracks(items)


def _write(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sr, subtype="FLOAT")


def _sha256_audio(audio: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(audio, dtype=np.float32))
    return hashlib.sha256(a.tobytes()).hexdigest()


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n <= 0:
        return 0.0
    x = np.asarray(a[:n], dtype=np.float64).reshape(-1)
    y = np.asarray(b[:n], dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denom) if denom > 1e-12 else 0.0


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2) + 1e-12))


def _collect_outputs(root: Path, names: tuple[str, ...]) -> dict[str, Path]:
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".wav", ".flac"}]
    out: dict[str, Path] = {}
    for name in names:
        exact = [p for p in files if _safe_key(p.stem) == _safe_key(name)]
        fuzzy = [p for p in files if re.search(rf"(^|_){re.escape(_safe_key(name))}($|_)", _safe_key(p.stem))]
        matches = exact or fuzzy
        if matches:
            out[name] = matches[0]
    return out


def _run_mdx23c(input_audio: np.ndarray, sr: int, root: Path, label: str, config: Path, checkpoint: Path, timeout: int) -> tuple[dict[str, np.ndarray], float, str]:
    inp = root / label / "input"
    out = root / label / "output"
    inp.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    _write(inp / "drums.wav", input_audio, sr)
    cmd = [
        "python", str(MSS_REPO / "inference.py"),
        "--model_type", "mdx23c",
        "--config_path", str(config),
        "--start_check_point", str(checkpoint),
        "--input_folder", str(inp),
        "--store_dir", str(out),
        "--device_ids", "0",
        "--disable_detailed_pbar",
        "--filename_template", "{file_name}/{instr}",
    ]
    started = time.monotonic()
    proc = subprocess.run(cmd, cwd=MSS_REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    elapsed = time.monotonic() - started
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed: {' | '.join((proc.stdout or '').splitlines()[-20:])}")
    paths = _collect_outputs(out, ("kick", "snare", "toms", "hh", "cymbals", "ride", "crash"))
    loaded: dict[str, np.ndarray] = {}
    for name, path in paths.items():
        loaded[name] = mt._load(path)[0]
    if "cymbals" not in loaded and "ride" in loaded and "crash" in loaded:
        loaded["cymbals"] = _sum([loaded["ride"], loaded["crash"]])
    return {k: v for k, v in loaded.items() if k in CHILDREN}, elapsed, "\n".join((proc.stdout or "").splitlines()[-25:])


def _build_child_references(tracks: dict[str, np.ndarray], work: Path, sr: int) -> tuple[dict[str, Path], dict, list[dict]]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    members: dict[str, list[str]] = defaultdict(list)
    warnings: list[dict] = []
    for name, audio in tracks.items():
        child, warning = child_category(name)
        if child:
            grouped[child].append(audio)
            members[child].append(name)
            if warning:
                warnings.append({"track": name, "mapped_child": child, "warning": warning})
    refs: dict[str, Path] = {}
    stats: dict[str, dict] = {}
    for child in CHILDREN:
        if not grouped.get(child):
            stats[child] = {"available": False, "members": []}
            continue
        audio = _sum(grouped[child])
        path = work / "child_refs" / f"{child}.wav"
        _write(path, audio, sr)
        refs[child] = path
        stats[child] = {
            "available": True,
            "members": members[child],
            "rms_dbfs": round(20.0 * np.log10(max(_rms(audio), 1e-12)), 3),
            "peak": round(float(np.max(np.abs(audio))), 9),
        }
    return refs, stats, warnings


def _score_candidate(name: str, outputs: dict[str, np.ndarray], refs: dict[str, Path], parent: np.ndarray, sr: int, work: Path) -> dict:
    score_rows: dict[str, dict] = {}
    matrix: dict[str, dict[str, float]] = {}
    refs_audio = {k: mt._load(v)[0] for k, v in refs.items()}
    for out_name, audio in outputs.items():
        matrix[out_name] = {ref_name: round(_cos(audio, ref_audio), 6) for ref_name, ref_audio in refs_audio.items()}
        if out_name in refs_audio:
            try:
                score_rows[out_name] = {"ok": True, **_score(audio, refs_audio[out_name], sr)}
            except Exception as exc:
                score_rows[out_name] = {"ok": False, "error": str(exc), "error_type": exc.__class__.__name__}
    by_reference = {}
    for ref_name in refs_audio:
        choices = [(out_name, matrix.get(out_name, {}).get(ref_name, -1.0)) for out_name in outputs]
        best_name, best_corr = max(choices, key=lambda x: x[1]) if choices else (None, None)
        by_reference[ref_name] = {
            "best_output": best_name,
            "correlation": best_corr,
            "expected_output": ref_name,
            "landed_as_expected": best_name == ref_name,
        }

    if outputs:
        n = min([len(parent)] + [len(a) for a in outputs.values()])
        child_sum = _sum([a[:n] for a in outputs.values()])
        parent_n = parent[:n]
        residual = parent_n - child_sum
        parent_rms = _rms(parent_n)
        residual_db = 20.0 * np.log10(max(_rms(residual) / max(parent_rms, 1e-12), 1e-12))
        reconstruction = {
            "parent_vs_children_sum_correlation": round(_cos(parent_n, child_sum), 6),
            "residual_relative_to_parent_db": round(float(residual_db), 3),
        }
    else:
        reconstruction = None

    direct_corrs = [matrix[c][c] for c in refs_audio if c in matrix and c in matrix[c]]
    energy_weights = []
    weighted_values = []
    for child, ref_audio in refs_audio.items():
        if child in matrix and child in matrix[child]:
            w = _rms(ref_audio) ** 2
            energy_weights.append(w)
            weighted_values.append(w * matrix[child][child])
    macro_corr = float(np.mean(direct_corrs)) if direct_corrs else None
    weighted_corr = float(sum(weighted_values) / max(sum(energy_weights), 1e-12)) if weighted_values else None
    return {
        "candidate": name,
        "scores": score_rows,
        "routing_correlation_matrix": matrix,
        "routing_by_reference": by_reference,
        "reconstruction": reconstruction,
        "macro_direct_correlation": round(macro_corr, 6) if macro_corr is not None else None,
        "energy_weighted_direct_correlation": round(weighted_corr, 6) if weighted_corr is not None else None,
        "output_fingerprints": {k: _sha256_audio(v) for k, v in outputs.items()},
    }


def run() -> dict:
    zip_url = str(os.getenv("LITELABS_BENCHMARK_ZIP_URL", "")).strip()
    if not zip_url:
        raise RuntimeError("LITELABS_BENCHMARK_ZIP_URL is required")
    output = Path(os.getenv("LITELABS_BENCHMARK_OUTPUT", "/workspace/litelabs-research/experimental_ground_truth.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    timeout = max(300, min(3300, int(os.getenv("LITELABS_BENCHMARK_MODEL_TIMEOUT", "1800"))))
    sw_model_dir = Path(os.getenv("STEMFORGE_MODEL_DIR", "/models/bs_roformer_sw"))
    started = time.monotonic()

    if not CURRENT_5_CONFIG.is_file() or not CURRENT_5_CHECKPOINT.is_file():
        raise RuntimeError("Current experimental DrumSep 5-stem model is not installed in this image")
    six_config, six_checkpoint, _ = _ensure_drumsep(Path("/models/drumsep_mdx23c"))

    with tempfile.TemporaryDirectory(prefix="litelabs_exp_gt_") as temp:
        root = Path(temp)
        archive = root / (Path(urlparse(zip_url).path).name or "multitracks.zip")
        extracted = root / "extracted"
        work = root / "work"
        progress("Downloading studio multitrack pack", 2)
        mt._download(zip_url, archive)
        mt._safe_extract(archive, extracted)
        sgt._remove_macos_metadata(extracted)

        progress("Building exact synthetic source", 6)
        mt._track_category = sgt.track_category
        tracks, sr, inventory = mt._build_tracks(extracted, work)
        source_path, parent_refs, scale = mt._build_references(tracks, sr, work)
        integrity = sgt._reference_integrity(source_path, parent_refs, sr)
        if not integrity["exact_partition"]:
            raise RuntimeError(f"Synthetic reference integrity failed: {integrity}")
        child_refs, child_ref_stats, mapping_warnings = _build_child_references(tracks, work, sr)
        if not child_refs:
            raise RuntimeError("No usable experimental drum child references were found in this pack")
        log("available child references: " + ", ".join(child_refs.keys()))

        progress("Running production BS-RoFormer parent separation", 12)
        sw_config = sw_model_dir / "BS-Roformer-SW.yaml"
        sw_checkpoint = sw_model_dir / "BS-Roformer-SW.ckpt"
        if not sw_config.is_file() or not sw_checkpoint.is_file():
            raise RuntimeError("BS-Roformer-SW model/config missing")
        sw_source = root / "sw_source"
        sw_out = root / "sw_out"
        sw_source.mkdir(); sw_out.mkdir()
        source_audio, _ = mt._load(source_path)
        _write(sw_source / "studio_sum.wav", source_audio, sr)
        sw_started = time.monotonic()
        sw = subprocess.run([
            "bs-roformer-infer", "--config_path", str(sw_config), "--model_path", str(sw_checkpoint),
            "--input_folder", str(sw_source), "--store_dir", str(sw_out),
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        sw_elapsed = time.monotonic() - sw_started
        if sw.returncode != 0:
            raise RuntimeError("BS-RoFormer failed: " + " | ".join((sw.stdout or "").splitlines()[-30:]))
        stems = {}
        for p in sw_out.rglob("*.wav"):
            low = p.stem.lower()
            for target in ["vocals", "bass", "drums", "piano", "guitar", "other"]:
                if low.endswith("_" + target):
                    stems[target] = p
        if "drums" not in stems:
            raise RuntimeError("BS-RoFormer drums parent not found")
        parent_drums, _ = mt._load(stems["drums"])
        true_drums, _ = mt._load(parent_refs["drums"])
        parent_score = _score(parent_drums, true_drums, sr)

        candidates = []
        candidate_defs = [
            ("current_5stem_on_bs_parent", parent_drums, CURRENT_5_CONFIG, CURRENT_5_CHECKPOINT),
            ("current_5stem_on_full_mix", source_audio, CURRENT_5_CONFIG, CURRENT_5_CHECKPOINT),
            ("candidate_6stem_on_bs_parent", parent_drums, six_config, six_checkpoint),
            ("candidate_6stem_on_full_mix", source_audio, six_config, six_checkpoint),
        ]
        for idx, (label, audio_in, config, checkpoint) in enumerate(candidate_defs):
            progress(f"Testing {label}", 30 + idx * 15)
            child_audio, elapsed, runtime_tail = _run_mdx23c(audio_in, sr, root / "candidates", label, config, checkpoint, timeout)
            scored = _score_candidate(label, child_audio, child_refs, parent_drums, sr, work)
            scored["elapsed_seconds"] = round(elapsed, 3)
            scored["runtime_tail"] = runtime_tail
            scored["model_config"] = str(config)
            scored["model_checkpoint"] = str(checkpoint)
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
                "candidates_complete": len(candidates),
                "candidates_total": len(candidate_defs),
                "candidates": candidates,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            output.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
            log(f"checkpoint written: {output}")

        ranked = sorted(
            [{"candidate": c["candidate"], "energy_weighted_direct_correlation": c["energy_weighted_direct_correlation"], "macro_direct_correlation": c["macro_direct_correlation"]} for c in candidates],
            key=lambda x: (x["energy_weighted_direct_correlation"] if x["energy_weighted_direct_correlation"] is not None else -999),
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
    output = Path(os.getenv("LITELABS_BENCHMARK_OUTPUT", "/workspace/litelabs-research/experimental_ground_truth.json"))
    log(f"complete in {result.get('elapsed_seconds')}s")
    log(f"retrieve {output} before stopping the Pod")
    while True:
        time.sleep(3600)
