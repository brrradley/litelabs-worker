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
import soundfile as sf
import torch

import multitrack_ground_truth_campaign as mt
import synthetic_ground_truth_campaign as sgt
from ground_truth_benchmark import _score

MODE = "experimental_drum_refinement_v1"
CHILDREN = ("kick", "snare", "toms", "hh", "cymbals")
CURRENT_5_CONFIG = Path("/models/drumsep_5stem/mdx23c_drumsep_5stem_aufr33_jarredou_config.yaml")
CURRENT_5_CHECKPOINT = Path("/models/drumsep_5stem/mdx23c_drumsep_5stem_aufr33_jarredou.ckpt")
MSS_REPO = Path(os.getenv("LITELABS_MSS_REPO_DIR", "/opt/music-source-separation-training"))


def log(message: str) -> None:
    print(f"[LiteLABS drum refine] {message}", flush=True)


def progress(message: str, percent: int) -> None:
    print(f"[LiteLABS drum refine] {percent:3d}% {message}", flush=True)


def _safe_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _write(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sr, subtype="FLOAT")


def _sum(items: list[np.ndarray]) -> np.ndarray:
    return mt._sum_tracks(items)


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2) + 1e-12))


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n <= 0:
        return 0.0
    x = np.asarray(a[:n], dtype=np.float64).reshape(-1)
    y = np.asarray(b[:n], dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denom) if denom > 1e-12 else 0.0


def child_category(name: str) -> tuple[str | None, str | None]:
    n = _safe_key(name)
    if "kick" in n or re.search(r"(^|_)kik($|_)", n):
        return "kick", None
    if "snare" in n or re.search(r"(^|_)sn($|_)", n):
        return "snare", None
    if "tom" in n:
        return "toms", "Track also contains percussion; toms reference is not perfectly isolated." if "perc" in n else None
    if any(x in n for x in ["hihat", "hi_hat", "hat", "hh"]):
        return "hh", "Track also contains agogo/percussion; hi-hat reference is not perfectly isolated." if "agogo" in n else None
    if any(x in n for x in ["cymbal", "cymbals", "ride", "crash"]):
        return "cymbals", None
    return None, None


def _build_child_refs(tracks: dict[str, np.ndarray], work: Path, sr: int):
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    members: dict[str, list[str]] = defaultdict(list)
    warnings = []
    for name, audio in tracks.items():
        child, warning = child_category(name)
        if child:
            grouped[child].append(audio)
            members[child].append(name)
            if warning:
                warnings.append({"track": name, "mapped_child": child, "warning": warning})
    refs = {}
    stats = {}
    for child in CHILDREN:
        if not grouped.get(child):
            stats[child] = {"available": False, "members": []}
            continue
        audio = _sum(grouped[child])
        path = work / "refs" / f"{child}.wav"
        _write(path, audio, sr)
        refs[child] = path
        stats[child] = {"available": True, "members": members[child], "rms_dbfs": round(20*np.log10(max(_rms(audio), 1e-12)), 3)}
    return refs, stats, warnings


def _collect_outputs(root: Path) -> dict[str, Path]:
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".wav", ".flac"}]
    out = {}
    for child in CHILDREN:
        exact = [p for p in files if _safe_key(p.stem) == _safe_key(child)]
        fuzzy = [p for p in files if re.search(rf"(^|_){re.escape(_safe_key(child))}($|_)", _safe_key(p.stem))]
        matches = exact or fuzzy
        if matches:
            out[child] = matches[0]
    return out


def _run_current_drumsep(parent: np.ndarray, sr: int, root: Path, timeout: int) -> tuple[dict[str, np.ndarray], float, str]:
    inp = root / "input"
    out = root / "output"
    inp.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    _write(inp / "drums.wav", parent, sr)
    cmd = [
        "python", str(MSS_REPO / "inference.py"), "--model_type", "mdx23c",
        "--config_path", str(CURRENT_5_CONFIG), "--start_check_point", str(CURRENT_5_CHECKPOINT),
        "--input_folder", str(inp), "--store_dir", str(out), "--device_ids", "0",
        "--disable_detailed_pbar", "--filename_template", "{file_name}/{instr}",
    ]
    started = time.monotonic()
    proc = subprocess.run(cmd, cwd=MSS_REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    elapsed = time.monotonic() - started
    if proc.returncode != 0:
        raise RuntimeError("current DrumSep failed: " + " | ".join((proc.stdout or "").splitlines()[-30:]))
    paths = _collect_outputs(out)
    missing = [c for c in CHILDREN if c not in paths]
    if missing:
        raise RuntimeError(f"current DrumSep missing outputs: {missing}; found={[p.name for p in out.rglob('*') if p.is_file()]}")
    return {k: mt._load(v)[0] for k, v in paths.items()}, elapsed, "\n".join((proc.stdout or "").splitlines()[-25:])


def _wiener_refine(parent: np.ndarray, children: dict[str, np.ndarray], power: float, n_fft: int, hop: int) -> dict[str, np.ndarray]:
    """Repartition the parent using masks derived from current child magnitudes.

    Masks sum to one at every TF bin, so the refinement redistributes leakage while
    remaining tied to the existing high-quality parent rather than inventing audio.
    """
    n = min([len(parent)] + [len(children[c]) for c in CHILDREN])
    p = np.asarray(parent[:n], dtype=np.float32)
    c = {k: np.asarray(v[:n], dtype=np.float32) for k, v in children.items()}
    window = torch.hann_window(n_fft, periodic=True)
    result = {k: np.zeros_like(p, dtype=np.float32) for k in CHILDREN}
    eps = 1e-10
    with torch.no_grad():
        for ch in range(2):
            parent_t = torch.from_numpy(p[:, ch])
            parent_spec = torch.stft(parent_t, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, center=True, return_complex=True)
            mags = []
            for name in CHILDREN:
                t = torch.from_numpy(c[name][:, ch])
                s = torch.stft(t, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, center=True, return_complex=True)
                mags.append(torch.abs(s).clamp_min(eps).pow(power))
            den = torch.stack(mags, dim=0).sum(dim=0).clamp_min(eps)
            for idx, name in enumerate(CHILDREN):
                mask = mags[idx] / den
                spec = parent_spec * mask
                wav = torch.istft(spec, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, center=True, length=n)
                result[name][:, ch] = wav.cpu().numpy().astype(np.float32)
    return result


def _score_candidate(label: str, outputs: dict[str, np.ndarray], refs: dict[str, Path], parent: np.ndarray, sr: int) -> dict:
    refs_audio = {k: mt._load(v)[0] for k, v in refs.items()}
    scores = {}
    matrix = {}
    for out_name, audio in outputs.items():
        matrix[out_name] = {ref: round(_cos(audio, ra), 6) for ref, ra in refs_audio.items()}
        if out_name in refs_audio:
            try:
                scores[out_name] = {"ok": True, **_score(audio, refs_audio[out_name], sr)}
            except Exception as exc:
                scores[out_name] = {"ok": False, "error": str(exc), "error_type": exc.__class__.__name__}
    by_ref = {}
    for ref in refs_audio:
        choices = [(out_name, matrix.get(out_name, {}).get(ref, -1.0)) for out_name in outputs]
        best_name, best_corr = max(choices, key=lambda x: x[1]) if choices else (None, None)
        by_ref[ref] = {"best_output": best_name, "correlation": best_corr, "expected_output": ref, "landed_as_expected": best_name == ref}
    n = min([len(parent)] + [len(outputs[c]) for c in outputs])
    rebuilt = _sum([outputs[c][:n] for c in outputs])
    parent_n = parent[:n]
    residual = parent_n - rebuilt
    residual_db = 20*np.log10(max(_rms(residual)/max(_rms(parent_n), 1e-12), 1e-12))
    direct = [matrix[c][c] for c in refs_audio if c in matrix and c in matrix[c]]
    weights = []
    weighted = []
    for c, ref_audio in refs_audio.items():
        if c in matrix and c in matrix[c]:
            w = _rms(ref_audio)**2
            weights.append(w)
            weighted.append(w*matrix[c][c])
    return {
        "candidate": label,
        "scores": scores,
        "routing_correlation_matrix": matrix,
        "routing_by_reference": by_ref,
        "macro_direct_correlation": round(float(np.mean(direct)), 6) if direct else None,
        "energy_weighted_direct_correlation": round(float(sum(weighted)/max(sum(weights), 1e-12)), 6) if weighted else None,
        "reconstruction": {"parent_vs_children_sum_correlation": round(_cos(parent_n, rebuilt), 6), "residual_relative_to_parent_db": round(float(residual_db), 3)},
    }


def run() -> dict:
    zip_url = str(os.getenv("LITELABS_BENCHMARK_ZIP_URL", "")).strip()
    if not zip_url:
        raise RuntimeError("LITELABS_BENCHMARK_ZIP_URL is required")
    output = Path(os.getenv("LITELABS_BENCHMARK_OUTPUT", "/workspace/litelabs-research/drum_refinement.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    timeout = max(300, min(3300, int(os.getenv("LITELABS_BENCHMARK_MODEL_TIMEOUT", "1800"))))
    model_dir = Path(os.getenv("LITELABS_AUDIO_SEPARATOR_MODEL_DIR", "/models/audio_separator"))
    started = time.monotonic()

    if not CURRENT_5_CONFIG.is_file() or not CURRENT_5_CHECKPOINT.is_file():
        raise RuntimeError("Current 5-stem DrumSep model missing from research image")

    with tempfile.TemporaryDirectory(prefix="litelabs_drum_refine_") as temp:
        root = Path(temp)
        archive = root / (Path(urlparse(zip_url).path).name or "multitracks.zip")
        extracted = root / "extracted"
        work = root / "work"
        progress("Downloading studio multitrack pack", 2)
        mt._download(zip_url, archive)
        mt._safe_extract(archive, extracted)
        sgt._remove_macos_metadata(extracted)

        progress("Building exact synthetic source and child references", 7)
        mt._track_category = sgt.track_category
        tracks, sr, inventory = mt._build_tracks(extracted, work)
        source_path, parent_refs, scale = mt._build_references(tracks, sr, work)
        integrity = sgt._reference_integrity(source_path, parent_refs, sr)
        if not integrity["exact_partition"]:
            raise RuntimeError(f"Synthetic reference integrity failed: {integrity}")
        child_refs, child_stats, mapping_warnings = _build_child_refs(tracks, work, sr)
        if not child_refs:
            raise RuntimeError("No usable drum child references in pack")

        progress("Running fixed BS-RoFormer parent separation", 14)
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

        progress("Running current Experimental DrumSep", 30)
        baseline_children, drumsep_elapsed, drumsep_tail = _run_current_drumsep(parent, sr, root / "drumsep", timeout)
        candidates = [_score_candidate("baseline_current_5stem", baseline_children, child_refs, parent, sr)]

        powers = [1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
        fft_configs = [(2048, 512), (4096, 1024)]
        total = len(powers) * len(fft_configs)
        idx = 0
        for n_fft, hop in fft_configs:
            for power in powers:
                idx += 1
                pct = 38 + int(50 * idx / total)
                label = f"wiener_p{power:g}_fft{n_fft}"
                progress(f"Testing {label}", pct)
                refined = _wiener_refine(parent, baseline_children, power=power, n_fft=n_fft, hop=hop)
                candidates.append(_score_candidate(label, refined, child_refs, parent, sr))

        ranked = sorted(
            [{"candidate": c["candidate"], "energy_weighted_direct_correlation": c["energy_weighted_direct_correlation"], "macro_direct_correlation": c["macro_direct_correlation"], "reconstruction": c["reconstruction"]} for c in candidates],
            key=lambda x: x["energy_weighted_direct_correlation"] if x["energy_weighted_direct_correlation"] is not None else -999,
            reverse=True,
        )
        result = {
            "ok": True,
            "mode": MODE,
            "source_zip_url": zip_url,
            "sample_rate": sr,
            "global_scale": scale,
            "reference_integrity": integrity,
            "inventory": inventory,
            "child_reference_stats": child_stats,
            "mapping_warnings": mapping_warnings,
            "bs_parent_elapsed_seconds": round(sw_elapsed, 3),
            "drumsep_elapsed_seconds": round(drumsep_elapsed, 3),
            "drumsep_runtime_tail": drumsep_tail,
            "candidates": candidates,
            "ranking_by_energy_weighted_correlation": ranked,
            "complete": True,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        progress("Experimental drum refinement benchmark complete", 100)
        log(f"retrieve {output} before stopping the Pod")
        return result


if __name__ == "__main__":
    result = run()
    log(f"complete in {result.get('elapsed_seconds')}s")
    while True:
        time.sleep(3600)
