from __future__ import annotations

import os
import re
import subprocess
import tempfile
import threading
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
import soundfile as sf

from ground_truth_benchmark import _score

AUDIO_EXTS = {".wav", ".aif", ".aiff", ".flac"}
DEFAULT_EXPERIMENTS = [
    {"id": "prod_base", "model": "BS-Roformer-SW.ckpt", "segment": 256, "overlap": 8, "batch_size": 1, "autocast": True},
    {"id": "overlap_12", "model": "BS-Roformer-SW.ckpt", "segment": 256, "overlap": 12, "batch_size": 1, "autocast": True},
    {"id": "overlap_16", "model": "BS-Roformer-SW.ckpt", "segment": 256, "overlap": 16, "batch_size": 1, "autocast": True},
    {"id": "segment_384", "model": "BS-Roformer-SW.ckpt", "segment": 384, "overlap": 8, "batch_size": 1, "autocast": True},
    {"id": "segment_512", "model": "BS-Roformer-SW.ckpt", "segment": 512, "overlap": 8, "batch_size": 1, "autocast": True},
    {"id": "segment_512_overlap_12", "model": "BS-Roformer-SW.ckpt", "segment": 512, "overlap": 12, "batch_size": 1, "autocast": True},
    {"id": "fp32", "model": "BS-Roformer-SW.ckpt", "segment": 256, "overlap": 8, "batch_size": 1, "autocast": False},
]
DEFAULT_TARGETS = ["vocals", "bass", "drums", "piano", "guitar", "other"]


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=600) as response:
        response.raise_for_status()
        with destination.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)


def _safe_extract(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        root = destination.resolve()
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        zf.extractall(destination)


def _normalise_file(source: Path, destination: Path, sample_rate: int = 44100) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(source), "-ar", str(sample_rate), "-ac", "2",
        "-c:a", "pcm_f32le", str(destination),
    ], check=True)


def _load(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return np.asarray(audio, dtype=np.float32), int(sr)


def _write(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="FLOAT")


def _base_name(path: Path) -> str:
    name = path.stem
    name = re.sub(r"\.(L|R)$", "", name, flags=re.I)
    return name.strip()


def _pair_key(path: Path) -> tuple[str, str | None]:
    stem = path.stem
    match = re.search(r"\.(L|R)$", stem, flags=re.I)
    side = match.group(1).upper() if match else None
    base = re.sub(r"\.(L|R)$", "", stem, flags=re.I).strip().lower()
    return base, side


def _track_category(name: str) -> str:
    n = name.lower()
    if any(x in n for x in ["lead vocal", "background vocal", "pitched voc", "breath", "robot", "scream"]):
        return "vocals"
    if any(x in n for x in ["bass"]):
        return "bass"
    if any(x in n for x in ["kik", "kick", "sn_", "snare", "hat", "clap"]):
        return "drums"
    if any(x in n for x in ["pno", "piano", "org", "arp"]):
        return "piano"
    if any(x in n for x in ["gtr", "guitar"]):
        return "guitar"
    return "other"


def _build_tracks(extracted: Path, work: Path, sample_rate: int = 44100) -> tuple[dict[str, np.ndarray], int, list[dict]]:
    wavs = [p for p in extracted.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"]
    grouped: dict[str, dict[str, Path]] = defaultdict(dict)
    singles: list[Path] = []
    for path in wavs:
        key, side = _pair_key(path)
        if side:
            grouped[key][side] = path
        else:
            singles.append(path)

    tracks: dict[str, np.ndarray] = {}
    inventory: list[dict] = []
    sr_out = sample_rate

    for key, sides in sorted(grouped.items()):
        if "L" not in sides or "R" not in sides:
            for side, path in sides.items():
                singles.append(path)
            continue
        lnorm = work / "normalised" / f"{key}_L.wav"
        rnorm = work / "normalised" / f"{key}_R.wav"
        _normalise_file(sides["L"], lnorm, sample_rate)
        _normalise_file(sides["R"], rnorm, sample_rate)
        la, lsr = _load(lnorm)
        ra, rsr = _load(rnorm)
        if lsr != rsr:
            raise ValueError(f"Sample rate mismatch in pair {key}")
        n = max(len(la), len(ra))
        stereo = np.zeros((n, 2), dtype=np.float32)
        stereo[: len(la), 0] = la[:, 0]
        stereo[: len(ra), 1] = ra[:, 0]
        tracks[key] = stereo
        inventory.append({"name": key, "category": _track_category(key), "source": [sides["L"].name, sides["R"].name]})
        sr_out = lsr

    for index, path in enumerate(sorted(singles)):
        key = _base_name(path).lower()
        norm = work / "normalised" / f"single_{index:03d}.wav"
        _normalise_file(path, norm, sample_rate)
        audio, sr = _load(norm)
        tracks[key] = audio
        inventory.append({"name": key, "category": _track_category(key), "source": [path.name]})
        sr_out = sr

    if not tracks:
        raise ValueError("No multitrack WAV files found in ZIP")
    return tracks, sr_out, inventory


def _sum_tracks(items: list[np.ndarray]) -> np.ndarray:
    if not items:
        return np.zeros((1, 2), dtype=np.float32)
    n = max(len(x) for x in items)
    out = np.zeros((n, 2), dtype=np.float64)
    for item in items:
        out[: len(item)] += item.astype(np.float64)
    return out.astype(np.float32)


def _build_references(tracks: dict[str, np.ndarray], sr: int, work: Path) -> tuple[Path, dict[str, Path], float]:
    by_category: dict[str, list[np.ndarray]] = defaultdict(list)
    for name, audio in tracks.items():
        by_category[_track_category(name)].append(audio)

    all_sum = _sum_tracks(list(tracks.values()))
    peak = float(np.max(np.abs(all_sum))) if all_sum.size else 0.0
    global_scale = 0.98 / peak if peak > 0.98 else 1.0
    all_sum = all_sum * global_scale

    source_path = work / "references" / "studio_sum.wav"
    _write(source_path, all_sum, sr)

    refs: dict[str, Path] = {}
    for category in ["vocals", "bass", "drums", "piano", "guitar", "other"]:
        audio = _sum_tracks(by_category.get(category, [])) * global_scale
        path = work / "references" / f"{category}.wav"
        _write(path, audio, sr)
        refs[category] = path

    instrumental = _sum_tracks([audio for name, audio in tracks.items() if _track_category(name) != "vocals"]) * global_scale
    ipath = work / "references" / "instrumental.wav"
    _write(ipath, instrumental, sr)
    refs["instrumental"] = ipath
    return source_path, refs, global_scale


def _gpu_used_mib() -> int | None:
    try:
        completed = subprocess.run(["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5)
        values = [int(line.strip()) for line in completed.stdout.splitlines() if line.strip().isdigit()]
        return sum(values) if values else 0
    except Exception:
        return None


def _run(cmd: list[str], timeout: int) -> tuple[int, str, float, int | None]:
    peak = _gpu_used_mib()
    stop = threading.Event()
    def monitor():
        nonlocal peak
        while not stop.wait(0.25):
            value = _gpu_used_mib()
            if value is not None:
                peak = value if peak is None else max(peak, value)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        completed = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        return completed.returncode, completed.stdout or "", time.monotonic() - started, peak
    finally:
        stop.set(); thread.join(timeout=2)


def _find_generated(output_dir: Path, stem: str) -> Path | None:
    aliases = {stem, "drum" if stem == "drums" else stem, "vocal" if stem == "vocals" else stem}
    files = [p for p in output_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".wav", ".flac"}]
    for path in files:
        tokens = [x.strip().lower() for x in re.findall(r"\(([^)]+)\)", path.name)]
        if any(alias in tokens for alias in aliases):
            return path
    for path in files:
        low = path.name.lower()
        if any(re.search(rf"(?:^|[_\s-]){re.escape(alias)}(?:[_\s.-]|$)", low) for alias in aliases):
            return path
    return None


def _normalise_experiments(value) -> list[dict]:
    source = value if isinstance(value, list) and value else DEFAULT_EXPERIMENTS
    result = []
    for i, item in enumerate(source):
        if not isinstance(item, dict):
            continue
        result.append({
            "id": str(item.get("id") or f"experiment_{i:02d}"),
            "model": str(item.get("model") or "BS-Roformer-SW.ckpt"),
            "segment": max(32, min(4096, int(item.get("segment") or 256))),
            "overlap": max(2, min(50, int(item.get("overlap") or 8))),
            "batch_size": max(1, min(16, int(item.get("batch_size") or 1))),
            "autocast": bool(item.get("autocast", True)),
        })
    return result or list(DEFAULT_EXPERIMENTS)


def build_multitrack_ground_truth_campaign(payload: dict, progress=None) -> dict:
    zip_url = str(payload.get("zip_url") or payload.get("url") or "").strip()
    if not zip_url:
        return {"ok": False, "mode": "multitrack_ground_truth_campaign", "error": "zip_url is required"}
    targets = [str(x).lower().strip() for x in (payload.get("targets") or DEFAULT_TARGETS)]
    targets = [x for x in targets if x in DEFAULT_TARGETS]
    experiments = _normalise_experiments(payload.get("experiments"))
    pairs = [(target, exp) for target in targets for exp in experiments]
    cursor = max(0, int(payload.get("cursor") or 0))
    max_runs = max(1, min(100, int(payload.get("max_runs") or 12)))
    time_budget = max(120, min(3300, int(payload.get("time_budget_seconds") or 3000)))
    timeout = max(300, min(3300, int(payload.get("model_timeout_seconds") or 1800)))
    end = min(len(pairs), cursor + max_runs)
    model_dir = Path(os.getenv("LITELABS_AUDIO_SEPARATOR_MODEL_DIR", "/models/audio_separator"))
    model_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    started_campaign = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="litelabs_multitrack_gt_") as temp:
        root = Path(temp)
        archive = root / (Path(urlparse(zip_url).path).name or "multitracks.zip")
        extracted = root / "extracted"
        if progress: progress("Downloading multitrack benchmark pack", 2)
        _download(zip_url, archive)
        _safe_extract(archive, extracted)
        if progress: progress("Building studio references", 5)
        tracks, sr, inventory = _build_tracks(extracted, root / "work")
        source_path, refs, scale = _build_references(tracks, sr, root / "work")

        masters = [p for p in extracted.rglob("*") if p.is_file() and p.suffix.lower() in {".aif", ".aiff"}]
        master_compatibility = None
        master_name = None
        if masters:
            master = max(masters, key=lambda p: p.stat().st_size)
            master_name = master.name
            master_wav = root / "master.wav"
            _normalise_file(master, master_wav, sr)
            master_audio, _ = _load(master_wav)
            studio_audio, _ = _load(source_path)
            try:
                master_compatibility = _score(studio_audio, master_audio, sr)
            except Exception as exc:
                master_compatibility = {"error": str(exc), "error_type": exc.__class__.__name__}

        for position in range(cursor, end):
            if position > cursor and time.monotonic() - started_campaign >= time_budget:
                end = position
                break
            target, exp = pairs[position]
            out = root / "outputs" / f"{position:03d}"
            out.mkdir(parents=True, exist_ok=True)
            if progress:
                progress(f"{target}: {exp['id']}", int(8 + 88 * (position - cursor) / max(1, end - cursor)))
            cmd = [
                "audio-separator", str(source_path),
                "--model_filename", exp["model"],
                "--model_file_dir", str(model_dir),
                "--output_dir", str(out),
                "--output_format", "FLAC",
                "--single_stem", target.capitalize(),
                "--mdxc_segment_size", str(exp["segment"]),
                "--mdxc_overlap", str(exp["overlap"]),
                "--mdxc_batch_size", str(exp["batch_size"]),
            ]
            if exp["autocast"]:
                cmd.append("--use_autocast")
            row = {"pair_index": position, "target_stem": target, "experiment_id": exp["id"], "experiment": exp, "ok": False}
            try:
                rc, output, runtime, peak = _run(cmd, timeout)
                row.update({"runtime_seconds": round(runtime, 3), "peak_gpu_memory_mib": peak, "returncode": rc, "log_tail": output[-4000:]})
                if rc != 0:
                    raise RuntimeError(f"audio-separator exited with code {rc}")
                generated = _find_generated(out, target)
                if generated is None:
                    raise FileNotFoundError(f"Could not identify generated {target} file")
                generated_wav = root / "scored" / f"{position:03d}_{target}.wav"
                _normalise_file(generated, generated_wav, sr)
                estimate, est_sr = _load(generated_wav)
                reference, ref_sr = _load(refs[target])
                if est_sr != ref_sr:
                    raise ValueError("Sample-rate mismatch after normalisation")
                row["score"] = _score(reference, estimate, est_sr)
                row["ok"] = True
            except Exception as exc:
                row.update({"error": str(exc), "error_type": exc.__class__.__name__})
            rows.append(row)

    leaderboards = {}
    for target in targets:
        entries = [r for r in rows if r.get("ok") and r["target_stem"] == target]
        leaderboards[target] = sorted(entries, key=lambda r: (-float((r.get("score") or {}).get("quality_score") or 0), float(r.get("runtime_seconds") or 999999)))
    next_cursor = end if end < len(pairs) else None
    return {
        "ok": True,
        "mode": "multitrack_ground_truth_campaign",
        "schema_version": 1,
        "zip_url": zip_url,
        "master_file": master_name,
        "master_vs_studio_sum": master_compatibility,
        "reference_scale": round(scale, 8),
        "track_inventory": inventory,
        "targets": targets,
        "experiments": experiments,
        "cursor": cursor,
        "processed_runs": len(rows),
        "total_runs": len(pairs),
        "next_cursor": next_cursor,
        "complete": next_cursor is None,
        "results": rows,
        "leaderboards_by_stem": leaderboards,
        "metric_note": "Every candidate is scored against a reference stem built directly from the supplied multitracks. The separator input is the exact summed multitrack mix, so reference/source correspondence is deterministic.",
        "master_note": "The supplied AIF is scored against the summed multitrack mix separately; this indicates how closely the multitrack pack matches the mastered source.",
        "no_audio_exported": True,
    }
