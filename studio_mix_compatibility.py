from __future__ import annotations

import json
import math
import subprocess
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests

from ground_truth_benchmark import _load, _normalise_audio, _score

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".aiff", ".aif"}


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with destination.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)


def _safe_extract(zip_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            name = member.filename.replace("\\", "/")
            if name.startswith("/") or "../" in name:
                continue
            archive.extract(member, destination)


def _valid_audio_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        relative = path.relative_to(root).as_posix()
        if relative.startswith("__MACOSX/") or path.name.startswith("._"):
            continue
        if path.stat().st_size < 4096:
            continue
        files.append(path)
    return sorted(files)


def _match_file(files: list[Path], requested: str) -> Path | None:
    requested_lower = requested.lower().replace("\\", "/")
    exact = [p for p in files if p.relative_to(p.parents[len(p.parts) - len(p.parts)] if False else p.parent).as_posix().lower() == requested_lower]
    # Match by full relative suffix first, then basename. This supports nested ZIP roots.
    for path in files:
        if path.as_posix().lower().endswith(requested_lower):
            return path
    requested_name = Path(requested).name.lower()
    matches = [p for p in files if p.name.lower() == requested_name]
    return matches[0] if len(matches) == 1 else None


def _run(cmd: list[str]) -> None:
    completed = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError((completed.stdout or "ffmpeg failed")[-4000:])


def _mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float64, copy=False)
    return np.mean(audio, axis=1, dtype=np.float64)


def _best_offset(reference: np.ndarray, estimate: np.ndarray, max_shift: int) -> tuple[int, float]:
    ref = _mono(reference)
    est = _mono(estimate)
    length = min(len(ref), len(est))
    if length < 4096:
        return 0, 0.0
    ref = ref[:length]
    est = est[:length]
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    ref_norm = float(np.linalg.norm(ref)) + 1e-12
    best_offset = 0
    best_corr = -1.0
    step = max(1, max_shift // 256)
    for offset in range(-max_shift, max_shift + 1, step):
        if offset >= 0:
            a = ref[offset:]
            b = est[: len(a)]
        else:
            b = est[-offset:]
            a = ref[: len(b)]
        if len(a) < 4096:
            continue
        denom = (float(np.linalg.norm(a)) * float(np.linalg.norm(b))) + 1e-12
        corr = abs(float(np.dot(a, b)) / denom)
        if corr > best_corr:
            best_corr = corr
            best_offset = offset
    return best_offset, max(0.0, best_corr)


def _window_alignment(reference: np.ndarray, estimate: np.ndarray, sample_rate: int, centre_fraction: float, window_seconds: float = 20.0) -> dict:
    total = min(len(reference), len(estimate))
    window = min(total, int(window_seconds * sample_rate))
    centre = int(total * centre_fraction)
    start = max(0, min(total - window, centre - window // 2))
    ref_window = reference[start : start + window]
    est_window = estimate[start : start + window]
    max_shift = int(sample_rate * 0.75)
    offset, corr = _best_offset(ref_window, est_window, max_shift)
    return {
        "position_fraction": centre_fraction,
        "position_seconds": round(start / sample_rate, 3),
        "offset_samples": int(offset),
        "offset_ms": round(offset * 1000.0 / sample_rate, 3),
        "absolute_correlation": round(corr, 6),
    }


def build_studio_mix_compatibility(payload: dict, progress=None) -> dict:
    source_url = str(payload.get("source_url") or "").strip()
    official_url = str(payload.get("official_url") or "").strip()
    official_stems = payload.get("official_stems") or [
        "Bass.flac",
        "Drum OH.flac",
        "Guitar.flac",
        "Lead Vocal.flac",
        "Piano.flac",
    ]
    if not source_url or not official_url:
        return {"ok": False, "mode": "studio_mix_compatibility", "error": "source_url and official_url are required"}
    if not isinstance(official_stems, list) or not official_stems:
        return {"ok": False, "mode": "studio_mix_compatibility", "error": "official_stems must be a non-empty list"}

    with tempfile.TemporaryDirectory(prefix="litelabs_studio_compat_") as temp:
        root = Path(temp)
        source_raw = root / (Path(urlparse(source_url).path).name or "source.flac")
        official_zip = root / "official.zip"
        official_root = root / "official"
        official_root.mkdir(parents=True, exist_ok=True)

        if progress:
            progress("Downloading submitted source mix", 5)
        _download(source_url, source_raw)
        if progress:
            progress("Downloading official studio stems", 15)
        _download(official_url, official_zip)
        _safe_extract(official_zip, official_root)
        files = _valid_audio_files(official_root)

        selected: list[Path] = []
        missing: list[str] = []
        for requested in official_stems:
            matched = _match_file(files, str(requested))
            if matched is None:
                missing.append(str(requested))
            else:
                selected.append(matched)
        if missing:
            return {
                "ok": False,
                "mode": "studio_mix_compatibility",
                "error": "Missing requested official stems",
                "missing": missing,
                "available_files": [p.relative_to(official_root).as_posix() for p in files],
            }

        source_wav = root / "normalised" / "source.wav"
        _normalise_audio(source_raw, source_wav)
        normalised_stems: list[Path] = []
        for index, stem in enumerate(selected):
            destination = root / "normalised" / f"stem_{index:02d}.wav"
            _normalise_audio(stem, destination)
            normalised_stems.append(destination)

        reconstructed = root / "reconstructed.wav"
        inputs: list[str] = []
        for stem in normalised_stems:
            inputs.extend(["-i", str(stem)])
        filter_complex = f"amix=inputs={len(normalised_stems)}:duration=longest:normalize=0"
        if progress:
            progress("Reconstructing official studio mix", 35)
        _run(["ffmpeg", "-y", *inputs, "-filter_complex", filter_complex, "-c:a", "pcm_f32le", str(reconstructed)])

        source_audio, source_rate = _load(source_wav)
        reconstructed_audio, reconstructed_rate = _load(reconstructed)
        if source_rate != reconstructed_rate:
            raise ValueError("Sample-rate mismatch after normalisation")

        if progress:
            progress("Measuring timing and master compatibility", 60)
        fractions = [0.05, 0.25, 0.50, 0.75, 0.95]
        windows = [_window_alignment(source_audio, reconstructed_audio, source_rate, fraction) for fraction in fractions]
        offsets = [float(item["offset_ms"]) for item in windows]
        drift_ms = offsets[-1] - offsets[0] if len(offsets) >= 2 else 0.0
        duration_source = len(source_audio) / source_rate
        duration_reconstructed = len(reconstructed_audio) / reconstructed_rate
        speed_ratio = duration_reconstructed / duration_source if duration_source else 1.0
        ppm_difference = (speed_ratio - 1.0) * 1_000_000.0

        global_score = _score(source_audio, reconstructed_audio, source_rate)
        median_window_corr = float(np.median([item["absolute_correlation"] for item in windows]))
        fixed_offset = float(np.median(offsets))
        timing_stable = max(offsets) - min(offsets) <= 15.0
        likely_same_performance = median_window_corr >= 0.35
        directly_usable = timing_stable and median_window_corr >= 0.70
        usable_after_compensation = likely_same_performance and (abs(drift_ms) <= 250.0 or abs(ppm_difference) <= 1500.0)

        if directly_usable:
            verdict = "compatible_directly"
            recommendation = "The official stems reconstruct the submitted source closely enough for direct ground-truth scoring after fixed-offset alignment."
        elif usable_after_compensation:
            verdict = "compatible_after_alignment_or_speed_compensation"
            recommendation = "The official stems appear related to the submitted source, but timing/master compensation is required before stem scoring."
        else:
            verdict = "not_reliable_ground_truth_for_this_source"
            recommendation = "The official reconstruction does not match the submitted source closely enough for trustworthy waveform ground-truth scoring."

        return {
            "ok": True,
            "mode": "studio_mix_compatibility",
            "schema_version": 1,
            "source_url": source_url,
            "official_url": official_url,
            "official_stems_used": [p.relative_to(official_root).as_posix() for p in selected],
            "source_duration_seconds": round(duration_source, 6),
            "reconstructed_duration_seconds": round(duration_reconstructed, 6),
            "duration_difference_ms": round((duration_reconstructed - duration_source) * 1000.0, 3),
            "estimated_speed_ratio": round(speed_ratio, 9),
            "estimated_speed_difference_ppm": round(ppm_difference, 3),
            "window_alignment": windows,
            "median_window_correlation": round(median_window_corr, 6),
            "median_fixed_offset_ms": round(fixed_offset, 3),
            "estimated_drift_ms_across_track": round(drift_ms, 3),
            "timing_stable": timing_stable,
            "global_score_without_compensation": global_score,
            "verdict": verdict,
            "recommendation": recommendation,
            "no_audio_exported": True,
        }
