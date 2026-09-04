from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf

QA_VERSION = 1


def _read(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    return np.asarray(audio, dtype=np.float32), int(sr)


def _mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    return np.mean(audio, axis=1, dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n <= 0:
        return 0.0
    a = a[:n].astype(np.float64, copy=False)
    b = b[:n].astype(np.float64, copy=False)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _db(value: float) -> float:
    return 20.0 * math.log10(max(float(value), 1e-12))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _signal_metrics(audio: np.ndarray, sr: int) -> dict:
    mono = _mono(audio)
    if mono.size == 0:
        return {
            "rms_dbfs": -120.0,
            "peak_dbfs": -120.0,
            "active_ratio": 0.0,
            "clipping_fraction": 0.0,
            "spectral_centroid_hz": 0.0,
        }

    rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2) + 1e-12))
    peak = float(np.max(np.abs(mono)))

    # Cheap activity estimate using 50 ms RMS frames.
    frame = max(1, int(sr * 0.05))
    usable = (len(mono) // frame) * frame
    if usable:
        framed = mono[:usable].reshape(-1, frame).astype(np.float64)
        frame_rms = np.sqrt(np.mean(framed * framed, axis=1) + 1e-12)
        threshold = max(10 ** (-50.0 / 20.0), rms * 0.18)
        active_ratio = float(np.mean(frame_rms >= threshold))
    else:
        active_ratio = 1.0 if rms > 10 ** (-50.0 / 20.0) else 0.0

    clipping_fraction = float(np.mean(np.abs(mono) >= 0.999))

    # Spectral fingerprint on at most ~12 seconds of decimated audio. This keeps QA quick.
    if len(mono) > sr * 12:
        idx = np.linspace(0, len(mono) - 1, sr * 12, dtype=np.int64)
        sample = mono[idx]
    else:
        sample = mono
    if sample.size > 8192:
        sample = sample[:: max(1, sample.size // 8192)]
    if sample.size >= 64:
        spectrum = np.abs(np.fft.rfft(sample.astype(np.float64)))
        freqs = np.fft.rfftfreq(sample.size, d=1.0 / sr)
        denom = float(np.sum(spectrum))
        centroid = float(np.sum(freqs * spectrum) / denom) if denom > 1e-12 else 0.0
    else:
        centroid = 0.0

    return {
        "rms_dbfs": round(_db(rms), 3),
        "peak_dbfs": round(_db(peak), 3),
        "active_ratio": round(active_ratio, 6),
        "clipping_fraction": round(clipping_fraction, 8),
        "spectral_centroid_hz": round(centroid, 2),
    }


def _stem_score(metrics: dict, distinctness: float, reconstruction: float | None) -> tuple[float, dict]:
    rms_db = float(metrics.get("rms_dbfs", -120.0))
    active = float(metrics.get("active_ratio", 0.0))
    clipping = float(metrics.get("clipping_fraction", 0.0))

    # These are heuristic research signals, not source-separation ground truth.
    activity_score = _clamp01((active - 0.02) / 0.48)
    level_score = _clamp01((rms_db + 55.0) / 35.0)
    useful_signal = 0.65 * activity_score + 0.35 * level_score
    technical_health = _clamp01(1.0 - min(1.0, clipping * 250.0))
    distinctness_score = _clamp01(distinctness)
    reconstruction_score = _clamp01(reconstruction) if reconstruction is not None else 0.75

    score = (
        0.35 * reconstruction_score
        + 0.25 * distinctness_score
        + 0.20 * useful_signal
        + 0.20 * technical_health
    )
    components = {
        "reconstruction_integrity": round(reconstruction_score, 6),
        "separation_distinctness": round(distinctness_score, 6),
        "useful_signal": round(useful_signal, 6),
        "technical_health": round(technical_health, 6),
    }
    return round(_clamp01(score), 6), components


def build_research_qa(
    *,
    source: Path,
    stems: dict[str, Path],
    model_by_stem: dict[str, str],
    filename: str,
    input_size_bytes: int,
    input_format: str,
    genre: str,
    preset: str,
    pipeline_revision: str,
    job_id: object = None,
    extra: dict | None = None,
) -> dict:
    """Build a lightweight silent QA record for admin research.

    Scores are deliberately heuristic. Store the component metrics so later QA versions
    can be recalibrated without pretending the score is studio-multitrack ground truth.
    """
    source_audio, source_sr = _read(source)
    source_mono = _mono(source_audio)

    loaded: dict[str, np.ndarray] = {}
    metrics: dict[str, dict] = {}
    sample_rates: dict[str, int] = {}
    for stem, path in stems.items():
        if not path or not Path(path).is_file():
            continue
        audio, sr = _read(Path(path))
        loaded[stem] = _mono(audio)
        sample_rates[stem] = sr
        metrics[stem] = _signal_metrics(audio, sr)

    reconstruction_cosine: float | None = None
    if loaded:
        n = min([len(source_mono)] + [len(v) for v in loaded.values()])
        if n > 0:
            summed = np.sum(np.stack([v[:n] for v in loaded.values()], axis=0), axis=0)
            reconstruction_cosine = _clamp01(abs(_cosine(source_mono[:n], summed)))

    stem_records: dict[str, dict] = {}
    names = list(loaded)
    for name in names:
        target = loaded[name]
        others = [loaded[other] for other in names if other != name]
        if others:
            n = min([len(target)] + [len(v) for v in others])
            rest = np.sum(np.stack([v[:n] for v in others], axis=0), axis=0)
            leakage_corr = abs(_cosine(target[:n], rest))
            distinctness = _clamp01(1.0 - leakage_corr)
        else:
            leakage_corr = 0.0
            distinctness = 0.75

        score, components = _stem_score(metrics[name], distinctness, reconstruction_cosine)
        stem_records[name] = {
            "model": model_by_stem.get(name, "unknown"),
            "score": score,
            "components": components,
            "metrics": metrics[name],
            "correlation_with_other_stems": round(leakage_corr, 6),
        }

    record = {
        "qa_version": QA_VERSION,
        "score_type": "heuristic_research_signal",
        "ground_truth_available": False,
        "job_id": job_id,
        "filename": filename,
        "input_size_bytes": int(input_size_bytes),
        "input_format": str(input_format or "").lower(),
        "duration_seconds": round(len(source_mono) / max(source_sr, 1), 3),
        "genre": genre,
        "preset": preset,
        "pipeline_revision": pipeline_revision,
        "reconstruction_cosine": round(reconstruction_cosine, 6) if reconstruction_cosine is not None else None,
        "stems": stem_records,
    }
    if extra:
        record["pipeline_metrics"] = extra
    return record
