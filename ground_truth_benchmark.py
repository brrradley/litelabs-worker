from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests
import soundfile as sf


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with destination.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)


def _normalise_audio(source: Path, destination: Path, sample_rate: int = 44100) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(source), "-ar", str(sample_rate), "-ac", "2",
        "-c:a", "pcm_f32le", str(destination),
    ], check=True)


def _load(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return audio, sample_rate


def _mono_envelope(audio: np.ndarray, hop: int = 256) -> np.ndarray:
    mono = np.mean(audio, axis=1)
    usable = len(mono) - (len(mono) % hop)
    if usable <= 0:
        return np.zeros(1, dtype=np.float32)
    blocks = mono[:usable].reshape(-1, hop)
    return np.sqrt(np.mean(blocks * blocks, axis=1) + 1e-12)


def _best_offset(reference: np.ndarray, estimate: np.ndarray, sample_rate: int, max_shift_seconds: float = 0.75) -> int:
    hop = 256
    ref_env = _mono_envelope(reference, hop)
    est_env = _mono_envelope(estimate, hop)
    count = min(len(ref_env), len(est_env), 12000)
    if count < 8:
        return 0
    ref_env = ref_env[:count] - np.mean(ref_env[:count])
    est_env = est_env[:count] - np.mean(est_env[:count])
    max_shift = max(1, int(max_shift_seconds * sample_rate / hop))
    best_lag = 0
    best_score = -float("inf")
    for lag in range(-max_shift, max_shift + 1):
        if lag < 0:
            a, b = ref_env[-lag:], est_env[: count + lag]
        elif lag > 0:
            a, b = ref_env[: count - lag], est_env[lag:]
        else:
            a, b = ref_env, est_env
        if len(a) < 8:
            continue
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        score = float(np.dot(a, b) / denominator)
        if score > best_score:
            best_score = score
            best_lag = lag
    return int(best_lag * hop)


def _align(reference: np.ndarray, estimate: np.ndarray, offset: int) -> tuple[np.ndarray, np.ndarray]:
    if offset > 0:
        estimate = estimate[offset:]
    elif offset < 0:
        reference = reference[-offset:]
    length = min(len(reference), len(estimate))
    return reference[:length], estimate[:length]


def _safe_db(value: float) -> float:
    return 10.0 * math.log10(max(value, 1e-12))


def _score(reference: np.ndarray, estimate: np.ndarray, sample_rate: int) -> dict:
    offset = _best_offset(reference, estimate, sample_rate)
    reference, estimate = _align(reference, estimate, offset)
    if len(reference) < sample_rate:
        raise ValueError("Aligned audio is too short to score")

    ref = reference.reshape(-1).astype(np.float64)
    est = estimate.reshape(-1).astype(np.float64)
    ref_energy = float(np.dot(ref, ref)) + 1e-12

    gain = float(np.dot(est, ref) / ref_energy)
    polarity_flipped = gain < 0
    gain = abs(gain)
    if polarity_flipped:
        est = -est
    gain = float(np.dot(est, ref) / ref_energy)
    gain = gain if abs(gain) > 1e-12 else 1.0
    gain_matched = est / gain

    residual = ref - gain_matched
    residual_energy = float(np.dot(residual, residual)) + 1e-12
    correlation = float(np.corrcoef(ref, gain_matched)[0, 1]) if np.std(ref) > 1e-12 and np.std(gain_matched) > 1e-12 else 0.0
    si_target = (float(np.dot(gain_matched, ref)) / ref_energy) * ref
    si_noise = gain_matched - si_target
    si_sdr = _safe_db(float(np.dot(si_target, si_target)) / (float(np.dot(si_noise, si_noise)) + 1e-12))
    residual_db = _safe_db(residual_energy / ref_energy)
    nrmse = math.sqrt(residual_energy / ref_energy)

    segment = min(len(ref), sample_rate * 90 * 2)
    ref_fft = np.abs(np.fft.rfft(ref[:segment]))
    est_fft = np.abs(np.fft.rfft(gain_matched[:segment]))
    spectral_similarity = float(np.dot(ref_fft, est_fft) / ((np.linalg.norm(ref_fft) * np.linalg.norm(est_fft)) + 1e-12))

    quality_score = max(0.0, min(100.0,
        45.0 * max(0.0, min(1.0, correlation))
        + 25.0 * max(0.0, min(1.0, spectral_similarity))
        + 30.0 * max(0.0, min(1.0, (si_sdr + 10.0) / 30.0))
    ))

    return {
        "quality_score": round(quality_score, 2),
        "si_sdr_db": round(si_sdr, 3),
        "residual_db_vs_reference": round(residual_db, 3),
        "correlation": round(correlation, 6),
        "spectral_similarity": round(spectral_similarity, 6),
        "normalised_rmse": round(nrmse, 6),
        "alignment_offset_samples": offset,
        "alignment_offset_ms": round(offset * 1000.0 / sample_rate, 3),
        "estimated_gain_db": round(20.0 * math.log10(max(abs(gain), 1e-12)), 3),
        "polarity_flipped": polarity_flipped,
        "duration_scored_seconds": round(len(reference) / sample_rate, 3),
    }


def _url(base_url: str, relative_path: str) -> str:
    return base_url.rstrip("/") + "/" + quote(relative_path.lstrip("/"), safe="/")


def build_ground_truth_benchmark(payload: dict, progress=None) -> dict:
    base_url = str(payload.get("base_url") or "").strip()
    tests = payload.get("tests") or []
    if not base_url or not isinstance(tests, list) or not tests:
        return {"ok": False, "mode": "ground_truth_benchmark", "error": "base_url and tests[] are required"}

    results: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="litelabs_ground_truth_") as temp:
        root = Path(temp)
        for test_index, test in enumerate(tests):
            test_id = str(test.get("id") or f"test-{test_index + 1}")
            references = test.get("references") or {}
            candidates = test.get("candidates") or []
            if progress:
                progress(f"Preparing ground-truth test {test_id}", 5)

            reference_cache: dict[str, tuple[np.ndarray, int]] = {}
            for stem, path in references.items():
                raw = root / "raw" / test_id / "references" / f"{stem}_{Path(path).name}"
                wav = root / "wav" / test_id / "references" / f"{stem}.wav"
                wav.parent.mkdir(parents=True, exist_ok=True)
                _download(_url(base_url, str(path)), raw)
                _normalise_audio(raw, wav)
                reference_cache[str(stem)] = _load(wav)

            test_rows: list[dict] = []
            for candidate_index, candidate in enumerate(candidates):
                model = str(candidate.get("model") or "unknown")
                stem = str(candidate.get("stem") or "unknown")
                path = str(candidate.get("path") or "")
                row = {"model": model, "stem": stem, "path": path, "ok": False}
                try:
                    if stem not in reference_cache:
                        raise ValueError(f"No reference supplied for stem {stem}")
                    raw = root / "raw" / test_id / "candidates" / f"{candidate_index:03d}_{Path(path).name}"
                    wav = root / "wav" / test_id / "candidates" / f"{candidate_index:03d}.wav"
                    wav.parent.mkdir(parents=True, exist_ok=True)
                    _download(_url(base_url, path), raw)
                    _normalise_audio(raw, wav)
                    estimate, sample_rate = _load(wav)
                    reference, reference_rate = reference_cache[stem]
                    if sample_rate != reference_rate:
                        raise ValueError("Unexpected sample-rate mismatch after normalisation")
                    row.update(_score(reference, estimate, sample_rate))
                    row["ok"] = True
                except Exception as exc:
                    row["error"] = str(exc)
                    row["error_type"] = exc.__class__.__name__
                test_rows.append(row)
                if progress:
                    completed = candidate_index + 1
                    progress(f"Scored {model} {stem} on {test_id}", int(10 + 85 * completed / max(1, len(candidates))))

            leaderboards: dict[str, list[dict]] = {}
            for stem in references:
                ranked = [row for row in test_rows if row.get("ok") and row.get("stem") == stem]
                leaderboards[stem] = sorted(ranked, key=lambda row: (-float(row.get("quality_score") or 0.0), -float(row.get("si_sdr_db") or -999.0)))
            results.append({
                "id": test_id,
                "genre": test.get("genre"),
                "traits": test.get("traits") or [],
                "results": test_rows,
                "leaderboards": leaderboards,
            })

    return {
        "ok": True,
        "mode": "ground_truth_benchmark",
        "schema_version": 1,
        "test_count": len(results),
        "tests": results,
        "no_audio_exported": True,
        "metric_note": "Candidates are timing-, polarity- and gain-aligned to genuine reference stems before SI-SDR, residual, correlation and spectral scoring.",
    }
