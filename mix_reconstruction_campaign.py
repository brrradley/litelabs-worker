from __future__ import annotations

import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
import soundfile as sf

from ground_truth_benchmark import _score

AUDIO_EXTS = {'.wav', '.aif', '.aiff', '.flac', '.mp3', '.m4a'}


def _progress(cb, message: str, percent: int) -> None:
    if cb:
        cb(message, percent)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=600) as response:
        response.raise_for_status()
        with destination.open('wb') as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)


def _safe_extract(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        root = destination.resolve()
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f'Unsafe ZIP member: {member.filename}')
        zf.extractall(destination)


def _real_audio(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
        return False
    if path.name.startswith('._'):
        return False
    if any(part == '__MACOSX' for part in path.parts):
        return False
    return True


def _normalise(source: Path, destination: Path, sr: int = 44100) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-i', str(source),
        '-ar', str(sr), '-ac', '2', '-c:a', 'pcm_f32le', str(destination),
    ], check=True)


def _load(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype='float32', always_2d=True)
    return np.asarray(audio, dtype=np.float32), int(sr)


def _write(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype='FLOAT')


def _choose_master(files: list[Path], explicit: str | None = None) -> Path:
    if explicit:
        explicit_low = explicit.lower()
        matches = [p for p in files if p.name.lower() == explicit_low or str(p).lower().endswith(explicit_low)]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(f'Explicit master filename not found uniquely: {explicit}')

    non_wav = [p for p in files if p.suffix.lower() != '.wav']
    if non_wav:
        return max(non_wav, key=lambda p: p.stat().st_size)

    keywords = ('master', 'original', 'full mix', 'full_mix', 'mixdown', 'reference')
    named = [p for p in files if any(k in p.name.lower() for k in keywords)]
    if named:
        return max(named, key=lambda p: p.stat().st_size)

    raise ValueError('Could not identify the original/master automatically. Set LITELABS_MASTER_FILENAME.')


def _pair_key(path: Path) -> tuple[str, str | None]:
    stem = path.stem
    lower = stem.lower()
    for suffix, side in (('.l', 'L'), ('.r', 'R')):
        if lower.endswith(suffix):
            return stem[:-2].strip().lower(), side
    return stem.strip().lower(), None


def _build_tracks(paths: list[Path], work: Path, sr: int = 44100) -> tuple[dict[str, np.ndarray], list[dict]]:
    grouped: dict[str, dict[str, Path]] = {}
    singles: list[Path] = []
    for path in paths:
        key, side = _pair_key(path)
        if side:
            grouped.setdefault(key, {})[side] = path
        else:
            singles.append(path)

    tracks: dict[str, np.ndarray] = {}
    inventory: list[dict] = []
    for key, sides in sorted(grouped.items()):
        if 'L' not in sides or 'R' not in sides:
            singles.extend(sides.values())
            continue
        lp = work / 'norm' / f'{key}_L.wav'
        rp = work / 'norm' / f'{key}_R.wav'
        _normalise(sides['L'], lp, sr)
        _normalise(sides['R'], rp, sr)
        la, _ = _load(lp)
        ra, _ = _load(rp)
        n = max(len(la), len(ra))
        stereo = np.zeros((n, 2), dtype=np.float32)
        stereo[:len(la), 0] = la[:, 0]
        stereo[:len(ra), 1] = ra[:, 0]
        tracks[key] = stereo
        inventory.append({'name': key, 'source': [sides['L'].name, sides['R'].name]})

    for i, path in enumerate(sorted(singles)):
        key = path.stem.strip().lower()
        norm = work / 'norm' / f'single_{i:03d}.wav'
        _normalise(path, norm, sr)
        audio, _ = _load(norm)
        tracks[key] = audio
        inventory.append({'name': key, 'source': [path.name]})

    if not tracks:
        raise ValueError('No studio multitrack files found')
    return tracks, inventory


def _sum_tracks(tracks: dict[str, np.ndarray], gains: dict[str, float] | None = None) -> np.ndarray:
    n = max(len(x) for x in tracks.values())
    out = np.zeros((n, 2), dtype=np.float64)
    for name, audio in tracks.items():
        gain = 1.0 if gains is None else float(gains.get(name, 1.0))
        out[:len(audio)] += audio.astype(np.float64) * gain
    return out.astype(np.float32)


def _mono(audio: np.ndarray) -> np.ndarray:
    return np.mean(audio, axis=1, dtype=np.float64).astype(np.float32)


def _estimate_global_offset(reference: np.ndarray, candidate: np.ndarray, sr: int, max_seconds: float = 8.0) -> int:
    # Downsample for a cheap broad alignment search, then return full-rate samples.
    step = 8
    ref = _mono(reference)[::step]
    cand = _mono(candidate)[::step]
    maxlag = int(max_seconds * sr / step)
    n = min(len(ref), len(cand), int(180 * sr / step))
    ref = ref[:n]
    cand = cand[:n]
    ref = ref - np.mean(ref)
    cand = cand - np.mean(cand)
    size = 1
    while size < len(ref) + len(cand):
        size <<= 1
    corr = np.fft.irfft(np.fft.rfft(ref, size) * np.conj(np.fft.rfft(cand, size)), size)
    corr = np.concatenate((corr[-maxlag:], corr[:maxlag + 1]))
    lags = np.arange(-maxlag, maxlag + 1)
    lag_ds = int(lags[int(np.argmax(np.abs(corr)))])
    return lag_ds * step


def _shift(audio: np.ndarray, offset: int, length: int) -> np.ndarray:
    out = np.zeros((length, 2), dtype=np.float32)
    if offset >= 0:
        src_start = 0
        dst_start = offset
    else:
        src_start = -offset
        dst_start = 0
    count = min(len(audio) - src_start, length - dst_start)
    if count > 0:
        out[dst_start:dst_start + count] = audio[src_start:src_start + count]
    return out


def _fit_track_gains(tracks: dict[str, np.ndarray], master: np.ndarray, offset: int, sr: int) -> dict[str, float]:
    names = list(tracks)
    length = len(master)
    step = max(1, sr // 400)  # ~400 samples/sec -> manageable least-squares matrix.
    y = _mono(master)[::step]
    columns = []
    for name in names:
        shifted = _shift(tracks[name], offset, length)
        columns.append(_mono(shifted)[::step])
    X = np.stack(columns, axis=1).astype(np.float64)
    y = y.astype(np.float64)
    finite = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[finite]
    y = y[finite]
    if len(y) < max(1000, len(names) * 20):
        raise ValueError('Not enough aligned samples for gain fitting')
    gains, *_ = np.linalg.lstsq(X, y, rcond=1e-6)
    gains = np.clip(gains, -8.0, 8.0)
    return {name: float(value) for name, value in zip(names, gains)}


def _fit_stereo_matrix(candidate: np.ndarray, master: np.ndarray) -> np.ndarray:
    n = min(len(candidate), len(master))
    step = max(1, n // 120000)
    X = candidate[:n:step].astype(np.float64)
    Y = master[:n:step].astype(np.float64)
    matrix, *_ = np.linalg.lstsq(X, Y, rcond=1e-6)
    return matrix


def build_mix_reconstruction_campaign(payload: dict, progress=None) -> dict:
    zip_url = str(payload.get('zip_url') or payload.get('url') or '').strip()
    if not zip_url:
        return {'ok': False, 'mode': 'mix_reconstruction_campaign', 'error': 'zip_url is required'}
    explicit_master = str(payload.get('master_filename') or '').strip() or None
    sr = int(payload.get('sample_rate') or 44100)

    with tempfile.TemporaryDirectory(prefix='litelabs_mix_reconstruct_') as temp:
        root = Path(temp)
        archive = root / (Path(urlparse(zip_url).path).name or 'multitracks.zip')
        extracted = root / 'extracted'
        work = root / 'work'

        _progress(progress, 'Downloading benchmark pack', 5)
        _download(zip_url, archive)
        _safe_extract(archive, extracted)
        files = [p for p in extracted.rglob('*') if _real_audio(p)]
        if not files:
            raise ValueError('No audio files found in archive')

        master_path = _choose_master(files, explicit_master)
        track_paths = [p for p in files if p != master_path and p.suffix.lower() == '.wav']
        if not track_paths:
            raise ValueError('No WAV multitracks found after excluding the master')

        _progress(progress, f'Normalising master: {master_path.name}', 12)
        master_wav = work / 'master.wav'
        _normalise(master_path, master_wav, sr)
        master, _ = _load(master_wav)

        _progress(progress, f'Loading {len(track_paths)} studio files', 22)
        tracks, inventory = _build_tracks(track_paths, work, sr)
        naive = _sum_tracks(tracks)
        naive_score = _score(naive, master, sr)

        _progress(progress, 'Finding global timeline alignment', 42)
        offset = _estimate_global_offset(master, naive, sr)
        aligned_naive = _shift(naive, offset, len(master))
        aligned_naive_score = _score(aligned_naive, master, sr)

        _progress(progress, 'Fitting per-track gain and polarity', 58)
        gains = _fit_track_gains(tracks, master, offset, sr)
        gained_unshifted = _sum_tracks(tracks, gains)
        gained = _shift(gained_unshifted, offset, len(master))
        gained_score = _score(gained, master, sr)

        _progress(progress, 'Fitting stereo mix matrix', 72)
        matrix = _fit_stereo_matrix(gained, master)
        calibrated = (gained.astype(np.float64) @ matrix).astype(np.float32)
        calibrated_score = _score(calibrated, master, sr)

        residual = master[:len(calibrated)] - calibrated[:len(master)]
        rms_master = float(np.sqrt(np.mean(master.astype(np.float64) ** 2)) + 1e-12)
        rms_residual = float(np.sqrt(np.mean(residual.astype(np.float64) ** 2)) + 1e-12)
        residual_db = 20.0 * np.log10(rms_residual / rms_master)

        _progress(progress, 'Writing calibrated reconstruction', 90)
        output_dir = Path(str(payload.get('artifact_dir') or '/workspace/litelabs-research/reconstruction'))
        output_dir.mkdir(parents=True, exist_ok=True)
        recon_path = output_dir / 'calibrated_reconstruction.wav'
        residual_path = output_dir / 'calibration_residual.wav'
        _write(recon_path, calibrated, sr)
        _write(residual_path, residual, sr)

        gain_rows = []
        for item in inventory:
            name = item['name']
            gain = gains.get(name, 1.0)
            gain_rows.append({
                **item,
                'gain_linear': round(float(gain), 8),
                'gain_db_abs': round(float(20.0 * np.log10(max(abs(gain), 1e-12))), 4),
                'polarity_flipped': bool(gain < 0),
            })

        _progress(progress, 'Reconstruction calibration complete', 100)
        return {
            'ok': True,
            'mode': 'mix_reconstruction_campaign',
            'zip_url': zip_url,
            'master_file': master_path.name,
            'sample_rate': sr,
            'studio_track_count': len(tracks),
            'inventory': gain_rows,
            'naive_score': naive_score,
            'aligned_naive_score': aligned_naive_score,
            'global_offset_samples': int(offset),
            'global_offset_ms': round(1000.0 * offset / sr, 4),
            'gain_fitted_score': gained_score,
            'stereo_matrix': [[round(float(x), 8) for x in row] for row in matrix.tolist()],
            'calibrated_score': calibrated_score,
            'residual_rms_db_vs_master': round(float(residual_db), 4),
            'reconstruction_wav': str(recon_path),
            'residual_wav': str(residual_path),
        }
