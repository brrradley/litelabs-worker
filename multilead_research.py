from __future__ import annotations

import math
import subprocess
import tempfile
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


UNMIXX_DIR = Path("/opt/unmixx")
UNMIXX_CKPT = UNMIXX_DIR / "ckpt" / "best.ckpt"
UNMIXX_CONF = UNMIXX_DIR / "ckpt" / "conf.yml"
MODEL_SAMPLE_RATE = 24000


def _db(value: float) -> float:
    return 20.0 * math.log10(max(float(value), 1e-12))


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n <= 0:
        return 0.0
    aa = np.asarray(a[:n], dtype=np.float64).reshape(-1)
    bb = np.asarray(b[:n], dtype=np.float64).reshape(-1)
    aa = aa - np.mean(aa)
    bb = bb - np.mean(bb)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / denom) if denom > 1e-12 else 0.0


def _run_unmixx_chunk(chunk_path: Path, out_dir: Path, timeout: int) -> tuple[int, float, str]:
    started = time.monotonic()
    proc = subprocess.run(
        [
            "python", str(UNMIXX_DIR / "inference.py"),
            "--conf_path", str(UNMIXX_CONF),
            "--ckpt_path", str(UNMIXX_CKPT),
            "--audio_path", str(chunk_path),
            "--output_dir", str(out_dir),
            "--device", "cuda",
        ],
        cwd=str(UNMIXX_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, time.monotonic() - started, proc.stdout or ""


def _resample_mono(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if int(orig_sr) == int(target_sr):
        return np.asarray(audio, dtype=np.float32)
    return np.asarray(
        librosa.resample(
            np.asarray(audio, dtype=np.float32),
            orig_sr=int(orig_sr),
            target_sr=int(target_sr),
            res_type="soxr_hq",
        ),
        dtype=np.float32,
    )


def run_multilead_research(
    vocal_parent: Path,
    output_dir: Path,
    track: str,
    progress=None,
    *,
    chunk_seconds: float = 20.0,
    overlap_seconds: float = 2.0,
    timeout: int = 1800,
) -> dict:
    """Split a vocal parent into two co-lead singers with chunked UNMIXX.

    The public Experimental interface remains compatible with the existing
    multi-lead route: it returns two files named lead_vocals_a/b and is invoked
    only when multi-lead separation is explicitly enabled.
    """
    started = time.monotonic()
    if progress:
        progress("Separating multiple lead singers", 60)

    if not UNMIXX_CKPT.is_file() or not UNMIXX_CONF.is_file():
        raise RuntimeError("UNMIXX model files are missing from the worker image")

    audio, parent_sr = sf.read(str(vocal_parent), always_2d=True, dtype="float32")
    if audio.shape[1] > 2:
        audio = audio[:, :2]
    mono_parent = np.mean(audio, axis=1, dtype=np.float32)
    mono_24 = _resample_mono(mono_parent, int(parent_sr), MODEL_SAMPLE_RATE)

    chunk = max(1, int(MODEL_SAMPLE_RATE * float(chunk_seconds)))
    overlap = max(0, min(chunk // 2, int(MODEL_SAMPLE_RATE * float(overlap_seconds))))
    step = max(1, chunk - overlap)
    starts = list(range(0, len(mono_24), step))

    assembled = [np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)]
    swaps = 0
    chunk_reports: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="litelabs_unmixx_") as temp:
        root = Path(temp)
        for index, start in enumerate(starts):
            segment = mono_24[start:min(len(mono_24), start + chunk)]
            if len(segment) < max(1, MODEL_SAMPLE_RATE // 2):
                continue

            chunk_dir = root / f"chunk_{index:03d}"
            out_dir = chunk_dir / "out"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            chunk_path = chunk_dir / "input.wav"
            sf.write(str(chunk_path), segment, MODEL_SAMPLE_RATE, subtype="PCM_16")

            rc, elapsed, log = _run_unmixx_chunk(chunk_path, out_dir, timeout)
            info = {
                "index": index,
                "start_seconds": round(start / MODEL_SAMPLE_RATE, 3),
                "duration_seconds": round(len(segment) / MODEL_SAMPLE_RATE, 3),
                "runtime_seconds": round(elapsed, 3),
                "returncode": rc,
            }
            if rc != 0:
                info["log_tail"] = "\n".join(log.splitlines()[-50:])
                chunk_reports.append(info)
                raise RuntimeError(
                    f"UNMIXX failed on chunk {index}: " + "\n".join(log.splitlines()[-15:])
                )

            speakers = sorted(out_dir.rglob("spk*.wav"))
            if len(speakers) < 2:
                raise RuntimeError(
                    f"UNMIXX chunk {index} produced {len(speakers)} speaker files; expected 2"
                )

            current: list[np.ndarray] = []
            for speaker in speakers[:2]:
                data, sr = sf.read(str(speaker), always_2d=True, dtype="float32")
                if int(sr) != MODEL_SAMPLE_RATE:
                    raise RuntimeError(f"UNMIXX output sample rate changed: {sr}")
                current.append(np.mean(data, axis=1, dtype=np.float32))

            swapped = False
            if not len(assembled[0]):
                assembled = [current[0], current[1]]
            else:
                ov = min(overlap, len(assembled[0]), len(current[0]), len(current[1]))
                if ov > 0:
                    same = abs(_cos(assembled[0][-ov:], current[0][:ov])) + abs(
                        _cos(assembled[1][-ov:], current[1][:ov])
                    )
                    crossed = abs(_cos(assembled[0][-ov:], current[1][:ov])) + abs(
                        _cos(assembled[1][-ov:], current[0][:ov])
                    )
                    swapped = crossed > same
                if swapped:
                    current = [current[1], current[0]]
                    swaps += 1

                for speaker_index in (0, 1):
                    ov = min(overlap, len(assembled[speaker_index]), len(current[speaker_index]))
                    if ov > 0:
                        fade_in = np.linspace(0.0, 1.0, ov, endpoint=False, dtype=np.float32)
                        fade_out = 1.0 - fade_in
                        joined = (
                            assembled[speaker_index][-ov:] * fade_out
                            + current[speaker_index][:ov] * fade_in
                        )
                        assembled[speaker_index] = np.concatenate(
                            [
                                assembled[speaker_index][:-ov],
                                joined,
                                current[speaker_index][ov:],
                            ]
                        )
                    else:
                        assembled[speaker_index] = np.concatenate(
                            [assembled[speaker_index], current[speaker_index]]
                        )

            info["speaker_swap"] = bool(swapped)
            chunk_reports.append(info)

    target = len(mono_24)
    sources_24 = []
    for item in assembled:
        item = item[:target]
        if len(item) < target:
            item = np.pad(item, (0, target - len(item)))
        sources_24.append(np.asarray(item, dtype=np.float32))

    source_a_mono = _resample_mono(sources_24[0], MODEL_SAMPLE_RATE, int(parent_sr))
    source_b_mono = _resample_mono(sources_24[1], MODEL_SAMPLE_RATE, int(parent_sr))
    n = min(len(mono_parent), len(source_a_mono), len(source_b_mono))
    mono_reference = mono_parent[:n]
    source_a_mono = source_a_mono[:n]
    source_b_mono = source_b_mono[:n]

    rebuilt = source_a_mono + source_b_mono
    residual = mono_reference - rebuilt
    parent_rms = float(np.sqrt(np.mean(mono_reference.astype(np.float64) ** 2) + 1e-12))
    residual_rms = float(np.sqrt(np.mean(residual.astype(np.float64) ** 2) + 1e-12))

    # Keep pack compatibility with the existing stereo co-lead stems. UNMIXX is
    # evaluated on a mono vocal parent, so these are intentional dual-mono files.
    source_a = np.repeat(source_a_mono[:, None], 2, axis=1)
    source_b = np.repeat(source_b_mono[:, None], 2, axis=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    a_path = output_dir / f"{track}_lead_vocals_a.flac"
    b_path = output_dir / f"{track}_lead_vocals_b.flac"
    sf.write(str(a_path), source_a, int(parent_sr), subtype="PCM_24")
    sf.write(str(b_path), source_b, int(parent_sr), subtype="PCM_24")

    report = {
        "ok": True,
        "research_only": True,
        "model": "UNMIXX chunked",
        "purpose": "two-singer / co-lead separation",
        "files": [a_path.name, b_path.name],
        "labels": {
            a_path.name: "Lead Vocals A",
            b_path.name: "Lead Vocals B",
        },
        "native_model_sample_rate": MODEL_SAMPLE_RATE,
        "export_sample_rate": int(parent_sr),
        "model_bandwidth_note": (
            "UNMIXX operates at 24 kHz; outputs are resampled to the parent sample "
            "rate and exported dual-mono for pack compatibility."
        ),
        "chunk_seconds": float(chunk_seconds),
        "overlap_seconds": float(overlap_seconds),
        "chunk_count": len(chunk_reports),
        "speaker_identity_swaps": int(swaps),
        "chunks": chunk_reports,
        "parent_vs_children_sum_cosine": round(float(_cos(mono_reference, rebuilt)), 6),
        "residual_relative_to_parent_db": round(
            _db(residual_rms / max(parent_rms, 1e-12)), 3
        ),
        "lead_a_vs_lead_b_correlation": round(
            abs(_cos(source_a_mono, source_b_mono)), 6
        ),
        "runtime_seconds": round(time.monotonic() - started, 3),
    }
    if progress:
        progress("Multi-lead separation complete", 66)
    return report
