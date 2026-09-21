from __future__ import annotations

import json
import math
import sys
import time
from argparse import Namespace
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

MEDLEYVOX_ROOT = Path("/opt/medleyvox")
MODEL_ROOT = Path("/models/medleyvox")
MODEL_NAME = "vocals 238"
TARGET = "vocals"


def _db(value: float) -> float:
    return 20.0 * math.log10(max(float(value), 1e-12))


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n <= 0:
        return 0.0
    aa = np.asarray(a[:n], dtype=np.float64)
    bb = np.asarray(b[:n], dtype=np.float64)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / denom) if denom > 1e-12 else 0.0


def _load_model(device: torch.device):
    sys.path.insert(0, str(MEDLEYVOX_ROOT))
    from models import load_model_with_args

    config_path = MODEL_ROOT / MODEL_NAME / f"{TARGET}.json"
    weights_path = MODEL_ROOT / MODEL_NAME / f"{TARGET}.pth"
    data = json.loads(config_path.read_text(encoding="utf-8"))
    args = Namespace(**data["args"])
    model = load_model_with_args(args)

    checkpoint = torch.load(str(weights_path), map_location=device)
    if bool(getattr(args, "ema", False)):
        model_dict = model.state_dict()
        filtered = {
            key.replace("ema_model.module.", ""): value
            for key, value in checkpoint.items()
            if key.replace("ema_model.module.", "") in model_dict
        }
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(checkpoint)

    # Keep the research separator in float32. The pretrained checkpoint can
    # contain half-precision buffers and its scatter/index operations reject a
    # Float source into a Half destination under autocast.
    model.to(device)
    model.float()
    model.eval()
    return model, args


def _infer_chunk(model, chunk: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(chunk, dtype=np.float32)).view(1, 1, -1).to(device)
    with torch.inference_mode():
        out = model.separate(tensor.float())
    arr = out.detach().float().cpu().numpy()
    if arr.ndim != 3 or arr.shape[1] < 2:
        raise RuntimeError(f"Unexpected MedleyVox output shape: {arr.shape}")
    return np.asarray(arr[0, :2, :], dtype=np.float32)


def _best_overlap_permutation(prev: np.ndarray, cur: np.ndarray, overlap: int) -> tuple[np.ndarray, bool]:
    if overlap <= 0 or prev.shape[-1] < overlap or cur.shape[-1] < overlap:
        return cur, False
    direct = abs(_cos(prev[0, -overlap:], cur[0, :overlap])) + abs(_cos(prev[1, -overlap:], cur[1, :overlap]))
    swapped = abs(_cos(prev[0, -overlap:], cur[1, :overlap])) + abs(_cos(prev[1, -overlap:], cur[0, :overlap]))
    if swapped > direct:
        return cur[[1, 0], :], True
    return cur, False


def _separate_channel(
    model,
    audio: np.ndarray,
    sr: int,
    device: torch.device,
    chunk_seconds: float = 12.0,
    overlap_seconds: float = 2.0,
) -> tuple[np.ndarray, int]:
    chunk = max(1, int(sr * chunk_seconds))
    overlap = max(1, int(sr * overlap_seconds))
    hop = max(1, chunk - overlap)

    total = len(audio)
    out = np.zeros((2, total), dtype=np.float64)
    weight = np.zeros(total, dtype=np.float64)
    previous = None
    swaps = 0

    starts = list(range(0, total, hop))
    for start in starts:
        end = min(total, start + chunk)
        piece = np.asarray(audio[start:end], dtype=np.float32)
        valid = len(piece)
        if valid < chunk:
            piece = np.pad(piece, (0, chunk - valid))

        separated = _infer_chunk(model, piece, device)[:, :valid]
        if previous is not None:
            separated, swapped = _best_overlap_permutation(previous, separated, min(overlap, valid))
            swaps += int(swapped)

        window = np.ones(valid, dtype=np.float64)
        actual_overlap = min(overlap, valid)
        if start > 0 and actual_overlap > 1:
            window[:actual_overlap] = np.linspace(0.0, 1.0, actual_overlap, endpoint=False)
        if end < total and actual_overlap > 1:
            window[-actual_overlap:] = np.minimum(
                window[-actual_overlap:],
                np.linspace(1.0, 0.0, actual_overlap, endpoint=False),
            )

        out[:, start:end] += separated[:, :valid] * window[None, :]
        weight[start:end] += window
        previous = separated

    out /= np.maximum(weight[None, :], 1e-8)
    return np.asarray(out, dtype=np.float32), swaps


def _align_stereo(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, bool]:
    direct = abs(_cos(left[0], right[0])) + abs(_cos(left[1], right[1]))
    swapped = abs(_cos(left[0], right[1])) + abs(_cos(left[1], right[0]))
    if swapped > direct:
        return right[[1, 0], :], True
    return right, False


def run_multilead_research(
    vocal_parent: Path,
    output_dir: Path,
    track: str,
    progress=None,
) -> dict:
    started = time.monotonic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if progress:
        progress("Research: separating multiple lead singers", 60)

    model, args = _load_model(device)
    model_sr = int(getattr(args, "sample_rate", 24000))

    audio, parent_sr = sf.read(str(vocal_parent), always_2d=True, dtype="float32")
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]

    resampled = []
    for channel in range(2):
        resampled.append(
            librosa.resample(
                np.asarray(audio[:, channel], dtype=np.float32),
                orig_sr=int(parent_sr),
                target_sr=model_sr,
                res_type="soxr_hq",
            )
        )

    left, left_swaps = _separate_channel(model, resampled[0], model_sr, device)
    right, right_swaps = _separate_channel(model, resampled[1], model_sr, device)
    right, stereo_swapped = _align_stereo(left, right)

    source_a_24 = np.stack([left[0], right[0]], axis=1)
    source_b_24 = np.stack([left[1], right[1]], axis=1)

    def back_to_parent_sr(item: np.ndarray) -> np.ndarray:
        channels = []
        for ch in range(item.shape[1]):
            channels.append(
                librosa.resample(
                    np.asarray(item[:, ch], dtype=np.float32),
                    orig_sr=model_sr,
                    target_sr=int(parent_sr),
                    res_type="soxr_hq",
                )
            )
        n = min(len(channels[0]), len(channels[1]), len(audio))
        return np.stack([channels[0][:n], channels[1][:n]], axis=1).astype(np.float32)

    source_a = back_to_parent_sr(source_a_24)
    source_b = back_to_parent_sr(source_b_24)
    parent = np.asarray(audio[: min(len(audio), len(source_a), len(source_b))], dtype=np.float32)
    n = min(len(parent), len(source_a), len(source_b))
    parent = parent[:n]
    source_a = source_a[:n]
    source_b = source_b[:n]

    rebuilt = source_a + source_b
    residual = parent - rebuilt
    parent_rms = float(np.sqrt(np.mean(parent.astype(np.float64) ** 2) + 1e-12))
    residual_rms = float(np.sqrt(np.mean(residual.astype(np.float64) ** 2) + 1e-12))

    output_dir.mkdir(parents=True, exist_ok=True)
    a_path = output_dir / f"{track}_lead_vocals_a.flac"
    b_path = output_dir / f"{track}_lead_vocals_b.flac"
    sf.write(str(a_path), source_a, int(parent_sr), subtype="PCM_24")
    sf.write(str(b_path), source_b, int(parent_sr), subtype="PCM_24")

    report = {
        "ok": True,
        "research_only": True,
        "model": "MedleyVox vocals 238 (Cyru5)",
        "purpose": "two-singer / co-lead separation",
        "files": [a_path.name, b_path.name],
        "labels": {
            a_path.name: "Lead Vocals A",
            b_path.name: "Lead Vocals B",
        },
        "native_model_sample_rate": model_sr,
        "export_sample_rate": int(parent_sr),
        "model_bandwidth_note": "MedleyVox operates at 24 kHz; exports are resampled to the parent sample rate for pack compatibility.",
        "chunk_seconds": 12.0,
        "overlap_seconds": 2.0,
        "left_identity_swaps": int(left_swaps),
        "right_identity_swaps": int(right_swaps),
        "stereo_channel_source_swap_applied": bool(stereo_swapped),
        "parent_vs_children_sum_cosine": round(float(_cos(parent.reshape(-1), rebuilt.reshape(-1))), 6),
        "residual_relative_to_parent_db": round(_db(residual_rms / max(parent_rms, 1e-12)), 3),
        "lead_a_vs_lead_b_correlation": round(abs(_cos(source_a.reshape(-1), source_b.reshape(-1))), 6),
        "runtime_seconds": round(time.monotonic() - started, 3),
    }
    if progress:
        progress("Research: multi-lead separation complete", 66)
    return report
