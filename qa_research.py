from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf

QA_VERSION = 3

PARENT_STEMS = ("vocals", "percussion", "bass", "strings", "keys", "other")
CHILD_GROUPS = {
    "vocal_children": ("lead_vocals", "backing_vocals"),
    "drum_children": ("kick", "snare", "toms", "hi_hats", "cymbals"),
    "wind_children": ("wind_brass", "wind_brass_residual"),
    "sax_children": ("saxophone", "sax_residual"),
}
GROUP_PARENT = {
    "vocal_children": "vocals",
    "drum_children": "percussion",
    "wind_children": "other",
    "sax_children": "wind_brass",
}


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


def _sum_aligned(items: list[np.ndarray]) -> np.ndarray | None:
    items = [item for item in items if item is not None and len(item) > 0]
    if not items:
        return None
    n = min(len(item) for item in items)
    if n <= 0:
        return None
    return np.sum(np.stack([item[:n] for item in items], axis=0), axis=0)


def _similarity(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None or len(a) <= 0 or len(b) <= 0:
        return None
    return _clamp01(abs(_cosine(a, b)))


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


def _stem_score(
    metrics: dict,
    distinctness: float,
    reconstruction: float | None,
    leakage_corr: float,
) -> tuple[float, dict, list[str]]:
    """Score one stem using v2 research heuristics.

    V2 deliberately gives separation distinctness and useful signal most of the weight.
    Reconstruction is a group/reference consistency check rather than a global free pass,
    and technical health can no longer inflate an otherwise poor separation.
    """
    rms_db = float(metrics.get("rms_dbfs", -120.0))
    active = float(metrics.get("active_ratio", 0.0))
    clipping = float(metrics.get("clipping_fraction", 0.0))

    activity_score = _clamp01((active - 0.02) / 0.48)
    level_score = _clamp01((rms_db + 55.0) / 35.0)
    useful_signal = 0.65 * activity_score + 0.35 * level_score
    technical_health = _clamp01(1.0 - min(1.0, clipping * 250.0))
    distinctness_score = _clamp01(distinctness)
    reconstruction_score = _clamp01(reconstruction) if reconstruction is not None else 0.75

    score = (
        0.45 * distinctness_score
        + 0.25 * useful_signal
        + 0.20 * reconstruction_score
        + 0.10 * technical_health
    )

    cap_reasons: list[str] = []
    cap = 1.0

    # Near-empty stems can look perfectly "distinct" simply because there is almost
    # nothing in them. They must not receive a high quality score.
    if useful_signal <= 0.05:
        cap = min(cap, 0.30)
        cap_reasons.append("near-empty signal")
    elif useful_signal < 0.10:
        cap = min(cap, 0.40)
        cap_reasons.append("very weak signal")
    elif useful_signal < 0.20:
        cap = min(cap, 0.70)
        cap_reasons.append("weak/sparse signal")

    # High peer correlation is the most useful bleed warning we have without studio
    # ground truth. Hard caps stop reconstruction/technical-health scores masking it.
    if leakage_corr > 0.75:
        cap = min(cap, 0.45)
        cap_reasons.append("very high peer bleed")
    elif leakage_corr > 0.60:
        cap = min(cap, 0.60)
        cap_reasons.append("high peer bleed")

    if distinctness_score < 0.25:
        cap = min(cap, 0.45)
        cap_reasons.append("very low separation distinctness")
    elif distinctness_score < 0.40:
        cap = min(cap, 0.60)
        cap_reasons.append("low separation distinctness")

    score = min(score, cap)
    components = {
        "reconstruction_integrity": round(reconstruction_score, 6),
        "separation_distinctness": round(distinctness_score, 6),
        "useful_signal": round(useful_signal, 6),
        "technical_health": round(technical_health, 6),
    }
    return round(_clamp01(score), 6), components, cap_reasons


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
    genre_reason: str = "",
) -> dict:
    """Build a lightweight silent QA record for admin research.

    V2 compares stems only with meaningful peers. Parent stems are compared with other
    parent stems; drum/vocal/wind/sax children are compared with siblings. This avoids
    the v1 error of mixing overlapping parent and child hierarchies into one leakage sum.
    """
    source_audio, source_sr = _read(source)
    source_mono = _mono(source_audio)

    loaded: dict[str, np.ndarray] = {}
    metrics: dict[str, dict] = {}
    for stem, path in stems.items():
        if not path or not Path(path).is_file():
            continue
        audio, sr = _read(Path(path))
        loaded[stem] = _mono(audio)
        metrics[stem] = _signal_metrics(audio, sr)

    parent_arrays = [loaded[name] for name in PARENT_STEMS if name in loaded]
    parent_sum = _sum_aligned(parent_arrays)
    reconstruction_cosine = _similarity(source_mono, parent_sum)

    group_reconstruction: dict[str, float | None] = {}
    for group_name, members in CHILD_GROUPS.items():
        children_sum = _sum_aligned([loaded[name] for name in members if name in loaded])
        parent_name = GROUP_PARENT[group_name]
        parent_audio = loaded.get(parent_name)
        group_reconstruction[group_name] = _similarity(parent_audio, children_sum)

    # Instrumental is derived from source minus vocals, so compare it with that target
    # instead of the sum of every overlapping stem in an Experimental pack.
    instrumental_reconstruction: float | None = None
    if "instrumental" in loaded and "vocals" in loaded:
        n = min(len(source_mono), len(loaded["vocals"]), len(loaded["instrumental"]))
        if n > 0:
            instrumental_target = source_mono[:n] - loaded["vocals"][:n]
            instrumental_reconstruction = _similarity(loaded["instrumental"][:n], instrumental_target)

    stem_to_group: dict[str, str] = {}
    for group_name, members in CHILD_GROUPS.items():
        for member in members:
            stem_to_group[member] = group_name

    stem_records: dict[str, dict] = {}
    for name, target in loaded.items():
        qa_group = "other"
        reconstruction_reference = "fallback"

        if name in PARENT_STEMS:
            qa_group = "parent"
            peers = [loaded[other] for other in PARENT_STEMS if other != name and other in loaded]
            reconstruction = reconstruction_cosine
            reconstruction_reference = "source_vs_parent_sum"
        elif name == "instrumental":
            qa_group = "derived_instrumental"
            peers = [loaded["vocals"]] if "vocals" in loaded else []
            reconstruction = instrumental_reconstruction
            reconstruction_reference = "source_minus_vocals"
        elif name in stem_to_group:
            qa_group = stem_to_group[name]
            members = CHILD_GROUPS[qa_group]
            peers = [loaded[other] for other in members if other != name and other in loaded]
            reconstruction = group_reconstruction.get(qa_group)
            reconstruction_reference = f"children_sum_vs_{GROUP_PARENT[qa_group]}"
        else:
            peers = [loaded[other] for other in loaded if other != name]
            reconstruction = reconstruction_cosine
            reconstruction_reference = "source_vs_parent_sum"

        rest = _sum_aligned(peers)
        if rest is not None:
            leakage_corr = abs(_cosine(target, rest))
            distinctness = _clamp01(1.0 - leakage_corr)
        else:
            leakage_corr = 0.0
            distinctness = 0.75

        score, components, cap_reasons = _stem_score(
            metrics[name], distinctness, reconstruction, leakage_corr
        )
        semantic_role = (
            "complement_residual"
            if name in {"wind_brass_residual", "sax_residual"}
            else "child_stem"
            if name in stem_to_group
            else "derived_stem"
            if name == "instrumental"
            else "parent_stem"
            if name in PARENT_STEMS
            else "other"
        )
        stem_records[name] = {
            "model": model_by_stem.get(name, "unknown"),
            "semantic_role": semantic_role,
            "score": score,
            "components": components,
            "metrics": metrics[name],
            "correlation_with_other_stems": round(leakage_corr, 6),
            "qa_group": qa_group,
            "reconstruction_reference": reconstruction_reference,
            "score_cap_reasons": cap_reasons,
        }

    record = {
        "qa_version": QA_VERSION,
        "score_type": "heuristic_stem_confidence_v3",
        "score_interpretation": "QA confidence/routing evidence; not an audible-fidelity percentage or studio-ground-truth accuracy estimate",
        "ground_truth_available": False,
        "job_id": job_id,
        "filename": filename,
        "input_size_bytes": int(input_size_bytes),
        "input_format": str(input_format or "").lower(),
        "duration_seconds": round(len(source_mono) / max(source_sr, 1), 3),
        "genre": genre,
        "genre_reason": genre_reason,
        "preset": preset,
        "pipeline_revision": pipeline_revision,
        "reconstruction_cosine": round(reconstruction_cosine, 6) if reconstruction_cosine is not None else None,
        "scoring_profile": {
            "distinctness_weight": 0.45,
            "useful_signal_weight": 0.25,
            "reconstruction_weight": 0.20,
            "technical_health_weight": 0.10,
            "peer_group_aware": True,
            "hard_quality_caps": True,
            "score_semantics": "confidence_not_fidelity",
            "residuals_are_complements": True,
        },
        "group_reconstruction": {
            key: round(value, 6) if value is not None else None
            for key, value in group_reconstruction.items()
        },
        "stems": stem_records,
    }
    if extra:
        record["pipeline_metrics"] = extra
    return record
