from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MODEL_ROOT = Path("/models/essentia")
EMBEDDING = MODEL_ROOT / "discogs-effnet-bs64-1.pb"
INSTRUMENT_HEAD = MODEL_ROOT / "mtg_jamendo_instrument-discogs-effnet-1.pb"
INSTRUMENT_META = MODEL_ROOT / "mtg_jamendo_instrument-discogs-effnet-1.json"
GENRE_HEAD = MODEL_ROOT / "genre_discogs400-discogs-effnet-1.pb"
GENRE_META = MODEL_ROOT / "genre_discogs400-discogs-effnet-1.json"


def _aggregate(predictions: np.ndarray, classes: list[str]) -> dict[str, dict[str, float]]:
    arr = np.asarray(predictions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[-1] != len(classes):
        raise RuntimeError(f"Essentia output/class mismatch: {arr.shape} vs {len(classes)}")
    result = {}
    for idx, name in enumerate(classes):
        values = arr[:, idx]
        result[name] = {
            "mean": round(float(np.mean(values)), 6),
            "max": round(float(np.max(values)), 6),
            "p90": round(float(np.quantile(values, 0.90)), 6),
        }
    return result


def _classify_confidence(item: dict[str, float]) -> str:
    mean = float(item.get("mean", 0.0))
    p90 = float(item.get("p90", 0.0))
    maximum = float(item.get("max", 0.0))
    if mean >= 0.30 or (p90 >= 0.50 and maximum >= 0.60):
        return "strong"
    if mean >= 0.12 or p90 >= 0.28:
        return "likely"
    return "weak"


def run_essentia_research(instrument_audio: Path, genre_audio: Path, progress=None) -> dict:
    from essentia.standard import (
        MonoLoader,
        TensorflowPredict2D,
        TensorflowPredictEffnetDiscogs,
    )

    instrument_meta = json.loads(INSTRUMENT_META.read_text(encoding="utf-8"))
    genre_meta = json.loads(GENRE_META.read_text(encoding="utf-8"))
    instrument_classes = list(instrument_meta["classes"])
    genre_classes = list(genre_meta["classes"])

    embedder = TensorflowPredictEffnetDiscogs(
        graphFilename=str(EMBEDDING),
        output="PartitionedCall:1",
    )

    # These frozen classifier heads expose the legacy graph node names below.
    # Keep them explicit so TensorflowPredict2D does not guess SavedModel-style
    # serving names that are not present in the packaged .pb graphs.
    instrument_model = TensorflowPredict2D(
        graphFilename=str(INSTRUMENT_HEAD),
        input="model/Placeholder",
        output="model/Sigmoid",
    )
    genre_model = TensorflowPredict2D(
        graphFilename=str(GENRE_HEAD),
        input="model/Placeholder",
        output="model/Sigmoid",
    )

    if progress:
        progress("LiteLABS Inst-MTG", 41)

    instrument_signal = MonoLoader(
        filename=str(instrument_audio),
        sampleRate=16000,
        resampleQuality=4,
    )()
    genre_signal = MonoLoader(
        filename=str(genre_audio),
        sampleRate=16000,
        resampleQuality=4,
    )()

    instrument_embeddings = embedder(instrument_signal)
    instrument_predictions = instrument_model(instrument_embeddings)
    instrument_scores = _aggregate(instrument_predictions, instrument_classes)

    detected = {}
    for name, item in instrument_scores.items():
        confidence = _classify_confidence(item)
        if confidence != "weak":
            detected[name] = {**item, "confidence": confidence}

    if progress:
        progress("LiteLABS G400", 43)

    genre_embeddings = embedder(genre_signal)
    genre_predictions = genre_model(genre_embeddings)
    genre_scores = _aggregate(genre_predictions, genre_classes)
    genre_ranked = sorted(
        (
            {
                "label": name,
                "mean": score["mean"],
                "p90": score["p90"],
                "max": score["max"],
            }
            for name, score in genre_scores.items()
        ),
        key=lambda item: (item["mean"], item["p90"], item["max"]),
        reverse=True,
    )

    broad = {}
    for item in genre_ranked:
        label = str(item["label"])
        family = label.split("---", 1)[0] if "---" in label else "Other"
        broad[family] = broad.get(family, 0.0) + float(item["mean"])
    broad_ranked = [
        {"family": family, "score": round(score, 6)}
        for family, score in sorted(broad.items(), key=lambda kv: kv[1], reverse=True)
    ]

    return {
        "ok": True,
        "engine": "Essentia TensorFlow",
        "embedding_model": "discogs-effnet-bs64-1",
        "instrument_model": "mtg_jamendo_instrument-discogs-effnet-1",
        "genre_model": "genre_discogs400-discogs-effnet-1",
        "detected_instruments": detected,
        "instrument_top10": [
            {
                "label": name,
                **score,
                "confidence": _classify_confidence(score),
            }
            for name, score in sorted(
                instrument_scores.items(),
                key=lambda kv: (kv[1]["mean"], kv[1]["p90"], kv[1]["max"]),
                reverse=True,
            )[:10]
        ],
        "genre_top10": genre_ranked[:10],
        "genre_broad_families": broad_ranked[:8],
        "research_only": True,
    }
