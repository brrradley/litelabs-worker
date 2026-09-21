from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

MODEL_ROOT = Path("/models/essentia")
EMBEDDING = MODEL_ROOT / "discogs-effnet-bs64-1.pb"
INSTRUMENT_HEAD = MODEL_ROOT / "mtg_jamendo_instrument-discogs-effnet-1.pb"
INSTRUMENT_META = MODEL_ROOT / "mtg_jamendo_instrument-discogs-effnet-1.json"


def _confidence(item: dict[str, float]) -> str:
    mean = float(item.get("mean", 0.0))
    p90 = float(item.get("p90", 0.0))
    maximum = float(item.get("max", 0.0))

    if mean >= 0.30 or (p90 >= 0.50 and maximum >= 0.60):
        return "strong"
    if mean >= 0.12 or p90 >= 0.28:
        return "likely"
    return "weak"


def _aggregate(predictions: np.ndarray, classes: list[str]) -> list[dict]:
    arr = np.asarray(predictions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[-1] != len(classes):
        raise RuntimeError(
            "Instrument output/class mismatch: "
            f"predictions={arr.shape}, classes={len(classes)}"
        )

    ranked = []
    for idx, label in enumerate(classes):
        values = arr[:, idx]
        item = {
            "label": str(label),
            "mean": round(float(np.mean(values)), 6),
            "p90": round(float(np.quantile(values, 0.90)), 6),
            "max": round(float(np.max(values)), 6),
        }
        item["confidence"] = _confidence(item)
        ranked.append(item)

    ranked.sort(
        key=lambda item: (item["mean"], item["p90"], item["max"]),
        reverse=True,
    )
    return ranked


def run_instrument_probe(audio_path: Path) -> dict:
    from essentia.standard import (
        MonoLoader,
        TensorflowPredict2D,
        TensorflowPredictEffnetDiscogs,
    )

    started = time.monotonic()

    for path in (EMBEDDING, INSTRUMENT_HEAD, INSTRUMENT_META):
        if not path.is_file():
            raise RuntimeError(f"Missing instrument asset: {path}")

    meta = json.loads(INSTRUMENT_META.read_text(encoding="utf-8"))
    classes = list(meta.get("classes") or [])
    if len(classes) < 20:
        raise RuntimeError(
            f"Instrument metadata looks incomplete: {len(classes)} classes"
        )

    signal = MonoLoader(
        filename=str(audio_path),
        sampleRate=16000,
        resampleQuality=4,
    )()
    if len(signal) < 16000:
        raise RuntimeError("Audio is too short for instrument analysis")

    embedder = TensorflowPredictEffnetDiscogs(
        graphFilename=str(EMBEDDING),
        output="PartitionedCall:1",
    )

    # Official MTG-Jamendo instrument example leaves TensorflowPredict2D's
    # endpoints unspecified. Do not force SavedModel node names here.
    classifier = TensorflowPredict2D(
        graphFilename=str(INSTRUMENT_HEAD),
    )

    embeddings = np.asarray(embedder(signal), dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise RuntimeError(
            f"Invalid EffNet embedding shape: {embeddings.shape}"
        )

    predictions = np.asarray(classifier(embeddings), dtype=np.float32)
    ranked = _aggregate(predictions, classes)

    detected = [
        item
        for item in ranked
        if item["confidence"] in {"strong", "likely"}
    ]

    return {
        "ok": True,
        "mode": "instrument_probe",
        "model": "Inst-MTG",
        "detected_instruments": detected,
        "top10": ranked[:10],
        "class_count": len(classes),
        "embedding_frames": int(embeddings.shape[0]),
        "runtime_seconds": round(time.monotonic() - started, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    args = parser.parse_args()

    result = run_instrument_probe(Path(args.audio))
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
