from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

MODEL_ROOT = Path("/models/essentia")
EMBEDDING = MODEL_ROOT / "discogs-effnet-bs64-1.pb"
GENRE_HEAD = MODEL_ROOT / "genre_discogs400-discogs-effnet-1.pb"
GENRE_META = MODEL_ROOT / "genre_discogs400-discogs-effnet-1.json"


def _aggregate(predictions: np.ndarray, classes: list[str]) -> list[dict]:
    arr = np.asarray(predictions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[-1] != len(classes):
        raise RuntimeError(
            f"G400 output/class mismatch: predictions={arr.shape}, classes={len(classes)}"
        )

    ranked = []
    for idx, label in enumerate(classes):
        values = arr[:, idx]
        ranked.append(
            {
                "label": str(label),
                "mean": round(float(np.mean(values)), 6),
                "p90": round(float(np.quantile(values, 0.90)), 6),
                "max": round(float(np.max(values)), 6),
            }
        )
    ranked.sort(
        key=lambda item: (item["mean"], item["p90"], item["max"]),
        reverse=True,
    )
    return ranked


def _public_label(raw_label: str) -> str:
    label = str(raw_label or "").strip()
    if "---" in label:
        label = label.split("---", 1)[1]
    return label.replace("_", " ").strip()


def run_genre_probe(audio_path: Path) -> dict:
    from essentia.standard import (
        MonoLoader,
        TensorflowPredict2D,
        TensorflowPredictEffnetDiscogs,
    )

    started = time.monotonic()

    for path in (EMBEDDING, GENRE_HEAD, GENRE_META):
        if not path.is_file():
            raise RuntimeError(f"Missing G400 asset: {path}")

    meta = json.loads(GENRE_META.read_text(encoding="utf-8"))
    classes = list(meta.get("classes") or [])
    if len(classes) != 400:
        raise RuntimeError(f"Expected 400 G400 classes, found {len(classes)}")

    signal = MonoLoader(
        filename=str(audio_path),
        sampleRate=16000,
        resampleQuality=4,
    )()
    if len(signal) < 16000:
        raise RuntimeError("Audio is too short for genre analysis")

    embedder = TensorflowPredictEffnetDiscogs(
        graphFilename=str(EMBEDDING),
        output="PartitionedCall:1",
    )
    classifier = TensorflowPredict2D(
        graphFilename=str(GENRE_HEAD),
        input="serving_default_model_Placeholder",
        output="PartitionedCall:0",
    )

    embeddings = np.asarray(embedder(signal), dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise RuntimeError(f"Invalid EffNet embedding shape: {embeddings.shape}")

    predictions = np.asarray(classifier(embeddings), dtype=np.float32)
    ranked = _aggregate(predictions, classes)

    broad_scores: dict[str, float] = {}
    for item in ranked:
        raw = str(item["label"])
        family = raw.split("---", 1)[0] if "---" in raw else "Other"
        broad_scores[family] = broad_scores.get(family, 0.0) + float(item["mean"])

    broad = [
        {"family": family, "score": round(score, 6)}
        for family, score in sorted(
            broad_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]

    top = ranked[0]
    return {
        "ok": True,
        "mode": "genre_probe",
        "model": "G400",
        "raw_genre": top["label"],
        "genre": _public_label(top["label"]),
        "confidence": {
            "mean": top["mean"],
            "p90": top["p90"],
            "max": top["max"],
        },
        "top10": ranked[:10],
        "broad_families": broad[:8],
        "embedding_frames": int(embeddings.shape[0]),
        "runtime_seconds": round(time.monotonic() - started, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    args = parser.parse_args()

    result = run_genre_probe(Path(args.audio))
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
