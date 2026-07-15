from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests

CLASSES = [
    "accordion", "acousticbassguitar", "acousticguitar", "bass", "beat", "bell", "bongo", "brass",
    "cello", "clarinet", "classicalguitar", "computer", "doublebass", "drummachine", "drums",
    "electricguitar", "electricpiano", "flute", "guitar", "harmonica", "harp", "horn", "keyboard",
    "oboe", "orchestra", "organ", "pad", "percussion", "piano", "pipeorgan", "rhodes", "sampler",
    "saxophone", "strings", "synthesizer", "trombone", "trumpet", "viola", "violin", "voice",
]

PRIMARY_MAP = {
    "vocals": {"voice"},
    "drums": {"drums", "beat", "drummachine", "percussion", "bongo"},
    "bass": {"bass", "acousticbassguitar", "doublebass"},
    "guitar": {"guitar", "acousticguitar", "classicalguitar", "electricguitar"},
    "piano": {"piano", "electricpiano", "keyboard", "organ", "pipeorgan", "rhodes"},
    "other": {"accordion", "bell", "brass", "cello", "clarinet", "computer", "flute", "harmonica", "harp", "horn", "oboe", "orchestra", "pad", "sampler", "saxophone", "strings", "synthesizer", "trombone", "trumpet", "viola", "violin"},
}


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with destination.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)


def _predict_window(audio: np.ndarray, embedder, classifier) -> np.ndarray:
    embeddings = embedder(audio.astype(np.float32, copy=False))
    predictions = np.asarray(classifier(embeddings), dtype=np.float32)
    if predictions.ndim == 1:
        return predictions
    return np.max(predictions, axis=0)


def _primary_scores(tag_scores: dict[str, float]) -> dict[str, float]:
    return {
        primary: round(max((tag_scores.get(tag, 0.0) for tag in tags), default=0.0), 6)
        for primary, tags in PRIMARY_MAP.items()
    }


def build_instrument_wireframe(payload: dict, progress=None) -> dict:
    audio_url = str(payload.get("audio_url") or payload.get("source_url") or "").strip()
    if not audio_url:
        return {"ok": False, "mode": "instrument_wireframe", "error": "audio_url is required"}

    window_seconds = max(5.0, min(30.0, float(payload.get("window_seconds") or 10.0)))
    hop_seconds = max(2.5, min(window_seconds, float(payload.get("hop_seconds") or 5.0)))
    threshold = max(0.05, min(0.95, float(payload.get("threshold") or 0.20)))
    model_dir = Path(os.getenv("LITELABS_INSTRUMENT_MODEL_DIR", "/models/instrument_wireframe"))
    embedding_model = model_dir / "discogs-effnet-bs64-1.pb"
    classifier_model = model_dir / "mtg_jamendo_instrument-discogs-effnet-1.pb"
    for required in (embedding_model, classifier_model):
        if not required.exists():
            return {"ok": False, "mode": "instrument_wireframe", "error": f"Missing model file: {required}"}

    try:
        from essentia.standard import MonoLoader, TensorflowPredict2D, TensorflowPredictEffnetDiscogs
    except Exception as exc:
        return {"ok": False, "mode": "instrument_wireframe", "error": f"Essentia TensorFlow unavailable: {exc}", "error_type": exc.__class__.__name__}

    with tempfile.TemporaryDirectory(prefix="litelabs_wireframe_") as temp:
        root = Path(temp)
        source = root / (Path(urlparse(audio_url).path).name or "track.flac")
        if progress:
            progress("Downloading source audio", 5)
        _download(audio_url, source)

        audio = MonoLoader(filename=str(source), sampleRate=16000, resampleQuality=4)()
        embedder = TensorflowPredictEffnetDiscogs(graphFilename=str(embedding_model), output="PartitionedCall:1")
        classifier = TensorflowPredict2D(graphFilename=str(classifier_model), output="model/Sigmoid")

        rate = 16000
        window = int(window_seconds * rate)
        hop = int(hop_seconds * rate)
        starts = list(range(0, max(1, len(audio) - window + 1), hop))
        if not starts or starts[-1] + window < len(audio):
            starts.append(max(0, len(audio) - window))

        timeline = []
        per_tag: dict[str, list[float]] = {tag: [] for tag in CLASSES}
        for index, start in enumerate(starts):
            clip = audio[start:start + window]
            if len(clip) < window:
                clip = np.pad(clip, (0, window - len(clip)))
            scores = _predict_window(clip, embedder, classifier)
            tagged = {tag: float(scores[i]) for i, tag in enumerate(CLASSES)}
            for tag, score in tagged.items():
                per_tag[tag].append(score)
            detected = [
                {"instrument": tag, "confidence": round(score, 6)}
                for tag, score in sorted(tagged.items(), key=lambda item: -item[1])
                if score >= threshold
            ]
            timeline.append({
                "start_seconds": round(start / rate, 3),
                "end_seconds": round(min(len(audio), start + window) / rate, 3),
                "detected": detected,
            })
            if progress:
                progress("Mapping instrument timeline", 10 + int(75 * (index + 1) / max(1, len(starts))))

        instruments = []
        for tag, values in per_tag.items():
            arr = np.asarray(values, dtype=np.float32)
            peak = float(np.max(arr)) if len(arr) else 0.0
            median = float(np.median(arr)) if len(arr) else 0.0
            present = arr >= threshold
            if peak < threshold:
                continue
            active_indexes = np.flatnonzero(present)
            instruments.append({
                "instrument": tag,
                "peak_confidence": round(peak, 6),
                "median_confidence": round(median, 6),
                "window_presence_ratio": round(float(np.mean(present)), 6),
                "first_seen_seconds": round(starts[int(active_indexes[0])] / rate, 3) if len(active_indexes) else None,
                "last_seen_seconds": round(min(len(audio), starts[int(active_indexes[-1])] + window) / rate, 3) if len(active_indexes) else None,
            })
        instruments.sort(key=lambda item: (-item["peak_confidence"], -item["window_presence_ratio"]))

        tag_peaks = {tag: (max(values) if values else 0.0) for tag, values in per_tag.items()}
        expected_scores = _primary_scores(tag_peaks)
        expected = [stem for stem, score in expected_scores.items() if score >= threshold]
        routing_hints = []
        for stem in expected:
            routing_hints.append({"stem": stem, "action": "ensure_export_present", "confidence": expected_scores[stem]})
        if expected_scores.get("piano", 0.0) >= threshold:
            routing_hints.append({"stem": "piano", "action": "never_omit_and_compare_alternate_candidate", "confidence": expected_scores["piano"]})
        if expected_scores.get("other", 0.0) >= threshold:
            routing_hints.append({"stem": "other", "action": "preserve_other_and_check_absorption", "confidence": expected_scores["other"]})

        return {
            "ok": True,
            "mode": "instrument_wireframe",
            "schema_version": 1,
            "research_only": True,
            "licensing_note": "MTG-created model is non-commercial unless separately licensed.",
            "audio_url": audio_url,
            "duration_seconds": round(len(audio) / rate, 3),
            "window_seconds": window_seconds,
            "hop_seconds": hop_seconds,
            "threshold": threshold,
            "instruments": instruments,
            "primary_stem_scores": expected_scores,
            "expected_primary_stems": expected,
            "timeline": timeline,
            "routing_hints": routing_hints,
            "model_family": "MTG-Jamendo instrument classifier",
        }
