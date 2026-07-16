from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests

from instrument_wireframe import build_instrument_wireframe


ROUTING_FAMILY_ALIASES = {
    "Rock": "rock", "Pop": "pop", "Electronic": "electronic", "Hip Hop": "hip_hop",
    "Funk / Soul": "soul_funk", "Reggae": "reggae", "Jazz": "jazz", "Classical": "classical",
    "Folk, World, & Country": "folk_country", "Latin": "latin", "Stage & Screen": "stage_screen",
    "Blues": "blues", "Brass & Military": "orchestral_brass", "Non-Music": "non_music",
    "Children's": "other",
}


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with destination.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)


def _genre_hints(routing_family: str) -> list[dict]:
    hints: list[dict] = []
    if routing_family == "rock":
        hints.extend([
            {"action": "protect_drum_transients"},
            {"action": "prioritise_bass_definition"},
            {"action": "compare_guitar_candidate_for_vocal_bleed"},
        ])
    elif routing_family == "electronic":
        hints.extend([
            {"action": "never_omit_other"},
            {"action": "inspect_kick_bass_overlap"},
            {"action": "preserve_synth_harmonics"},
        ])
    elif routing_family == "hip_hop":
        hints.extend([
            {"action": "prioritise_lead_and_backing_vocals"},
            {"action": "protect_sub_bass"},
            {"action": "inspect_kick_bass_overlap"},
        ])
    elif routing_family in {"folk_country", "blues"}:
        hints.extend([
            {"action": "protect_acoustic_instrument_harmonics"},
            {"action": "avoid_aggressive_denoising"},
        ])
    elif routing_family in {"classical", "orchestral_brass", "stage_screen"}:
        hints.extend([
            {"action": "never_omit_other"},
            {"action": "preserve_long_sustains_and_room_tails"},
        ])
    elif routing_family in {"soul_funk", "jazz"}:
        hints.extend([
            {"action": "protect_bass_articulation"},
            {"action": "protect_drum_transients"},
            {"action": "preserve_keys_and_brass_in_other"},
        ])
    elif routing_family == "reggae":
        hints.extend([
            {"action": "protect_bass_weight"},
            {"action": "preserve_guitar_skank_transients"},
        ])
    return hints


def _dedupe_hints(hints: list[dict]) -> list[dict]:
    result: list[dict] = []
    positions: dict[tuple[str, str], int] = {}
    for hint in hints:
        action = str(hint.get("action") or "")
        stem = str(hint.get("stem") or "")
        key = (action, stem)
        if key not in positions:
            positions[key] = len(result)
            result.append(dict(hint))
            continue
        existing = result[positions[key]]
        if "confidence" in hint and "confidence" not in existing:
            existing["confidence"] = hint["confidence"]
    return result


def build_track_profile(payload: dict, progress=None) -> dict:
    audio_url = str(payload.get("audio_url") or payload.get("source_url") or "").strip()
    if not audio_url:
        return {"ok": False, "mode": "track_profile", "error": "audio_url is required"}

    style_threshold = max(0.01, min(0.95, float(payload.get("style_threshold") or 0.08)))
    top_styles = max(3, min(25, int(payload.get("top_styles") or 10)))
    include_timeline = bool(payload.get("include_timeline", False))

    model_dir = Path(os.getenv("LITELABS_INSTRUMENT_MODEL_DIR", "/models/instrument_wireframe"))
    embedding_model = model_dir / "discogs-effnet-bs64-1.pb"
    genre_model = model_dir / "genre_discogs400-discogs-effnet-1.pb"
    genre_metadata = model_dir / "genre_discogs400-discogs-effnet-1.json"
    for required in (embedding_model, genre_model, genre_metadata):
        if not required.exists():
            return {"ok": False, "mode": "track_profile", "error": f"Missing model file: {required}"}

    try:
        from essentia.standard import MonoLoader, TensorflowPredict2D, TensorflowPredictEffnetDiscogs
    except Exception as exc:
        return {"ok": False, "mode": "track_profile", "error": f"Essentia TensorFlow unavailable: {exc}", "error_type": exc.__class__.__name__}

    if progress:
        progress("Building instrument wireframe", 5)
    wireframe_payload = dict(payload)
    wireframe_payload["audio_url"] = audio_url
    wireframe = build_instrument_wireframe(wireframe_payload, progress=None)
    if not wireframe.get("ok"):
        return {"ok": False, "mode": "track_profile", "error": "Instrument wireframe failed", "wireframe_error": wireframe}

    with tempfile.TemporaryDirectory(prefix="litelabs_track_profile_") as temp:
        source = Path(temp) / (Path(urlparse(audio_url).path).name or "track.flac")
        if progress:
            progress("Downloading audio for genre analysis", 35)
        _download(audio_url, source)

        audio = MonoLoader(filename=str(source), sampleRate=16000, resampleQuality=4)()
        embedder = TensorflowPredictEffnetDiscogs(graphFilename=str(embedding_model), output="PartitionedCall:1")
        classifier = TensorflowPredict2D(
            graphFilename=str(genre_model),
            input="serving_default_model_Placeholder",
            output="PartitionedCall:0",
        )
        embeddings = embedder(audio.astype(np.float32, copy=False))
        predictions = np.asarray(classifier(embeddings), dtype=np.float32)
        aggregate = predictions if predictions.ndim == 1 else np.mean(predictions, axis=0)

        metadata = json.loads(genre_metadata.read_text())
        classes = list(metadata.get("classes") or [])
        if len(classes) != len(aggregate):
            return {
                "ok": False, "mode": "track_profile", "error": "Genre class count does not match model output",
                "class_count": len(classes), "prediction_count": int(len(aggregate)),
            }

        ranked = sorted(zip(classes, aggregate.tolist()), key=lambda item: -item[1])
        styles = []
        broad_scores: dict[str, float] = defaultdict(float)
        for label, score in ranked:
            broad, _, style = label.partition("---")
            broad_scores[broad] += float(score)
            if len(styles) < top_styles and float(score) >= style_threshold:
                styles.append({
                    "broad_genre": broad, "style": style or broad, "label": label,
                    "confidence": round(float(score), 6),
                })

        broad_ranked = sorted(broad_scores.items(), key=lambda item: -item[1])
        broad_total = sum(score for _, score in broad_ranked) or 1.0
        broad_genres = [
            {
                "genre": genre,
                "aggregate_score": round(score, 6),
                "normalized_share": round(score / broad_total, 6),
            }
            for genre, score in broad_ranked[:8]
        ]
        primary_genre = broad_ranked[0][0] if broad_ranked else "Unknown"
        routing_family = ROUTING_FAMILY_ALIASES.get(primary_genre, "other")
        expected = list(wireframe.get("expected_primary_stems") or [])

        if progress:
            progress("Combining genre and instrument routing", 90)

        routing_hints = _dedupe_hints(_genre_hints(routing_family) + list(wireframe.get("routing_hints") or []))
        result = {
            "ok": True,
            "mode": "track_profile",
            "schema_version": 2,
            "research_only": True,
            "licensing_note": "MTG-created models are non-commercial unless separately licensed.",
            "audio_url": audio_url,
            "duration_seconds": wireframe.get("duration_seconds"),
            "primary_genre": primary_genre,
            "routing_family": routing_family,
            "broad_genres": broad_genres,
            "styles": styles,
            "instruments": wireframe.get("instruments", []),
            "primary_stem_scores": wireframe.get("primary_stem_scores", {}),
            "expected_primary_stems": expected,
            "routing_hints": routing_hints,
            "model_families": ["Discogs400 genre classifier", "MTG-Jamendo instrument classifier"],
        }
        if include_timeline:
            result["instrument_timeline"] = wireframe.get("timeline", [])
        return result
