from __future__ import annotations

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

INSTRUMENT_GROUPS = {
    "voice": "Voice",
    "drums": "Rhythm", "beat": "Rhythm", "drummachine": "Rhythm", "percussion": "Rhythm", "bongo": "Rhythm",
    "bass": "Bass", "acousticbassguitar": "Bass", "doublebass": "Bass",
    "guitar": "Guitars", "acousticguitar": "Guitars", "classicalguitar": "Guitars", "electricguitar": "Guitars",
    "piano": "Keys", "electricpiano": "Keys", "keyboard": "Keys", "organ": "Keys", "pipeorgan": "Keys", "rhodes": "Keys",
    "saxophone": "Woodwind / Brass", "clarinet": "Woodwind / Brass", "flute": "Woodwind / Brass", "oboe": "Woodwind / Brass",
    "trumpet": "Woodwind / Brass", "trombone": "Woodwind / Brass", "horn": "Woodwind / Brass", "brass": "Woodwind / Brass",
    "violin": "Strings", "viola": "Strings", "cello": "Strings", "strings": "Strings", "harp": "Strings",
    "synthesizer": "Electronic", "pad": "Electronic", "sampler": "Electronic", "computer": "Electronic",
    "accordion": "Other", "bell": "Other", "harmonica": "Other", "orchestra": "Ensemble",
}

SPECIALIST_ROUTING = {
    "drums": "decompose_drums",
    "percussion": "decompose_drums",
    "drummachine": "decompose_drums",
    "piano": "extract_piano",
    "electricpiano": "extract_electric_piano",
    "rhodes": "extract_electric_piano",
    "organ": "extract_organ",
    "pipeorgan": "extract_organ",
    "saxophone": "extract_saxophone",
    "trumpet": "extract_trumpet",
    "trombone": "extract_trombone",
    "clarinet": "extract_clarinet",
    "flute": "extract_flute",
    "oboe": "extract_oboe",
    "horn": "extract_horn",
    "brass": "extract_brass_family",
    "strings": "extract_strings",
    "violin": "extract_violin",
    "viola": "extract_viola",
    "cello": "extract_cello",
    "synthesizer": "preserve_synth_residual",
    "pad": "preserve_synth_residual",
}

DISPLAY_NAMES = {
    "acousticbassguitar": "Acoustic Bass Guitar",
    "acousticguitar": "Acoustic Guitar",
    "classicalguitar": "Classical Guitar",
    "doublebass": "Double Bass",
    "drummachine": "Drum Machine",
    "electricguitar": "Electric Guitar",
    "electricpiano": "Electric Piano",
    "pipeorgan": "Pipe Organ",
    "saxophone": "Saxophone",
    "synthesizer": "Synthesizer",
}


def _display_name(tag: str) -> str:
    return DISPLAY_NAMES.get(tag, tag.replace("_", " ").title())


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


def _confidence_tier(peak: float, presence: float, threshold: float) -> str:
    if peak >= max(0.70, threshold + 0.35) and presence >= 0.20:
        return "high"
    if peak >= max(0.45, threshold + 0.15) and presence >= 0.10:
        return "medium"
    return "low"


def _merge_ranges(starts: list[int], present: np.ndarray, rate: int, window: int, total_samples: int) -> list[dict]:
    indexes = np.flatnonzero(present)
    if not len(indexes):
        return []
    ranges: list[list[float]] = []
    for index in indexes:
        start = starts[int(index)] / rate
        end = min(total_samples, starts[int(index)] + window) / rate
        if ranges and start <= ranges[-1][1] + 0.05:
            ranges[-1][1] = max(ranges[-1][1], end)
        else:
            ranges.append([start, end])
    return [
        {"start_seconds": round(start, 3), "end_seconds": round(end, 3), "duration_seconds": round(end - start, 3)}
        for start, end in ranges
    ]


def build_instrument_wireframe(payload: dict, progress=None) -> dict:
    audio_url = str(payload.get("audio_url") or payload.get("source_url") or "").strip()
    if not audio_url:
        return {"ok": False, "mode": "instrument_wireframe", "error": "audio_url is required"}

    window_seconds = max(5.0, min(30.0, float(payload.get("window_seconds") or 10.0)))
    hop_seconds = max(2.5, min(window_seconds, float(payload.get("hop_seconds") or 5.0)))
    threshold = max(0.05, min(0.95, float(payload.get("threshold") or 0.20)))
    routing_presence_floor = max(0.02, min(0.50, float(payload.get("routing_presence_floor") or 0.08)))
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
                {"instrument": tag, "display_name": _display_name(tag), "confidence": round(score, 6), "group": INSTRUMENT_GROUPS.get(tag, "Other")}
                for tag, score in sorted(tagged.items(), key=lambda item: -item[1])
                if score >= threshold
            ]
            timeline.append({
                "start_seconds": round(start / rate, 3),
                "end_seconds": round(min(len(audio), start + window) / rate, 3),
                "detected": detected,
            })
            if progress:
                progress("Mapping detailed instrument timeline", 10 + int(75 * (index + 1) / max(1, len(starts))))

        instruments = []
        routing_plan = []
        grouped: dict[str, list[dict]] = {}
        for tag, values in per_tag.items():
            arr = np.asarray(values, dtype=np.float32)
            peak = float(np.max(arr)) if len(arr) else 0.0
            median = float(np.median(arr)) if len(arr) else 0.0
            present = arr >= threshold
            if peak < threshold:
                continue
            presence = float(np.mean(present))
            active_indexes = np.flatnonzero(present)
            item = {
                "instrument": tag,
                "display_name": _display_name(tag),
                "group": INSTRUMENT_GROUPS.get(tag, "Other"),
                "peak_confidence": round(peak, 6),
                "median_confidence": round(median, 6),
                "window_presence_ratio": round(presence, 6),
                "confidence_tier": _confidence_tier(peak, presence, threshold),
                "first_seen_seconds": round(starts[int(active_indexes[0])] / rate, 3) if len(active_indexes) else None,
                "last_seen_seconds": round(min(len(audio), starts[int(active_indexes[-1])] + window) / rate, 3) if len(active_indexes) else None,
                "time_ranges": _merge_ranges(starts, present, rate, window, len(audio)),
            }
            instruments.append(item)
            grouped.setdefault(item["group"], []).append(item)

            action = SPECIALIST_ROUTING.get(tag)
            if action and presence >= routing_presence_floor and item["confidence_tier"] != "low":
                routing_plan.append({
                    "instrument": tag,
                    "display_name": item["display_name"],
                    "action": action,
                    "confidence_tier": item["confidence_tier"],
                    "peak_confidence": item["peak_confidence"],
                    "window_presence_ratio": item["window_presence_ratio"],
                    "time_ranges": item["time_ranges"],
                    "status": "candidate_for_specialist_extraction",
                })

        instruments.sort(key=lambda item: (-item["peak_confidence"], -item["window_presence_ratio"]))
        for values in grouped.values():
            values.sort(key=lambda item: (-item["peak_confidence"], -item["window_presence_ratio"]))
        routing_plan.sort(key=lambda item: ({"high": 0, "medium": 1, "low": 2}.get(item["confidence_tier"], 3), -item["peak_confidence"]))

        tag_peaks = {tag: (max(values) if values else 0.0) for tag, values in per_tag.items()}
        expected_scores = _primary_scores(tag_peaks)
        expected = [stem for stem, score in expected_scores.items() if score >= threshold]

        parent_policy = {
            "drums": "replace_parent_if_child_stems_reconstruct_and_pass_quality_gate",
            "piano": "replace_generic_piano_keys_label_with_verified_specific_key_instruments",
            "other": "extract_verified_instruments_then_keep_only_remaining_residual_as_synth_other",
        }

        if progress:
            progress("Instrument inventory complete", 100)

        return {
            "ok": True,
            "mode": "instrument_wireframe",
            "schema_version": 2,
            "research_only": True,
            "licensing_note": "MTG-created model is non-commercial unless separately licensed.",
            "audio_url": audio_url,
            "duration_seconds": round(len(audio) / rate, 3),
            "window_seconds": window_seconds,
            "hop_seconds": hop_seconds,
            "threshold": threshold,
            "routing_presence_floor": routing_presence_floor,
            "instrument_inventory": instruments,
            "instrument_groups": grouped,
            "primary_stem_scores": expected_scores,
            "expected_primary_stems": expected,
            "timeline": timeline,
            "specialist_routing_plan": routing_plan,
            "parent_export_policy": parent_policy,
            "analysis_policy": {
                "genre_is_not_used_for_instrument_routing": True,
                "specific_instrument_identity_preferred_over_generic_parent_label": True,
                "parent_stem_removed_after_successful_verified_decomposition": True,
                "partial_decomposition_keeps_only_parent_residual": True,
                "mega53_crosscheck": "next_stage",
            },
            "model_family": "MTG-Jamendo instrument classifier",
        }
