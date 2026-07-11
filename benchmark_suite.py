from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from research_tools import (
    analyse_source_features,
    classify_genre_hint,
    model_label,
    run_model_spec,
    safe_folder_name,
    safe_track_name,
    stem_role,
)

DEFAULT_MODELS = [
    "current_litelabs",
    "demucs:htdemucs_ft",
    "demucs:htdemucs_6s",
    "audio_separator:BS-Roformer-SW.ckpt",
    "audio_separator:mel_band_roformer_vocals_becruily.ckpt",
    "audio_separator:mel_band_roformer_instrumental_becruily.ckpt",
    "audio_separator:mel_band_roformer_bleed_suppressor_v1.ckpt",
    "audio_separator:mel_band_roformer_instrumental_bleedless_v2_gabox.ckpt",
    "audio_separator:mel_band_roformer_instrumental_bleedless_v3_gabox.ckpt",
    "audio_separator:mel_band_roformer_instrumental_fullness_v3_gabox.ckpt",
    "audio_separator:mel_band_roformer_instrumental_instv8n_gabox.ckpt",
    "audio_separator:mel_band_roformer_instrumental_fvx_gabox.ckpt",
]


def _download(url: str, destination: Path) -> None:
    with requests.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        with destination.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)


def _filename(track: dict, index: int) -> str:
    supplied = str(track.get("filename") or "").strip()
    if supplied:
        return supplied
    parsed = Path(urlparse(str(track.get("url") or "")).path).name
    return parsed or f"track-{index + 1}.flac"


def _technical_score(metric: dict) -> float:
    """Technical hygiene only; this is deliberately not claimed as perceptual quality."""
    max_db = float(metric.get("max_db", -99.0) or -99.0)
    active = float(metric.get("active_ratio", 0.0) or 0.0)
    headroom = 25.0 if -12.0 <= max_db <= -0.3 else (15.0 if max_db < -0.3 else 5.0)
    activity = max(0.0, min(50.0, active * 50.0))
    file_ok = 25.0 if float(metric.get("duration_seconds", 0.0) or 0.0) > 1.0 else 0.0
    return round(headroom + activity + file_ok, 2)


def _stem_rows(run_result: dict) -> list[dict]:
    rows: list[dict] = []
    for metric in run_result.get("metrics") or []:
        relative = str(metric.get("relative_path") or metric.get("file") or "")
        role = stem_role(Path(relative))
        if role == "unknown":
            continue
        rows.append({
            "stem": role,
            "file": relative,
            "duration_seconds": metric.get("duration_seconds"),
            "mean_db": metric.get("mean_db"),
            "max_db": metric.get("max_db"),
            "active_ratio": metric.get("active_ratio"),
            "technical_score": _technical_score(metric),
        })
    return rows


def _leaderboard(results: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for item in results:
        if not item.get("ok"):
            continue
        for stem in item.get("stems") or []:
            grouped.setdefault(stem["stem"], []).append({
                "model": item["model"],
                "track_id": item["track_id"],
                "genre": item.get("genre"),
                "technical_score": stem.get("technical_score"),
                "runtime_seconds": item.get("runtime_seconds"),
            })
    output: dict[str, list[dict]] = {}
    for stem, rows in grouped.items():
        by_model: dict[str, dict] = {}
        for row in rows:
            bucket = by_model.setdefault(row["model"], {"model": row["model"], "scores": [], "runtimes": [], "tracks": 0})
            bucket["scores"].append(float(row.get("technical_score") or 0.0))
            bucket["runtimes"].append(float(row.get("runtime_seconds") or 0.0))
            bucket["tracks"] += 1
        ranked = []
        for bucket in by_model.values():
            ranked.append({
                "model": bucket["model"],
                "tracks": bucket["tracks"],
                "average_technical_score": round(sum(bucket["scores"]) / max(1, len(bucket["scores"])), 2),
                "average_runtime_seconds": round(sum(bucket["runtimes"]) / max(1, len(bucket["runtimes"])), 3),
            })
        output[stem] = sorted(ranked, key=lambda row: (-row["average_technical_score"], row["average_runtime_seconds"]))
    return output


def build_benchmark_suite(payload: dict, progress=None) -> dict:
    tracks = payload.get("tracks") or []
    models = payload.get("models") or DEFAULT_MODELS
    if not tracks:
        return {"ok": False, "mode": "benchmark_suite", "error": "Missing input.tracks"}
    if not isinstance(tracks, list) or not isinstance(models, list):
        return {"ok": False, "mode": "benchmark_suite", "error": "tracks and models must be arrays"}

    cursor = max(0, int(payload.get("cursor") or 0))
    max_pairs = max(1, int(payload.get("max_pairs") or 6))
    time_budget_seconds = max(60, min(3300, int(payload.get("time_budget_seconds") or 480)))
    output_format = str(payload.get("output_format") or "flac").lower().strip()
    pairs = [(track_index, model_index) for track_index in range(len(tracks)) for model_index in range(len(models))]
    end = min(len(pairs), cursor + max_pairs)
    started = time.monotonic()
    results: list[dict] = []
    track_cache: dict[int, tuple[Path, dict, dict]] = {}

    with tempfile.TemporaryDirectory(prefix="litelabs_suite_") as temp:
        root = Path(temp)
        for position in range(cursor, end):
            if position > cursor and time.monotonic() - started >= time_budget_seconds:
                end = position
                break
            track_index, model_index = pairs[position]
            track = tracks[track_index]
            model = models[model_index]
            url = str(track.get("url") or track.get("audio_url") or "").strip()
            if not url:
                results.append({"ok": False, "track_id": f"track-{track_index + 1}", "model": model_label(model), "error": "Missing track URL"})
                continue

            if track_index not in track_cache:
                filename = _filename(track, track_index)
                input_path = root / f"track_{track_index:03d}_{filename}"
                if progress:
                    progress(f"Downloading benchmark track {track_index + 1}/{len(tracks)}", 5)
                _download(url, input_path)
                features = analyse_source_features(input_path)
                genre = {
                    "hint": track.get("genre") or track.get("genre_hint"),
                    "subgenre": track.get("subgenre"),
                    "tags": track.get("tags") or [],
                }
                if not genre["hint"]:
                    detected = classify_genre_hint(filename, features, [])
                    genre["hint"] = detected.get("hint")
                    genre["confidence"] = detected.get("confidence")
                    genre["reason"] = detected.get("reason")
                track_cache[track_index] = (input_path, features, genre)

            input_path, features, genre = track_cache[track_index]
            label = model_label(model)
            scratch = root / "scratch" / f"{track_index:03d}_{model_index:03d}_{safe_folder_name(label)}"
            review = root / "review" / f"{track_index:03d}_{model_index:03d}_{safe_folder_name(label)}"
            if progress:
                percent = int(10 + (position - cursor) * 80 / max(1, end - cursor))
                progress(f"Benchmarking {label} on track {track_index + 1}/{len(tracks)}", percent)
            run_result = run_model_spec(input_path, scratch, review, model, output_format)
            item = {
                "pair_index": position,
                "track_index": track_index,
                "track_id": str(track.get("id") or safe_track_name(_filename(track, track_index))),
                "filename": _filename(track, track_index),
                "genre": genre,
                "source_features": features,
                "model": label,
                "ok": bool(run_result.get("ok")),
                "runtime_seconds": run_result.get("runtime_seconds"),
                "error": run_result.get("error"),
                "error_type": run_result.get("error_type"),
                "stems": _stem_rows(run_result),
            }
            results.append(item)
            shutil.rmtree(scratch, ignore_errors=True)
            shutil.rmtree(review, ignore_errors=True)

    next_cursor = end if end < len(pairs) else None
    report = {
        "ok": True,
        "mode": "benchmark_suite",
        "schema_version": 1,
        "models": models,
        "track_count": len(tracks),
        "model_count": len(models),
        "total_pairs": len(pairs),
        "cursor": cursor,
        "processed_pairs": len(results),
        "next_cursor": next_cursor,
        "complete": next_cursor is None,
        "results": results,
        "stem_leaderboard_this_batch": _leaderboard(results),
        "scoring_warning": "technical_score measures file validity, activity and headroom only. It is not a perceptual-quality verdict. Ground-truth/reference stems are required for objective separation-quality scoring.",
        "no_audio_exported": True,
    }
    return report
