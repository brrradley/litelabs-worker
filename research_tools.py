from __future__ import annotations

import json
import math
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a"}


def safe_track_name(filename: str) -> str:
    stem = Path(filename).stem or "track"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem or "track"


def run(cmd: list[str | Path]) -> None:
    print("\nRUN:", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)


def run_capture(cmd: list[str | Path], timeout: int | None = None) -> str:
    print("\nRUN CAPTURE:", " ".join(str(x) for x in cmd), flush=True)
    completed = subprocess.run(
        [str(x) for x in cmd],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    return completed.stdout or ""


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def zip_folder(source_dir: Path, archive: Path, archive_root: str) -> None:
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zip_file:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                zip_file.write(path, arcname=str(Path(archive_root) / path.relative_to(source_dir)))


def probe_duration(path: Path) -> float:
    output = run_capture([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ], timeout=60).strip()
    try:
        return float(output)
    except ValueError:
        return 0.0


def parse_float(pattern: str, text: str, default: float) -> float:
    match = re.search(pattern, text)
    if not match:
        return default
    try:
        return float(match.group(1))
    except ValueError:
        return default


def analyse_audio(path: Path) -> dict:
    duration = probe_duration(path)
    size_bytes = path.stat().st_size if path.exists() else 0
    vol = run_capture([
        "ffmpeg", "-hide_banner", "-nostats", "-i", path,
        "-af", "volumedetect", "-f", "null", "-",
    ], timeout=120)
    mean_db = parse_float(r"mean_volume:\s*(-?\d+(?:\.\d+)?) dB", vol, -99.0)
    max_db = parse_float(r"max_volume:\s*(-?\d+(?:\.\d+)?) dB", vol, -99.0)
    silence = run_capture([
        "ffmpeg", "-hide_banner", "-nostats", "-i", path,
        "-af", "silencedetect=noise=-45dB:d=0.30", "-f", "null", "-",
    ], timeout=120)
    silence_total = 0.0
    for value in re.findall(r"silence_duration:\s*(\d+(?:\.\d+)?)", silence):
        try:
            silence_total += float(value)
        except ValueError:
            pass
    active_ratio = max(0.0, min(1.0, (duration - silence_total) / duration)) if duration > 0 else 0.0
    return {
        "file": path.name,
        "size_bytes": size_bytes,
        "duration_seconds": round(duration, 3),
        "mean_db": mean_db,
        "max_db": max_db,
        "silence_seconds": round(silence_total, 3),
        "active_ratio": round(active_ratio, 4),
    }


def collect_audio_metrics(root: Path) -> list[dict]:
    metrics: list[dict] = []
    for file in sorted(root.rglob("*")):
        if file.is_file() and file.suffix.lower() in AUDIO_EXTS:
            try:
                item = analyse_audio(file)
                item["relative_path"] = str(file.relative_to(root))
                metrics.append(item)
            except Exception as exc:
                metrics.append({"file": file.name, "relative_path": str(file.relative_to(root)), "error": str(exc)})
    return metrics


def command_available(command: str) -> bool:
    return shutil.which(command) is not None


def build_system_info() -> dict:
    info: dict = {
        "ok": True,
        "mode": "system_info",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "commands": {
            "ffmpeg": command_available("ffmpeg"),
            "ffprobe": command_available("ffprobe"),
            "demucs": command_available("demucs"),
            "bs-roformer-infer": command_available("bs-roformer-infer"),
            "audio-separator": command_available("audio-separator"),
            "nvidia-smi": command_available("nvidia-smi"),
        },
        "env": {
            "STEMFORGE_MODEL_DIR": os.getenv("STEMFORGE_MODEL_DIR", ""),
            "LITELABS_AUDIO_SEPARATOR_MODEL_DIR": os.getenv("LITELABS_AUDIO_SEPARATOR_MODEL_DIR", ""),
        },
    }
    for name, cmd in {
        "ffmpeg_version": ["ffmpeg", "-version"],
        "demucs_version": ["demucs", "--version"],
        "audio_separator_help": ["audio-separator", "--help"],
        "nvidia_smi": ["nvidia-smi"],
    }.items():
        if command_available(cmd[0]):
            try:
                info[name] = "\n".join(run_capture(cmd, timeout=20).splitlines()[:20])
            except Exception as exc:
                info[name] = f"error: {exc}"
    try:
        import torch
        info["torch"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        }
    except Exception as exc:
        info["torch"] = {"error": str(exc)}
    for label, folder in {
        "stemforge_model_dir": Path(os.getenv("STEMFORGE_MODEL_DIR", "/models/bs_roformer_sw")),
        "audio_separator_model_dir": Path(os.getenv("LITELABS_AUDIO_SEPARATOR_MODEL_DIR", "/models/audio_separator")),
    }.items():
        try:
            info[label] = sorted(str(p.name) for p in folder.glob("*"))[:80] if folder.exists() else []
        except Exception as exc:
            info[label] = [f"error: {exc}"]
    return info


def analyse_source_features(path: Path) -> dict:
    try:
        import librosa
        import numpy as np
        y, sr = librosa.load(path, sr=22050, mono=True, duration=180)
        if y.size < sr:
            return {}
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo_value = float(np.asarray(tempo).reshape(-1)[0]) if np.asarray(tempo).size else 0.0
        harmonic, percussive = librosa.effects.hpss(y)
        harmonic_rms = float(np.mean(librosa.feature.rms(y=harmonic)))
        percussive_rms = float(np.mean(librosa.feature.rms(y=percussive)))
        percussive_ratio = percussive_rms / (harmonic_rms + percussive_rms + 1e-9)
        spectrum = np.abs(librosa.stft(y, n_fft=2048))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        total = float(spectrum.sum()) + 1e-9
        bass_mask = (freqs >= 55) & (freqs < 250)
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        return {
            "tempo": round(tempo_value, 2),
            "beat_count": int(len(beats)),
            "percussive_ratio": round(float(percussive_ratio), 3),
            "bass_ratio": round(float(spectrum[bass_mask].sum() / total), 3),
            "spectral_centroid": round(centroid, 2),
        }
    except Exception as exc:
        return {"error": str(exc)}


def model_label(spec: object) -> str:
    if isinstance(spec, dict):
        return str(spec.get("name") or spec.get("model") or spec.get("type") or "model")
    return str(spec)


def safe_folder_name(value: str) -> str:
    value = value.replace(":", "_")
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return value or "model"


def normalise_model_specs(models: list | None) -> list:
    if not models:
        return ["current_litelabs", "demucs:htdemucs_ft", "demucs:htdemucs_6s"]
    return models


def copy_review_files(source_root: Path, review_dir: Path) -> list[str]:
    review_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for file in sorted(source_root.rglob("*")):
        if not file.is_file():
            continue
        if file.suffix.lower() not in AUDIO_EXTS and file.name != "README.txt":
            continue
        target = review_dir / file.relative_to(source_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)
        copied.append(str(target.relative_to(review_dir.parent)))
    return copied


def run_current_litelabs(input_path: Path, scratch_dir: Path, review_dir: Path, output_format: str, progress=None) -> dict:
    from master_pack import build_master_pack
    model_dir = Path(os.getenv("STEMFORGE_MODEL_DIR", "/models/bs_roformer_sw"))
    result = build_master_pack(
        input_audio=input_path,
        work_root=scratch_dir / "work",
        model_dir=model_dir,
        output_root=scratch_dir / "output",
        output_format=output_format,
        progress=progress,
    )
    archive_path = Path(result["archive_path"])
    extracted = scratch_dir / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zip_file:
        zip_file.extractall(extracted)
    inner_root = next((p for p in extracted.iterdir() if p.is_dir()), extracted)
    review_files = copy_review_files(inner_root, review_dir)
    result["metrics"] = collect_audio_metrics(review_dir)
    result["review_files"] = review_files
    return result


def run_demucs_model(input_path: Path, scratch_dir: Path, review_dir: Path, model_name: str) -> dict:
    output_dir = scratch_dir / "demucs_output"
    run(["demucs", "-n", model_name, "-d", "cuda", "--flac", "-o", output_dir, input_path])
    review_files = copy_review_files(output_dir, review_dir / "demucs_output")
    metrics = collect_audio_metrics(review_dir)
    return {"model_name": model_name, "stems": [m["relative_path"] for m in metrics], "metrics": metrics, "review_files": review_files}


def run_audio_separator_model(input_path: Path, scratch_dir: Path, review_dir: Path, model_filename: str, output_format: str) -> dict:
    output_dir = scratch_dir / "audio_separator_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(os.getenv("LITELABS_AUDIO_SEPARATOR_MODEL_DIR", "/models/audio_separator"))
    model_dir.mkdir(parents=True, exist_ok=True)
    run([
        "audio-separator", input_path,
        "--model_filename", model_filename,
        "--model_file_dir", model_dir,
        "--output_dir", output_dir,
        "--output_format", output_format.upper(),
    ])
    review_files = copy_review_files(output_dir, review_dir / "audio_separator_output")
    metrics = collect_audio_metrics(review_dir)
    return {"model_filename": model_filename, "stems": [m["relative_path"] for m in metrics], "metrics": metrics, "review_files": review_files}


def run_model_spec(input_path: Path, scratch_dir: Path, review_dir: Path, spec: object, output_format: str, progress=None) -> dict:
    label = model_label(spec)
    started = time.monotonic()
    try:
        if isinstance(spec, dict):
            spec_type = str(spec.get("type") or "").lower()
            if spec_type == "demucs":
                result = run_demucs_model(input_path, scratch_dir, review_dir, str(spec.get("model") or "htdemucs_ft"))
            elif spec_type in {"audio_separator", "audio-separator"}:
                result = run_audio_separator_model(input_path, scratch_dir, review_dir, str(spec.get("model") or spec.get("model_filename")), output_format)
            elif spec_type in {"current", "current_litelabs", "master_pack"}:
                result = run_current_litelabs(input_path, scratch_dir, review_dir, output_format, progress=progress)
            else:
                raise ValueError(f"Unknown model spec type: {spec_type}")
        else:
            text = str(spec)
            if text == "current_litelabs":
                result = run_current_litelabs(input_path, scratch_dir, review_dir, output_format, progress=progress)
            elif text.startswith("demucs:"):
                result = run_demucs_model(input_path, scratch_dir, review_dir, text.split(":", 1)[1])
            elif text.startswith("audio_separator:") or text.startswith("audio-separator:"):
                result = run_audio_separator_model(input_path, scratch_dir, review_dir, text.split(":", 1)[1], output_format)
            else:
                raise ValueError(f"Unknown model spec: {text}")
        result.update({"label": label, "ok": True, "runtime_seconds": round(time.monotonic() - started, 3)})
        return result
    except Exception as exc:
        return {"label": label, "ok": False, "runtime_seconds": round(time.monotonic() - started, 3), "error": str(exc), "error_type": exc.__class__.__name__}


def stem_role(path: Path) -> str:
    lower = path.name.lower()

    # audio-separator names often include the model name after the actual stem,
    # for example: track_(other)_mel_band_roformer_vocals_becruily.flac.
    # Read explicit parenthesised stem tokens before scanning the model name.
    match = re.search(r"[_\s-]\((vocals?|instrumental|drums?|bass|guitar|piano|keys|other)\)", lower)
    if match:
        token = match.group(1)
        if token.startswith("vocal"):
            return "vocals"
        if token == "instrumental":
            return "instrumental"
        if token.startswith("drum"):
            return "drums"
        if token == "bass":
            return "bass"
        if token == "guitar":
            return "guitar"
        if token in {"piano", "keys"}:
            return "piano_keys"
        if token == "other":
            return "other"

    # Use leading LiteLABS numeric prefixes before checking model names.
    prefix_match = re.match(r"^\d+_[^_]+.*?_(vocals|drums|bass|guitar|piano|keys|instrumental|other|synth|strings)", lower)
    if prefix_match:
        token = prefix_match.group(1)
        if token == "vocals":
            return "vocals"
        if token == "drums":
            return "drums"
        if token == "bass":
            return "bass"
        if token == "guitar":
            return "guitar"
        if token in {"piano", "keys"}:
            return "piano_keys"
        if token == "instrumental":
            return "instrumental"
        return "other"

    if "instrumental" in lower:
        return "instrumental"
    if "drum" in lower:
        return "drums"
    if "bass" in lower:
        return "bass"
    if "guitar" in lower:
        return "guitar"
    if "piano" in lower or "keys" in lower:
        return "piano_keys"
    if "other" in lower or "synth" in lower or "strings" in lower:
        return "other"
    if "vocal" in lower:
        return "vocals"
    return "unknown"


def review_audio_files(review_dir: Path) -> list[Path]:
    return sorted(p for p in review_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


def find_role_files(review_dir: Path) -> dict[str, list[Path]]:
    roles: dict[str, list[Path]] = {}
    for file in review_audio_files(review_dir):
        roles.setdefault(stem_role(file), []).append(file)
    return roles


def ffmpeg_mix(files: list[Path], output: Path) -> None:
    if not files:
        raise ValueError("No files supplied for mix")
    cmd: list[str | Path] = ["ffmpeg", "-y"]
    for file in files:
        cmd.extend(["-i", file])
    inputs = "".join(f"[{i}:a]" for i in range(len(files)))
    cmd.extend(["-filter_complex", f"{inputs}amix=inputs={len(files)}:duration=first:normalize=0[out]", "-map", "[out]", "-c:a", "pcm_f32le", output])
    run(cmd)


def ffmpeg_residual(source: Path, stem_sum: Path, output: Path) -> None:
    run([
        "ffmpeg", "-y", "-i", source, "-i", stem_sum,
        "-filter_complex", "[1:a]volume=-1[inv];[0:a][inv]amix=inputs=2:duration=first:normalize=0[out]",
        "-map", "[out]", "-c:a", "pcm_f32le", output,
    ])


def reconstruction_tests(source: Path, review_root: Path, runs: list[dict]) -> list[dict]:
    tests: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="litelabs_analysis_") as temp:
        temp_root = Path(temp)
        source_stats = analyse_audio(source)
        for run_result in runs:
            label = str(run_result.get("label") or "model")
            folder = review_root / safe_folder_name(label)
            candidates = [p for p in review_root.iterdir() if p.is_dir() and p.name.endswith(safe_folder_name(label))]
            if candidates:
                folder = candidates[0]
            roles = find_role_files(folder)
            scenarios: list[tuple[str, list[Path]]] = []
            component_files = [f for f in review_audio_files(folder) if stem_role(f) != "instrumental"]
            if component_files:
                scenarios.append(("sum_component_stems", component_files))
            if roles.get("vocals") and roles.get("instrumental"):
                scenarios.append(("vocals_plus_instrumental", [roles["vocals"][0], roles["instrumental"][0]]))
            for scenario, files in scenarios:
                try:
                    mix_path = temp_root / f"{safe_folder_name(label)}_{scenario}_sum.wav"
                    residual_path = temp_root / f"{safe_folder_name(label)}_{scenario}_residual.wav"
                    ffmpeg_mix(files, mix_path)
                    ffmpeg_residual(source, mix_path, residual_path)
                    mix_stats = analyse_audio(mix_path)
                    residual_stats = analyse_audio(residual_path)
                    residual_delta = residual_stats.get("mean_db", -99.0) - source_stats.get("mean_db", -99.0)
                    tests.append({
                        "label": label,
                        "scenario": scenario,
                        "stem_count": len(files),
                        "stem_roles": [stem_role(f) for f in files],
                        "sum_max_db": mix_stats.get("max_db"),
                        "sum_mean_db": mix_stats.get("mean_db"),
                        "residual_mean_db": residual_stats.get("mean_db"),
                        "residual_delta_db_vs_source": round(float(residual_delta), 3),
                        "clipping_risk": bool(float(mix_stats.get("max_db", -99.0)) >= -0.1),
                    })
                except Exception as exc:
                    tests.append({"label": label, "scenario": scenario, "error": str(exc)})
    return tests


def audio_similarity(a: Path, b: Path, duration: int = 90) -> dict:
    try:
        import librosa
        import numpy as np
        ya, sr = librosa.load(a, sr=22050, mono=True, duration=duration)
        yb, _ = librosa.load(b, sr=22050, mono=True, duration=duration)
        n = min(len(ya), len(yb))
        if n < 1024:
            return {"error": "too short"}
        ya = ya[:n]
        yb = yb[:n]
        denom = float(np.linalg.norm(ya) * np.linalg.norm(yb)) + 1e-12
        cosine = float(np.dot(ya, yb) / denom)
        return {"cosine_similarity_90s": round(cosine, 6)}
    except Exception as exc:
        return {"error": str(exc)}


def stem_similarity_matrix(review_root: Path, runs: list[dict]) -> list[dict]:
    model_roles: dict[str, dict[str, Path]] = {}
    for run_result in runs:
        label = str(run_result.get("label") or "model")
        candidates = [p for p in review_root.iterdir() if p.is_dir() and p.name.endswith(safe_folder_name(label))]
        if not candidates:
            continue
        roles = find_role_files(candidates[0])
        model_roles[label] = {role: files[0] for role, files in roles.items() if files and role != "unknown"}
    rows: list[dict] = []
    labels = sorted(model_roles)
    for i, left in enumerate(labels):
        for right in labels[i + 1:]:
            for role in sorted(set(model_roles[left]) & set(model_roles[right])):
                rows.append({"left": left, "right": right, "role": role, **audio_similarity(model_roles[left][role], model_roles[right][role])})
    return rows


def classify_genre_hint(filename: str, source_features: dict, runs: list[dict]) -> dict:
    name = filename.lower()
    tempo = float(source_features.get("tempo", 0.0) or 0.0) if isinstance(source_features, dict) else 0.0
    percussive = float(source_features.get("percussive_ratio", 0.0) or 0.0) if isinstance(source_features, dict) else 0.0
    bass = float(source_features.get("bass_ratio", 0.0) or 0.0) if isinstance(source_features, dict) else 0.0
    reason: list[str] = []
    if any(token in name for token in ["streets", "fit_but", "fit but"]):
        return {"hint": "indie_rock_garage_vocal_alt_pop", "confidence": "medium", "reason": "known stress-track style: guitar-led UK garage/indie vocal production, not a pure rock-band demix"}
    if 118 <= tempo <= 136 and percussive >= 0.48 and bass >= 0.13:
        reason.append(f"dance tempo/rhythm profile ({tempo:.0f} BPM, percussive {percussive:.2f}, bass {bass:.2f})")
        return {"hint": "electronic_dance", "confidence": "medium", "reason": "; ".join(reason)}
    if percussive >= 0.42 and bass >= 0.10:
        reason.append(f"rhythmic mixed production ({tempo:.0f} BPM, percussive {percussive:.2f})")
        return {"hint": "guitar_pop_or_rhythm_heavy_vocal", "confidence": "low", "reason": "; ".join(reason)}
    return {"hint": "mixed_or_unknown", "confidence": "low", "reason": "source features did not strongly match a research genre bucket"}


def summarise_bakeoff(runs: list[dict], reconstruction: list[dict]) -> dict:
    ok_runs = [r for r in runs if r.get("ok")]
    fastest = min(ok_runs, key=lambda r: float(r.get("runtime_seconds", 999999))) if ok_runs else None
    best_reconstruction = None
    valid = [r for r in reconstruction if "residual_delta_db_vs_source" in r and r.get("scenario") == "sum_component_stems"]
    if valid:
        best_reconstruction = min(valid, key=lambda r: float(r.get("residual_delta_db_vs_source", 999999)))
    return {
        "fastest_model": fastest.get("label") if fastest else None,
        "fastest_runtime_seconds": fastest.get("runtime_seconds") if fastest else None,
        "best_component_reconstruction_model": best_reconstruction.get("label") if best_reconstruction else None,
        "best_component_reconstruction_delta_db": best_reconstruction.get("residual_delta_db_vs_source") if best_reconstruction else None,
        "clipping_risks": [r for r in reconstruction if r.get("clipping_risk")],
        "notes": [
            "Lower residual_delta_db_vs_source is better for reconstruction/null testing.",
            "Clean instrumental stems are treated as convenience stems; component-stem reconstruction is scored separately.",
            "Chas/MVSEP-style weighted ensemble work should use these metrics as guidance, not as a copy of any private flow diagram.",
        ],
    }


def build_model_bakeoff(input_path: Path, output_root: Path, filename: str, models: list | None = None, output_format: str = "flac", progress=None) -> dict:
    track = safe_track_name(filename)
    output_format = (output_format or "flac").lower().strip()
    if output_format not in {"flac", "mp3"}:
        output_format = "flac"
    bakeoff_dir = output_root / f"{track}-model-bakeoff"
    scratch_root = output_root / f"{track}-scratch"
    bakeoff_dir.mkdir(parents=True, exist_ok=True)
    scratch_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, bakeoff_dir / f"00_source_{input_path.name}")
    specs = normalise_model_specs(models)
    runs: list[dict] = []
    total = max(1, len(specs))
    for index, spec in enumerate(specs, start=1):
        label = model_label(spec)
        if progress:
            progress(f"Running research model {index}/{total}: {label}", int(10 + (index - 1) * (65 / total)))
        review_dir = bakeoff_dir / f"{index:02d}_{safe_folder_name(label)}"
        scratch_dir = scratch_root / f"{index:02d}_{safe_folder_name(label)}"
        result = run_model_spec(input_path, scratch_dir, review_dir, spec, output_format, progress=progress)
        write_json(review_dir / "run_result.json", result)
        runs.append(result)
    if progress:
        progress("Running forensic bake-off analysis", 82)
    source_features = analyse_source_features(input_path)
    reconstruction = reconstruction_tests(input_path, bakeoff_dir, runs)
    similarities = stem_similarity_matrix(bakeoff_dir, runs)
    genre_hint = classify_genre_hint(filename, source_features, runs)
    benchmark = {
        "benchmark_id": f"001_{track}" if "Streets" in filename or "streets" in filename.lower() else track,
        "keep_for_comparison": True,
        "source_file": input_path.name,
        "purpose": "LiteLABS research bake-off; compare future model reports against this baseline when relevant.",
    }
    report = {
        "track": track,
        "source_file": input_path.name,
        "output_format": output_format,
        "models_requested": specs,
        "benchmark": benchmark,
        "genre_hint": genre_hint,
        "source_features": source_features,
        "runs": runs,
        "forensic_analysis": {
            "reconstruction_tests": reconstruction,
            "stem_similarity_matrix": similarities,
            "summary": summarise_bakeoff(runs, reconstruction),
        },
        "system_info": build_system_info(),
    }
    write_json(bakeoff_dir / "research_report.json", report)
    write_json(bakeoff_dir / "benchmark_history.json", {"benchmarks": [benchmark]})
    lines = [
        "LiteLABS research model bake-off", "",
        f"Track: {track}",
        f"Source: {input_path.name}",
        f"Models tested: {len(specs)}", "",
        f"Genre hint: {genre_hint.get('hint')} ({genre_hint.get('confidence')})",
        f"Genre reason: {genre_hint.get('reason')}", "",
        "Runs:",
    ]
    for result in runs:
        status = "OK" if result.get("ok") else "FAILED"
        runtime = result.get("runtime_seconds", "?")
        lines.append(f"- {result.get('label')}: {status} ({runtime}s)")
        if not result.get("ok"):
            lines.append(f"  Error: {result.get('error')}")
    summary = report["forensic_analysis"]["summary"]
    lines.extend([
        "", "Forensic summary:",
        f"- Fastest model: {summary.get('fastest_model')} ({summary.get('fastest_runtime_seconds')}s)",
        f"- Best component reconstruction: {summary.get('best_component_reconstruction_model')} ({summary.get('best_component_reconstruction_delta_db')} dB delta vs source)",
        f"- Clipping-risk checks: {len(summary.get('clipping_risks') or [])}",
        "", "Use research_report.json for full null/reconstruction tests, clipping risk, similarity matrix and metrics.",
    ])
    write_text(bakeoff_dir / "README.txt", "\n".join(lines) + "\n")
    archive = output_root / f"{track}-model-bakeoff.zip"
    if progress:
        progress("Creating research ZIP", 92)
    zip_folder(bakeoff_dir, archive, bakeoff_dir.name)
    return {
        "track": track,
        "archive_path": str(archive),
        "archive_size_bytes": archive.stat().st_size,
        "files": sorted(str(p.relative_to(bakeoff_dir)) for p in bakeoff_dir.rglob("*") if p.is_file()),
        "runs": [{"label": r.get("label"), "ok": r.get("ok"), "runtime_seconds": r.get("runtime_seconds"), "error": r.get("error"), "review_files": r.get("review_files", [])} for r in runs],
    }


def build_vocal_residual_test(vocals_path: Path, lead_path: Path, output_root: Path, filename: str) -> dict:
    """Create lead/backing/check files from an existing full vocal stem and lead vocal stem."""
    if not vocals_path.exists():
        raise FileNotFoundError(f"Missing full vocals input: {vocals_path}")
    if not lead_path.exists():
        raise FileNotFoundError(f"Missing lead vocals input: {lead_path}")
    track = safe_track_name(filename)
    pack_dir = output_root / f"{track}-vocal-residual-test"
    pack_dir.mkdir(parents=True, exist_ok=True)
    lead_out = pack_dir / f"01_{track}_lead_vocals.flac"
    backing_out = pack_dir / f"02_{track}_backing_vocals_residual.flac"
    check_out = pack_dir / f"03_{track}_lead_plus_backing_check.flac"
    full_copy = pack_dir / f"00_{track}_full_vocals_input.flac"
    run(["ffmpeg", "-y", "-i", vocals_path, "-c:a", "flac", full_copy])
    run(["ffmpeg", "-y", "-i", lead_path, "-c:a", "flac", lead_out])
    run([
        "ffmpeg", "-y", "-i", vocals_path, "-i", lead_path,
        "-filter_complex", "[1:a]volume=-1[invlead];[0:a][invlead]amix=inputs=2:duration=first:normalize=0[out]",
        "-map", "[out]", "-c:a", "flac", backing_out,
    ])
    run([
        "ffmpeg", "-y", "-i", lead_out, "-i", backing_out,
        "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first:normalize=0[out]",
        "-map", "[out]", "-c:a", "flac", check_out,
    ])
    readme = pack_dir / "README.txt"
    write_text(
        readme,
        "LiteLABS research vocal residual test\n\n"
        f"Track: {track}\n\n"
        "Files included:\n"
        f"00_{track}_full_vocals_input.flac\n"
        f"01_{track}_lead_vocals.flac\n"
        f"02_{track}_backing_vocals_residual.flac\n"
        f"03_{track}_lead_plus_backing_check.flac\n\n"
        "Method:\n"
        "backing_vocals_residual = full_vocals_input - lead_vocals\n"
        "lead_plus_backing_check = lead_vocals + backing_vocals_residual\n\n"
        "The check file should sound very close to the original full vocal input if the lead/backing split is mathematically tidy.\n",
    )
    archive = output_root / f"{track}-vocal-residual-test.zip"
    zip_folder(pack_dir, archive, pack_dir.name)
    return {"track": track, "archive_path": str(archive), "files": sorted(p.name for p in pack_dir.iterdir() if p.is_file())}
