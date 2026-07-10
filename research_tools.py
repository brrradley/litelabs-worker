from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import time
import zipfile
from pathlib import Path


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
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: dict | list) -> None:
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
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]).strip()
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
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i", path,
        "-af", "volumedetect",
        "-f", "null",
        "-",
    ])
    mean_db = parse_float(r"mean_volume:\s*(-?\d+(?:\.\d+)?) dB", vol, -99.0)
    max_db = parse_float(r"max_volume:\s*(-?\d+(?:\.\d+)?) dB", vol, -99.0)

    silence = run_capture([
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i", path,
        "-af", "silencedetect=noise=-45dB:d=0.30",
        "-f", "null",
        "-",
    ])
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
    audio_exts = {".wav", ".flac", ".mp3", ".m4a"}
    metrics: list[dict] = []
    for file in sorted(root.rglob("*")):
        if file.is_file() and file.suffix.lower() in audio_exts:
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
                output = run_capture(cmd, timeout=20)
                info[name] = "\n".join(output.splitlines()[:20])
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


def run_current_litelabs(input_path: Path, run_dir: Path, output_format: str, progress=None) -> dict:
    from master_pack import build_master_pack

    model_dir = Path(os.getenv("STEMFORGE_MODEL_DIR", "/models/bs_roformer_sw"))
    result = build_master_pack(
        input_audio=input_path,
        work_root=run_dir / "work",
        model_dir=model_dir,
        output_root=run_dir / "output",
        output_format=output_format,
        progress=progress,
    )
    archive_path = Path(result["archive_path"])
    extracted = run_dir / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zip_file:
        zip_file.extractall(extracted)
    shutil.copy2(archive_path, run_dir / archive_path.name)
    result["archive_copy"] = str(run_dir / archive_path.name)
    result["metrics"] = collect_audio_metrics(extracted)
    return result


def run_demucs_model(input_path: Path, run_dir: Path, model_name: str) -> dict:
    output_dir = run_dir / "demucs_output"
    run(["demucs", "-n", model_name, "-d", "cuda", "--flac", "-o", output_dir, input_path])
    metrics = collect_audio_metrics(output_dir)
    archive = run_dir / f"{safe_folder_name(model_name)}.zip"
    zip_folder(output_dir, archive, f"{safe_folder_name(model_name)}")
    return {"model_name": model_name, "archive_copy": str(archive), "stems": [m["relative_path"] for m in metrics], "metrics": metrics}


def run_audio_separator_model(input_path: Path, run_dir: Path, model_filename: str, output_format: str) -> dict:
    output_dir = run_dir / "audio_separator_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(os.getenv("LITELABS_AUDIO_SEPARATOR_MODEL_DIR", "/models/audio_separator"))
    model_dir.mkdir(parents=True, exist_ok=True)
    run([
        "audio-separator",
        input_path,
        "--model_filename", model_filename,
        "--model_file_dir", model_dir,
        "--output_dir", output_dir,
        "--output_format", output_format.upper(),
    ])
    metrics = collect_audio_metrics(output_dir)
    archive = run_dir / f"{safe_folder_name(model_filename)}.zip"
    zip_folder(output_dir, archive, safe_folder_name(model_filename))
    return {"model_filename": model_filename, "archive_copy": str(archive), "stems": [m["relative_path"] for m in metrics], "metrics": metrics}


def run_model_spec(input_path: Path, run_dir: Path, spec: object, output_format: str, progress=None) -> dict:
    label = model_label(spec)
    started = time.monotonic()
    try:
        if isinstance(spec, dict):
            spec_type = str(spec.get("type") or "").lower()
            if spec_type == "demucs":
                result = run_demucs_model(input_path, run_dir, str(spec.get("model") or "htdemucs_ft"))
            elif spec_type in {"audio_separator", "audio-separator"}:
                result = run_audio_separator_model(input_path, run_dir, str(spec.get("model") or spec.get("model_filename")), output_format)
            elif spec_type in {"current", "current_litelabs", "master_pack"}:
                result = run_current_litelabs(input_path, run_dir, output_format, progress=progress)
            else:
                raise ValueError(f"Unknown model spec type: {spec_type}")
        else:
            text = str(spec)
            if text == "current_litelabs":
                result = run_current_litelabs(input_path, run_dir, output_format, progress=progress)
            elif text.startswith("demucs:"):
                result = run_demucs_model(input_path, run_dir, text.split(":", 1)[1])
            elif text.startswith("audio_separator:") or text.startswith("audio-separator:"):
                result = run_audio_separator_model(input_path, run_dir, text.split(":", 1)[1], output_format)
            else:
                raise ValueError(f"Unknown model spec: {text}")

        result.update({
            "label": label,
            "ok": True,
            "runtime_seconds": round(time.monotonic() - started, 3),
        })
        return result
    except Exception as exc:
        return {
            "label": label,
            "ok": False,
            "runtime_seconds": round(time.monotonic() - started, 3),
            "error": str(exc),
            "error_type": exc.__class__.__name__,
        }


def build_model_bakeoff(input_path: Path, output_root: Path, filename: str, models: list | None = None, output_format: str = "flac", progress=None) -> dict:
    track = safe_track_name(filename)
    output_format = (output_format or "flac").lower().strip()
    if output_format not in {"flac", "mp3"}:
        output_format = "flac"

    bakeoff_dir = output_root / f"{track}-model-bakeoff"
    bakeoff_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, bakeoff_dir / f"00_source_{input_path.name}")

    specs = normalise_model_specs(models)
    runs: list[dict] = []
    total = max(1, len(specs))

    for index, spec in enumerate(specs, start=1):
        label = model_label(spec)
        if progress:
            progress(f"Running research model {index}/{total}: {label}", int(10 + (index - 1) * (75 / total)))
        run_dir = bakeoff_dir / f"{index:02d}_{safe_folder_name(label)}"
        run_dir.mkdir(parents=True, exist_ok=True)
        result = run_model_spec(input_path, run_dir, spec, output_format, progress=progress)
        write_json(run_dir / "run_result.json", result)
        runs.append(result)

    report = {
        "track": track,
        "source_file": input_path.name,
        "output_format": output_format,
        "models_requested": specs,
        "runs": runs,
        "system_info": build_system_info(),
    }
    write_json(bakeoff_dir / "research_report.json", report)

    lines = [
        "LiteLABS research model bake-off",
        "",
        f"Track: {track}",
        f"Source: {input_path.name}",
        f"Models tested: {len(specs)}",
        "",
        "Runs:",
    ]
    for result in runs:
        status = "OK" if result.get("ok") else "FAILED"
        runtime = result.get("runtime_seconds", "?")
        lines.append(f"- {result.get('label')}: {status} ({runtime}s)")
        if not result.get("ok"):
            lines.append(f"  Error: {result.get('error')}")
    lines.extend([
        "",
        "Use research_report.json for detailed metrics. The audio files are grouped by model folder.",
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
        "runs": [{"label": r.get("label"), "ok": r.get("ok"), "runtime_seconds": r.get("runtime_seconds"), "error": r.get("error")} for r in runs],
    }


def build_vocal_residual_test(vocals_path: Path, lead_path: Path, output_root: Path, filename: str) -> dict:
    """Create lead/backing/check files from an existing full vocal stem and lead vocal stem.

    This does not run a vocal model. It is for testing Chas's residual method:

    backing vocals = full vocals - lead vocals
    check file = lead vocals + backing vocals

    If the method is behaving, the check file should sound very close to the original full vocals stem.
    """

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

    print("Creating normalised full vocal reference", flush=True)
    run(["ffmpeg", "-y", "-i", vocals_path, "-c:a", "flac", full_copy])

    print("Creating lead vocal reference", flush=True)
    run(["ffmpeg", "-y", "-i", lead_path, "-c:a", "flac", lead_out])

    print("Creating backing vocal residual", flush=True)
    run([
        "ffmpeg", "-y",
        "-i", vocals_path,
        "-i", lead_path,
        "-filter_complex", "[1:a]volume=-1[invlead];[0:a][invlead]amix=inputs=2:duration=first:normalize=0[out]",
        "-map", "[out]",
        "-c:a", "flac",
        backing_out,
    ])

    print("Creating lead plus backing check file", flush=True)
    run([
        "ffmpeg", "-y",
        "-i", lead_out,
        "-i", backing_out,
        "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first:normalize=0[out]",
        "-map", "[out]",
        "-c:a", "flac",
        check_out,
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

    return {
        "track": track,
        "archive_path": str(archive),
        "files": sorted(p.name for p in pack_dir.iterdir() if p.is_file()),
    }
