from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from ground_truth_benchmark import _load, _normalise_audio, _score


DEFAULT_MODELS = [
    "BS-Roformer-SW.ckpt",
    "melband_roformer_big_beta4.ckpt",
    "melband_roformer_big_beta5e.ckpt",
    "MelBandRoformerBigSYHFTV1.ckpt",
    "mel_band_roformer_vocals_becruily.ckpt",
]


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with destination.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)


def _filename_from_url(url: str, fallback: str) -> str:
    return Path(urlparse(url).path).name or fallback


def _gpu_used_mib() -> int | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        values = [int(line.strip()) for line in completed.stdout.splitlines() if line.strip().isdigit()]
        return sum(values) if values else 0
    except Exception:
        return None


def _run_with_gpu_monitor(cmd: list[str], timeout: int) -> tuple[int, str, float, int | None]:
    peak = _gpu_used_mib()
    stop = threading.Event()

    def monitor() -> None:
        nonlocal peak
        while not stop.wait(0.25):
            value = _gpu_used_mib()
            if value is not None:
                peak = value if peak is None else max(peak, value)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return completed.returncode, completed.stdout or "", time.monotonic() - started, peak
    finally:
        stop.set()
        thread.join(timeout=2)


def _find_vocals(output_dir: Path) -> Path | None:
    tokens = ("(vocals)", "_vocals", " vocals")
    files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".wav", ".flac", ".mp3", ".m4a"}
    )
    for path in files:
        lower = path.name.lower()
        if any(token in lower for token in tokens):
            return path
    return None


def _normalise_models(models: object) -> list[str]:
    if not isinstance(models, list) or not models:
        return list(DEFAULT_MODELS)
    result = []
    for item in models:
        if isinstance(item, dict):
            value = item.get("model") or item.get("model_filename") or item.get("name")
        else:
            value = item
        if value:
            result.append(str(value))
    return result or list(DEFAULT_MODELS)


def build_model_ground_truth_bakeoff(payload: dict, progress=None) -> dict:
    audio_url = str(payload.get("audio_url") or "").strip()
    references = payload.get("references") or {}
    if not audio_url:
        return {"ok": False, "mode": "model_ground_truth_bakeoff", "error": "audio_url is required"}
    if not references.get("vocals") or not references.get("instrumental"):
        return {
            "ok": False,
            "mode": "model_ground_truth_bakeoff",
            "error": "references.vocals and references.instrumental are required",
        }

    models = _normalise_models(payload.get("models"))
    timeout_seconds = max(300, min(3300, int(payload.get("model_timeout_seconds") or 1800)))
    overlap = max(2, min(50, int(payload.get("mdxc_overlap") or 8)))
    segment_size = max(32, min(4096, int(payload.get("mdxc_segment_size") or 256)))
    batch_size = max(1, min(16, int(payload.get("mdxc_batch_size") or 1)))
    use_autocast = bool(payload.get("use_autocast", True))
    model_dir = Path(os.getenv("LITELABS_AUDIO_SEPARATOR_MODEL_DIR", "/models/audio_separator"))
    model_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="litelabs_model_gt_") as temp:
        root = Path(temp)
        source = root / _filename_from_url(audio_url, "source.wav")
        source_wav = root / "source_wav" / "source.wav"
        if progress:
            progress("Downloading source mix", 3)
        _download(audio_url, source)
        _normalise_audio(source, source_wav)
        source_audio, source_rate = _load(source_wav)

        reference_audio = {}
        for stem in ("vocals", "instrumental"):
            raw = root / "references" / _filename_from_url(str(references[stem]), f"{stem}.wav")
            wav = root / "references_wav" / f"{stem}.wav"
            _download(str(references[stem]), raw)
            _normalise_audio(raw, wav)
            reference_audio[stem] = _load(wav)

        for index, model in enumerate(models):
            output_dir = root / "outputs" / f"{index:02d}"
            output_dir.mkdir(parents=True, exist_ok=True)
            if progress:
                progress(f"Running {model}", int(8 + 78 * index / max(1, len(models))))
            cmd = [
                "audio-separator",
                str(source),
                "--model_filename",
                model,
                "--model_file_dir",
                str(model_dir),
                "--output_dir",
                str(output_dir),
                "--output_format",
                "FLAC",
                "--mdxc_segment_size",
                str(segment_size),
                "--mdxc_overlap",
                str(overlap),
                "--mdxc_batch_size",
                str(batch_size),
            ]
            if use_autocast:
                cmd.append("--use_autocast")

            row = {"model": model, "ok": False, "command": cmd}
            try:
                returncode, output, runtime, peak_vram = _run_with_gpu_monitor(cmd, timeout_seconds)
                row.update(
                    {
                        "runtime_seconds": round(runtime, 3),
                        "peak_gpu_memory_mib": peak_vram,
                        "returncode": returncode,
                        "log_tail": output[-6000:],
                    }
                )
                if returncode != 0:
                    raise RuntimeError(f"audio-separator exited with code {returncode}")

                generated_vocals = _find_vocals(output_dir)
                if generated_vocals is None:
                    raise FileNotFoundError("Could not identify generated vocals file")

                vocals_wav = root / "normalised" / f"{index:02d}_vocals.wav"
                _normalise_audio(generated_vocals, vocals_wav)
                vocals_estimate, vocals_rate = _load(vocals_wav)
                if vocals_rate != source_rate:
                    raise ValueError("Source/vocal sample-rate mismatch after normalisation")

                vocals_reference, vocals_ref_rate = reference_audio["vocals"]
                instrumental_reference, instrumental_ref_rate = reference_audio["instrumental"]
                if vocals_rate != vocals_ref_rate or source_rate != instrumental_ref_rate:
                    raise ValueError("Reference sample-rate mismatch after normalisation")

                length = min(len(source_audio), len(vocals_estimate))
                if length < source_rate:
                    raise ValueError("Generated vocals are too short to derive an instrumental")
                instrumental_estimate = source_audio[:length] - vocals_estimate[:length]

                scores = {
                    "vocals": _score(vocals_reference, vocals_estimate, vocals_rate),
                    "instrumental": _score(instrumental_reference, instrumental_estimate, source_rate),
                }
                row.update(
                    {
                        "ok": True,
                        "scores": scores,
                        "generated_files": {
                            "vocals": generated_vocals.name,
                            "instrumental": "derived: source mix minus generated vocals",
                        },
                        "instrumental_derivation": "source_minus_vocals",
                    }
                )
            except Exception as exc:
                row.update({"error": str(exc), "error_type": exc.__class__.__name__})
            rows.append(row)

    leaderboards = {}
    for stem in ("vocals", "instrumental"):
        leaderboard = []
        for row in rows:
            if row.get("ok") and stem in (row.get("scores") or {}):
                entry = {
                    "model": row["model"],
                    "runtime_seconds": row.get("runtime_seconds"),
                    "peak_gpu_memory_mib": row.get("peak_gpu_memory_mib"),
                }
                entry.update(row["scores"][stem])
                leaderboard.append(entry)
        leaderboards[stem] = sorted(
            leaderboard,
            key=lambda item: (
                -float(item.get("quality_score") or 0),
                -float(item.get("si_sdr_db") or -999),
            ),
        )

    return {
        "ok": True,
        "mode": "model_ground_truth_bakeoff",
        "schema_version": 2,
        "models_requested": models,
        "runs": rows,
        "leaderboards": leaderboards,
        "settings": {
            "model_timeout_seconds": timeout_seconds,
            "mdxc_segment_size": segment_size,
            "mdxc_overlap": overlap,
            "mdxc_batch_size": batch_size,
            "use_autocast": use_autocast,
        },
        "no_audio_exported": True,
        "metric_note": "Generated vocals are scored directly. Instrumentals are derived as source mix minus generated vocals, then both are timing-, polarity- and gain-aligned to genuine references before scoring.",
    }
