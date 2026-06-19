from __future__ import annotations

import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests
import runpod

print("StemForge worker booting", flush=True)


def download_file(url: str, destination: Path) -> None:
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def upload_file_put(url: str, file_path: Path) -> None:
    headers = {"Content-Type": "application/gzip"}
    with file_path.open("rb") as file:
        response = requests.put(url, data=file, headers=headers, timeout=300)
    response.raise_for_status()


def handler(job: dict) -> dict:
    print("StemForge received job", flush=True)
    payload = job.get("input") or {}

    if payload.get("healthcheck") is True:
        return {"ok": True, "status": "ready", "service": "stemforge-worker"}

    audio_url = payload.get("audio_url")
    if not audio_url:
        return {"ok": False, "error": "Missing required input.audio_url"}

    # Lazy import so empty health checks do not load the full audio pipeline.
    from master_pack import build_master_pack

    filename = payload.get("filename")
    if not filename:
        parsed_name = Path(urlparse(audio_url).path).name
        filename = parsed_name or "track.mp3"

    model_dir = Path(
        payload.get("model_dir")
        or os.getenv("STEMFORGE_MODEL_DIR", "/models/bs_roformer_sw")
    )

    result_put_url = payload.get("result_put_url")
    result_public_url = payload.get("result_public_url")

    try:
        with tempfile.TemporaryDirectory(prefix="stemforge_") as temp_dir:
            temp_root = Path(temp_dir)
            input_path = temp_root / filename
            work_root = temp_root / "work"
            output_root = temp_root / "output"

            download_file(audio_url, input_path)

            result = build_master_pack(
                input_audio=input_path,
                work_root=work_root,
                model_dir=model_dir,
                output_root=output_root,
            )

            archive_path = Path(result["archive_path"])
            archive_size = archive_path.stat().st_size

            uploaded = False
            if result_put_url:
                upload_file_put(result_put_url, archive_path)
                uploaded = True

            return {
                "ok": True,
                "track": result["track"],
                "archive_size_bytes": archive_size,
                "uploaded": uploaded,
                "result_url": result_public_url,
                "stems": result["stems"],
            }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "error_type": exc.__class__.__name__,
        }


print("StemForge handler ready", flush=True)
runpod.serverless.start({"handler": handler})
