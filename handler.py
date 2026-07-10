from __future__ import annotations

import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests
import runpod

print("LiteLABS research worker booting", flush=True)


def post_progress(url: str | None, token: str | None, job_id: str | int | None, message: str, percent: int) -> None:
    print(f"LiteLABS progress {percent}%: {message}", flush=True)
    if not url or not token or not job_id:
        return
    try:
        requests.post(
            url,
            json={
                "token": token,
                "job_id": job_id,
                "message": message,
                "percent": max(0, min(100, int(percent))),
            },
            timeout=8,
        )
    except Exception as exc:
        print(f"LiteLABS progress callback failed: {exc}", flush=True)


def download_file(url: str, destination: Path) -> None:
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def content_type_for(file_path: Path) -> str:
    lower = file_path.name.lower()
    if lower.endswith(".zip"):
        return "application/zip"
    if lower.endswith(".tar.gz") or lower.endswith(".tgz"):
        return "application/gzip"
    return "application/octet-stream"


def upload_file_put(url: str, file_path: Path) -> None:
    headers = {"Content-Type": content_type_for(file_path)}
    with file_path.open("rb") as file:
        response = requests.put(url, data=file, headers=headers, timeout=300)
    response.raise_for_status()


def infer_filename(url: str, fallback: str) -> str:
    parsed_name = Path(urlparse(url).path).name
    return parsed_name or fallback


def handler(job: dict) -> dict:
    print("LiteLABS research job received", flush=True)
    payload = job.get("input") or {}

    if payload.get("healthcheck") is True:
        return {
            "ok": True,
            "status": "ready",
            "service": "litelabs-research-worker",
            "modes": ["system_info", "master_pack", "model_bakeoff", "vocal_residual_test"],
        }

    mode = payload.get("mode") or "master_pack"
    result_put_url = payload.get("result_put_url")
    result_public_url = payload.get("result_public_url")
    progress_url = payload.get("progress_url")
    progress_token = payload.get("progress_token")
    progress_job_id = payload.get("progress_job_id")

    def progress(message: str, percent: int) -> None:
        post_progress(progress_url, progress_token, progress_job_id, message, percent)

    try:
        if mode == "system_info":
            from research_tools import build_system_info

            return build_system_info()

        with tempfile.TemporaryDirectory(prefix="litelabs_research_") as temp_dir:
            temp_root = Path(temp_dir)
            output_root = temp_root / "output"
            output_root.mkdir(parents=True, exist_ok=True)

            if mode == "model_bakeoff":
                audio_url = payload.get("audio_url")
                if not audio_url:
                    return {"ok": False, "error": "Missing required input.audio_url"}

                filename = payload.get("filename") or infer_filename(audio_url, "track.mp3")
                input_path = temp_root / filename
                output_format = str(payload.get("output_format") or "flac").lower().strip()
                models = payload.get("models")

                progress("Downloading research audio", 5)
                download_file(audio_url, input_path)
                progress("Research audio downloaded", 8)

                from research_tools import build_model_bakeoff

                result = build_model_bakeoff(
                    input_path=input_path,
                    output_root=output_root,
                    filename=filename,
                    models=models,
                    output_format=output_format,
                    progress=progress,
                )
                archive_path = Path(result["archive_path"])
                archive_size = archive_path.stat().st_size

                uploaded = False
                if result_put_url:
                    progress("Uploading research bake-off ZIP", 95)
                    upload_file_put(result_put_url, archive_path)
                    uploaded = True

                progress("Research bake-off ready", 100)
                return {
                    "ok": True,
                    "mode": mode,
                    "track": result["track"],
                    "archive_size_bytes": archive_size,
                    "uploaded": uploaded,
                    "result_url": result_public_url,
                    "runs": result["runs"],
                    "files": result["files"],
                }

            if mode == "vocal_residual_test":
                vocals_url = payload.get("vocals_url") or payload.get("audio_url")
                lead_vocals_url = payload.get("lead_vocals_url") or payload.get("lead_url")
                if not vocals_url:
                    return {"ok": False, "error": "Missing input.vocals_url or input.audio_url"}
                if not lead_vocals_url:
                    return {"ok": False, "error": "Missing input.lead_vocals_url"}

                filename = payload.get("filename") or infer_filename(vocals_url, "vocals.flac")
                vocals_path = temp_root / filename
                lead_path = temp_root / (payload.get("lead_filename") or infer_filename(lead_vocals_url, "lead-vocals.flac"))

                progress("Downloading full vocal stem", 10)
                download_file(vocals_url, vocals_path)
                progress("Downloading lead vocal stem", 20)
                download_file(lead_vocals_url, lead_path)

                from research_tools import build_vocal_residual_test

                progress("Creating backing vocal residual", 45)
                result = build_vocal_residual_test(
                    vocals_path=vocals_path,
                    lead_path=lead_path,
                    output_root=output_root,
                    filename=filename,
                )
                archive_path = Path(result["archive_path"])
                archive_size = archive_path.stat().st_size

                uploaded = False
                if result_put_url:
                    progress("Uploading research ZIP", 90)
                    upload_file_put(result_put_url, archive_path)
                    uploaded = True

                progress("Research pack ready", 100)
                return {
                    "ok": True,
                    "mode": mode,
                    "track": result["track"],
                    "archive_size_bytes": archive_size,
                    "uploaded": uploaded,
                    "result_url": result_public_url,
                    "files": result["files"],
                }

            if mode != "master_pack":
                return {"ok": False, "error": f"Unknown research mode: {mode}"}

            audio_url = payload.get("audio_url")
            if not audio_url:
                return {"ok": False, "error": "Missing required input.audio_url"}

            from master_pack import build_master_pack

            filename = payload.get("filename") or infer_filename(audio_url, "track.mp3")
            input_path = temp_root / filename
            work_root = temp_root / "work"
            model_dir = Path(
                payload.get("model_dir")
                or os.getenv("STEMFORGE_MODEL_DIR", "/models/bs_roformer_sw")
            )

            progress("Worker starting", 12)
            progress("Downloading audio", 15)
            download_file(audio_url, input_path)
            progress("Audio downloaded", 17)

            result = build_master_pack(
                input_audio=input_path,
                work_root=work_root,
                model_dir=model_dir,
                output_root=output_root,
                progress=progress,
            )

            archive_path = Path(result["archive_path"])
            archive_size = archive_path.stat().st_size

            uploaded = False
            if result_put_url:
                progress("Uploading ZIP back to LiteRECORDS", 94)
                upload_file_put(result_put_url, archive_path)
                uploaded = True
                progress("Finalising download", 98)

            return {
                "ok": True,
                "mode": mode,
                "track": result["track"],
                "archive_size_bytes": archive_size,
                "uploaded": uploaded,
                "result_url": result_public_url,
                "stems": result["stems"],
            }
    except Exception as exc:
        post_progress(progress_url, progress_token, progress_job_id, f"Worker error: {exc}", 100)
        return {
            "ok": False,
            "mode": mode,
            "error": str(exc),
            "error_type": exc.__class__.__name__,
        }


print("LiteLABS research handler ready", flush=True)
runpod.serverless.start({"handler": handler})
