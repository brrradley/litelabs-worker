from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests
import runpod

print("LiteLABS research worker booting", flush=True)


def post_progress(url, token, job_id, message: str, percent: int) -> None:
    print(f"LiteLABS progress {percent}%: {message}", flush=True)
    if not url or not token or not job_id:
        return
    try:
        requests.post(url, json={"token": token, "job_id": job_id, "message": message, "percent": max(0, min(100, int(percent)))}, timeout=8)
    except Exception as exc:
        print(f"LiteLABS progress callback failed: {exc}", flush=True)


def download_file(url: str, destination: Path) -> None:
    with requests.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        with destination.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def content_type_for(path: Path) -> str:
    return "application/zip" if path.name.lower().endswith(".zip") else "application/octet-stream"


def upload_file_put(url: str, path: Path) -> None:
    with path.open("rb") as file:
        response = requests.put(url, data=file, headers={"Content-Type": content_type_for(path)}, timeout=300)
    response.raise_for_status()


def infer_filename(url: str, fallback: str) -> str:
    return Path(urlparse(url).path).name or fallback


def run_discovery_command(cmd: list[str], timeout: int = 90, output_limit: int = 12000) -> dict:
    try:
        completed = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        output = completed.stdout or ""
        if output_limit > 0 and len(output) > output_limit:
            output = output[-output_limit:]
        return {"command": cmd, "returncode": completed.returncode, "ok": completed.returncode == 0, "output": output}
    except Exception as exc:
        return {"command": cmd, "ok": False, "error": str(exc), "error_type": exc.__class__.__name__}


def build_audio_separator_discovery(payload: dict | None = None) -> dict:
    payload = payload or {}
    model_dir = Path(os.getenv("LITELABS_AUDIO_SEPARATOR_MODEL_DIR", "/models/audio_separator"))
    model_dir.mkdir(parents=True, exist_ok=True)
    list_filter = str(payload.get("list_filter") or payload.get("stem") or "").strip().lower()
    list_limit = max(1, min(250, int(payload.get("list_limit") or 100)))
    list_format = str(payload.get("list_format") or "pretty").strip().lower()
    if list_format not in {"pretty", "json"}:
        list_format = "pretty"
    command = ["audio-separator", "--list_models", "--list_limit", str(list_limit), "--list_format", list_format]
    if list_filter:
        command.extend(["--list_filter", list_filter])
    commands = []
    if bool(payload.get("include_help", False)):
        commands.append(run_discovery_command(["audio-separator", "--help"], output_limit=20000))
    commands.append(run_discovery_command(command, output_limit=100000))
    files = sorted(str(path.relative_to(model_dir)) for path in model_dir.rglob("*") if path.is_file())[:250]
    return {"ok": True, "mode": "audio_separator_discovery", "list_filter": list_filter or None, "list_limit": list_limit, "list_format": list_format, "env": {"LITELABS_AUDIO_SEPARATOR_MODEL_DIR": str(model_dir), "STEMFORGE_MODEL_DIR": os.getenv("STEMFORGE_MODEL_DIR", "")}, "model_dir_files": files, "commands": commands}


def load_ground_truth_builder():
    try:
        from ground_truth_benchmark import build_ground_truth_benchmark
        return build_ground_truth_benchmark
    except ModuleNotFoundError:
        fallback_url = "https://raw.githubusercontent.com/brrradley/litelabs-worker/research/ground_truth_benchmark.py"
        response = requests.get(fallback_url, timeout=60)
        response.raise_for_status()
        namespace = {"__name__": "ground_truth_benchmark_runtime", "__file__": fallback_url}
        exec(compile(response.text, fallback_url, "exec"), namespace)
        return namespace["build_ground_truth_benchmark"]


def handler(job: dict) -> dict:
    print("LiteLABS research job received", flush=True)
    payload = job.get("input") or {}
    modes = ["system_info", "master_pack", "model_bakeoff", "benchmark_suite", "ground_truth_benchmark", "model_ground_truth_bakeoff", "cascade_ground_truth_bakeoff", "multi_case_ground_truth_bakeoff", "adaptive_research_campaign", "stem_pack_inventory", "stem_pack_compare", "studio_mix_compatibility", "vocal_residual_test", "audio_separator_discovery"]

    if payload.get("healthcheck") is True:
        status = {}
        checks = {
            "ground_truth_benchmark": ("ground_truth_benchmark", "build_ground_truth_benchmark"),
            "model_ground_truth_bakeoff": ("model_ground_truth_bakeoff", "build_model_ground_truth_bakeoff"),
            "cascade_ground_truth_bakeoff": ("cascade_ground_truth_bakeoff", "build_cascade_ground_truth_bakeoff"),
            "multi_case_ground_truth_bakeoff": ("multi_case_ground_truth_bakeoff", "build_multi_case_ground_truth_bakeoff"),
            "adaptive_research_campaign": ("adaptive_research_campaign", "build_adaptive_research_campaign"),
            "stem_pack_inventory": ("stem_pack_inventory", "build_stem_pack_inventory"),
            "stem_pack_compare": ("stem_pack_compare", "build_stem_pack_compare"),
            "studio_mix_compatibility": ("studio_mix_compatibility", "build_studio_mix_compatibility"),
        }
        for key, (module_name, attribute) in checks.items():
            try:
                module = __import__(module_name, fromlist=[attribute])
                getattr(module, attribute)
                status[key] = True
            except Exception as exc:
                status[key] = False
                status[f"{key}_error"] = str(exc)
        return {"ok": True, "status": "ready", "service": "litelabs-research-worker", "modes": modes, "module_status": status}

    mode = payload.get("mode") or "master_pack"
    progress_url = payload.get("progress_url")
    progress_token = payload.get("progress_token")
    progress_job_id = payload.get("progress_job_id")
    result_put_url = payload.get("result_put_url")
    result_public_url = payload.get("result_public_url")

    def progress(message: str, percent: int) -> None:
        post_progress(progress_url, progress_token, progress_job_id, message, percent)

    try:
        if mode == "system_info":
            from research_tools import build_system_info
            return build_system_info()
        if mode == "audio_separator_discovery":
            return build_audio_separator_discovery(payload)
        if mode == "benchmark_suite":
            from benchmark_suite import build_benchmark_suite
            return build_benchmark_suite(payload, progress=progress)
        if mode == "ground_truth_benchmark":
            return load_ground_truth_builder()(payload, progress=progress)
        if mode == "model_ground_truth_bakeoff":
            from model_ground_truth_bakeoff import build_model_ground_truth_bakeoff
            return build_model_ground_truth_bakeoff(payload, progress=progress)
        if mode == "cascade_ground_truth_bakeoff":
            from cascade_ground_truth_bakeoff import build_cascade_ground_truth_bakeoff
            return build_cascade_ground_truth_bakeoff(payload, progress=progress)
        if mode == "multi_case_ground_truth_bakeoff":
            from multi_case_ground_truth_bakeoff import build_multi_case_ground_truth_bakeoff
            return build_multi_case_ground_truth_bakeoff(payload, progress=progress)
        if mode == "adaptive_research_campaign":
            from adaptive_research_campaign import build_adaptive_research_campaign
            return build_adaptive_research_campaign(payload, progress=progress)
        if mode == "stem_pack_inventory":
            from stem_pack_inventory import build_stem_pack_inventory
            return build_stem_pack_inventory(payload, progress=progress)
        if mode == "stem_pack_compare":
            from stem_pack_compare import build_stem_pack_compare
            return build_stem_pack_compare(payload, progress=progress)
        if mode == "studio_mix_compatibility":
            from studio_mix_compatibility import build_studio_mix_compatibility
            return build_studio_mix_compatibility(payload, progress=progress)

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
                download_file(audio_url, input_path)
                from research_tools import build_model_bakeoff
                result = build_model_bakeoff(input_path=input_path, output_root=output_root, filename=filename, models=payload.get("models"), output_format=str(payload.get("output_format") or "flac").lower().strip(), progress=progress)
                archive_path = Path(result["archive_path"])
                uploaded = False
                if result_put_url:
                    upload_file_put(result_put_url, archive_path)
                    uploaded = True
                return {"ok": True, "mode": mode, "track": result["track"], "archive_size_bytes": archive_path.stat().st_size, "uploaded": uploaded, "result_url": result_public_url, "runs": result["runs"], "files": result["files"]}
            if mode == "vocal_residual_test":
                vocals_url = payload.get("vocals_url") or payload.get("audio_url")
                lead_url = payload.get("lead_vocals_url") or payload.get("lead_url")
                if not vocals_url or not lead_url:
                    return {"ok": False, "error": "Missing vocal URLs"}
                filename = payload.get("filename") or infer_filename(vocals_url, "vocals.flac")
                vocals_path = temp_root / filename
                lead_path = temp_root / (payload.get("lead_filename") or infer_filename(lead_url, "lead-vocals.flac"))
                download_file(vocals_url, vocals_path)
                download_file(lead_url, lead_path)
                from research_tools import build_vocal_residual_test
                result = build_vocal_residual_test(vocals_path=vocals_path, lead_path=lead_path, output_root=output_root, filename=filename)
                archive_path = Path(result["archive_path"])
                uploaded = False
                if result_put_url:
                    upload_file_put(result_put_url, archive_path)
                    uploaded = True
                return {"ok": True, "mode": mode, "track": result["track"], "archive_size_bytes": archive_path.stat().st_size, "uploaded": uploaded, "result_url": result_public_url, "files": result["files"]}
            if mode != "master_pack":
                return {"ok": False, "error": f"Unknown research mode: {mode}"}
            audio_url = payload.get("audio_url")
            if not audio_url:
                return {"ok": False, "error": "Missing required input.audio_url"}
            filename = payload.get("filename") or infer_filename(audio_url, "track.mp3")
            input_path = temp_root / filename
            download_file(audio_url, input_path)
            from master_pack import build_master_pack
            result = build_master_pack(input_audio=input_path, work_root=temp_root / "work", model_dir=Path(payload.get("model_dir") or os.getenv("STEMFORGE_MODEL_DIR", "/models/bs_roformer_sw")), output_root=output_root, progress=progress)
            archive_path = Path(result["archive_path"])
            uploaded = False
            if result_put_url:
                upload_file_put(result_put_url, archive_path)
                uploaded = True
            return {"ok": True, "mode": mode, "track": result["track"], "archive_size_bytes": archive_path.stat().st_size, "uploaded": uploaded, "result_url": result_public_url, "stems": result["stems"]}
    except Exception as exc:
        post_progress(progress_url, progress_token, progress_job_id, f"Worker error: {exc}", 100)
        return {"ok": False, "mode": mode, "error": str(exc), "error_type": exc.__class__.__name__}


print("LiteLABS research handler ready", flush=True)
runpod.serverless.start({"handler": handler})
