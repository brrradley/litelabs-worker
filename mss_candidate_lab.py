from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests

CANONICAL_STEMS = ("vocals", "drums", "bass", "guitar", "piano", "other")
SUPPORTED_MODEL_TYPES = (
    "mdx23c", "htdemucs", "segm_models", "torchseg", "bs_roformer",
    "mel_band_roformer", "swin_upernet", "bandit", "scnet",
    "bandit_v2", "apollo", "bs_mamba2", "conformer", "bs_conformer",
    "scnet_tran", "scnet_masked",
)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with destination.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)


def _load_registry() -> list[dict]:
    registry_path = Path(os.getenv("LITELABS_MSS_MODEL_REGISTRY", "/models/mss_training/model_registry.json"))
    if not registry_path.exists():
        return []
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    return list(payload.get("models") or payload if isinstance(payload, list) else [])


def _validate_model(entry: dict) -> dict:
    model_id = str(entry.get("id") or "").strip()
    target = str(entry.get("target_stem") or "").strip().lower()
    model_type = str(entry.get("model_type") or "").strip()
    config_path = Path(str(entry.get("config_path") or ""))
    checkpoint_path = Path(str(entry.get("checkpoint_path") or ""))
    licence = str(entry.get("licence") or "unknown").strip()
    commercial_ok = bool(entry.get("commercial_use_allowed", False))
    errors = []
    if not model_id:
        errors.append("missing id")
    if target not in CANONICAL_STEMS:
        errors.append(f"unsupported target_stem: {target}")
    if model_type not in SUPPORTED_MODEL_TYPES:
        errors.append(f"unsupported model_type: {model_type}")
    if not config_path.is_file():
        errors.append(f"missing config: {config_path}")
    if not checkpoint_path.is_file():
        errors.append(f"missing checkpoint: {checkpoint_path}")
    if licence.lower() in {"", "unknown", "unspecified"}:
        errors.append("checkpoint licence is not declared")
    if not commercial_ok:
        errors.append("commercial use is not approved")
    return {
        "id": model_id,
        "target_stem": target,
        "model_type": model_type,
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "licence": licence,
        "commercial_use_allowed": commercial_ok,
        "enabled": bool(entry.get("enabled", True)),
        "ready": not errors and bool(entry.get("enabled", True)),
        "errors": errors,
    }


def _inventory() -> dict:
    repo_dir = Path(os.getenv("LITELABS_MSS_REPO_DIR", "/opt/music-source-separation-training"))
    registry = [_validate_model(entry) for entry in _load_registry()]
    ready_by_stem = {
        stem: [item["id"] for item in registry if item["ready"] and item["target_stem"] == stem]
        for stem in CANONICAL_STEMS
    }
    return {
        "ok": True,
        "mode": "mss_candidate_lab",
        "schema_version": 1,
        "action": "inventory",
        "research_only": True,
        "goal": "Produce the fullest and highest-quality canonical stem kit from one uploaded mixture.",
        "framework": {
            "name": "ZFTurbo/Music-Source-Separation-Training",
            "repo_dir": str(repo_dir),
            "available": (repo_dir / "inference.py").is_file(),
            "pinned_commit": os.getenv("LITELABS_MSS_REPO_COMMIT", "unknown"),
            "supported_model_types": list(SUPPORTED_MODEL_TYPES),
        },
        "canonical_stems": list(CANONICAL_STEMS),
        "registered_models": registry,
        "ready_candidates_by_stem": ready_by_stem,
        "quality_gate": {
            "checkpoint_licence_required": True,
            "commercial_approval_required": True,
            "reference_free_scoring_is_not_ground_truth": True,
            "promotion_requires_benchmark_and_listening_win": True,
            "production_auto_selection_enabled": False,
        },
        "next_priority": ["piano", "other", "vocals", "guitar", "drums", "bass"],
    }


def _run_candidate(payload: dict, progress=None) -> dict:
    inventory = _inventory()
    model_id = str(payload.get("model_id") or "").strip()
    models = {item["id"]: item for item in inventory["registered_models"]}
    model = models.get(model_id)
    if not model:
        return {"ok": False, "mode": "mss_candidate_lab", "action": "run", "error": f"Unknown model_id: {model_id}", "inventory": inventory}
    if not model["ready"]:
        return {"ok": False, "mode": "mss_candidate_lab", "action": "run", "error": "Model is not approved and ready", "model": model}

    audio_url = str(payload.get("audio_url") or payload.get("source_url") or "").strip()
    if not audio_url:
        return {"ok": False, "mode": "mss_candidate_lab", "action": "run", "error": "audio_url is required"}

    repo_dir = Path(inventory["framework"]["repo_dir"])
    with tempfile.TemporaryDirectory(prefix="litelabs_mss_candidate_") as temp:
        root = Path(temp)
        input_dir = root / "input"
        output_dir = root / "output"
        filename = unquote(Path(urlparse(audio_url).path).name) or "track.flac"
        source = input_dir / filename
        if progress:
            progress(f"Downloading source for {model_id}", 10)
        _download(audio_url, source)
        command = [
            "python", str(repo_dir / "inference.py"),
            "--model_type", model["model_type"],
            "--config_path", model["config_path"],
            "--start_check_point", model["checkpoint_path"],
            "--input_folder", str(input_dir),
            "--store_dir", str(output_dir),
            "--device_ids", "0",
            "--disable_detailed_pbar",
            "--filename_template", "{file_name}/{instr}",
        ]
        if progress:
            progress(f"Running candidate {model_id}", 25)
        completed = subprocess.run(command, cwd=repo_dir, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=int(payload.get("timeout_seconds") or 1800))
        files = sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file())
        return {
            "ok": completed.returncode == 0,
            "mode": "mss_candidate_lab",
            "schema_version": 1,
            "action": "run",
            "model": model,
            "target_stem": model["target_stem"],
            "return_code": completed.returncode,
            "output_files": files,
            "log_tail": "\n".join((completed.stdout or "").splitlines()[-80:]),
            "promotion_status": "candidate_only",
            "next_action": "Score against the current LiteLABS baseline, mixture consistency, contamination metrics and listening tests.",
        }


def build_mss_candidate_lab(payload: dict, progress=None) -> dict:
    action = str(payload.get("action") or "inventory").strip().lower()
    if action == "inventory":
        return _inventory()
    if action == "run":
        return _run_candidate(payload, progress=progress)
    return {"ok": False, "mode": "mss_candidate_lab", "error": f"Unsupported action: {action}"}
