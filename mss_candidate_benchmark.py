from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import requests
import soundfile as sf


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=(30, 600)) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    return audio[:, :2], int(sr)


def _align(*arrays: np.ndarray) -> list[np.ndarray]:
    length = min(array.shape[0] for array in arrays)
    return [array[:length] for array in arrays]


def _metrics(path: Path, mixture: np.ndarray) -> dict:
    audio, sr = _read_audio(path)
    audio, mix = _align(audio, mixture)
    flat = audio.reshape(-1).astype(np.float64)
    mix_flat = mix.reshape(-1).astype(np.float64)
    rms = float(np.sqrt(np.mean(flat * flat) + 1e-12))
    peak = float(np.max(np.abs(flat)))
    threshold = max(rms * 0.1, 1e-5)
    active_ratio = float(np.mean(np.abs(flat) > threshold))
    denom = float(np.linalg.norm(flat) * np.linalg.norm(mix_flat) + 1e-12)
    cosine = float(np.dot(flat, mix_flat) / denom)
    return {
        "file": str(path),
        "sample_rate": sr,
        "duration_seconds": round(audio.shape[0] / sr, 3),
        "rms_dbfs": round(20.0 * np.log10(rms + 1e-12), 3),
        "peak_dbfs": round(20.0 * np.log10(peak + 1e-12), 3),
        "active_ratio": round(active_ratio, 6),
        "mixture_cosine": round(cosine, 6),
    }


def _residual_metrics(mixture: np.ndarray, stem_paths: list[Path]) -> dict:
    stems = [_read_audio(path)[0] for path in stem_paths]
    aligned = _align(mixture, *stems)
    mix = aligned[0].astype(np.float64)
    summed = np.sum(np.stack(aligned[1:], axis=0).astype(np.float64), axis=0)
    residual = mix - summed
    mix_rms = float(np.sqrt(np.mean(mix * mix) + 1e-12))
    residual_rms = float(np.sqrt(np.mean(residual * residual) + 1e-12))
    return {
        "residual_db_relative_to_mix": round(20.0 * np.log10((residual_rms + 1e-12) / (mix_rms + 1e-12)), 3),
        "reconstruction_cosine": round(float(np.dot(mix.reshape(-1), summed.reshape(-1)) / (np.linalg.norm(mix) * np.linalg.norm(summed) + 1e-12)), 6),
    }


def _run_candidate(model_id: str, source: Path, output_dir: Path, timeout_seconds: int) -> dict:
    import mss_candidate_lab

    raw_models = {str(item.get("id") or ""): item for item in mss_candidate_lab._load_registry()}
    entry = raw_models.get(model_id)
    if not entry:
        raise RuntimeError(f"Unknown MSS model: {model_id}")
    validated = mss_candidate_lab._validate_model(entry)
    auto_installed = False
    if not validated["research_ready"]:
        install = mss_candidate_lab._install_candidate({"model_id": model_id})
        if not install.get("ok"):
            raise RuntimeError(f"Could not install {model_id}: {install}")
        auto_installed = True
        validated = mss_candidate_lab._validate_model(entry)

    input_dir = output_dir.parent / f"{model_id}-input"
    input_dir.mkdir(parents=True, exist_ok=True)
    local_source = input_dir / source.name
    if not local_source.exists():
        local_source.symlink_to(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = Path(os.getenv("LITELABS_MSS_REPO_DIR", "/opt/music-source-separation-training"))
    command = [
        "python", str(repo_dir / "inference.py"),
        "--model_type", validated["model_type"],
        "--config_path", validated["config_path"],
        "--start_check_point", validated["checkpoint_path"],
        "--input_folder", str(input_dir),
        "--store_dir", str(output_dir),
        "--device_ids", "0",
        "--disable_detailed_pbar",
        "--filename_template", "{file_name}/{instr}",
    ]
    completed = subprocess.run(command, cwd=repo_dir, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout_seconds)
    files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    if completed.returncode != 0 or not files:
        raise RuntimeError(f"Candidate {model_id} failed: {(completed.stdout or '')[-6000:]}")
    return {
        "model": validated,
        "auto_installed_on_worker": auto_installed,
        "files": files,
        "log_tail": "\n".join((completed.stdout or "").splitlines()[-30:]),
    }


def _find_named(root: Path, token: str) -> Path:
    matches = [path for path in root.rglob("*") if path.is_file() and token.lower() in path.name.lower()]
    if not matches:
        raise FileNotFoundError(f"No output matching {token} under {root}")
    return sorted(matches)[0]


def build_mss_candidate_benchmark(payload: dict, progress=None) -> dict:
    audio_url = str(payload.get("audio_url") or payload.get("source_url") or "").strip()
    if not audio_url:
        return {"ok": False, "mode": "mss_candidate_benchmark", "error": "audio_url is required"}
    timeout_seconds = int(payload.get("timeout_seconds") or 1800)

    with tempfile.TemporaryDirectory(prefix="litelabs_mss_benchmark_") as temp:
        root = Path(temp)
        source_name = unquote(Path(urlparse(audio_url).path).name) or "track.flac"
        source = root / source_name
        if progress:
            progress("Downloading benchmark source", 3)
        _download(audio_url, source)
        mixture, sr = _read_audio(source)

        if progress:
            progress("Building current LiteLABS baseline", 10)
        from master_pack import build_master_pack
        baseline_output = root / "baseline-output"
        baseline = build_master_pack(
            input_audio=source,
            work_root=root / "baseline-work",
            model_dir=Path(payload.get("model_dir") or os.getenv("STEMFORGE_MODEL_DIR", "/models/bs_roformer_sw")),
            output_root=baseline_output,
            progress=None,
        )
        baseline_root = baseline_output / f"{baseline['track']}-litelabs-stem-pack"
        baseline_piano = _find_named(baseline_root, "piano_keys")
        baseline_other = _find_named(baseline_root, "synth_strings_other")

        if progress:
            progress("Running HTDemucs6 challenger", 48)
        piano_run = _run_candidate("htdemucs6-piano-challenger", source, root / "htdemucs6-output", timeout_seconds)
        challenger_piano = _find_named(root / "htdemucs6-output", "piano")

        if progress:
            progress("Running dedicated Other challenger", 72)
        other_run = _run_candidate("viperx-bs-roformer-other-challenger", source, root / "viperx-other-output", timeout_seconds)
        challenger_other = _find_named(root / "viperx-other-output", "other")

        htdemucs_stems = [_find_named(root / "htdemucs6-output", stem) for stem in ("vocals", "drums", "bass", "guitar", "piano", "other")]

        if progress:
            progress("Scoring candidate outputs", 92)
        result = {
            "ok": True,
            "mode": "mss_candidate_benchmark",
            "schema_version": 1,
            "track": baseline["track"],
            "sample_rate": sr,
            "quality_warning": "Reference-free measurements cannot establish true stem fidelity; listening and ground-truth tests remain required.",
            "piano": {
                "baseline": _metrics(baseline_piano, mixture),
                "challenger": _metrics(challenger_piano, mixture),
                "challenger_model": piano_run["model"]["id"],
            },
            "other": {
                "baseline": _metrics(baseline_other, mixture),
                "challenger": _metrics(challenger_other, mixture),
                "challenger_model": other_run["model"]["id"],
            },
            "htdemucs6_reconstruction": _residual_metrics(mixture, htdemucs_stems),
            "worker_installation": {
                "piano_auto_installed": piano_run["auto_installed_on_worker"],
                "other_auto_installed": other_run["auto_installed_on_worker"],
            },
            "logs": {
                "piano": piano_run["log_tail"],
                "other": other_run["log_tail"],
            },
            "decision_rule": "Do not promote from these metrics alone. Promote only after reference-free sanity checks, listening, and compatible ground-truth wins.",
        }
        if progress:
            progress("Candidate benchmark complete", 100)
        return result
