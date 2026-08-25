from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import requests
import soundfile as sf

from sw_residual_allocator import _download, _resolve_model_files

DRUMSEP_CONFIG_URL = "https://github.com/openmirlab/mdxnet-infer/releases/download/weights-drumsep-v1/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.yaml"
DRUMSEP_CHECKPOINT_URL = "https://github.com/openmirlab/mdxnet-infer/releases/download/weights-drumsep-v1/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt"
DRUMSEP_CONFIG_SHA256 = "17d1649a227f841165bdb4c11a42082898192a1ea3ceab7e7e0b9293d6589dd6"
DRUMSEP_CHECKPOINT_SHA256 = "d2a4aa53eb584d21eead358a4e66d1882ad182911be018f052b5da73be9096d0"
CHILDREN = ("kick", "snare", "toms", "hh", "ride", "crash")


def _read(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    return audio[:, :2].astype(np.float64), int(sr)


def _db(value: float) -> float:
    return round(20.0 * np.log10(max(float(value), 1e-12)), 6)


def _metrics(audio: np.ndarray) -> dict:
    rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))
    peak = float(np.max(np.abs(audio)) if audio.size else 0.0)
    active = float(np.mean(np.max(np.abs(audio), axis=1) > 1e-4)) if len(audio) else 0.0
    return {"rms_dbfs": _db(rms), "peak_dbfs": _db(peak), "active_ratio": round(active, 6)}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = a.reshape(-1)
    y = b.reshape(-1)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(x, y) / denom)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_verified(url: str, path: Path, expected_sha256: str, progress=None, progress_message: str = "Downloading model") -> bool:
    if path.is_file() and _sha256(path) == expected_sha256:
        return False
    if path.exists():
        path.unlink()
    if progress:
        progress(progress_message, 5)
    _download(url, path)
    actual = _sha256(path)
    if actual != expected_sha256:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for {path.name}: expected {expected_sha256}, got {actual}")
    return True


def _ensure_drumsep(model_dir: Path, progress=None) -> tuple[Path, Path, bool]:
    model_dir.mkdir(parents=True, exist_ok=True)
    config = model_dir / "aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.yaml"
    checkpoint = model_dir / "aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt"
    config_installed = _ensure_verified(
        DRUMSEP_CONFIG_URL,
        config,
        DRUMSEP_CONFIG_SHA256,
        progress=progress,
        progress_message="Downloading verified DrumSep 6-stem configuration",
    )
    checkpoint_installed = _ensure_verified(
        DRUMSEP_CHECKPOINT_URL,
        checkpoint,
        DRUMSEP_CHECKPOINT_SHA256,
        progress=progress,
        progress_message="Downloading verified DrumSep 6-stem checkpoint",
    )
    return config, checkpoint, bool(config_installed or checkpoint_installed)


def build_drum_decomposition_v1(payload: dict, progress=None) -> dict:
    audio_url = str(payload.get("audio_url") or payload.get("source_url") or "").strip()
    if not audio_url:
        return {"ok": False, "mode": "drum_decomposition_v1", "error": "audio_url is required"}

    timeout = int(payload.get("timeout_seconds") or 1800)
    sw_model_dir = Path(str(payload.get("model_dir") or "/models/bs_roformer_sw"))
    drumsep_dir = Path(str(payload.get("drumsep_model_dir") or "/models/drumsep_mdx23c"))
    repo_dir = Path(str(payload.get("mss_repo_dir") or "/opt/music-source-separation-training"))
    residual_gate_db = float(payload.get("reconstruction_residual_gate_db") or -20.0)
    cosine_gate = float(payload.get("reconstruction_cosine_gate") or 0.98)

    try:
        sw_config, sw_checkpoint, sw_installed = _resolve_model_files(sw_model_dir, progress=progress)
        drum_config, drum_checkpoint, drum_installed = _ensure_drumsep(drumsep_dir, progress=progress)
    except Exception as exc:
        return {"ok": False, "mode": "drum_decomposition_v1", "failed_stage": "model_setup", "error": str(exc)}

    with tempfile.TemporaryDirectory(prefix="litelabs_drum_decomp_") as temp:
        root = Path(temp)
        original_name = unquote(Path(urlparse(audio_url).path).name) or "track.flac"
        downloaded = root / original_name
        source_dir = root / "source"
        sw_out = root / "sw"
        drum_input = root / "drum_input"
        drumsep_out = root / "drumsep"
        for folder in (source_dir, sw_out, drum_input, drumsep_out):
            folder.mkdir(parents=True, exist_ok=True)

        if progress:
            progress("Preparing source", 10)
        _download(audio_url, downloaded)
        source_wav = source_dir / f"{Path(original_name).stem}.wav"
        converted = subprocess.run(
            ["ffmpeg", "-y", "-i", str(downloaded), "-ar", "44100", "-ac", "2", str(source_wav)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=300,
        )
        if converted.returncode != 0:
            return {"ok": False, "mode": "drum_decomposition_v1", "failed_stage": "convert", "runtime_log": "\n".join((converted.stdout or "").splitlines()[-80:])}

        if progress:
            progress("Isolating parent drum stem", 20)
        sw = subprocess.run(
            [
                "bs-roformer-infer",
                "--config_path", str(sw_config),
                "--model_path", str(sw_checkpoint),
                "--input_folder", str(source_dir),
                "--store_dir", str(sw_out),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        if sw.returncode != 0:
            return {"ok": False, "mode": "drum_decomposition_v1", "failed_stage": "parent_separation", "runtime_log": "\n".join((sw.stdout or "").splitlines()[-100:])}

        drum_matches = [p for p in sw_out.rglob("*.wav") if p.name.lower().endswith("_drums.wav")]
        if not drum_matches:
            drum_matches = [p for p in sw_out.rglob("*.wav") if "drum" in p.name.lower()]
        if not drum_matches:
            return {"ok": False, "mode": "drum_decomposition_v1", "failed_stage": "collect_parent", "error": "SW drum stem not found"}
        parent_path = drum_input / "drums.wav"
        parent_audio, parent_sr = _read(drum_matches[0])
        sf.write(parent_path, parent_audio.astype(np.float32), parent_sr, subtype="FLOAT")

        if progress:
            progress("Decomposing drums into six children", 48)
        command = [
            "python", str(repo_dir / "inference.py"),
            "--model_type", "mdx23c",
            "--config_path", str(drum_config),
            "--start_check_point", str(drum_checkpoint),
            "--input_folder", str(drum_input),
            "--store_dir", str(drumsep_out),
            "--device_ids", "0",
            "--disable_detailed_pbar",
            "--filename_template", "{file_name}/{instr}",
        ]
        run = subprocess.run(
            command,
            cwd=repo_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        if run.returncode != 0:
            return {"ok": False, "mode": "drum_decomposition_v1", "failed_stage": "drumsep", "runtime_log": "\n".join((run.stdout or "").splitlines()[-120:])}

        files = [p for p in drumsep_out.rglob("*") if p.is_file() and p.suffix.lower() in {".wav", ".flac"}]
        child_paths: dict[str, Path] = {}
        for child in CHILDREN:
            exact = [p for p in files if p.stem.lower().replace("_", "-") == child]
            fuzzy = [p for p in files if child in p.stem.lower().replace("_", "-")]
            matches = exact or fuzzy
            if not matches:
                return {"ok": False, "mode": "drum_decomposition_v1", "failed_stage": "collect_children", "missing_child": child, "files": [str(p.relative_to(drumsep_out)) for p in files]}
            child_paths[child] = matches[0]

        loaded = {name: _read(path)[0] for name, path in child_paths.items()}
        length = min([len(parent_audio)] + [len(a) for a in loaded.values()])
        parent = parent_audio[:length]
        loaded = {name: audio[:length] for name, audio in loaded.items()}
        rebuilt = np.sum(np.stack([loaded[name] for name in CHILDREN], axis=0), axis=0)
        residual = parent - rebuilt

        parent_rms = float(np.sqrt(np.mean(parent * parent) + 1e-12))
        residual_rms = float(np.sqrt(np.mean(residual * residual) + 1e-12))
        relative_residual_db = _db(residual_rms / max(parent_rms, 1e-12))
        correlation = _cosine(parent, rebuilt)
        replace_parent = bool(relative_residual_db <= residual_gate_db and correlation >= cosine_gate)

        children = []
        for name in CHILDREN:
            item = _metrics(loaded[name])
            item["instrument"] = name
            item["output_file"] = str(child_paths[name].relative_to(drumsep_out))
            children.append(item)

        report = {
            "parent": {**_metrics(parent), "source": "BS-RoFormer-SW drums"},
            "children": children,
            "reconstruction": {
                "residual_relative_to_parent_db": relative_residual_db,
                "parent_vs_children_sum_cosine": round(correlation, 6),
                "residual_gate_db": residual_gate_db,
                "cosine_gate": cosine_gate,
                "passed": replace_parent,
            },
            "export_decision": {
                "replace_parent_drums": replace_parent,
                "if_passed": list(CHILDREN),
                "if_failed": ["drums"],
            },
        }

        if progress:
            progress("Drum decomposition validation complete", 100)
        return {
            "ok": True,
            "mode": "drum_decomposition_v1",
            "schema_version": 2,
            "research_only": True,
            "audio_url": audio_url,
            "model": "DrumSep MDX23C 6-stem (aufr33/jarredou; verified openmirlab mirror)",
            "sw_model_auto_installed": sw_installed,
            "drumsep_model_auto_installed": drum_installed,
            "report": report,
            "warning": "Research validation only. Original DrumSep checkpoint license terms are not formally documented; do not assume commercial suitability.",
        }
