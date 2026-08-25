from __future__ import annotations

import json
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import requests
import soundfile as sf

from drum_decomposition_v1 import CHILDREN as DRUM_CHILDREN, _ensure_drumsep
from sw_residual_allocator import STEMS as SW_STEMS, _download, _resolve_model_files
from wind_brass_decomposition_v2 import MEGA53_CHECKPOINT, MEGA53_CONFIG, _cos, _metrics, _run_polled

MODE = "routed_extraction_v1"
WIND_CHILDREN = ("saxophone", "trumpet")


def _read(path: Path):
    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    return audio[:, :2].astype(np.float64), int(sr)


def _db(value: float) -> float:
    return round(20.0 * np.log10(max(float(value), 1e-12)), 6)


def _safe_name(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value.strip())
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or "track"


def _collect_sw_stems(swout: Path) -> dict[str, Path]:
    files = [p for p in swout.rglob("*.wav") if p.is_file()]
    found: dict[str, Path] = {}
    for stem in SW_STEMS:
        matches = [p for p in files if p.name.lower().endswith(f"_{stem}.wav")]
        if not matches:
            matches = [p for p in files if stem in p.name.lower() and "instrumental" not in p.name.lower()]
        if matches:
            found[stem] = matches[0]
    return found


def _collect_named_outputs(root: Path, names: tuple[str, ...]) -> dict[str, Path]:
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".wav", ".flac"}]
    by_name = {p.stem.lower().replace("_", "-"): p for p in files}
    result: dict[str, Path] = {}
    for name in names:
        exact = by_name.get(name)
        if exact:
            result[name] = exact
            continue
        fuzzy = [p for p in files if name in p.stem.lower().replace("_", "-")]
        if fuzzy:
            result[name] = fuzzy[0]
    return result


def _write_flac(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), sr, format="FLAC", subtype="PCM_24")


def _copy_as_flac(source: Path, destination: Path) -> None:
    audio, sr = _read(source)
    _write_flac(destination, audio, sr)


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def build_routed_extraction_v1(payload: dict, progress=None) -> dict:
    audio_url = str(payload.get("audio_url") or payload.get("source_url") or "").strip()
    if not audio_url:
        return {"ok": False, "mode": MODE, "error": "audio_url is required"}

    timeout = int(payload.get("timeout_seconds") or 1800)
    heartbeat_seconds = max(5, int(payload.get("heartbeat_seconds") or 15))
    rms_gate = float(payload.get("child_rms_gate_dbfs") or -40.0)
    parent_cos_gate = float(payload.get("child_parent_cosine_gate") or 0.55)
    overlap_gate = float(payload.get("child_pairwise_cosine_gate") or 0.20)
    residual_keep_gate = float(payload.get("residual_keep_gate_db") or -35.0)
    drum_residual_gate = float(payload.get("drum_reconstruction_residual_gate_db") or -20.0)
    drum_cosine_gate = float(payload.get("drum_reconstruction_cosine_gate") or 0.98)

    model_dir = Path(str(payload.get("model_dir") or "/models/bs_roformer_sw"))
    drumsep_dir = Path(str(payload.get("drumsep_model_dir") or "/models/drumsep_mdx23c"))
    repo_dir = Path(str(payload.get("mss_repo_dir") or "/opt/music-source-separation-training"))

    if not MEGA53_CONFIG.is_file() or not MEGA53_CHECKPOINT.is_file():
        return {"ok": False, "mode": MODE, "failed_stage": "model_setup", "error": "Baked Mega53 model files are missing"}

    started = time.monotonic()
    timings: dict[str, float] = {}

    def emit(message: str, percent: int) -> None:
        print(f"[routed_extraction_v1] {message} ({percent}%)", flush=True)
        if progress:
            progress(message, percent)

    try:
        t0 = time.monotonic()
        emit("Resolving baked separation models", 2)
        sw_config, sw_checkpoint, sw_installed = _resolve_model_files(model_dir, progress=progress)
        drum_config, drum_checkpoint, drum_installed = _ensure_drumsep(drumsep_dir, progress=progress)
        timings["model_setup"] = round(time.monotonic() - t0, 3)
    except Exception as exc:
        return {"ok": False, "mode": MODE, "failed_stage": "model_setup", "error": str(exc)}

    with tempfile.TemporaryDirectory(prefix="litelabs_routed_v1_") as temp:
        root = Path(temp)
        srcdir = root / "source"
        swout = root / "sw"
        drum_input = root / "drum_input"
        drumout = root / "drumsep"
        other_input = root / "other_input"
        megaout = root / "mega53"
        finaldir = root / "final"
        logdir = root / "logs"
        for d in (srcdir, swout, drum_input, drumout, other_input, megaout, finaldir, logdir):
            d.mkdir(parents=True, exist_ok=True)

        raw_name = unquote(Path(urlparse(audio_url).path).name) or "track.flac"
        requested_name = str(payload.get("filename") or Path(raw_name).stem)
        track = _safe_name(Path(requested_name).stem)
        downloaded = root / raw_name

        emit("Downloading source", 4)
        t0 = time.monotonic()
        _download(audio_url, downloaded)
        timings["download"] = round(time.monotonic() - t0, 3)

        emit("Preparing source audio", 6)
        t0 = time.monotonic()
        source = srcdir / f"{track}.wav"
        import subprocess
        with (logdir / "ffmpeg.log").open("w", encoding="utf-8") as log:
            conv = subprocess.run(
                ["ffmpeg", "-y", "-i", str(downloaded), "-ar", "44100", "-ac", "2", str(source)],
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=300,
            )
        if conv.returncode != 0:
            return {"ok": False, "mode": MODE, "failed_stage": "convert"}
        timings["convert"] = round(time.monotonic() - t0, 3)

        emit("Running primary separation once", 10)
        rc, elapsed = _run_polled(
            ["bs-roformer-infer", "--config_path", str(sw_config), "--model_path", str(sw_checkpoint), "--input_folder", str(srcdir), "--store_dir", str(swout)],
            cwd=None,
            timeout=timeout,
            log_path=logdir / "sw.log",
            progress=progress,
            stage_name="Primary BS-RoFormer separation",
            start_percent=10,
            end_percent=35,
            heartbeat_seconds=heartbeat_seconds,
        )
        timings["primary_separation"] = round(elapsed, 3)
        if rc != 0:
            return {"ok": False, "mode": MODE, "failed_stage": "primary_separation"}

        stems = _collect_sw_stems(swout)
        missing = [stem for stem in SW_STEMS if stem not in stems]
        if missing:
            return {"ok": False, "mode": MODE, "failed_stage": "collect_primary_stems", "missing": missing}

        # Preserve the non-decomposed primary stems in the flat final directory.
        for stem in ("vocals", "bass", "guitar", "piano"):
            _copy_as_flac(stems[stem], finaldir / f"{track}_{stem}.flac")

        # Derived instrumental remains useful and costs almost nothing once vocals exists.
        mixture, mix_sr = _read(source)
        vocals, _ = _read(stems["vocals"])
        n_inst = min(len(mixture), len(vocals))
        instrumental = mixture[:n_inst] - vocals[:n_inst]
        _write_flac(finaldir / f"{track}_instrumental.flac", instrumental, mix_sr)

        # Drum branch: reuse the already-created SW drums parent.
        emit("Decomposing drum parent", 38)
        drum_parent, drum_sr = _read(stems["drums"])
        drum_parent_path = drum_input / "drums.wav"
        sf.write(drum_parent_path, drum_parent.astype(np.float32), drum_sr, subtype="FLOAT")
        rc, elapsed = _run_polled(
            [
                "python", str(repo_dir / "inference.py"), "--model_type", "mdx23c",
                "--config_path", str(drum_config), "--start_check_point", str(drum_checkpoint),
                "--input_folder", str(drum_input), "--store_dir", str(drumout), "--device_ids", "0",
                "--disable_detailed_pbar", "--filename_template", "{file_name}/{instr}",
            ],
            cwd=repo_dir,
            timeout=timeout,
            log_path=logdir / "drumsep.log",
            progress=progress,
            stage_name="DrumSep decomposition",
            start_percent=38,
            end_percent=62,
            heartbeat_seconds=heartbeat_seconds,
        )
        timings["drumsep"] = round(elapsed, 3)
        if rc != 0:
            return {"ok": False, "mode": MODE, "failed_stage": "drumsep"}

        drum_paths = _collect_named_outputs(drumout, DRUM_CHILDREN)
        if len(drum_paths) != len(DRUM_CHILDREN):
            return {"ok": False, "mode": MODE, "failed_stage": "collect_drum_children", "found": sorted(drum_paths)}
        drum_loaded = {name: _read(path)[0] for name, path in drum_paths.items()}
        dn = min([len(drum_parent)] + [len(a) for a in drum_loaded.values()])
        drum_reference = drum_parent[:dn]
        drum_sum = np.sum(np.stack([drum_loaded[name][:dn] for name in DRUM_CHILDREN], axis=0), axis=0)
        drum_residual = drum_reference - drum_sum
        drum_parent_rms = float(np.sqrt(np.mean(drum_reference * drum_reference) + 1e-12))
        drum_residual_rms = float(np.sqrt(np.mean(drum_residual * drum_residual) + 1e-12))
        drum_residual_db = _db(drum_residual_rms / max(drum_parent_rms, 1e-12))
        drum_cos = float(_cos(drum_reference, drum_sum))
        drums_passed = bool(drum_residual_db <= drum_residual_gate and drum_cos >= drum_cosine_gate)

        if drums_passed:
            for child in DRUM_CHILDREN:
                _write_flac(finaldir / f"{track}_drums_{child}.flac", drum_loaded[child][:dn], drum_sr)
        else:
            _write_flac(finaldir / f"{track}_drums.flac", drum_reference, drum_sr)

        # Other branch: reuse the already-created SW Other parent.
        emit("Decomposing Other into saxophone and trumpet", 65)
        other_parent, other_sr = _read(stems["other"])
        other_parent_path = other_input / "other.wav"
        sf.write(other_parent_path, other_parent.astype(np.float32), other_sr, subtype="FLOAT")
        rc, elapsed = _run_polled(
            [
                "python", str(repo_dir / "inference.py"), "--model_type", "bs_roformer",
                "--config_path", str(MEGA53_CONFIG), "--start_check_point", str(MEGA53_CHECKPOINT),
                "--input_folder", str(other_input), "--store_dir", str(megaout), "--device_ids", "0",
                "--disable_detailed_pbar", "--filename_template", "{file_name}/{instr}",
            ],
            cwd=repo_dir,
            timeout=timeout,
            log_path=logdir / "mega53.log",
            progress=progress,
            stage_name="Mega53 saxophone/trumpet decomposition",
            start_percent=65,
            end_percent=88,
            heartbeat_seconds=heartbeat_seconds,
        )
        timings["mega53"] = round(elapsed, 3)
        if rc != 0:
            return {"ok": False, "mode": MODE, "failed_stage": "mega53"}

        wind_paths = _collect_named_outputs(megaout, WIND_CHILDREN)
        wind_loaded: dict[str, np.ndarray] = {}
        child_report = []
        for child in WIND_CHILDREN:
            path = wind_paths.get(child)
            if not path:
                child_report.append({"instrument": child, "approved": False, "reason": "missing_output"})
                continue
            audio, _ = _read(path)
            wind_loaded[child] = audio
            metrics = _metrics(audio, other_parent)
            child_report.append({"instrument": child, **metrics})

        overlap = None
        if all(name in wind_loaded for name in WIND_CHILDREN):
            overlap = float(_cos(wind_loaded["saxophone"], wind_loaded["trumpet"]))

        approved: list[str] = []
        for item in child_report:
            if "rms_dbfs" not in item:
                continue
            reasons = []
            if item["rms_dbfs"] < rms_gate:
                reasons.append("too_quiet")
            if item["parent_cosine"] < parent_cos_gate:
                reasons.append("weak_parent_relation")
            if overlap is not None and abs(overlap) > overlap_gate:
                reasons.append("pairwise_overlap_too_high")
            item["approved"] = not reasons
            item["rejection_reasons"] = reasons
            if item["approved"]:
                approved.append(item["instrument"])

        partial_other = False
        other_residual_db = 0.0
        other_sum_cos = 0.0
        if approved:
            on = min([len(other_parent)] + [len(wind_loaded[name]) for name in approved])
            parent = other_parent[:on]
            child_sum = np.sum(np.stack([wind_loaded[name][:on] for name in approved], axis=0), axis=0)
            residual = parent - child_sum
            parent_rms = float(np.sqrt(np.mean(parent * parent) + 1e-12))
            residual_rms = float(np.sqrt(np.mean(residual * residual) + 1e-12))
            other_residual_db = _db(residual_rms / max(parent_rms, 1e-12))
            other_sum_cos = float(_cos(parent, child_sum))
            residual_meaningful = bool(other_residual_db >= residual_keep_gate)
            partial_other = bool(approved) and residual_meaningful
            if partial_other:
                for child in approved:
                    _write_flac(finaldir / f"{track}_other_{child}.flac", wind_loaded[child][:on], other_sr)
                _write_flac(finaldir / f"{track}_other_residual.flac", residual, other_sr)

        if not partial_other:
            _write_flac(finaldir / f"{track}_other.flac", other_parent, other_sr)

        emit("Packaging flat routed stem set", 92)
        report = {
            "schema_version": 1,
            "mode": MODE,
            "track": track,
            "primary_separation_runs": 1,
            "flat_output": True,
            "drums": {
                "decomposition_passed": drums_passed,
                "parent_vs_children_sum_cosine": round(drum_cos, 6),
                "residual_relative_to_parent_db": drum_residual_db,
                "exported": [f"drums_{name}" for name in DRUM_CHILDREN] if drums_passed else ["drums"],
            },
            "other": {
                "partial_decomposition_passed": partial_other,
                "approved_children": approved,
                "pairwise_sax_trumpet_cosine": round(overlap, 6) if overlap is not None else None,
                "approved_children_sum_vs_parent_cosine": round(other_sum_cos, 6),
                "residual_relative_to_parent_db": other_residual_db,
                "exported": [f"other_{name}" for name in approved] + (["other_residual"] if partial_other else []) if partial_other else ["other"],
                "children": child_report,
            },
            "model_setup": {
                "sw_auto_installed": bool(sw_installed),
                "drumsep_auto_installed": bool(drum_installed),
                "mega53_baked": True,
            },
            "timings_seconds": timings,
        }
        report = _json_safe(report)
        (finaldir / f"{track}_ROUTED_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        archive = root / f"{track}_litelabs_routed_stems.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
            for path in sorted(finaldir.iterdir()):
                bundle.write(path, arcname=path.name)

        uploaded = False
        put_url = str(payload.get("result_put_url") or "").strip()
        if put_url:
            emit("Uploading routed stem pack", 96)
            with archive.open("rb") as handle:
                response = requests.put(put_url, data=handle, headers={"Content-Type": "application/zip"}, timeout=(30, 1800))
            response.raise_for_status()
            uploaded = True

        timings["total"] = round(time.monotonic() - started, 3)
        report["timings_seconds"] = timings
        emit("Routed extraction complete", 100)

        result = {
            "ok": True,
            "mode": MODE,
            "schema_version": 1,
            "research_only": True,
            "track": track,
            "primary_separation_runs": 1,
            "archive_name": archive.name,
            "archive_size_bytes": archive.stat().st_size,
            "uploaded": uploaded,
            "result_url": payload.get("result_public_url"),
            "files": sorted(path.name for path in finaldir.iterdir()),
            "report": report,
            "warning": "Research validation only. DrumSep checkpoint licensing remains unresolved and Mega53 remains discovery/decomposition evidence; do not assume commercial suitability yet.",
        }
        return _json_safe(result)
