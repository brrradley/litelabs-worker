from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import soundfile as sf
import torch

from routed_extraction_v1 import _collect_named_outputs, _collect_sw_stems, _copy_as_flac, _db, _read, _safe_name, _write_flac
from sw_residual_allocator import STEMS as SW_STEMS, _download, _resolve_model_files
from wind_brass_decomposition_v2 import _cos, _run_polled

MODE = "experimental_children_v1"
DRUM5 = ("kick", "snare", "toms", "hh", "cymbals")
DRUM5_CONFIG = Path("/models/drumsep_5stem/mdx23c_drumsep_5stem_aufr33_jarredou_config.yaml")
DRUM5_CHECKPOINT = Path("/models/drumsep_5stem/mdx23c_drumsep_5stem_aufr33_jarredou.ckpt")
WIND_MODEL = "17_HP-Wind_Inst-UVR.pth"
SAX_MODEL_DIR = Path("/models/sax_demucs")
SAX_MODEL = "filosax_demucs_v3_14.22_SDR.th"


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _copy_audio_tree(source_root: Path, destination_root: Path, prefix: str) -> list[str]:
    copied: list[str] = []
    for src in sorted(source_root.rglob("*")):
        if not src.is_file() or src.suffix.lower() not in {".wav", ".flac", ".mp3"}:
            continue
        rel = "_".join(src.relative_to(source_root).with_suffix("").parts)
        dest = destination_root / f"{prefix}_{_safe_name(rel)}.flac"
        _copy_as_flac(src, dest)
        copied.append(dest.name)
    return copied


def _wiener_kick(parent: np.ndarray, children: dict[str, np.ndarray], n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    """Return only the conservative p=1 Wiener-refined kick from the drums parent."""
    n = min([len(parent)] + [len(children[name]) for name in DRUM5])
    p = np.asarray(parent[:n], dtype=np.float32)
    c = {name: np.asarray(children[name][:n], dtype=np.float32) for name in DRUM5}
    window = torch.hann_window(n_fft, periodic=True)
    result = np.zeros_like(p, dtype=np.float32)
    eps = 1e-10
    with torch.no_grad():
        for ch in range(p.shape[1]):
            parent_t = torch.from_numpy(p[:, ch])
            parent_spec = torch.stft(parent_t, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, center=True, return_complex=True)
            mags = []
            for name in DRUM5:
                stem_t = torch.from_numpy(c[name][:, ch])
                stem_spec = torch.stft(stem_t, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, center=True, return_complex=True)
                mags.append(torch.abs(stem_spec).clamp_min(eps))
            denom = torch.stack(mags, dim=0).sum(dim=0).clamp_min(eps)
            kick_mask = mags[0] / denom
            kick_spec = parent_spec * kick_mask
            wav = torch.istft(kick_spec, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, center=True, length=n)
            result[:, ch] = wav.cpu().numpy().astype(np.float32)
    return result


def _apply_mass_conserving_kick_refinement(parent: np.ndarray, children: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict]:
    """Replace only kick with the proven mild Wiener version and preserve the drums parent exactly.

    The correction required after replacing kick is redistributed across the four untouched
    children according to their instantaneous energy. This keeps the refined kick intact while
    avoiding a global Wiener pass that our supervised tests showed can hurt snare and hi-hat.
    """
    n = min([len(parent)] + [len(children[name]) for name in DRUM5])
    parent_n = np.asarray(parent[:n], dtype=np.float32)
    baseline = {name: np.asarray(children[name][:n], dtype=np.float32).copy() for name in DRUM5}
    refined_kick = _wiener_kick(parent_n, baseline, n_fft=2048, hop=512)

    out = {name: audio.copy() for name, audio in baseline.items()}
    out["kick"] = refined_kick
    others = [name for name in DRUM5 if name != "kick"]

    current_sum = np.sum(np.stack([out[name] for name in DRUM5], axis=0), axis=0)
    correction = parent_n - current_sum
    energies = np.stack([np.square(baseline[name], dtype=np.float32) for name in others], axis=0)
    denom = np.sum(energies, axis=0)
    tiny = 1e-12
    weights = energies / np.maximum(denom[None, ...], tiny)
    silent = denom <= tiny
    if np.any(silent):
        weights[:, silent] = 1.0 / len(others)
    for idx, name in enumerate(others):
        out[name] = out[name] + correction * weights[idx]

    rebuilt = np.sum(np.stack([out[name] for name in DRUM5], axis=0), axis=0)
    residual = parent_n - rebuilt
    parent_rms = float(np.sqrt(np.mean(parent_n * parent_n) + 1e-12))
    residual_rms = float(np.sqrt(np.mean(residual * residual) + 1e-12))
    kick_delta_rms = float(np.sqrt(np.mean((refined_kick - baseline["kick"]) ** 2) + 1e-12))
    return out, {
        "applied": True,
        "method": "kick_only_wiener_p1_fft2048_mass_conserving",
        "kick_delta_rms": kick_delta_rms,
        "parent_vs_children_sum_cosine": round(float(_cos(parent_n, rebuilt)), 6),
        "residual_relative_to_parent_db": _db(residual_rms / max(parent_rms, 1e-12)),
    }


def build_experimental_children_v1(payload: dict, progress=None) -> dict:
    audio_url = str(payload.get("audio_url") or payload.get("source_url") or "").strip()
    if not audio_url:
        return {"ok": False, "mode": MODE, "error": "audio_url is required"}

    timeout = int(payload.get("timeout_seconds") or 1800)
    heartbeat = max(5, int(payload.get("heartbeat_seconds") or 15))
    repo_dir = Path(str(payload.get("mss_repo_dir") or "/opt/music-source-separation-training"))
    model_dir = Path(str(payload.get("model_dir") or "/models/bs_roformer_sw"))
    audio_separator_model_dir = Path(str(payload.get("audio_separator_model_dir") or "/models/audio_separator"))

    required = [DRUM5_CONFIG, DRUM5_CHECKPOINT, SAX_MODEL_DIR / SAX_MODEL, audio_separator_model_dir / WIND_MODEL]
    missing_models = [str(p) for p in required if not p.is_file()]
    if missing_models:
        return {"ok": False, "mode": MODE, "failed_stage": "model_setup", "missing_models": missing_models}

    started = time.monotonic()
    timings: dict[str, float] = {}

    def emit(message: str, percent: int) -> None:
        print(f"[{MODE}] {message} ({percent}%)", flush=True)
        if progress:
            progress(message, percent)

    sw_config, sw_checkpoint, sw_installed = _resolve_model_files(model_dir, progress=progress)

    with tempfile.TemporaryDirectory(prefix="litelabs_exp_children_") as temp:
        root = Path(temp)
        srcdir = root / "source"
        swout = root / "sw"
        drum_in = root / "drum_in"
        drum_out = root / "drum5"
        wind_out = root / "wind"
        sax_out = root / "sax"
        final = root / "pack"
        experimental = final / "experimental"
        logs = root / "logs"
        for d in (srcdir, swout, drum_in, drum_out, wind_out, sax_out, final, experimental, logs):
            d.mkdir(parents=True, exist_ok=True)

        raw_name = unquote(Path(urlparse(audio_url).path).name) or "track.flac"
        track = _safe_name(Path(str(payload.get("filename") or raw_name)).stem)
        downloaded = root / raw_name

        emit("Initiating stem separation", 2)
        t0 = time.monotonic()
        _download(audio_url, downloaded)
        timings["download"] = round(time.monotonic() - t0, 3)

        source = srcdir / f"{track}.wav"
        with (logs / "ffmpeg.log").open("w", encoding="utf-8") as log:
            conv = subprocess.run(["ffmpeg", "-y", "-i", str(downloaded), "-ar", "44100", "-ac", "2", str(source)], stdout=log, stderr=subprocess.STDOUT, text=True, timeout=300)
        if conv.returncode != 0:
            return {"ok": False, "mode": MODE, "failed_stage": "convert"}

        emit("Running BS-RoFormer parent separation", 10)
        rc, elapsed = _run_polled(
            ["bs-roformer-infer", "--config_path", str(sw_config), "--model_path", str(sw_checkpoint), "--input_folder", str(srcdir), "--store_dir", str(swout)],
            cwd=None, timeout=timeout, log_path=logs / "sw.log", progress=None,
            stage_name="BS-RoFormer parent separation", start_percent=10, end_percent=28, heartbeat_seconds=heartbeat,
        )
        timings["bs_roformer"] = round(elapsed, 3)
        if rc != 0:
            return {"ok": False, "mode": MODE, "failed_stage": "bs_roformer"}

        stems = _collect_sw_stems(swout)
        missing = [s for s in SW_STEMS if s not in stems]
        if missing:
            return {"ok": False, "mode": MODE, "failed_stage": "collect_parents", "missing": missing}

        # Quality baseline: untouched BS-RoFormer parents remain at ZIP root.
        for stem in SW_STEMS:
            _copy_as_flac(stems[stem], final / f"{track}_{stem}.flac")
        mixture, mix_sr = _read(source)
        vocals, _ = _read(stems["vocals"])
        n = min(len(mixture), len(vocals))
        _write_flac(final / f"{track}_instrumental.flac", mixture[:n] - vocals[:n], mix_sr)

        # Experimental drums: current 5-stem DrumSep plus targeted, supervised kick refinement.
        emit("Running DrumSep 5-Stem Experiment", 32)
        drums, drum_sr = _read(stems["drums"])
        sf.write(drum_in / "drums.wav", drums.astype(np.float32), drum_sr, subtype="FLOAT")
        rc, elapsed = _run_polled(
            ["python", str(repo_dir / "inference.py"), "--model_type", "mdx23c", "--config_path", str(DRUM5_CONFIG), "--start_check_point", str(DRUM5_CHECKPOINT), "--input_folder", str(drum_in), "--store_dir", str(drum_out), "--device_ids", "0", "--disable_detailed_pbar", "--filename_template", "{file_name}/{instr}"],
            cwd=repo_dir, timeout=timeout, log_path=logs / "drum5.log", progress=None,
            stage_name="DrumSep 5-stem", start_percent=32, end_percent=52, heartbeat_seconds=heartbeat,
        )
        timings["drumsep_5stem"] = round(elapsed, 3)
        drum_report = {"ok": rc == 0, "files": []}
        if rc == 0:
            drum_paths = _collect_named_outputs(drum_out, DRUM5)
            loaded = {name: _read(path)[0] for name, path in drum_paths.items()}
            if len(loaded) == len(DRUM5):
                dn = min([len(drums)] + [len(a) for a in loaded.values()])
                parent = drums[:dn]
                baseline = {name: loaded[name][:dn] for name in DRUM5}
                baseline_sum = np.sum(np.stack([baseline[name] for name in DRUM5], axis=0), axis=0)
                baseline_residual = parent - baseline_sum
                parent_rms = float(np.sqrt(np.mean(parent * parent) + 1e-12))
                baseline_residual_rms = float(np.sqrt(np.mean(baseline_residual * baseline_residual) + 1e-12))
                refined, kick_refinement = _apply_mass_conserving_kick_refinement(parent, baseline)
                drum_report.update({
                    "baseline_parent_vs_children_sum_cosine": round(float(_cos(parent, baseline_sum)), 6),
                    "baseline_residual_relative_to_parent_db": _db(baseline_residual_rms / max(parent_rms, 1e-12)),
                    "kick_refinement": kick_refinement,
                })
                for name in DRUM5:
                    dest = experimental / f"{track}_drums_5stem_{name}.flac"
                    _write_flac(dest, refined[name], drum_sr)
                    drum_report["files"].append(dest.name)
            else:
                drum_report["missing"] = [name for name in DRUM5 if name not in loaded]

        # Candidate B: family-level UVR Wind extraction from the RoFormer Other parent.
        emit("Running Wind Family Experiment", 56)
        other_parent = root / "other_parent.flac"
        _copy_as_flac(stems["other"], other_parent)
        t0 = time.monotonic()
        wind_cmd = ["audio-separator", str(other_parent), "--model_filename", WIND_MODEL, "--model_file_dir", str(audio_separator_model_dir), "--output_dir", str(wind_out), "--output_format", "FLAC"]
        wind = subprocess.run(wind_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        timings["wind_uvr"] = round(time.monotonic() - t0, 3)
        wind_files = _copy_audio_tree(wind_out, experimental, f"{track}_wind_uvr") if wind.returncode == 0 else []

        # Candidate C: dedicated saxophone model. It is evaluated beside the family stem, not promoted automatically.
        emit("Running Saxophone Specialist Experiment", 74)
        t0 = time.monotonic()
        sax_cmd = ["python", "-m", "demucs", "--repo", str(SAX_MODEL_DIR), "-n", SAX_MODEL, "-o", str(sax_out), str(other_parent)]
        sax = subprocess.run(sax_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        timings["sax_demucs"] = round(time.monotonic() - t0, 3)
        sax_files = _copy_audio_tree(sax_out, experimental, f"{track}_sax_specialist") if sax.returncode == 0 else []

        report = {
            "schema_version": 2,
            "mode": MODE,
            "quality_baseline": "BS-RoFormer-SW parent stems at ZIP root",
            "experimental_policy": "Children remain experimental; drums use the validated 5-stem path with kick-only mild Wiener refinement and mass-conserving compensation.",
            "models": {
                "drums": "MDX23C DrumSep 5-stem mirror (aufr33/jarredou) + kick-only Wiener p=1 FFT2048",
                "wind": WIND_MODEL,
                "saxophone": SAX_MODEL,
            },
            "drums_5stem": drum_report,
            "wind": {"ok": wind.returncode == 0, "files": wind_files, "runtime_tail": "\n".join((wind.stdout or "").splitlines()[-25:]) if wind.returncode else ""},
            "saxophone": {"ok": sax.returncode == 0, "files": sax_files, "runtime_tail": "\n".join((sax.stdout or "").splitlines()[-25:]) if sax.returncode else ""},
            "sw_auto_installed": bool(sw_installed),
            "timings_seconds": timings,
            "licensing": {
                "drumsep_5stem": "research only; original checkpoint terms unresolved",
                "wind_uvr": "research pending provenance/license confirmation despite public mirrors",
                "sax_demucs": "model card currently tagged MIT",
            },
        }
        (final / f"{track}_EXPERIMENTAL_REPORT.json").write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")

        emit("Packaging parent and experimental stems", 94)
        archive = root / f"{track}_parent_plus_experimental.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
            for p in sorted(final.rglob("*")):
                if p.is_file():
                    bundle.write(p, arcname=str(p.relative_to(final)))

        uploaded = False
        put_url = str(payload.get("result_put_url") or "").strip()
        if put_url:
            import requests
            with archive.open("rb") as handle:
                response = requests.put(put_url, data=handle, headers={"Content-Type": "application/zip"}, timeout=(30, 1800))
            response.raise_for_status()
            uploaded = True

        timings["total"] = round(time.monotonic() - started, 3)
        emit("Experimental child comparison complete", 100)
        return _json_safe({
            "ok": True,
            "mode": MODE,
            "track": track,
            "archive_name": archive.name,
            "archive_size_bytes": archive.stat().st_size,
            "uploaded": uploaded,
            "result_url": payload.get("result_public_url"),
            "root_parent_files": sorted(p.name for p in final.iterdir() if p.is_file()),
            "experimental_files": sorted(p.name for p in experimental.iterdir() if p.is_file()),
            "report": report,
        })
