from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import requests
import soundfile as sf

from routed_extraction_v1 import _collect_sw_stems, _safe_name
from sw_residual_allocator import _resolve_model_files
from wind_brass_decomposition_v2 import _run_polled

MODE = "research_benchmark_v2"
MODEL_DIR = Path("/models/audio_separator")
VOCAL_CHALLENGER = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
KARAOKE_BASELINE = "mel_band_roformer_karaoke_becruily.ckpt"
KARAOKE_CHALLENGER = "bs_roformer_karaoke_anvuew.ckpt"
UNMIXX_DIR = Path("/opt/unmixx")
UNMIXX_CONF = UNMIXX_DIR / "ckpt/conf.yml"
UNMIXX_CKPT = UNMIXX_DIR / "ckpt/best.ckpt"

PUBLISHED = {
    "vocal_challenger": {
        "label": "Viperx BS-RoFormer ep317",
        "filename_metric": 12.9755,
        "multisong_vocals_sdr": 10.87,
        "note": "12.9755 is the checkpoint/private-validation label; ZFTurbo publishes 10.87 on Multisong.",
    },
    "karaoke_challenger": {
        "label": "Anvuew BS-RoFormer Karaoke",
        "published_lead_sdr": 10.22,
    },
    "unmixx": {
        "label": "UNMIXX",
        "published_claim": ">~2.2 dB SDRi over the prior MedleyVox method on the MedleyVox evaluation set",
    },
}


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _download(url: str, destination: Path) -> None:
    with requests.get(url, stream=True, timeout=(30, 1800)) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(4 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)


def _run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 1800) -> tuple[int, float, str]:
    started = time.monotonic()
    completed = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )
    return completed.returncode, time.monotonic() - started, completed.stdout or ""


def _mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    return np.mean(audio, axis=1, dtype=np.float32), int(sr)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n <= 0:
        return 0.0
    aa = a[:n].astype(np.float64, copy=False)
    bb = b[:n].astype(np.float64, copy=False)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / denom) if denom > 1e-12 else 0.0


def _db_ratio(num: np.ndarray, den: np.ndarray) -> float:
    n = min(len(num), len(den))
    if n <= 0:
        return -120.0
    nr = float(np.sqrt(np.mean(num[:n].astype(np.float64) ** 2) + 1e-12))
    dr = float(np.sqrt(np.mean(den[:n].astype(np.float64) ** 2) + 1e-12))
    return 20.0 * math.log10(max(nr, 1e-12) / max(dr, 1e-12))


def _stats(path: Path) -> dict:
    audio, sr = _mono(path)
    if not len(audio):
        return {"duration_seconds": 0.0, "rms_dbfs": -120.0, "active_ratio": 0.0}
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-12))
    frame = max(1, int(sr * 0.05))
    usable = (len(audio) // frame) * frame
    if usable:
        framed = audio[:usable].reshape(-1, frame).astype(np.float64)
        frame_rms = np.sqrt(np.mean(framed * framed, axis=1) + 1e-12)
        threshold = max(10 ** (-50.0 / 20.0), rms * 0.18)
        active = float(np.mean(frame_rms >= threshold))
    else:
        active = float(rms > 10 ** (-50.0 / 20.0))
    return {
        "duration_seconds": round(len(audio) / sr, 3),
        "rms_dbfs": round(20.0 * math.log10(max(rms, 1e-12)), 3),
        "active_ratio": round(active, 6),
    }


def _pair_metrics(parent: Path, a: Path, b: Path) -> dict:
    p, _ = _mono(parent)
    x, _ = _mono(a)
    y, _ = _mono(b)
    n = min(len(p), len(x), len(y))
    if n <= 0:
        return {}
    summed = x[:n] + y[:n]
    residual = p[:n] - summed
    return {
        "reconstruction_cosine": round(_cos(p[:n], summed), 6),
        "residual_relative_to_parent_db": round(_db_ratio(residual, p[:n]), 3),
        "sibling_abs_correlation": round(abs(_cos(x[:n], y[:n])), 6),
        "a": _stats(a),
        "b": _stats(b),
    }


def _similarity(reference: Path, candidate: Path) -> dict:
    a, _ = _mono(reference)
    b, _ = _mono(candidate)
    n = min(len(a), len(b))
    if n <= 0:
        return {}
    diff = a[:n] - b[:n]
    return {
        "cosine": round(_cos(a[:n], b[:n]), 8),
        "difference_relative_to_reference_db": round(_db_ratio(diff, a[:n]), 3),
    }


def _audio_files(root: Path) -> list[Path]:
    return sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".wav", ".flac", ".mp3", ".m4a"}],
        key=lambda p: p.stat().st_size,
        reverse=True,
    )


def _classify_pair(root: Path) -> tuple[Path, Path]:
    files = _audio_files(root)
    primary = next((p for p in files if "vocals" in p.name.lower() and "instrumental" not in p.name.lower()), None)
    secondary = next((p for p in files if "instrumental" in p.name.lower() or "other" in p.name.lower()), None)
    if not primary or not secondary:
        raise RuntimeError(f"Could not classify separator pair: {[p.name for p in files]}")
    return primary, secondary


def _run_separator(input_path: Path, output_dir: Path, model: str, flags: list[str], timeout: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "audio-separator", str(input_path),
        "--model_filename", model,
        "--model_file_dir", str(MODEL_DIR),
        "--output_dir", str(output_dir),
        "--output_format", "FLAC",
        "--use_soundfile",
        *flags,
    ]
    rc, elapsed, log = _run(cmd, timeout=timeout)
    result = {
        "returncode": rc,
        "runtime_seconds": round(elapsed, 3),
        "flags": flags,
        "log_tail": "\n".join(log.splitlines()[-80:]),
        "fallback_detected": any(token in log.lower() for token in ("falling back", "fallback", "not supported", "disabled")),
    }
    if rc != 0:
        return result
    a, b = _classify_pair(output_dir)
    result.update({"primary": str(a), "secondary": str(b)})
    return result


def _copy_named(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _crop_center(input_path: Path, output_path: Path, seconds: float = 60.0) -> None:
    try:
        info = sf.info(str(input_path))
        duration = float(info.frames) / float(info.samplerate)
    except Exception:
        duration = seconds
    length = min(seconds, duration)
    start = max(0.0, (duration - length) / 2.0)
    cmd = [
        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{length:.3f}",
        "-i", str(input_path), "-ar", "44100", "-ac", "2", str(output_path),
    ]
    rc, _, log = _run(cmd, timeout=300)
    if rc != 0:
        raise RuntimeError("Failed to create benchmark crop: " + log[-2000:])


def _version(command: list[str]) -> str:
    try:
        rc, _, out = _run(command, timeout=60)
        if rc == 0:
            return (out.strip().splitlines() or [""])[-1]
    except Exception:
        pass
    return "unknown"


def run_research_benchmark(payload: dict, progress=None) -> dict:
    audio_url = str(payload.get("audio_url") or payload.get("source_url") or "").strip()
    if not audio_url:
        return {"ok": False, "mode": MODE, "error": "audio_url is required"}

    timeout = max(300, int(payload.get("timeout_seconds") or 2400))
    heartbeat = max(5, int(payload.get("heartbeat_seconds") or 15))
    filename = str(payload.get("filename") or unquote(Path(urlparse(audio_url).path).name) or "track.wav")
    track = _safe_name(Path(filename).stem)
    model_dir = Path(str(payload.get("model_dir") or "/models/bs_roformer_sw"))

    def emit(message: str, percent: int) -> None:
        print(f"[{MODE}] {message} ({percent}%)", flush=True)
        if progress:
            progress(message, percent)

    build_sha = os.getenv("LITELABS_BUILD_SHA", "unknown")
    report: dict = {
        "schema_version": 2,
        "mode": MODE,
        "build_sha": build_sha,
        "track": track,
        "published_reference_metrics": PUBLISHED,
        "environment": {
            "audio_separator": _version(["audio-separator", "--version"]),
            "bs_roformer_infer": _version(["bs-roformer-infer", "--version"]),
        },
        "tests": {},
    }
    try:
        import torch
        report["environment"]["torch"] = torch.__version__
        report["environment"]["cuda"] = torch.version.cuda
        report["environment"]["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except Exception as exc:
        report["environment"]["torch_error"] = str(exc)

    started = time.monotonic()
    archive = Path("/tmp") / f"litelabs-research-{uuid.uuid4().hex[:10]}-{track}.zip"
    with tempfile.TemporaryDirectory(prefix="litelabs_research_v2_") as temp:
        root = Path(temp)
        source_dir = root / "source"
        sw_dir = root / "sw"
        outputs = root / "outputs"
        source_dir.mkdir()
        sw_dir.mkdir()
        outputs.mkdir()

        raw_name = unquote(Path(urlparse(audio_url).path).name) or "input.audio"
        downloaded = root / raw_name
        source = source_dir / f"{track}.wav"

        emit("Downloading benchmark source", 2)
        _download(audio_url, downloaded)
        rc, elapsed, log = _run(["ffmpeg", "-y", "-i", str(downloaded), "-ar", "44100", "-ac", "2", str(source)], timeout=300)
        if rc != 0:
            raise RuntimeError("Source conversion failed: " + log[-2000:])
        report["tests"]["source_conversion"] = {"runtime_seconds": round(elapsed, 3)}

        emit("Running current SW vocal parent baseline", 8)
        sw_config, sw_checkpoint, _ = _resolve_model_files(model_dir, progress=None)
        rc, sw_elapsed = _run_polled(
            ["bs-roformer-infer", "--config_path", str(sw_config), "--model_path", str(sw_checkpoint),
             "--input_folder", str(source_dir), "--store_dir", str(sw_dir)],
            cwd=None, timeout=timeout, log_path=root / "sw.log", progress=None,
            stage_name="Research SW baseline", start_percent=8, end_percent=22, heartbeat_seconds=heartbeat,
        )
        if rc != 0:
            raise RuntimeError("Current SW parent baseline failed")
        sw_stems = _collect_sw_stems(sw_dir)
        sw_vocals = sw_stems.get("vocals")
        if not sw_vocals:
            raise RuntimeError("SW baseline did not produce vocals")
        report["tests"]["sw_parent_baseline"] = {
            "runtime_seconds": round(sw_elapsed, 3),
            "vocals": _stats(sw_vocals),
        }
        _copy_named(sw_vocals, outputs / "01_current_sw_vocals.flac")

        emit("Testing public vocal-parent challenger", 24)
        challenger_dir = root / "vocal_challenger"
        challenger = _run_separator(source, challenger_dir, VOCAL_CHALLENGER, [], timeout)
        report["tests"]["vocal_parent_challenger"] = challenger
        if challenger.get("returncode") == 0:
            challenger_vocals = Path(challenger["primary"])
            challenger_inst = Path(challenger["secondary"])
            challenger["pair_metrics"] = _pair_metrics(source, challenger_vocals, challenger_inst)
            challenger["vs_current_sw_vocals"] = _similarity(sw_vocals, challenger_vocals)
            _copy_named(challenger_vocals, outputs / "02_challenger_vocals.flac")
            _copy_named(challenger_inst, outputs / "02_challenger_instrumental.flac")

        emit("Running current lead/back baseline", 36)
        baseline_dir = root / "karaoke_baseline"
        baseline = _run_separator(sw_vocals, baseline_dir, KARAOKE_BASELINE, [], timeout)
        report["tests"]["lead_back_current"] = baseline
        baseline_lead = baseline_back = None
        if baseline.get("returncode") == 0:
            baseline_lead = Path(baseline["primary"])
            baseline_back = Path(baseline["secondary"])
            baseline["pair_metrics"] = _pair_metrics(sw_vocals, baseline_lead, baseline_back)
            _copy_named(baseline_lead, outputs / "03_current_lead_vocals.flac")
            _copy_named(baseline_back, outputs / "03_current_backing_vocals.flac")

        emit("Running Anvuew lead/back challenger", 46)
        anvuew_dir = root / "karaoke_anvuew"
        anvuew = _run_separator(sw_vocals, anvuew_dir, KARAOKE_CHALLENGER, [], timeout)
        report["tests"]["lead_back_anvuew"] = anvuew
        if anvuew.get("returncode") == 0:
            anvuew_lead = Path(anvuew["primary"])
            anvuew_back = Path(anvuew["secondary"])
            anvuew["pair_metrics"] = _pair_metrics(sw_vocals, anvuew_lead, anvuew_back)
            if baseline_lead and baseline_back:
                anvuew["vs_current"] = {
                    "lead": _similarity(baseline_lead, anvuew_lead),
                    "backing": _similarity(baseline_back, anvuew_back),
                }
            _copy_named(anvuew_lead, outputs / "04_anvuew_lead_vocals.flac")
            _copy_named(anvuew_back, outputs / "04_anvuew_backing_vocals.flac")

        emit("Benchmarking RoFormer precision/runtime modes", 58)
        crop = root / "sw_vocals_60s.wav"
        _crop_center(sw_vocals, crop, 60.0)
        speed_modes = [
            ("fp32_eager", []),
            ("autocast_eager", ["--use_autocast"]),
            ("native_fp16_eager", ["--use_native_fp16"]),
            ("autocast_compile", ["--use_autocast", "--use_torch_compile"]),
            ("native_fp16_compile", ["--use_native_fp16", "--use_torch_compile"]),
        ]
        speed_results = {}
        speed_reference = None
        for index, (name, flags) in enumerate(speed_modes):
            emit(f"Speed test: {name}", min(74, 58 + index * 4))
            out_dir = root / f"speed_{name}"
            result = _run_separator(crop, out_dir, KARAOKE_CHALLENGER, flags, timeout)
            if result.get("returncode") == 0:
                lead = Path(result["primary"])
                back = Path(result["secondary"])
                result["pair_metrics"] = _pair_metrics(crop, lead, back)
                if speed_reference is None:
                    speed_reference = (lead, back)
                else:
                    result["vs_fp32_eager"] = {
                        "lead": _similarity(speed_reference[0], lead),
                        "backing": _similarity(speed_reference[1], back),
                    }
            speed_results[name] = result
        report["tests"]["anvuew_speed_matrix_60s"] = speed_results

        emit("Testing UNMIXX multi-singer separation", 80)
        unmixx_input = root / "unmixx_vocals.wav"
        rc, _, log = _run(["ffmpeg", "-y", "-i", str(sw_vocals), "-ar", "24000", "-ac", "1", str(unmixx_input)], timeout=300)
        if rc != 0:
            raise RuntimeError("UNMIXX input conversion failed: " + log[-2000:])
        unmixx_out = root / "unmixx_out"
        rc, unmixx_elapsed, unmixx_log = _run(
            ["python", str(UNMIXX_DIR / "inference.py"), "--conf_path", str(UNMIXX_CONF),
             "--ckpt_path", str(UNMIXX_CKPT), "--audio_path", str(unmixx_input),
             "--output_dir", str(unmixx_out), "--device", "cuda"],
            cwd=UNMIXX_DIR, timeout=timeout,
        )
        unmixx_result = {
            "returncode": rc,
            "runtime_seconds": round(unmixx_elapsed, 3),
            "log_tail": "\n".join(unmixx_log.splitlines()[-80:]),
        }
        if rc == 0:
            speakers = sorted(unmixx_out.rglob("spk*.wav"))
            if len(speakers) >= 2:
                unmixx_result["pair_metrics"] = _pair_metrics(unmixx_input, speakers[0], speakers[1])
                _copy_named(speakers[0], outputs / "05_unmixx_voice_a.wav")
                _copy_named(speakers[1], outputs / "05_unmixx_voice_b.wav")
                unmixx_result["files"] = [p.name for p in speakers[:2]]
        report["tests"]["unmixx"] = unmixx_result

        report["total_runtime_seconds"] = round(time.monotonic() - started, 3)
        report_path = outputs / "research_benchmark_report.json"
        report_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
        readme = outputs / "README.txt"
        readme.write_text(
            "LiteLABS Research Benchmark v2\n"
            "==============================\n\n"
            "This pack is an A/B research artifact, not a production stem pack.\n"
            "Listen to numbered candidates alongside research_benchmark_report.json.\n"
            "Published SDR values in the report are external metadata; local metrics are\n"
            "reconstruction/similarity signals and must not be interpreted as SDR.\n",
            encoding="utf-8",
        )

        emit("Packaging research comparison", 92)
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
            for path in sorted(outputs.rglob("*")):
                if path.is_file():
                    bundle.write(path, arcname=path.name)

    archive_size = archive.stat().st_size
    uploaded = False
    put_url = str(payload.get("result_put_url") or "").strip()
    if put_url:
        emit("Uploading research comparison", 96)
        with archive.open("rb") as handle:
            response = requests.put(put_url, data=handle, headers={"Content-Type": "application/zip"}, timeout=(30, 1800))
        response.raise_for_status()
        uploaded = True

    emit("Research benchmark complete", 100)
    result = {
        "ok": True,
        "mode": MODE,
        "track": track,
        "build_sha": build_sha,
        "archive_name": archive.name,
        "archive_size_bytes": archive_size,
        "uploaded": uploaded,
        "result_url": payload.get("result_public_url"),
        "report": report,
    }
    if uploaded:
        archive.unlink(missing_ok=True)
    return _json_safe(result)
