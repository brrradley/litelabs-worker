from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
import soundfile as sf

from ground_truth_benchmark import _load, _normalise_audio, _score
from model_ground_truth_bakeoff import _find_stem

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a"}
SUPPORTED_TARGETS = {"bass", "drums", "guitar", "piano", "other"}


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


def _run(cmd: list[str], timeout: int) -> tuple[int, str, float, int | None]:
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


def _separator_command(
    source: Path,
    model: str,
    output_dir: Path,
    stem: str,
    model_dir: Path,
    segment_size: int,
    overlap: int,
    batch_size: int,
    use_autocast: bool,
) -> list[str]:
    cmd = [
        "audio-separator", str(source),
        "--model_filename", model,
        "--model_file_dir", str(model_dir),
        "--output_dir", str(output_dir),
        "--output_format", "FLAC",
        "--single_stem", stem.capitalize(),
        "--mdxc_segment_size", str(segment_size),
        "--mdxc_overlap", str(overlap),
        "--mdxc_batch_size", str(batch_size),
    ]
    if use_autocast:
        cmd.append("--use_autocast")
    return cmd


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate, subtype="PCM_16")


def _score_generated(generated: Path, reference: tuple, root: Path, label: str) -> dict:
    normalised = root / "normalised" / f"{label}.wav"
    _normalise_audio(generated, normalised)
    estimate, rate = _load(normalised)
    reference_audio, reference_rate = reference
    if rate != reference_rate:
        raise ValueError(f"Sample-rate mismatch for {label}")
    return _score(reference_audio, estimate, rate)


def build_cascade_ground_truth_bakeoff(payload: dict, progress=None) -> dict:
    audio_url = str(payload.get("audio_url") or "").strip()
    references = payload.get("references") or {}
    target_stems = payload.get("target_stems") or ["bass", "drums"]
    if not isinstance(target_stems, list):
        return {"ok": False, "mode": "cascade_ground_truth_bakeoff", "error": "target_stems must be a list"}
    target_stems = [str(stem).strip().lower() for stem in target_stems]
    if not audio_url:
        return {"ok": False, "mode": "cascade_ground_truth_bakeoff", "error": "audio_url is required"}
    if not target_stems or any(stem not in SUPPORTED_TARGETS for stem in target_stems):
        return {"ok": False, "mode": "cascade_ground_truth_bakeoff", "error": "Unsupported or empty target_stems"}
    for stem in target_stems:
        if not references.get(stem):
            return {"ok": False, "mode": "cascade_ground_truth_bakeoff", "error": f"references.{stem} is required"}

    split_model = str(payload.get("split_model") or "BS-Roformer-SW.ckpt")
    frontends = payload.get("vocal_models") or ["bs_roformer_vocals_revive_v3e_unwa.ckpt"]
    if not isinstance(frontends, list) or not frontends:
        return {"ok": False, "mode": "cascade_ground_truth_bakeoff", "error": "vocal_models must be a non-empty list"}
    frontends = [str(model) for model in frontends]

    timeout = max(300, min(3300, int(payload.get("model_timeout_seconds") or 1800)))
    overlap = max(2, min(50, int(payload.get("mdxc_overlap") or 8)))
    segment_size = max(32, min(4096, int(payload.get("mdxc_segment_size") or 256)))
    batch_size = max(1, min(16, int(payload.get("mdxc_batch_size") or 1)))
    use_autocast = bool(payload.get("use_autocast", True))
    model_dir = Path(os.getenv("LITELABS_AUDIO_SEPARATOR_MODEL_DIR", "/models/audio_separator"))
    model_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="litelabs_cascade_gt_") as temp:
        root = Path(temp)
        source = root / _filename_from_url(audio_url, "source.wav")
        source_wav = root / "normalised" / "source.wav"
        if progress:
            progress("Downloading source and references", 3)
        _download(audio_url, source)
        _normalise_audio(source, source_wav)
        source_audio, source_rate = _load(source_wav)

        reference_audio: dict[str, tuple] = {}
        for stem in target_stems:
            raw = root / "references" / _filename_from_url(str(references[stem]), f"{stem}.wav")
            wav = root / "references_normalised" / f"{stem}.wav"
            _download(str(references[stem]), raw)
            _normalise_audio(raw, wav)
            reference_audio[stem] = _load(wav)

        # Pipeline A: direct full mix -> split model.
        baseline = {"pipeline": "direct", "label": f"direct:{split_model}", "ok": False, "stages": []}
        try:
            total_runtime = 0.0
            peak_vram = 0
            scores = {}
            generated_files = {}
            for index, stem in enumerate(target_stems):
                output_dir = root / "direct" / stem
                output_dir.mkdir(parents=True, exist_ok=True)
                cmd = _separator_command(source, split_model, output_dir, stem, model_dir, segment_size, overlap, batch_size, use_autocast)
                code, output, runtime, peak = _run(cmd, timeout)
                baseline["stages"].append({"stage": f"direct_{stem}", "command": cmd, "returncode": code, "runtime_seconds": round(runtime, 3), "peak_gpu_memory_mib": peak, "log_tail": output[-3000:]})
                total_runtime += runtime
                peak_vram = max(peak_vram, int(peak or 0))
                if code != 0:
                    raise RuntimeError(f"Direct {stem} separation exited with code {code}")
                generated = _find_stem(output_dir, stem)
                if generated is None:
                    raise FileNotFoundError(f"Could not identify direct {stem} file")
                scores[stem] = _score_generated(generated, reference_audio[stem], root, f"direct_{stem}")
                generated_files[stem] = generated.name
            baseline.update({"ok": True, "scores": scores, "generated_files": generated_files, "runtime_seconds": round(total_runtime, 3), "peak_gpu_memory_mib": peak_vram})
        except Exception as exc:
            baseline.update({"error": str(exc), "error_type": exc.__class__.__name__})
        rows.append(baseline)

        # Pipeline B: full mix -> vocal model -> source-vocals residual -> split model.
        for frontend_index, vocal_model in enumerate(frontends):
            if progress:
                progress(f"Running cascade with {vocal_model}", int(25 + 65 * frontend_index / max(1, len(frontends))))
            row = {"pipeline": "vocal_first_cascade", "label": f"cascade:{vocal_model}->{split_model}", "vocal_model": vocal_model, "split_model": split_model, "ok": False, "stages": []}
            try:
                total_runtime = 0.0
                peak_vram = 0
                vocal_dir = root / "cascade" / f"{frontend_index:02d}" / "vocals"
                vocal_dir.mkdir(parents=True, exist_ok=True)
                vocal_cmd = _separator_command(source, vocal_model, vocal_dir, "vocals", model_dir, segment_size, overlap, batch_size, use_autocast)
                code, output, runtime, peak = _run(vocal_cmd, timeout)
                row["stages"].append({"stage": "vocal_extraction", "command": vocal_cmd, "returncode": code, "runtime_seconds": round(runtime, 3), "peak_gpu_memory_mib": peak, "log_tail": output[-3000:]})
                total_runtime += runtime
                peak_vram = max(peak_vram, int(peak or 0))
                if code != 0:
                    raise RuntimeError(f"Vocal extraction exited with code {code}")
                vocal_file = _find_stem(vocal_dir, "vocals")
                if vocal_file is None:
                    raise FileNotFoundError("Could not identify cascade vocal file")
                vocal_wav = root / "normalised" / f"cascade_{frontend_index:02d}_vocals.wav"
                _normalise_audio(vocal_file, vocal_wav)
                vocal_audio, vocal_rate = _load(vocal_wav)
                if vocal_rate != source_rate:
                    raise ValueError("Vocal/source sample-rate mismatch")
                length = min(len(source_audio), len(vocal_audio))
                instrumental_audio = source_audio[:length] - vocal_audio[:length]
                instrumental_path = root / "cascade" / f"{frontend_index:02d}" / "instrumental_residual.wav"
                _write_audio(instrumental_path, instrumental_audio, source_rate)

                scores = {}
                generated_files = {"vocals": vocal_file.name, "instrumental_input": instrumental_path.name}
                for stem in target_stems:
                    output_dir = root / "cascade" / f"{frontend_index:02d}" / stem
                    output_dir.mkdir(parents=True, exist_ok=True)
                    split_cmd = _separator_command(instrumental_path, split_model, output_dir, stem, model_dir, segment_size, overlap, batch_size, use_autocast)
                    code, output, runtime, peak = _run(split_cmd, timeout)
                    row["stages"].append({"stage": f"instrumental_split_{stem}", "command": split_cmd, "returncode": code, "runtime_seconds": round(runtime, 3), "peak_gpu_memory_mib": peak, "log_tail": output[-3000:]})
                    total_runtime += runtime
                    peak_vram = max(peak_vram, int(peak or 0))
                    if code != 0:
                        raise RuntimeError(f"Cascade {stem} split exited with code {code}")
                    generated = _find_stem(output_dir, stem)
                    if generated is None:
                        raise FileNotFoundError(f"Could not identify cascade {stem} file")
                    scores[stem] = _score_generated(generated, reference_audio[stem], root, f"cascade_{frontend_index:02d}_{stem}")
                    generated_files[stem] = generated.name
                row.update({"ok": True, "scores": scores, "generated_files": generated_files, "runtime_seconds": round(total_runtime, 3), "peak_gpu_memory_mib": peak_vram})
            except Exception as exc:
                row.update({"error": str(exc), "error_type": exc.__class__.__name__})
            rows.append(row)

    leaderboards = {}
    for stem in target_stems:
        entries = []
        for row in rows:
            if row.get("ok") and stem in (row.get("scores") or {}):
                entry = {"pipeline": row["pipeline"], "label": row["label"], "runtime_seconds": row.get("runtime_seconds"), "peak_gpu_memory_mib": row.get("peak_gpu_memory_mib")}
                entry.update(row["scores"][stem])
                entries.append(entry)
        leaderboards[stem] = sorted(entries, key=lambda item: (-float(item.get("quality_score") or 0), -float(item.get("si_sdr_db") or -999)))

    return {
        "ok": True,
        "mode": "cascade_ground_truth_bakeoff",
        "schema_version": 1,
        "target_stems": target_stems,
        "split_model": split_model,
        "vocal_models": frontends,
        "runs": rows,
        "leaderboards": leaderboards,
        "settings": {"model_timeout_seconds": timeout, "mdxc_segment_size": segment_size, "mdxc_overlap": overlap, "mdxc_batch_size": batch_size, "use_autocast": use_autocast},
        "no_audio_exported": True,
        "metric_note": "Compares direct full-mix instrument separation with a vocal-first cascade using source-minus-vocals as the instrumental input.",
    }
