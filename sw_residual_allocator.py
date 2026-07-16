from __future__ import annotations

import json
import subprocess
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import requests
import soundfile as sf

STEMS = ("vocals", "drums", "bass", "guitar", "piano", "other")


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=(30, 600)) as response:
        response.raise_for_status()
        with path.open("wb") as handle:
            for chunk in response.iter_content(4 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)


def _read(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    return audio[:, :2].astype(np.float64), int(sr)


def _db(value: float) -> float:
    return round(20.0 * np.log10(max(value, 1e-12)), 6)


def build_sw_residual_allocator(payload: dict, progress=None) -> dict:
    audio_url = str(payload.get("audio_url") or payload.get("source_url") or "").strip()
    if not audio_url:
        return {"ok": False, "mode": "sw_residual_allocator", "error": "audio_url is required"}

    timeout = int(payload.get("timeout_seconds") or 1800)
    strength = float(payload.get("allocation_strength") or 1.0)
    strength = max(0.0, min(1.0, strength))
    model_dir = Path(str(payload.get("model_dir") or "/models/bs_roformer_sw"))
    config = model_dir / "BS-Roformer-SW.yaml"
    checkpoint = model_dir / "BS-Roformer-SW.ckpt"

    if not config.is_file() or not checkpoint.is_file():
        return {"ok": False, "mode": "sw_residual_allocator", "error": "BS-RoFormer-SW model files are missing"}

    with tempfile.TemporaryDirectory(prefix="litelabs_sw_ra_") as temp:
        root = Path(temp)
        original_name = unquote(Path(urlparse(audio_url).path).name) or "track.flac"
        downloaded = root / original_name
        input_dir = root / "input"
        output_dir = root / "baseline"
        allocated_dir = root / "allocated"
        input_dir.mkdir()
        output_dir.mkdir()
        allocated_dir.mkdir()

        if progress:
            progress("Preparing SW/RA source", 5)
        _download(audio_url, downloaded)
        source_wav = input_dir / f"{Path(original_name).stem}.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(downloaded), "-ar", "44100", "-ac", "2", str(source_wav)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        if progress:
            progress("Running current BS-RoFormer-SW", 15)
        completed = subprocess.run(
            [
                "bs-roformer-infer",
                "--config_path", str(config),
                "--model_path", str(checkpoint),
                "--input_folder", str(input_dir),
                "--store_dir", str(output_dir),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        if completed.returncode != 0:
            return {
                "ok": False,
                "mode": "sw_residual_allocator",
                "failed_stage": "baseline",
                "return_code": completed.returncode,
                "runtime_log": "\n".join((completed.stdout or "").splitlines()[-100:]),
            }

        files = sorted(path for path in output_dir.rglob("*.wav") if path.is_file())
        stem_paths: dict[str, Path] = {}
        for stem in STEMS:
            matches = [path for path in files if path.name.lower().endswith(f"_{stem}.wav")]
            if not matches:
                matches = [path for path in files if stem in path.name.lower() and "instrumental" not in path.name.lower()]
            if not matches:
                return {"ok": False, "mode": "sw_residual_allocator", "failed_stage": "collect_stems", "missing_stem": stem}
            stem_paths[stem] = matches[0]

        mixture, sr = _read(source_wav)
        loaded = {stem: _read(path)[0] for stem, path in stem_paths.items()}
        length = min([len(mixture)] + [len(audio) for audio in loaded.values()])
        mixture = mixture[:length]
        loaded = {stem: audio[:length] for stem, audio in loaded.items()}

        stack = np.stack([loaded[stem] for stem in STEMS], axis=0)
        baseline_sum = np.sum(stack, axis=0)
        residual = mixture - baseline_sum
        baseline_residual_rms = float(np.sqrt(np.mean(residual * residual) + 1e-12))

        # Reference-free prototype: distribute residual according to each stem's
        # local absolute energy, with a floor so quiet stems are not starved.
        if progress:
            progress("Allocating mixture residual across SW stems", 65)
        weights = np.abs(stack) + 1e-5
        weights /= np.sum(weights, axis=0, keepdims=True)
        allocated_stack = stack + strength * weights * residual[None, :, :]

        # Any unallocated fraction is retained in Other so the exported set is exact.
        if strength < 1.0:
            allocated_stack[STEMS.index("other")] += (1.0 - strength) * residual

        for index, stem in enumerate(STEMS):
            sf.write(allocated_dir / f"{Path(original_name).stem}_{stem}.wav", allocated_stack[index].astype(np.float32), sr, subtype="FLOAT")

        allocated_sum = np.sum(allocated_stack, axis=0)
        final_residual = mixture - allocated_sum
        final_residual_rms = float(np.sqrt(np.mean(final_residual * final_residual) + 1e-12))
        exactness = 100.0 * max(0.0, 1.0 - final_residual_rms / (float(np.sqrt(np.mean(mixture * mixture) + 1e-12)) + 1e-12))

        report = {
            "prototype": True,
            "official_mvsep_ra": False,
            "warning": "This is a LiteLABS reference-free residual allocator prototype, not MVSEP's private RA/RAv2 implementation.",
            "allocation_strength": strength,
            "baseline_residual_rms_dbfs": _db(baseline_residual_rms),
            "final_residual_rms_dbfs": _db(final_residual_rms),
            "mixture_reconstruction_percent": round(exactness, 9),
        }
        (allocated_dir / "SW_RA_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        archive = root / "litelabs-sw-ra-prototype.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
            for path in sorted(allocated_dir.iterdir()):
                bundle.write(path, arcname=path.name)

        uploaded = False
        put_url = str(payload.get("result_put_url") or "").strip()
        if put_url:
            if progress:
                progress("Uploading SW/RA prototype", 90)
            with archive.open("rb") as handle:
                response = requests.put(put_url, data=handle, headers={"Content-Type": "application/zip"}, timeout=(30, 900))
            response.raise_for_status()
            uploaded = True

        if progress:
            progress("SW/RA prototype complete", 100)
        return {
            "ok": True,
            "mode": "sw_residual_allocator",
            "schema_version": 1,
            "prototype": True,
            "official_mvsep_ra": False,
            "report": report,
            "archive_name": archive.name,
            "archive_size_bytes": archive.stat().st_size,
            "uploaded": uploaded,
            "result_url": payload.get("result_public_url"),
            "files": sorted(path.name for path in allocated_dir.iterdir()),
        }
