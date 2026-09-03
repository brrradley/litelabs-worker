from __future__ import annotations

import json
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np

from routed_extraction_v1 import _collect_sw_stems, _copy_as_flac, _read, _safe_name, _write_flac
from sw_residual_allocator import STEMS as SW_STEMS, _download, _resolve_model_files
from wind_brass_decomposition_v2 import _run_polled

PRESETS = {
    "basic": ("instrumental", "vocals"),
    "core": ("vocals", "percussion", "bass", "strings", "keys", "other"),
    "experimental": (
        "lead_vocals", "backing_vocals", "kick", "snare", "toms", "hi_hats",
        "cymbals", "bass", "strings", "keys", "other", "wind_brass", "saxophone",
    ),
}

PRESET_LABELS = {
    "basic": "Basic",
    "core": "Core",
    "experimental": "Experimental",
}

STEM_LABELS = {
    "instrumental": "Instrumental",
    "vocals": "Vocals",
    "percussion": "Percussion",
    "bass": "Bass",
    "strings": "Strings",
    "keys": "Keys",
    "other": "Other",
    "lead_vocals": "Lead Vocals",
    "backing_vocals": "Backing Vocals",
    "kick": "Kick",
    "snare": "Snare",
    "toms": "Toms",
    "hi_hats": "Hi-Hats",
    "cymbals": "Cymbals",
    "wind_brass": "Wind / Brass",
    "saxophone": "Saxophone",
}


def preset_capabilities() -> dict:
    return {
        "ok": True,
        "service": "litelabs-worker",
        "schema_version": 1,
        "presets": [
            {
                "id": preset_id,
                "label": PRESET_LABELS[preset_id],
                "stems": [STEM_LABELS[stem] for stem in stems],
            }
            for preset_id, stems in PRESETS.items()
        ],
    }


PARENT_LABELS = {
    "vocals": "vocals",
    "drums": "percussion",
    "bass": "bass",
    "guitar": "strings",
    "piano": "keys",
    "other": "other",
}


def normalise_preset(value: object) -> str:
    return str(value or "").strip().lower()


def build_parent_preset(payload: dict, progress=None) -> dict:
    preset = normalise_preset(payload.get("preset"))
    if preset not in {"basic", "core"}:
        return {"ok": False, "error": f"Unsupported parent preset: {preset}", "preset": preset}

    audio_url = str(payload.get("audio_url") or payload.get("source_url") or "").strip()
    if not audio_url:
        return {"ok": False, "error": "audio_url is required", "preset": preset}

    timeout = int(payload.get("timeout_seconds") or 1800)
    heartbeat = max(5, int(payload.get("heartbeat_seconds") or 15))
    model_dir = Path(str(payload.get("model_dir") or "/models/bs_roformer_sw"))
    started = time.monotonic()
    timings: dict[str, float] = {}

    def emit(message: str, percent: int) -> None:
        print(f"[preset:{preset}] {message} ({percent}%)", flush=True)
        if progress:
            progress(message, percent)

    sw_config, sw_checkpoint, sw_installed = _resolve_model_files(model_dir, progress=progress)

    with tempfile.TemporaryDirectory(prefix=f"litelabs_{preset}_") as temp:
        root = Path(temp)
        srcdir = root / "source"
        swout = root / "sw"
        final = root / "pack"
        logs = root / "logs"
        for directory in (srcdir, swout, final, logs):
            directory.mkdir(parents=True, exist_ok=True)

        raw_name = unquote(Path(urlparse(audio_url).path).name) or "track.flac"
        track = _safe_name(Path(str(payload.get("filename") or raw_name)).stem)
        downloaded = root / raw_name

        emit("Initiating stem separation", 2)
        t0 = time.monotonic()
        _download(audio_url, downloaded)
        timings["download"] = round(time.monotonic() - t0, 3)

        emit("Preparing source audio", 6)
        source = srcdir / f"{track}.wav"
        import subprocess
        t0 = time.monotonic()
        with (logs / "ffmpeg.log").open("w", encoding="utf-8") as log:
            conv = subprocess.run(
                ["ffmpeg", "-y", "-i", str(downloaded), "-ar", "44100", "-ac", "2", str(source)],
                stdout=log, stderr=subprocess.STDOUT, text=True, timeout=300,
            )
        timings["convert"] = round(time.monotonic() - t0, 3)
        if conv.returncode != 0:
            return {"ok": False, "preset": preset, "failed_stage": "convert"}

        emit("Running BS-RoFormer Parent Separation", 12)
        rc, elapsed = _run_polled(
            ["bs-roformer-infer", "--config_path", str(sw_config), "--model_path", str(sw_checkpoint),
             "--input_folder", str(srcdir), "--store_dir", str(swout)],
            cwd=None, timeout=timeout, log_path=logs / "sw.log", progress=progress,
            stage_name="BS-RoFormer Parent Separation", start_percent=12, end_percent=82,
            heartbeat_seconds=heartbeat,
        )
        timings["bs_roformer"] = round(elapsed, 3)
        if rc != 0:
            return {"ok": False, "preset": preset, "failed_stage": "bs_roformer"}

        stems = _collect_sw_stems(swout)
        missing = [stem for stem in SW_STEMS if stem not in stems]
        if missing:
            return {"ok": False, "preset": preset, "failed_stage": "collect_parents", "missing": missing}

        exported: list[str] = []
        requested = PRESETS[preset]

        if preset == "basic":
            vocals_dest = final / f"{track}_vocals.flac"
            _copy_as_flac(stems["vocals"], vocals_dest)
            exported.append(vocals_dest.name)

            mixture, mix_sr = _read(source)
            vocals, _ = _read(stems["vocals"])
            n = min(len(mixture), len(vocals))
            instrumental_dest = final / f"{track}_instrumental.flac"
            _write_flac(instrumental_dest, mixture[:n] - vocals[:n], mix_sr)
            exported.append(instrumental_dest.name)
        else:
            for source_stem in SW_STEMS:
                label = PARENT_LABELS[source_stem]
                if label not in requested:
                    continue
                dest = final / f"{track}_{label}.flac"
                _copy_as_flac(stems[source_stem], dest)
                exported.append(dest.name)

        readme = final / "README.txt"
        readme.write_text(
            "LiteLABS Stem Pack\n"
            "==================\n\n"
            f"Track: {track}\n"
            f"Preset: {preset.upper()}\n"
            "Output format: FLAC\n\n"
            "Included stems:\n\n" + "\n".join(requested) + "\n\n"
            "Stem Extraction Tools by LiteLABS\n",
            encoding="utf-8",
        )

        report = {
            "schema_version": 1,
            "preset": preset,
            "requested_stems": list(requested),
            "exported_files": sorted(exported),
            "specialist_separators_run": [],
            "sw_auto_installed": bool(sw_installed),
            "timings_seconds": timings,
        }
        (final / f"{track}_PRESET_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        emit("Packaging Stem Pack", 90)
        archive = root / f"{track}_{preset}_stem_pack.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
            for file in sorted(final.rglob("*")):
                if file.is_file():
                    bundle.write(file, arcname=str(file.relative_to(final)))

        uploaded = False
        put_url = str(payload.get("result_put_url") or "").strip()
        if put_url:
            emit("Uploading Stem Pack", 95)
            import requests
            with archive.open("rb") as handle:
                response = requests.put(
                    put_url, data=handle, headers={"Content-Type": "application/zip"}, timeout=(30, 1800)
                )
            response.raise_for_status()
            uploaded = True

        timings["total"] = round(time.monotonic() - started, 3)
        emit("Stem Extraction Complete", 100)
        return {
            "ok": True,
            "mode": "preset_pack",
            "preset": preset,
            "track": track,
            "archive_name": archive.name,
            "archive_size_bytes": archive.stat().st_size,
            "uploaded": uploaded,
            "result_url": payload.get("result_public_url"),
            "requested_stems": list(requested),
            "files": sorted(exported),
            "specialist_separators_run": [],
            "timings_seconds": timings,
        }
