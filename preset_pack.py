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


def _detect_parent_genre(stems: dict[str, Path], source: Path) -> tuple[str, str]:
    """Reuse the established LiteLABS parent-stem heuristics for preset metadata."""
    try:
        import master_pack

        stats = {name: master_pack.analyse_audio(path) for name, path in stems.items()}
        score = lambda name: float((stats.get(name) or {}).get("score", 0.0))

        vocals = score("vocals")
        drums = score("drums")
        bass = score("bass")
        guitar = score("guitar")
        piano = score("piano")
        other = score("other")

        strong_rhythm = drums >= 0.44 and bass >= 0.30
        strong_vocal = vocals >= 0.45
        strong_guitar = guitar >= 0.42
        dominant_guitar = strong_guitar and guitar > max(other + 0.18, 0.66)
        strong_piano = piano >= 0.42
        strong_other = other >= 0.38

        if strong_rhythm and dominant_guitar:
            return "rock_band", "strong drums with dominant confident guitar activity"
        if strong_rhythm and (strong_other or not strong_guitar or bass >= 0.42):
            details = ["strong drums/bass"]
            if strong_other:
                details.append("active synth/other")
            if strong_guitar and not dominant_guitar:
                details.append("guitar appears secondary/sample-like")
            return "electronic_dance", ", ".join(details)
        if strong_piano and strong_vocal and drums < 0.42:
            return "piano_vocal_or_pop_ballad", "confident piano/keys with strong vocal and lighter drums"
        if strong_vocal and drums >= 0.35 and bass >= 0.25 and not dominant_guitar:
            return "vocal_pop", "strong vocal with moderate rhythm section and no dominant guitar"
        if strong_rhythm and not strong_vocal:
            return "instrumental_or_dance", "strong drums/bass with weaker vocal presence"

        original = master_pack.analyse_audio(source)
        if strong_vocal and float(original.get("active_ratio", 0.0)) > 0.35 and drums < 0.30 and bass < 0.30:
            return "acoustic_or_sparse", "strong vocal with low drum/bass activity"
        return "mixed_or_unknown", "audio features did not strongly match a known route"
    except Exception as exc:
        print(f"LiteLABS preset genre analysis skipped: {exc}", flush=True)
        return "mixed_or_unknown", "genre analysis unavailable"


def _format_execution_time(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _write_readme(
    path: Path,
    *,
    filename: str,
    preset: str,
    exported: list[str],
    genre: str,
    execution_seconds: float,
) -> None:
    included = "\n".join(f"- {name}" for name in sorted(exported))
    path.write_text(
        "LiteLABS Stem Extraction Tools\n"
        "==============================\n\n"
        "TRACK INFORMATION\n"
        "-----------------\n"
        f"Track: {filename}\n"
        f"Pack: {PRESET_LABELS[preset]}\n"
        "Output format: FLAC\n"
        f"Detected genre: {genre}\n"
        f"Execution time: {_format_execution_time(execution_seconds)}\n\n"
        "INCLUDED STEMS\n"
        "--------------\n"
        f"{included}\n\n"
        "ABOUT THIS PACK\n"
        "---------------\n"
        "This stem pack was created using LiteLABS Stem Extraction Tools.\n"
        "AI source separation is not the same as access to the original multitrack session.\n"
        "Depending on the source mix, some stems may contain bleed, shared ambience, effects\n"
        "or elements that overlap with neighbouring stems. This is normal for source separation.\n\n"
        "For best results, audition the stems together as well as in isolation before making\n"
        "production decisions. Phase, mastering, source quality and the original arrangement can\n"
        "all affect the apparent quality of an individual stem.\n\n"
        "USAGE\n"
        "-----\n"
        "These files are supplied for use with LiteRECORDS/LiteLABS workflows. You remain\n"
        "responsible for ensuring that your use of the source recording and extracted material\n"
        "is permitted by the relevant rights holder and applicable law.\n\n"
        "Stem Extraction Tools by LiteLABS\n"
        "https://literecords.com/\n",
        encoding="utf-8",
    )


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
        supplied_filename = str(payload.get("filename") or raw_name)
        track = _safe_name(Path(supplied_filename).stem)
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

        genre, genre_reason = _detect_parent_genre(stems, source)
        print(f"LiteLABS detected genre: {genre} ({genre_reason})", flush=True)

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

        execution_seconds = time.monotonic() - started
        _write_readme(
            final / "README.txt",
            filename=supplied_filename,
            preset=preset,
            exported=exported,
            genre=genre,
            execution_seconds=execution_seconds,
        )

        report = {
            "schema_version": 2,
            "preset": preset,
            "requested_stems": list(requested),
            "exported_files": sorted(exported),
            "detected_genre": genre,
            "genre_reason": genre_reason,
            "execution_seconds": round(execution_seconds, 3),
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
            "detected_genre": genre,
            "execution_seconds": round(execution_seconds, 3),
            "specialist_separators_run": [],
            "timings_seconds": timings,
        }
