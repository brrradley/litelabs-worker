from __future__ import annotations

import math
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import requests
import soundfile as sf

PRIMARY = ("vocals", "drums", "bass", "guitar", "piano", "other")
BASELINE = ("vocals", "drums", "bass", "guitar")


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with destination.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)


def _upload_put(url: str, path: Path) -> None:
    with path.open("rb") as source:
        response = requests.put(url, data=source, headers={"Content-Type": "application/zip"}, timeout=600)
    response.raise_for_status()


def _map_stems(root: Path) -> dict[str, Path]:
    from reference_free_stem_auditor import _find_audio, _role
    mapped: dict[str, Path] = {}
    for path in _find_audio(root):
        role = _role(path)
        if role in PRIMARY and role not in mapped:
            mapped[role] = path
    return mapped


def _align(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) >= length:
        return audio[:length]
    return np.pad(audio, ((0, length - len(audio)), (0, 0)))


def _candidate_metrics(name: str, audio: np.ndarray, baselines: dict[str, np.ndarray], role: str) -> dict:
    mono = np.mean(audio, axis=1, dtype=np.float64)
    rms = float(np.sqrt(np.mean(np.square(mono)) + 1e-12))
    rms_db = 20.0 * math.log10(max(rms, 1e-12))
    active_ratio = float(np.mean(np.abs(mono) >= 1e-4))
    correlations = {}
    for stem, baseline in baselines.items():
        left = mono - np.mean(mono)
        right = np.mean(baseline, axis=1, dtype=np.float64)
        right = right - np.mean(right)
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        correlations[stem] = abs(float(np.dot(left, right) / denom)) if denom > 1e-12 else 0.0
    max_corr = max(correlations.values(), default=0.0)
    audibility = max(0.0, min(1.0, (rms_db + 75.0) / 55.0))
    if role == "piano":
        score = 0.55 * audibility + 0.25 * active_ratio + 0.20 * (1.0 - max_corr)
    else:
        score = 0.45 * audibility + 0.20 * active_ratio + 0.35 * (1.0 - max_corr)
    return {
        "candidate": name,
        "role": role,
        "score": round(score, 6),
        "rms_dbfs": round(rms_db, 3),
        "active_ratio": round(active_ratio, 6),
        "max_baseline_correlation": round(max_corr, 6),
        "baseline_correlations": {key: round(value, 6) for key, value in correlations.items()},
    }


def _write_flac(path: Path, audio: np.ndarray, rate: int) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = float(np.mean(np.abs(audio) > 1.0))
    sf.write(path, np.clip(audio, -1.0, 1.0), rate, format="FLAC", subtype="PCM_24")
    return {"clipped_before_write_ratio": round(clipped, 8)}


def _zip_folder(folder: Path, archive: Path) -> None:
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zipped:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                zipped.write(path, arcname=str(Path(folder.name) / path.relative_to(folder)))


def build_adaptive_recovery_run(payload: dict, progress=None) -> dict:
    source_url = str(payload.get("source_url") or payload.get("audio_url") or "").strip()
    stem_pack_url = str(payload.get("stem_pack_url") or payload.get("pack_url") or "").strip()
    if not source_url or not stem_pack_url:
        return {"ok": False, "mode": "adaptive_recovery_run", "error": "source_url and stem_pack_url are required"}

    if progress:
        progress("Planning recovery", 3)
    from combined_recovery_planner import build_combined_recovery_planner
    initial_plan = build_combined_recovery_planner(payload, progress=None)
    if not initial_plan.get("ok"):
        return {"ok": False, "mode": "adaptive_recovery_run", "stage": "planning", "error": initial_plan.get("error", "planner failed")}

    mandatory = list(initial_plan.get("mandatory_retries") or [])
    if not mandatory:
        return {"ok": True, "mode": "adaptive_recovery_run", "schema_version": 2, "status": "no_recovery_needed", "initial_plan": initial_plan, "result_url": payload.get("result_public_url")}

    requested = {str(item.get("stem") or "") for item in mandatory}
    supported = requested.intersection({"piano", "other"})
    unsupported = sorted(requested - supported)

    with tempfile.TemporaryDirectory(prefix="litelabs_adaptive_recovery_") as temp:
        root = Path(temp)
        url_filename = unquote(Path(urlparse(source_url).path).name)
        filename = str(payload.get("filename") or url_filename or "track.flac")
        source_path = root / filename
        original_zip = root / "original_pack.zip"
        original_root = root / "original_pack"
        fresh_root = root / "fresh_pack"

        if progress:
            progress("Downloading source and baseline pack", 7)
        _download(source_url, source_path)
        _download(stem_pack_url, original_zip)

        from reference_free_stem_auditor import _load, _safe_extract
        _safe_extract(original_zip, original_root)
        original = _map_stems(original_root)
        missing_baseline = [stem for stem in BASELINE if stem not in original]
        if missing_baseline:
            return {"ok": False, "mode": "adaptive_recovery_run", "stage": "baseline", "error": f"Original pack is missing trusted baseline stems: {missing_baseline}"}

        if progress:
            progress("Generating fresh recovery candidates", 14)
        from master_pack import build_master_pack, safe_track_name
        fresh_result = build_master_pack(
            input_audio=source_path,
            work_root=root / "work",
            model_dir=Path(payload.get("model_dir") or os.getenv("STEMFORGE_MODEL_DIR", "/models/bs_roformer_sw")),
            output_root=root / "fresh_output",
            progress=(lambda message, percent: progress(message, 14 + int(percent * 0.50))) if progress else None,
        )
        fresh_archive = Path(fresh_result["archive_path"])
        _safe_extract(fresh_archive, fresh_root)
        fresh = _map_stems(fresh_root)

        source, rate = _load(source_path)
        length = len(source)
        baseline_audio = {stem: _align(_load(original[stem], rate)[0], length) for stem in BASELINE}

        track = safe_track_name(source_path.name)
        demucs_dir = root / "work" / track / "demucs6s" / "htdemucs_6s" / track
        piano_candidates: dict[str, np.ndarray] = {}
        other_candidates: dict[str, np.ndarray] = {}
        if "piano" in fresh:
            piano_candidates["bs_roformer_piano"] = _align(_load(fresh["piano"], rate)[0], length)
        if (demucs_dir / "piano.flac").exists():
            piano_candidates["demucs_piano"] = _align(_load(demucs_dir / "piano.flac", rate)[0], length)
        if "other" in fresh:
            other_candidates["bs_roformer_other"] = _align(_load(fresh["other"], rate)[0], length)
        if (demucs_dir / "other.flac").exists():
            other_candidates["demucs_other"] = _align(_load(demucs_dir / "other.flac", rate)[0], length)

        if not piano_candidates:
            return {"ok": False, "mode": "adaptive_recovery_run", "stage": "candidate_generation", "error": "No piano recovery candidate was generated"}

        if progress:
            progress("Ranking piano candidates", 72)
        piano_reports = [_candidate_metrics(name, audio, baseline_audio, "piano") for name, audio in piano_candidates.items()]
        piano_reports.sort(key=lambda item: -item["score"])
        selected_piano_name = piano_reports[0]["candidate"]
        selected_piano = piano_candidates[selected_piano_name]

        residual_other = source.copy()
        for audio in baseline_audio.values():
            residual_other -= audio
        residual_other -= selected_piano
        other_candidates["mixture_residual_other"] = residual_other

        if progress:
            progress("Ranking Other candidates", 76)
        other_reports = [_candidate_metrics(name, audio, baseline_audio | {"piano": selected_piano}, "other") for name, audio in other_candidates.items()]
        for report in other_reports:
            if report["candidate"] == "mixture_residual_other":
                report["score"] = round(report["score"] + 0.12, 6)
                report["consistency_bonus"] = 0.12
        other_reports.sort(key=lambda item: -item["score"])
        selected_other_name = other_reports[0]["candidate"]
        selected_other = other_candidates[selected_other_name]

        final_dir = root / "final" / f"{track}-litelabs-adaptive-stem-pack"
        final_dir.mkdir(parents=True, exist_ok=True)
        names = {
            "vocals": f"01_{track}_vocals.flac",
            "drums": f"02_{track}_drums.flac",
            "bass": f"03_{track}_bass.flac",
            "guitar": f"04_{track}_guitar.flac",
            "piano": f"05_{track}_piano_keys.flac",
            "other": f"06_{track}_synth_strings_other.flac",
            "instrumental": f"07_{track}_instrumental_clean.flac",
        }
        write_info = {}
        for stem in BASELINE:
            shutil.copy2(original[stem], final_dir / names[stem])
        write_info["piano"] = _write_flac(final_dir / names["piano"], selected_piano, rate)
        write_info["other"] = _write_flac(final_dir / names["other"], selected_other, rate)
        instrumental = baseline_audio["drums"] + baseline_audio["bass"] + baseline_audio["guitar"] + selected_piano + selected_other
        write_info["instrumental"] = _write_flac(final_dir / names["instrumental"], instrumental, rate)
        (final_dir / "README.txt").write_text(
            "LiteLABS adaptive recovery pack\n\n"
            "The original trusted vocals, drums, bass and guitar stems were preserved.\n"
            f"Selected piano candidate: {selected_piano_name}\n"
            f"Selected Other candidate: {selected_other_name}\n"
            "Dry main vocals remain disabled while dereverberation is reviewed.\n",
            encoding="utf-8",
        )

        archive = root / "output" / f"{track}-litelabs-adaptive-stem-pack.zip"
        archive.parent.mkdir(parents=True, exist_ok=True)
        _zip_folder(final_dir, archive)

        if progress:
            progress("Auditing final hybrid pack", 84)
        from reference_free_stem_auditor import build_reference_free_stem_auditor
        rebuilt_audit = build_reference_free_stem_auditor({"source_url": source_path.as_uri(), "stem_pack_url": archive.as_uri()}, progress=None)
        found = set(rebuilt_audit.get("primary_stems_found") or []) if rebuilt_audit.get("ok") else set()
        recovered = sorted(stem for stem in supported if stem in found)
        still_missing = sorted(stem for stem in requested if stem not in found)

        put_url = str(payload.get("result_put_url") or "").strip()
        uploaded = False
        if put_url:
            if progress:
                progress("Uploading recovered stem pack", 95)
            _upload_put(put_url, archive)
            uploaded = True

        if progress:
            progress("Adaptive recovery complete", 100)
        return {
            "ok": True,
            "mode": "adaptive_recovery_run",
            "schema_version": 2,
            "status": "recovered_pack_ready" if not still_missing else "recovery_incomplete",
            "source_url": source_url,
            "original_stem_pack_url": stem_pack_url,
            "requested_recovery_stems": sorted(requested),
            "supported_recovery_stems": sorted(supported),
            "unsupported_recovery_stems": unsupported,
            "recovered_stems": recovered,
            "still_missing_stems": still_missing,
            "strategy": {
                "baseline_preserved": True,
                "preserved_stems": list(BASELINE),
                "method": "Preserve the original trusted baseline and fill missing roles with ranked recovery candidates.",
                "candidate_selection": "Heuristic ranking uses audibility, activity, baseline correlation and a mixture-consistency bonus for residual Other.",
                "dry_main_vocals": "disabled",
            },
            "candidate_selection": {
                "piano": {"selected": selected_piano_name, "candidates": piano_reports},
                "other": {"selected": selected_other_name, "candidates": other_reports},
            },
            "write_safety": write_info,
            "initial_plan": initial_plan,
            "rebuilt_pack_audit": rebuilt_audit,
            "archive_name": archive.name,
            "archive_size_bytes": archive.stat().st_size,
            "stems": sorted(path.name for path in final_dir.iterdir() if path.is_file()),
            "uploaded": uploaded,
            "result_url": payload.get("result_public_url"),
            "limitations": [
                "Candidate ranking is reference-free and heuristic; listening tests remain required.",
                "Residual Other improves mixture consistency but can contain errors left by the preserved stems.",
                "No specialist third-party piano/Other model is ranked in this version beyond the available baseline and Demucs candidates.",
            ],
        }
