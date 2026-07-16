from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests


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
        return {
            "ok": True,
            "mode": "adaptive_recovery_run",
            "schema_version": 1,
            "status": "no_recovery_needed",
            "initial_plan": initial_plan,
            "result_url": payload.get("result_public_url"),
        }

    requested = {str(item.get("stem") or "") for item in mandatory}
    supported = requested.intersection({"piano", "other"})
    unsupported = sorted(requested - supported)

    with tempfile.TemporaryDirectory(prefix="litelabs_adaptive_recovery_") as temp:
        root = Path(temp)
        filename = str(payload.get("filename") or Path(urlparse(source_url).path).name or "track.flac")
        source = root / filename
        if progress:
            progress("Downloading source audio", 8)
        _download(source_url, source)

        if progress:
            progress("Running full six-stem recovery", 15)
        from master_pack import build_master_pack
        result = build_master_pack(
            input_audio=source,
            work_root=root / "work",
            model_dir=Path(payload.get("model_dir") or os.getenv("STEMFORGE_MODEL_DIR", "/models/bs_roformer_sw")),
            output_root=root / "output",
            progress=(lambda message, percent: progress(message, 15 + int(percent * 0.62))) if progress else None,
        )
        archive = Path(result["archive_path"])

        if progress:
            progress("Auditing rebuilt six-stem pack", 82)
        from reference_free_stem_auditor import build_reference_free_stem_auditor
        rebuilt_audit = build_reference_free_stem_auditor({
            "source_url": source.as_uri(),
            "stem_pack_url": archive.as_uri(),
        }, progress=None)

        found = set(rebuilt_audit.get("primary_stems_found") or []) if rebuilt_audit.get("ok") else set()
        recovered = sorted(stem for stem in supported if stem in found)
        still_missing = sorted(stem for stem in requested if stem not in found)

        put_url = str(payload.get("result_put_url") or "").strip()
        uploaded = False
        if put_url:
            if progress:
                progress("Uploading recovered stem pack", 94)
            _upload_put(put_url, archive)
            uploaded = True

        if progress:
            progress("Adaptive recovery complete", 100)

        return {
            "ok": True,
            "mode": "adaptive_recovery_run",
            "schema_version": 1,
            "status": "recovered_pack_ready" if not still_missing else "recovery_incomplete",
            "source_url": source_url,
            "original_stem_pack_url": stem_pack_url,
            "requested_recovery_stems": sorted(requested),
            "supported_recovery_stems": sorted(supported),
            "unsupported_recovery_stems": unsupported,
            "recovered_stems": recovered,
            "still_missing_stems": still_missing,
            "strategy": {
                "baseline_preserved": False,
                "method": "Rebuild the complete six-primary-stem pack with the current LiteLABS extraction pipeline.",
                "candidate_selection": "Current baseline outputs are retained; alternate candidate ranking is not yet automatic in this first execution mode.",
                "dry_main_vocals": "disabled",
            },
            "initial_plan": initial_plan,
            "rebuilt_pack_audit": rebuilt_audit,
            "archive_name": archive.name,
            "archive_size_bytes": archive.stat().st_size,
            "stems": result.get("stems") or [],
            "uploaded": uploaded,
            "result_url": payload.get("result_public_url"),
            "limitations": [
                "This first execution mode recovers the full six-stem baseline but does not yet rank specialist challenger models automatically.",
                "Reference-free audit consistency does not prove studio-stem correctness.",
            ],
        }
