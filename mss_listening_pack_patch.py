from pathlib import Path

path = Path('/app/mss_candidate_lab.py')
text = path.read_text(encoding='utf-8')

if 'import shutil\n' not in text:
    anchor = 'import subprocess\n'
    if anchor not in text:
        raise RuntimeError('Could not locate subprocess import')
    text = text.replace(anchor, anchor + 'import shutil\n', 1)

# Preserve the selected target stem outside _run_candidate's temporary directory.
if 'preserved_target = None' not in text:
    marker = '        target_metrics = _candidate_audio_metrics(target_matches[0], source) if succeeded and target_matches else None\n'
    if marker not in text:
        raise RuntimeError('Could not locate candidate target metrics')
    addition = marker + '''        preserved_target = None
        preserve_dir = str(payload.get("preserve_dir") or "").strip()
        if succeeded and target_matches and preserve_dir:
            preserve_root = Path(preserve_dir)
            preserve_root.mkdir(parents=True, exist_ok=True)
            preserved_target = preserve_root / f"{model_id}-{model['target_stem']}{target_matches[0].suffix}"
            shutil.copy2(target_matches[0], preserved_target)
'''
    text = text.replace(marker, addition, 1)

if '"preserved_target": str(preserved_target)' not in text:
    marker = '            "target_metrics": target_metrics,\n'
    if marker not in text:
        raise RuntimeError('Could not locate candidate return metrics')
    text = text.replace(
        marker,
        marker + '            "preserved_target": str(preserved_target) if preserved_target else None,\n',
        1,
    )

# Allow the current baseline runner to preserve piano and Other comparison stems.
old_sig = 'def _run_bs_baseline(audio_url: str, timeout_seconds: int = 1800) -> dict:'
new_sig = 'def _run_bs_baseline(audio_url: str, timeout_seconds: int = 1800, preserve_dir: str | None = None) -> dict:'
if old_sig in text:
    text = text.replace(old_sig, new_sig, 1)
elif new_sig not in text:
    raise RuntimeError('Could not extend baseline signature')

if '            "preserved": preserved,\n' not in text:
    marker = '        return {\n            "ok": completed.returncode == 0 and bool(piano_matches) and bool(other_matches),\n'
    if marker not in text:
        raise RuntimeError('Could not locate baseline return block')
    preservation = '''        preserved = {}
        if preserve_dir:
            preserve_root = Path(preserve_dir)
            preserve_root.mkdir(parents=True, exist_ok=True)
            if piano_matches:
                dest = preserve_root / f"current-bs-roformer-sw-piano{piano_matches[0].suffix}"
                shutil.copy2(piano_matches[0], dest)
                preserved["piano"] = str(dest)
            if other_matches:
                dest = preserve_root / f"current-bs-roformer-sw-other{other_matches[0].suffix}"
                shutil.copy2(other_matches[0], dest)
                preserved["other"] = str(dest)
'''
    text = text.replace(marker, preservation + marker, 1)
    return_code_marker = '            "return_code": completed.returncode,\n'
    if return_code_marker not in text:
        raise RuntimeError('Could not expose preserved baseline files')
    text = text.replace(return_code_marker, return_code_marker + '            "preserved": preserved,\n', 1)

builder_marker = '\n\ndef build_mss_candidate_lab(payload: dict, progress=None) -> dict:\n'
helper = '''

def _run_listening_pack(payload: dict, progress=None) -> dict:
    audio_url = str(payload.get("audio_url") or payload.get("source_url") or "").strip()
    if not audio_url:
        return {"ok": False, "mode": "mss_candidate_lab", "action": "listening_pack", "error": "audio_url is required"}
    put_url = str(payload.get("result_put_url") or "").strip()
    public_url = payload.get("result_public_url")
    if not put_url:
        return {
            "ok": False,
            "mode": "mss_candidate_lab",
            "action": "listening_pack",
            "error": "result_put_url is required so the ZIP survives serverless cleanup",
            "required_models": ["scnet-xl-ihf-other", "scnet-masked-xl-ihf-other", "mvsep-mega53-piano-keys"],
        }
    timeout_seconds = int(payload.get("timeout_seconds") or 1800)
    model_ids = payload.get("model_ids") or [
        "scnet-xl-ihf-other",
        "scnet-masked-xl-ihf-other",
        "mvsep-mega53-piano-keys",
    ]
    with tempfile.TemporaryDirectory(prefix="litelabs_listening_pack_") as temp:
        root = Path(temp)
        preserve_root = root / "finalists"
        preserve_root.mkdir(parents=True, exist_ok=True)
        if progress:
            progress("Creating current LiteLABS comparison stems", 5)
        baseline = _run_bs_baseline(audio_url, timeout_seconds=timeout_seconds, preserve_dir=str(preserve_root))
        if not baseline.get("ok"):
            return {"ok": False, "mode": "mss_candidate_lab", "action": "listening_pack", "failed_stage": "baseline", "result": baseline}
        runs = []
        total = max(1, len(model_ids))
        for index, model_id in enumerate(model_ids, start=1):
            if progress:
                progress(f"Rendering listening finalist {index}/{total}", int(15 + (index - 1) * 65 / total))
            result = _run_candidate({
                "action": "run",
                "model_id": model_id,
                "audio_url": audio_url,
                "timeout_seconds": timeout_seconds,
                "preserve_dir": str(preserve_root),
            }, progress=None)
            runs.append({"model_id": model_id, "ok": bool(result.get("ok")), "error": result.get("error"), "metrics": result.get("target_metrics")})
            if not result.get("ok") or not result.get("preserved_target"):
                return {"ok": False, "mode": "mss_candidate_lab", "action": "listening_pack", "failed_stage": model_id, "runs": runs}
        readme = preserve_root / "README.txt"
        readme.write_text(
            "LiteLABS finalist listening pack\\n\\n"
            "Compare current-bs-roformer-sw-piano with mvsep-mega53-piano-keys.\\n"
            "Compare scnet-xl-ihf-other with scnet-masked-xl-ihf-other.\\n"
            "Do not judge by loudness alone; listen for ownership, leakage, musical completeness and artifacts.\\n",
            encoding="utf-8",
        )
        archive = root / "litelabs-finalist-listening-pack.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
            for item in sorted(preserve_root.iterdir()):
                if item.is_file():
                    bundle.write(item, arcname=item.name)
        if progress:
            progress("Uploading finalist listening pack", 90)
        _upload_put(put_url, archive)
        if progress:
            progress("Finalist listening pack ready", 100)
        return {
            "ok": True,
            "mode": "mss_candidate_lab",
            "action": "listening_pack",
            "schema_version": 1,
            "uploaded": True,
            "result_url": public_url,
            "archive_name": archive.name,
            "archive_size_bytes": archive.stat().st_size,
            "files": sorted(item.name for item in preserve_root.iterdir() if item.is_file()),
            "runs": runs,
        }
'''
if 'def _run_listening_pack(' not in text:
    if builder_marker not in text:
        raise RuntimeError('Could not locate lab builder')
    text = text.replace(builder_marker, helper + builder_marker, 1)

if 'if action == "listening_pack"' not in text:
    unsupported_marker = '    return {"ok": False, "mode": "mss_candidate_lab", "error": f"Unsupported action: {action}"}\n'
    if unsupported_marker not in text:
        raise RuntimeError('Could not locate unsupported-action return')
    route = '    if action == "listening_pack":\n        return _run_listening_pack(payload, progress=progress)\n'
    text = text.replace(unsupported_marker, route + unsupported_marker, 1)

path.write_text(text, encoding='utf-8')
print('MSS finalist listening pack patch applied')
