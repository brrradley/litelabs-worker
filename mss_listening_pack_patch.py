from pathlib import Path
import re

path = Path('/app/mss_candidate_lab.py')
text = path.read_text(encoding='utf-8')

if 'import shutil\n' not in text:
    text = text.replace('import subprocess\n', 'import subprocess\nimport shutil\n', 1)

if 'preserved_target = None' not in text:
    match = re.search(r'^(?P<indent>\s*)target_metrics\s*=.*$', text, flags=re.MULTILINE)
    if not match:
        raise RuntimeError('Could not locate target_metrics assignment in generated candidate lab')
    indent = match.group('indent')
    insertion = (
        match.group(0) + '\n'
        + indent + 'preserved_target = None\n'
        + indent + 'preserve_dir = str(payload.get("preserve_dir") or "").strip()\n'
        + indent + 'if succeeded and target_matches and preserve_dir:\n'
        + indent + '    preserve_root = Path(preserve_dir)\n'
        + indent + '    preserve_root.mkdir(parents=True, exist_ok=True)\n'
        + indent + '    preserved_target = preserve_root / f"{model_id}-{model[\'target_stem\']}{target_matches[0].suffix}"\n'
        + indent + '    shutil.copy2(target_matches[0], preserved_target)'
    )
    text = text[:match.start()] + insertion + text[match.end():]

if '"preserved_target": str(preserved_target)' not in text:
    marker = '            "target_metrics": target_metrics,\n'
    if marker not in text:
        raise RuntimeError('Could not locate target_metrics return field')
    text = text.replace(marker, marker + '            "preserved_target": str(preserved_target) if preserved_target else None,\n', 1)

old_sig = 'def _run_bs_baseline(audio_url: str, timeout_seconds: int = 1800) -> dict:'
new_sig = 'def _run_bs_baseline(audio_url: str, timeout_seconds: int = 1800, preserve_dir: str | None = None) -> dict:'
if old_sig in text:
    text = text.replace(old_sig, new_sig, 1)
elif new_sig not in text:
    raise RuntimeError('Could not extend baseline signature')

if '"preserved": preserved,' not in text:
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
    text = text.replace('            "return_code": completed.returncode,\n', '            "return_code": completed.returncode,\n            "preserved": preserved,\n', 1)

builder_marker = '\n\ndef build_mss_candidate_lab(payload: dict, progress=None) -> dict:\n'
helper = '''

def _run_listening_pack(payload: dict, progress=None) -> dict:
    audio_url = str(payload.get("audio_url") or payload.get("source_url") or "").strip()
    if not audio_url:
        return {"ok": False, "mode": "mss_candidate_lab", "action": "listening_pack", "error": "audio_url is required"}
    put_url = str(payload.get("result_put_url") or "").strip()
    if not put_url:
        return {"ok": False, "mode": "mss_candidate_lab", "action": "listening_pack", "error": "result_put_url is required so the ZIP survives serverless cleanup"}
    timeout_seconds = int(payload.get("timeout_seconds") or 1800)
    model_ids = payload.get("model_ids") or ["scnet-xl-ihf-other", "scnet-masked-xl-ihf-other", "mvsep-mega53-piano-keys"]
    with tempfile.TemporaryDirectory(prefix="litelabs_listening_pack_") as temp:
        root = Path(temp)
        preserve_root = root / "finalists"
        preserve_root.mkdir(parents=True, exist_ok=True)
        baseline = _run_bs_baseline(audio_url, timeout_seconds=timeout_seconds, preserve_dir=str(preserve_root))
        if not baseline.get("ok"):
            return {"ok": False, "mode": "mss_candidate_lab", "action": "listening_pack", "failed_stage": "baseline", "result": baseline}
        runs = []
        for model_id in model_ids:
            result = _run_candidate({"action": "run", "model_id": model_id, "audio_url": audio_url, "timeout_seconds": timeout_seconds, "preserve_dir": str(preserve_root)}, progress=None)
            runs.append({"model_id": model_id, "ok": bool(result.get("ok")), "error": result.get("error"), "metrics": result.get("target_metrics")})
            if not result.get("ok") or not result.get("preserved_target"):
                return {"ok": False, "mode": "mss_candidate_lab", "action": "listening_pack", "failed_stage": model_id, "runs": runs}
        (preserve_root / "README.txt").write_text("LiteLABS finalist listening pack\\nCompare BS piano with Mega53 keys; compare both SCNet Other stems.\\n", encoding="utf-8")
        archive = root / "litelabs-finalist-listening-pack.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
            for item in sorted(preserve_root.iterdir()):
                if item.is_file():
                    bundle.write(item, arcname=item.name)
        _upload_put(put_url, archive)
        return {"ok": True, "mode": "mss_candidate_lab", "action": "listening_pack", "schema_version": 1, "uploaded": True, "result_url": payload.get("result_public_url"), "archive_name": archive.name, "archive_size_bytes": archive.stat().st_size, "files": sorted(item.name for item in preserve_root.iterdir() if item.is_file()), "runs": runs}
'''
if 'def _run_listening_pack(' not in text:
    if builder_marker not in text:
        raise RuntimeError('Could not locate lab builder')
    text = text.replace(builder_marker, helper + builder_marker, 1)

unsupported = '    return {"ok": False, "mode": "mss_candidate_lab", "error": f"Unsupported action: {action}"}\n'
if unsupported not in text:
    raise RuntimeError('Could not locate unsupported action return')
routes = ''
if 'if action == "listening_pack"' not in text:
    routes += '    if action == "listening_pack":\n        return _run_listening_pack(payload, progress=progress)\n'
if 'if action == "sw_ra"' not in text:
    routes += '    if action == "sw_ra":\n        from sw_residual_allocator import build_sw_residual_allocator\n        return build_sw_residual_allocator(payload, progress=progress)\n'
if routes:
    text = text.replace(unsupported, routes + unsupported, 1)

path.write_text(text, encoding='utf-8')
print('MSS listening pack and SW RA routes applied')
