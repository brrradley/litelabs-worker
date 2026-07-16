from pathlib import Path

path = Path('/app/mss_candidate_lab.py')
text = path.read_text(encoding='utf-8')

if 'import shutil\n' not in text:
    text = text.replace('import subprocess\n', 'import subprocess\nimport shutil\n', 1)

old_target = '''        target_matches = [path for path in output_paths if model["target_stem"] in path.name.lower()]\n        target_metrics = _candidate_audio_metrics(target_matches[0], source) if succeeded and target_matches else None\n        put_url = str(payload.get("result_put_url") or "").strip()\n'''
new_target = '''        target_matches = [path for path in output_paths if model["target_stem"] in path.name.lower()]\n        target_metrics = _candidate_audio_metrics(target_matches[0], source) if succeeded and target_matches else None\n        preserved_target = None\n        preserve_dir = str(payload.get("preserve_dir") or "").strip()\n        if succeeded and target_matches and preserve_dir:\n            preserve_root = Path(preserve_dir)\n            preserve_root.mkdir(parents=True, exist_ok=True)\n            preserved_target = preserve_root / f"{model_id}-{model['target_stem']}{target_matches[0].suffix}"\n            shutil.copy2(target_matches[0], preserved_target)\n        put_url = str(payload.get("result_put_url") or "").strip()\n'''
if old_target in text:
    text = text.replace(old_target, new_target, 1)
elif 'preserved_target = None' not in text:
    raise RuntimeError('Could not add candidate preservation')

old_return = '''            "target_metrics": target_metrics,\n            "archive_name": archive.name if archive else None,\n'''
new_return = '''            "target_metrics": target_metrics,\n            "preserved_target": str(preserved_target) if preserved_target else None,\n            "archive_name": archive.name if archive else None,\n'''
if old_return in text:
    text = text.replace(old_return, new_return, 1)
elif '"preserved_target":' not in text:
    raise RuntimeError('Could not expose preserved target')

old_sig = 'def _run_bs_baseline(audio_url: str, timeout_seconds: int = 1800) -> dict:'
new_sig = 'def _run_bs_baseline(audio_url: str, timeout_seconds: int = 1800, preserve_dir: str | None = None) -> dict:'
if old_sig in text:
    text = text.replace(old_sig, new_sig, 1)
elif new_sig not in text:
    raise RuntimeError('Could not extend baseline signature')

old_base_return = '''        return {\n            "ok": completed.returncode == 0 and bool(piano_matches) and bool(other_matches),\n            "return_code": completed.returncode,\n'''
new_base_return = '''        preserved = {}\n        if preserve_dir:\n            preserve_root = Path(preserve_dir)\n            preserve_root.mkdir(parents=True, exist_ok=True)\n            if piano_matches:\n                dest = preserve_root / f"current-bs-roformer-sw-piano{piano_matches[0].suffix}"\n                shutil.copy2(piano_matches[0], dest)\n                preserved["piano"] = str(dest)\n            if other_matches:\n                dest = preserve_root / f"current-bs-roformer-sw-other{other_matches[0].suffix}"\n                shutil.copy2(other_matches[0], dest)\n                preserved["other"] = str(dest)\n        return {\n            "ok": completed.returncode == 0 and bool(piano_matches) and bool(other_matches),\n            "return_code": completed.returncode,\n            "preserved": preserved,\n'''
if old_base_return in text:
    text = text.replace(old_base_return, new_base_return, 1)
elif '"preserved": preserved' not in text:
    raise RuntimeError('Could not add baseline preservation')

marker = '\n\ndef build_mss_candidate_lab(payload: dict, progress=None) -> dict:\n'
helper = '''\n\ndef _run_listening_pack(payload: dict, progress=None) -> dict:
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
    if marker not in text:
        raise RuntimeError('Could not locate lab builder')
    text = text.replace(marker, helper + marker, 1)

route = '''    if action == "campaign":\n        return _run_campaign(payload, progress=progress)\n'''
new_route = route + '''    if action == "listening_pack":\n        return _run_listening_pack(payload, progress=progress)\n'''
if route in text and 'if action == "listening_pack"' not in text:
    text = text.replace(route, new_route, 1)
elif 'if action == "listening_pack"' not in text:
    raise RuntimeError('Could not add listening pack route')

path.write_text(text, encoding='utf-8')
print('MSS finalist listening pack patch applied')
