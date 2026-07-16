from pathlib import Path

path = Path('/app/mss_candidate_lab.py')
text = path.read_text(encoding='utf-8')

if 'import zipfile\n' not in text:
    text = text.replace('import time\n', 'import time\nimport zipfile\n', 1)

upload_marker = '\n\ndef _load_registry() -> list[dict]:\n'
upload_helper = '''\n\ndef _upload_put(url: str, source: Path) -> None:
    with source.open("rb") as handle:
        response = requests.put(
            url,
            data=handle,
            headers={"Content-Type": "application/zip"},
            timeout=(30, 900),
        )
    response.raise_for_status()
'''
if 'def _upload_put(' not in text:
    if upload_marker not in text:
        raise RuntimeError('Could not locate registry marker for upload helper')
    text = text.replace(upload_marker, upload_helper + upload_marker, 1)

auto_old = '''    if not model["research_ready"]:
        return {"ok": False, "mode": "mss_candidate_lab", "action": "run", "error": "Model is not research-ready", "model": model}

    audio_url = str(payload.get("audio_url") or payload.get("source_url") or "").strip()
'''
auto_new = '''    auto_installed = False
    if not model["research_ready"]:
        missing_files_only = bool(model.get("validation_errors")) and all(
            str(error).startswith("missing config:") or str(error).startswith("missing checkpoint:")
            for error in model.get("validation_errors", [])
        )
        if missing_files_only:
            if progress:
                progress(f"Installing missing candidate files for {model_id}", 3)
            install_result = _install_candidate({"model_id": model_id}, progress=None)
            if not install_result.get("ok"):
                return {
                    "ok": False,
                    "mode": "mss_candidate_lab",
                    "action": "run",
                    "error": "Automatic candidate installation failed",
                    "install_result": install_result,
                }
            auto_installed = True
            inventory = _inventory()
            models = {item["id"]: item for item in inventory["registered_models"]}
            model = models.get(model_id)

        if not model or not model["research_ready"]:
            return {"ok": False, "mode": "mss_candidate_lab", "action": "run", "error": "Model is not research-ready", "model": model}

    audio_url = str(payload.get("audio_url") or payload.get("source_url") or "").strip()
'''
if auto_old in text:
    text = text.replace(auto_old, auto_new, 1)
elif 'auto_installed = False' not in text:
    raise RuntimeError('Could not locate MSS research-ready guard')

old = '''        files = sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file())
        return {
            "ok": completed.returncode == 0,
'''
new = '''        output_paths = sorted(path for path in output_dir.rglob("*") if path.is_file())
        files = [str(path.relative_to(output_dir)) for path in output_paths]
        succeeded = completed.returncode == 0 and bool(output_paths)
        archive = None
        uploaded = False
        put_url = str(payload.get("result_put_url") or "").strip()
        if succeeded:
            archive = root / f"{model_id}-candidate-output.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
                for output_path in output_paths:
                    bundle.write(output_path, arcname=str(output_path.relative_to(output_dir)))
            if put_url:
                if progress:
                    progress(f"Uploading candidate {model_id}", 92)
                _upload_put(put_url, archive)
                uploaded = True
        return {
            "ok": succeeded,
'''
if old in text:
    text = text.replace(old, new, 1)
elif 'output_paths = sorted(path for path in output_dir.rglob' not in text:
    raise RuntimeError('Could not locate candidate output block')

old_tail = '''            "output_files": files,
            "log_tail": "\\n".join((completed.stdout or "").splitlines()[-80:]),
            "promotion_status": "candidate_only",
'''
new_tail = '''            "output_files": files,
            "archive_name": archive.name if archive else None,
            "archive_size_bytes": archive.stat().st_size if archive else 0,
            "uploaded": uploaded,
            "result_url": payload.get("result_public_url"),
            "auto_installed_on_worker": auto_installed,
            "log_tail": "\\n".join((completed.stdout or "").splitlines()[-80:]),
            "promotion_status": "candidate_only",
'''
if old_tail in text:
    text = text.replace(old_tail, new_tail, 1)
elif '"auto_installed_on_worker": auto_installed' not in text:
    raise RuntimeError('Could not locate candidate return block')

path.write_text(text, encoding='utf-8')
print('MSS candidate output persistence and fresh-worker auto-install patch applied')
