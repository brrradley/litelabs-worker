from pathlib import Path

path = Path('/app/mss_candidate_lab.py')
text = path.read_text(encoding='utf-8')

if 'import zipfile\n' not in text:
    text = text.replace('import time\n', 'import time\nimport zipfile\n', 1)

upload_marker = '\n\ndef _load_registry() -> list[dict]:\n'
upload_helper = '''\n\ndef _upload_put(url: str, source: Path) -> None:\n    with source.open("rb") as handle:\n        response = requests.put(\n            url,\n            data=handle,\n            headers={"Content-Type": "application/zip"},\n            timeout=(30, 900),\n        )\n    response.raise_for_status()\n'''
if 'def _upload_put(' not in text:
    if upload_marker not in text:
        raise RuntimeError('Could not locate registry marker for upload helper')
    text = text.replace(upload_marker, upload_helper + upload_marker, 1)

old = '''        files = sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file())\n        return {\n            "ok": completed.returncode == 0,\n'''
new = '''        output_paths = sorted(path for path in output_dir.rglob("*") if path.is_file())\n        files = [str(path.relative_to(output_dir)) for path in output_paths]\n        succeeded = completed.returncode == 0 and bool(output_paths)\n        archive = None\n        uploaded = False\n        put_url = str(payload.get("result_put_url") or "").strip()\n        if succeeded:\n            archive = root / f"{model_id}-candidate-output.zip"\n            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:\n                for output_path in output_paths:\n                    bundle.write(output_path, arcname=str(output_path.relative_to(output_dir)))\n            if put_url:\n                if progress:\n                    progress(f"Uploading candidate {model_id}", 92)\n                _upload_put(put_url, archive)\n                uploaded = True\n        return {\n            "ok": succeeded,\n'''
if old in text:
    text = text.replace(old, new, 1)
elif 'output_paths = sorted(path for path in output_dir.rglob' not in text:
    raise RuntimeError('Could not locate candidate output block')

old_tail = '''            "output_files": files,\n            "log_tail": "\\n".join((completed.stdout or "").splitlines()[-80:]),\n            "promotion_status": "candidate_only",\n'''
new_tail = '''            "output_files": files,\n            "archive_name": archive.name if archive else None,\n            "archive_size_bytes": archive.stat().st_size if archive else 0,\n            "uploaded": uploaded,\n            "result_url": payload.get("result_public_url"),\n            "log_tail": "\\n".join((completed.stdout or "").splitlines()[-80:]),\n            "promotion_status": "candidate_only",\n'''
if old_tail in text:
    text = text.replace(old_tail, new_tail, 1)
elif '"archive_name": archive.name if archive else None' not in text:
    raise RuntimeError('Could not locate candidate return block')

path.write_text(text, encoding='utf-8')
print('MSS candidate output persistence patch applied')
