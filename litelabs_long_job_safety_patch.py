from pathlib import Path


def patch_experimental_upload() -> None:
    path = Path('/app/experimental_children_v1.py')
    text = path.read_text(encoding='utf-8')

    old = '''        uploaded = False\n        put_url = str(payload.get("result_put_url") or "").strip()\n        if put_url:\n            emit("Uploading Stem Pack", 96)\n            import requests\n            with archive.open("rb") as handle:\n                response = requests.put(put_url, data=handle, headers={"Content-Type": "application/zip"}, timeout=(30, 1800))\n            response.raise_for_status()\n            uploaded = True\n\n        timings["total"] = round(time.monotonic() - started, 3)\n'''

    new = '''        uploaded = False\n        archive_size_bytes = archive.stat().st_size\n        print(f"LiteLABS result archive ready: {archive.name} ({archive_size_bytes} bytes)", flush=True)\n        put_url = str(payload.get("result_put_url") or "").strip()\n        if put_url:\n            emit("Uploading Stem Pack", 96)\n            import requests\n            with archive.open("rb") as handle:\n                response = requests.put(put_url, data=handle, headers={"Content-Type": "application/zip"}, timeout=(30, 1800))\n            if response.status_code == 413:\n                timings["total"] = round(time.monotonic() - started, 3)\n                emit("Stem pack exceeds storage upload limit", 100)\n                return _json_safe({\n                    "ok": False,\n                    "mode": MODE,\n                    "failed_stage": "result_upload",\n                    "error_code": "result_too_large",\n                    "http_status": 413,\n                    "archive_name": archive.name,\n                    "archive_size_bytes": archive_size_bytes,\n                    "uploaded": False,\n                    "result_url": None,\n                    "timings_seconds": timings,\n                })\n            response.raise_for_status()\n            uploaded = True\n\n        timings["total"] = round(time.monotonic() - started, 3)\n'''

    if '"error_code": "result_too_large"' not in text:
        if old not in text:
            raise RuntimeError('Could not locate experimental upload block')
        text = text.replace(old, new, 1)

    text = text.replace('"archive_size_bytes": archive.stat().st_size,', '"archive_size_bytes": archive_size_bytes,')
    path.write_text(text, encoding='utf-8')


def patch_parent_upload() -> None:
    path = Path('/app/preset_pack.py')
    text = path.read_text(encoding='utf-8')

    old = '''        uploaded = False\n        put_url = str(payload.get("result_put_url") or "").strip()\n        if put_url:\n            emit("Uploading Stem Pack", 95)\n            import requests\n            with archive.open("rb") as handle:\n                response = requests.put(\n                    put_url, data=handle, headers={"Content-Type": "application/zip"}, timeout=(30, 1800)\n                )\n            response.raise_for_status()\n            uploaded = True\n\n        timings["total"] = round(time.monotonic() - started, 3)\n'''

    new = '''        uploaded = False\n        archive_size_bytes = archive.stat().st_size\n        print(f"LiteLABS result archive ready: {archive.name} ({archive_size_bytes} bytes)", flush=True)\n        put_url = str(payload.get("result_put_url") or "").strip()\n        if put_url:\n            emit("Uploading Stem Pack", 95)\n            import requests\n            with archive.open("rb") as handle:\n                response = requests.put(\n                    put_url, data=handle, headers={"Content-Type": "application/zip"}, timeout=(30, 1800)\n                )\n            if response.status_code == 413:\n                timings["total"] = round(time.monotonic() - started, 3)\n                emit("Stem pack exceeds storage upload limit", 100)\n                return {\n                    "ok": False,\n                    "mode": "preset_pack",\n                    "preset": preset,\n                    "failed_stage": "result_upload",\n                    "error_code": "result_too_large",\n                    "http_status": 413,\n                    "archive_name": archive.name,\n                    "archive_size_bytes": archive_size_bytes,\n                    "uploaded": False,\n                    "result_url": None,\n                    "research_qa": research_qa,\n                    "timings_seconds": timings,\n                }\n            response.raise_for_status()\n            uploaded = True\n\n        timings["total"] = round(time.monotonic() - started, 3)\n'''

    if '"error_code": "result_too_large"' not in text:
        if old not in text:
            raise RuntimeError('Could not locate parent preset upload block')
        text = text.replace(old, new, 1)

    text = text.replace('"archive_size_bytes": archive.stat().st_size,', '"archive_size_bytes": archive_size_bytes,')
    path.write_text(text, encoding='utf-8')


def patch_qa_metadata() -> None:
    targets = [
        (
            Path('/app/preset_pack.py'),
            '            genre=genre,\n            preset=preset,\n',
            '            genre=genre,\n            genre_reason=genre_reason,\n            preset=preset,\n',
        ),
        (
            Path('/app/experimental_children_v1.py'),
            '            genre=qa_genre,\n            preset="experimental",\n',
            '            genre=qa_genre,\n            genre_reason=qa_genre_reason,\n            preset="experimental",\n',
        ),
    ]

    for path, old, new in targets:
        text = path.read_text(encoding='utf-8')
        if 'genre_reason=' not in text:
            if old not in text:
                raise RuntimeError(f'Could not locate genre reason QA call in {path}')
            text = text.replace(old, new, 1)

        score_anchor = '        print(f"LiteLABS silent research QA complete for {len(qa_stems)} stems", flush=True)\n'
        score_line = '        print("LiteLABS QA scores: " + " ".join(f"{name}={float(record.get(\'score\', 0)):.3f}" for name, record in research_qa.get("stems", {}).items()), flush=True)\n'
        if 'LiteLABS QA scores:' not in text:
            if score_anchor not in text:
                raise RuntimeError(f'Could not locate QA log anchor in {path}')
            text = text.replace(score_anchor, score_anchor + score_line, 1)

        path.write_text(text, encoding='utf-8')


patch_experimental_upload()
patch_parent_upload()
patch_qa_metadata()
print('LiteLABS long-job/upload safety and QA metadata patch applied')
