from pathlib import Path

HELPER = r'''

def _litelabs_upload_archive(archive, put_url: str, payload: dict, archive_size_bytes: int) -> dict:
    """Upload a result archive, using retry-safe chunks when requested by the caller."""
    import math
    import time as _upload_time
    import requests

    mode = str(payload.get("result_upload_mode") or "").strip().lower()
    if mode != "chunked":
        with archive.open("rb") as handle:
            response = requests.put(
                put_url,
                data=handle,
                headers={"Content-Type": "application/zip"},
                timeout=(30, 1800),
            )
        if response.status_code == 413:
            return {"uploaded": False, "status_code": 413}
        response.raise_for_status()
        return {"uploaded": True, "mode": "single"}

    chunk_mb = int(payload.get("result_chunk_size_mb") or 16)
    chunk_mb = max(4, min(32, chunk_mb))
    chunk_bytes = chunk_mb * 1024 * 1024
    total = max(1, int(math.ceil(archive_size_bytes / chunk_bytes)))
    print(
        f"LiteLABS chunked result upload: {total} part(s) at up to {chunk_mb} MiB each",
        flush=True,
    )

    with archive.open("rb") as handle:
        for part in range(total):
            data = handle.read(chunk_bytes)
            if not data:
                raise RuntimeError(f"Result archive ended before chunk {part + 1}/{total}")

            for attempt in range(1, 4):
                try:
                    response = requests.post(
                        put_url,
                        params={"part": part, "total": total, "size": archive_size_bytes},
                        data=data,
                        headers={"Content-Type": "application/octet-stream"},
                        timeout=(30, 600),
                    )
                    if response.status_code == 413:
                        return {"uploaded": False, "status_code": 413, "part": part}
                    response.raise_for_status()

                    try:
                        body = response.json()
                    except ValueError:
                        body = {}
                    if body and not bool(body.get("ok", False)):
                        raise RuntimeError(str(body.get("error") or "Chunk receiver rejected upload"))
                    if part == total - 1 and body and not bool(body.get("complete", False)):
                        raise RuntimeError("Chunk receiver did not confirm final archive assembly")

                    print(
                        f"LiteLABS result upload chunk {part + 1}/{total} complete",
                        flush=True,
                    )
                    break
                except Exception as exc:
                    if attempt >= 3:
                        raise
                    wait = 2 ** (attempt - 1)
                    print(
                        f"LiteLABS result upload chunk {part + 1}/{total} attempt {attempt} failed: {exc}; retrying in {wait}s",
                        flush=True,
                    )
                    _upload_time.sleep(wait)

    return {"uploaded": True, "mode": "chunked", "chunks": total}
'''


def add_helper(text: str, marker: str) -> str:
    if 'def _litelabs_upload_archive(' in text:
        return text
    if marker not in text:
        raise RuntimeError(f'Could not locate helper insertion marker: {marker!r}')
    return text.replace(marker, HELPER + '\n\n' + marker, 1)


def replace_upload_block(path: Path, helper_marker: str, mode: str) -> None:
    text = path.read_text(encoding='utf-8')
    text = add_helper(text, helper_marker)

    start_marker = '        uploaded = False\n        archive_size_bytes = archive.stat().st_size\n'
    end_marker = '        timings["total"] = round(time.monotonic() - started, 3)\n'
    complete_marker = '        emit("Stem Extraction Complete", 100)\n'

    start = text.find(start_marker)
    if start < 0:
        raise RuntimeError(f'Could not locate patched result upload start in {path}')

    complete = text.find(complete_marker, start)
    if complete < 0:
        raise RuntimeError(f'Could not locate completion marker after result upload in {path}')

    # The long-job safety patch also writes timings["total"] inside its 413
    # error branch. We need the final, outer timing assignment immediately
    # before "Stem Extraction Complete", not that nested assignment.
    end = text.rfind(end_marker, start, complete)
    if end < 0:
        raise RuntimeError(f'Could not locate final patched result upload end in {path}')

    if mode == 'experimental':
        replacement = '''        uploaded = False\n        archive_size_bytes = archive.stat().st_size\n        print(f"LiteLABS result archive ready: {archive.name} ({archive_size_bytes} bytes)", flush=True)\n        put_url = str(payload.get("result_put_url") or "").strip()\n        if put_url:\n            emit("Uploading Stem Pack", 96)\n            upload_result = _litelabs_upload_archive(archive, put_url, payload, archive_size_bytes)\n            if int(upload_result.get("status_code") or 0) == 413:\n                timings["total"] = round(time.monotonic() - started, 3)\n                emit("Stem pack exceeds storage upload limit", 100)\n                return _json_safe({\n                    "ok": False,\n                    "mode": MODE,\n                    "failed_stage": "result_upload",\n                    "error_code": "result_too_large",\n                    "http_status": 413,\n                    "archive_name": archive.name,\n                    "archive_size_bytes": archive_size_bytes,\n                    "uploaded": False,\n                    "result_url": None,\n                    "timings_seconds": timings,\n                })\n            uploaded = bool(upload_result.get("uploaded"))\n\n'''
    elif mode == 'parent':
        replacement = '''        uploaded = False\n        archive_size_bytes = archive.stat().st_size\n        print(f"LiteLABS result archive ready: {archive.name} ({archive_size_bytes} bytes)", flush=True)\n        put_url = str(payload.get("result_put_url") or "").strip()\n        if put_url:\n            emit("Uploading Stem Pack", 95)\n            upload_result = _litelabs_upload_archive(archive, put_url, payload, archive_size_bytes)\n            if int(upload_result.get("status_code") or 0) == 413:\n                timings["total"] = round(time.monotonic() - started, 3)\n                emit("Stem pack exceeds storage upload limit", 100)\n                return {\n                    "ok": False,\n                    "mode": "preset_pack",\n                    "preset": preset,\n                    "failed_stage": "result_upload",\n                    "error_code": "result_too_large",\n                    "http_status": 413,\n                    "archive_name": archive.name,\n                    "archive_size_bytes": archive_size_bytes,\n                    "uploaded": False,\n                    "result_url": None,\n                    "research_qa": research_qa,\n                    "timings_seconds": timings,\n                }\n            uploaded = bool(upload_result.get("uploaded"))\n\n'''
    else:
        raise ValueError(mode)

    text = text[:start] + replacement + text[end:]
    path.write_text(text, encoding='utf-8')


replace_upload_block(Path('/app/experimental_children_v1.py'), 'MODE = "experimental_children_v1"', 'experimental')
replace_upload_block(Path('/app/preset_pack.py'), 'PRESETS = {', 'parent')
print('LiteLABS chunked result upload patch applied')
