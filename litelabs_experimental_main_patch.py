from pathlib import Path

path = Path('/app/handler.py')
text = path.read_text(encoding='utf-8')

text = text.replace('"default_mode": "routed_extraction_v1"', '"default_mode": "experimental_children_v1"', 1)
text = text.replace('mode = str(payload.get("mode") or "routed_extraction_v1").strip()', 'mode = str(payload.get("mode") or "experimental_children_v1").strip()', 1)

anchor = '    if mode == "routed_extraction_v1":\n'
route = '''    if mode == "experimental_children_v1":\n        try:\n            from experimental_children_v1 import build_experimental_children_v1\n            result = build_experimental_children_v1(payload, progress=progress)\n            if result.get("ok"):\n                result["production_test"] = True\n            return result\n        except Exception as exc:\n            post_progress(progress_url, progress_token, progress_job_id, f"Worker error: {exc}", 100)\n            return {"ok": False, "mode": mode, "error": str(exc), "error_type": exc.__class__.__name__}\n\n'''

if 'if mode == "experimental_children_v1"' not in text:
    if anchor not in text:
        raise RuntimeError('Could not locate routed mode anchor in production handler')
    text = text.replace(anchor, route + anchor, 1)

path.write_text(text, encoding='utf-8')
print('LiteLABS main experimental children route applied')
