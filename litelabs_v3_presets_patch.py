from pathlib import Path

path = Path('/app/handler.py')
text = path.read_text(encoding='utf-8')

# Advertise the neutral preset IDs in healthcheck output without changing the
# existing default mode used by the current XenForo integration.
old_health = 'return {"ok": True, "status": "ready", "service": "litelabs-worker", "default_mode": "experimental_children_v1"}'
new_health = 'return {"ok": True, "status": "ready", "service": "litelabs-worker", "default_mode": "experimental_children_v1", "presets": ["basic", "core", "experimental"]}'
if old_health not in text:
    raise RuntimeError('Could not locate patched healthcheck response')
text = text.replace(old_health, new_health, 1)

anchor = '    mode = str(payload.get("mode") or "experimental_children_v1").strip()\n'
insert = '''    preset = str(payload.get("preset") or "").strip().lower()\n    if preset:\n        if preset not in {"basic", "core", "experimental"}:\n            return {"ok": False, "error": f"Unknown preset: {preset}", "allowed_presets": ["basic", "core", "experimental"]}\n        if preset in {"basic", "core"}:\n            try:\n                from preset_pack import build_parent_preset\n                return build_parent_preset(payload, progress=progress)\n            except Exception as exc:\n                post_progress(progress_url, progress_token, progress_job_id, f"Worker error: {exc}", 100)\n                return {"ok": False, "mode": "preset_pack", "preset": preset, "error": str(exc), "error_type": exc.__class__.__name__}\n        # EXPERIMENTAL intentionally reuses the current deep v3 pipeline.\n        payload = dict(payload)\n        payload["mode"] = "experimental_children_v1"\n\n'''
if anchor not in text:
    raise RuntimeError('Could not locate mode selection anchor')
text = text.replace(anchor, insert + anchor, 1)

path.write_text(text, encoding='utf-8')
print('LiteLABS BASIC/CORE/EXPERIMENTAL preset dispatcher applied')
