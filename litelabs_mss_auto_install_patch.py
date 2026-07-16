from pathlib import Path

path = Path('/app/mss_candidate_lab.py')
text = path.read_text(encoding='utf-8')
old = '''    if not model["research_ready"]:
        return {"ok": False, "mode": "mss_candidate_lab", "action": "run", "error": "Model is not research-ready", "model": model}

    audio_url = str(payload.get("audio_url") or payload.get("source_url") or "").strip()
'''
new = '''    auto_installed = False
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
if old not in text:
    if 'auto_installed = False' in text:
        print('MSS auto-install patch already applied')
    else:
        raise RuntimeError('Could not locate MSS research-ready guard')
else:
    text = text.replace(old, new, 1)

return_old = '''            "production_eligible": model["production_ready"],
            "next_action": "Score against the current LiteLABS baseline, mixture consistency, contamination metrics and listening tests.",
'''
return_new = '''            "production_eligible": model["production_ready"],
            "auto_installed_on_worker": auto_installed,
            "next_action": "Score against the current LiteLABS baseline, mixture consistency, contamination metrics and listening tests.",
'''
if return_old in text:
    text = text.replace(return_old, return_new, 1)
elif '"auto_installed_on_worker": auto_installed' not in text:
    raise RuntimeError('Could not add auto-install result marker')

path.write_text(text, encoding='utf-8')
print('MSS fresh-worker auto-install patch applied')
