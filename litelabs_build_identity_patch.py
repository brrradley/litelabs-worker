from pathlib import Path

path = Path('/app/handler.py')
text = path.read_text(encoding='utf-8')

marker = 'print("LiteLABS handler ready", flush=True)\nrunpod.serverless.start({"handler": handler})\n'
if marker not in text:
    raise RuntimeError('Could not locate RunPod handler startup marker')

replacement = '''# Attach the exact image build revision to every dictionary response. This wraps\n# all existing branches, including health/capabilities requests injected by older\n# production layers, without changing their behaviour.\n_BUILD_SHA = os.getenv("LITELABS_BUILD_SHA", "unknown").strip() or "unknown"\n_original_handler = handler\n\ndef handler(job: dict) -> dict:\n    result = _original_handler(job)\n    if isinstance(result, dict):\n        result.setdefault("build_sha", _BUILD_SHA)\n    return result\n\nprint(f"LiteLABS handler ready (build {_BUILD_SHA})", flush=True)\nrunpod.serverless.start({"handler": handler})\n'''

text = text.replace(marker, replacement, 1)
path.write_text(text, encoding='utf-8')
print('LiteLABS build identity response wrapper applied')
