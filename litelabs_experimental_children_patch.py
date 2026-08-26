from pathlib import Path

path = Path('/app/handler.py')
text = path.read_text(encoding='utf-8')

if '"experimental_children_v1"' not in text:
    text = text.replace(
        '"audio_separator_discovery"]',
        '"audio_separator_discovery", "experimental_children_v1"]',
        1,
    )

anchor = '''        if mode == "audio_separator_discovery":\n            return build_audio_separator_discovery(payload)\n'''
route = anchor + '''        if mode == "experimental_children_v1":\n            from experimental_children_v1 import build_experimental_children_v1\n            return build_experimental_children_v1(payload, progress=progress)\n'''
if 'if mode == "experimental_children_v1"' not in text:
    if anchor not in text:
        raise RuntimeError('Could not locate audio_separator_discovery route')
    text = text.replace(anchor, route, 1)

path.write_text(text, encoding='utf-8')
print('Experimental children v1 route applied')
