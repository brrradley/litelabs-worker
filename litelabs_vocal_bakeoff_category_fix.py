from pathlib import Path

path = Path('/app/experimental_vocal_cascade_bakeoff.py')
text = path.read_text(encoding='utf-8')

old_missing = "f\"available vocal-like names: {[name for name in tracks if sgt.track_category(name)[0] == 'vocals']}\""
new_missing = "f\"available vocal-like names: {[name for name in tracks if sgt.track_category(name) == 'vocals']}\""
if old_missing not in text:
    raise RuntimeError('Could not locate vocal-like names category call')
text = text.replace(old_missing, new_missing, 1)

old_loop = '''    for name, audio in tracks.items():\n        category, _warning = sgt.track_category(name)\n        if category == \"vocals\":\n            vocal_members.append((name, audio))\n'''
new_loop = '''    for name, audio in tracks.items():\n        category = sgt.track_category(name)\n        if category == \"vocals\":\n            vocal_members.append((name, audio))\n'''
if old_loop not in text:
    raise RuntimeError('Could not locate vocal reference category loop')
text = text.replace(old_loop, new_loop, 1)

path.write_text(text, encoding='utf-8')
print('LiteLABS vocal bakeoff category API fix applied')
