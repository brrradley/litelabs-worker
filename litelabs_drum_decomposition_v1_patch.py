from pathlib import Path

path = Path('/app/handler.py')
text = path.read_text(encoding='utf-8')

if '"drum_decomposition_v1"' not in text:
    text = text.replace(
        '"audio_separator_discovery"]',
        '"audio_separator_discovery", "drum_decomposition_v1"]',
        1,
    )

anchor = '''        if mode == "audio_separator_discovery":\n            return build_audio_separator_discovery(payload)\n'''
route = anchor + '''        if mode == "drum_decomposition_v1":\n            from drum_decomposition_v1 import build_drum_decomposition_v1\n            return build_drum_decomposition_v1(payload, progress=progress)\n'''
if 'if mode == "drum_decomposition_v1"' not in text:
    if anchor not in text:
        raise RuntimeError('Could not locate audio_separator_discovery route')
    text = text.replace(anchor, route, 1)

path.write_text(text, encoding='utf-8')

import sys
sys.path.insert(0, '/app')
import drum_decomposition_v1
assert hasattr(drum_decomposition_v1, 'build_drum_decomposition_v1')
assert drum_decomposition_v1.CHILDREN == ('kick', 'snare', 'toms', 'hh', 'cymbals')
print('Drum decomposition v1 route and module self-test applied')
