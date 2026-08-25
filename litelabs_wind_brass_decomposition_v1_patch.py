from pathlib import Path

path = Path('/app/handler.py')
text = path.read_text(encoding='utf-8')

if '"wind_brass_decomposition_v1"' not in text:
    text = text.replace(
        '"audio_separator_discovery", "drum_decomposition_v1"]',
        '"audio_separator_discovery", "drum_decomposition_v1", "wind_brass_decomposition_v1"]',
        1,
    )

anchor = '''        if mode == "drum_decomposition_v1":\n            from drum_decomposition_v1 import build_drum_decomposition_v1\n            return build_drum_decomposition_v1(payload, progress=progress)\n'''
route = anchor + '''        if mode == "wind_brass_decomposition_v1":\n            from wind_brass_decomposition_v1 import build_wind_brass_decomposition_v1\n            return build_wind_brass_decomposition_v1(payload, progress=progress)\n'''
if 'if mode == "wind_brass_decomposition_v1"' not in text:
    if anchor not in text:
        raise RuntimeError('Could not locate drum decomposition route')
    text = text.replace(anchor, route, 1)

path.write_text(text, encoding='utf-8')

import sys
sys.path.insert(0, '/app')
import wind_brass_decomposition_v1
assert hasattr(wind_brass_decomposition_v1, 'build_wind_brass_decomposition_v1')
assert 'saxophone' in wind_brass_decomposition_v1.SPECIFIC
assert 'trumpet' in wind_brass_decomposition_v1.SPECIFIC
print('Wind/brass decomposition v1 route and module self-test applied')
