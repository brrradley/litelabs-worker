from pathlib import Path

path = Path('/app/handler.py')
text = path.read_text(encoding='utf-8')

if '"wind_brass_decomposition_v2"' not in text:
    text = text.replace(
        '"wind_brass_decomposition_v1"]',
        '"wind_brass_decomposition_v1", "wind_brass_decomposition_v2"]',
        1,
    )

anchor = '''        if mode == "wind_brass_decomposition_v1":\n            from wind_brass_decomposition_v1 import build_wind_brass_decomposition_v1\n            return build_wind_brass_decomposition_v1(payload, progress=progress)\n'''
route = anchor + '''        if mode == "wind_brass_decomposition_v2":\n            from wind_brass_decomposition_v2 import build_wind_brass_decomposition_v2\n            return build_wind_brass_decomposition_v2(payload, progress=progress)\n'''
if 'if mode == "wind_brass_decomposition_v2"' not in text:
    if anchor not in text:
        raise RuntimeError('Could not locate wind/brass v1 route')
    text = text.replace(anchor, route, 1)

path.write_text(text, encoding='utf-8')

import sys
sys.path.insert(0, '/app')
import wind_brass_decomposition_v2
assert hasattr(wind_brass_decomposition_v2, 'build_wind_brass_decomposition_v2')
assert wind_brass_decomposition_v2.TARGETS == ('saxophone', 'trumpet')
print('Wind/brass decomposition v2 route and self-test applied')
