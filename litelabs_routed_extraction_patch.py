from pathlib import Path

path = Path('/app/handler.py')
text = path.read_text(encoding='utf-8')

if '"routed_extraction_v1"' not in text:
    text = text.replace(
        '"wind_brass_decomposition_v2"]',
        '"wind_brass_decomposition_v2", "routed_extraction_v1"]',
        1,
    )

anchor = '''        if mode == "wind_brass_decomposition_v2":\n            from wind_brass_decomposition_v2 import build_wind_brass_decomposition_v2\n            return build_wind_brass_decomposition_v2(payload, progress=progress)\n'''
route = anchor + '''        if mode == "routed_extraction_v1":\n            from routed_extraction_v1 import build_routed_extraction_v1\n            return build_routed_extraction_v1(payload, progress=progress)\n'''
if 'if mode == "routed_extraction_v1"' not in text:
    if anchor not in text:
        raise RuntimeError('Could not locate wind/brass v2 route')
    text = text.replace(anchor, route, 1)

path.write_text(text, encoding='utf-8')
print('Routed extraction v1 route applied')
