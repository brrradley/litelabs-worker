from pathlib import Path

path = Path('/app/handler.py')
text = path.read_text(encoding='utf-8')

if '"instrument_inventory_v3"' not in text:
    text = text.replace(
        '"instrument_wireframe",',
        '"instrument_wireframe", "instrument_inventory_v3",',
        1,
    )

needle = '''        if mode == "instrument_wireframe":
            from instrument_wireframe import build_instrument_wireframe
            return build_instrument_wireframe(payload, progress=progress)
'''
route = needle + '''        if mode == "instrument_inventory_v3":
            from instrument_inventory_v3 import build_instrument_inventory_v3
            return build_instrument_inventory_v3(payload, progress=progress)
'''

if 'if mode == "instrument_inventory_v3"' not in text:
    if needle not in text:
        raise RuntimeError('Could not locate instrument_wireframe handler route')
    text = text.replace(needle, route, 1)

path.write_text(text, encoding='utf-8')
print('Instrument Inventory v3 route applied')
