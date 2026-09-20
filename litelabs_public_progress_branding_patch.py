from pathlib import Path

FILES = [
    Path('/app/handler.py'),
    Path('/app/experimental_children_v1.py'),
    Path('/app/preset_pack.py'),
]

REPLACEMENTS = {
    'BS-RoFormer Parent Separation': 'LiteLABS-RS Parent Separation',
    'DrumSep 5-Stem Decomposition': 'LiteLABS-DR Drum Decomposition',
    'Lead/Backing Vocal Separation': 'LiteLABS-VX Vocal Separation',
    'Mega53 Instrument Inventory': 'LiteLABS-IR Instrument Analysis',
    'Wind/Brass Family Separation': 'LiteLABS-WB Wind/Brass Separation',
    'Saxophone Specialist Separation': 'LiteLABS-SX Saxophone Separation',
}

changed = 0
for path in FILES:
    if not path.exists():
        continue
    text = path.read_text(encoding='utf-8')
    original = text
    for internal, public in REPLACEMENTS.items():
        text = text.replace(internal, public)
    if text != original:
        path.write_text(text, encoding='utf-8')
        changed += 1

if changed == 0:
    raise RuntimeError('No public progress labels were replaced')

print(f'LiteLABS public progress branding applied to {changed} file(s)')
