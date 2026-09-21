from pathlib import Path

path = Path('/app/experimental_children_v1.py')
text = path.read_text(encoding='utf-8')

# Public/readme copy must use LiteLABS product labels only. Internal reports and
# worker telemetry may retain implementation/model identities for research.
if 'public_readme_model_redaction_v1' not in text:
    old = '''    for family, model in models.items():
        lines.append(f"{family.title()}: {model}")
'''
    new = '''    # public_readme_model_redaction_v1
    public_model_labels = {
        "parent": "LiteLABS Core",
        "drums": "LiteLABS Perc-5",
        "wind": "LiteLABS Wind",
        "saxophone": "LiteLABS Sax",
        "lead_backing_vocals": "LiteLABS VX",
        "instrument_inventory": "LiteLABS Inst-MTG",
        "genre": "LiteLABS G400",
    }
    for family in models:
        lines.append(f"{family.title()}: {public_model_labels.get(family, 'LiteLABS Specialist')}")
'''
    if old not in text:
        raise RuntimeError('Could not locate experimental README model list')
    text = text.replace(old, new, 1)

# Never expose the third-party classifier name in the user-facing README.
text = text.replace(
    'genre_lines = ["ESSENTIA GENRE CANDIDATES", "-------------------------"]',
    'genre_lines = ["LITELABS G400 GENRE CANDIDATES", "-----------------------------"]',
)
text = text.replace(
    '# Add Essentia genre candidates to the README without overriding the existing',
    '# Add LiteLABS G400 genre candidates to the README without overriding the existing',
)

path.write_text(text, encoding='utf-8')

check = path.read_text(encoding='utf-8')
assert 'public_readme_model_redaction_v1' in check
assert 'LITELABS G400 GENRE CANDIDATES' in check
assert 'ESSENTIA GENRE CANDIDATES' not in check
print('LiteLABS public README model names redacted')
