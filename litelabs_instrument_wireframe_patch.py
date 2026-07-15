from pathlib import Path

path = Path('/app/handler.py')
text = path.read_text()

text = text.replace(
    '"reference_free_stem_auditor", "vocal_residual_test"',
    '"reference_free_stem_auditor", "instrument_wireframe", "vocal_residual_test"',
)
text = text.replace(
    '            "reference_free_stem_auditor": ("reference_free_stem_auditor", "build_reference_free_stem_auditor"),\n',
    '            "reference_free_stem_auditor": ("reference_free_stem_auditor", "build_reference_free_stem_auditor"),\n            "instrument_wireframe": ("instrument_wireframe", "build_instrument_wireframe"),\n',
)
needle = '''        if mode == "reference_free_stem_auditor":
            from reference_free_stem_auditor import build_reference_free_stem_auditor
            return build_reference_free_stem_auditor(payload, progress=progress)
'''
replacement = needle + '''        if mode == "instrument_wireframe":
            from instrument_wireframe import build_instrument_wireframe
            return build_instrument_wireframe(payload, progress=progress)
'''
if needle not in text:
    raise SystemExit('reference-free auditor handler block not found')
text = text.replace(needle, replacement, 1)
path.write_text(text)
