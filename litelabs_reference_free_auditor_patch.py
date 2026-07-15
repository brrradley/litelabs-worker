from pathlib import Path

path = Path('/app/handler.py')
text = path.read_text(encoding='utf-8')

text = text.replace(
    '"studio_mix_compatibility", "vocal_residual_test"',
    '"studio_mix_compatibility", "reference_free_stem_auditor", "vocal_residual_test"',
)

text = text.replace(
    '            "studio_mix_compatibility": ("studio_mix_compatibility", "build_studio_mix_compatibility"),\n',
    '            "studio_mix_compatibility": ("studio_mix_compatibility", "build_studio_mix_compatibility"),\n'
    '            "reference_free_stem_auditor": ("reference_free_stem_auditor", "build_reference_free_stem_auditor"),\n',
)

needle = '''        if mode == "studio_mix_compatibility":
            from studio_mix_compatibility import build_studio_mix_compatibility
            return build_studio_mix_compatibility(payload, progress=progress)
'''
replacement = needle + '''        if mode == "reference_free_stem_auditor":
            from reference_free_stem_auditor import build_reference_free_stem_auditor
            return build_reference_free_stem_auditor(payload, progress=progress)
'''
if needle not in text:
    raise SystemExit('studio_mix_compatibility handler block not found')
text = text.replace(needle, replacement, 1)

path.write_text(text, encoding='utf-8')
