from pathlib import Path

path = Path('/app/handler.py')
text = path.read_text()

old_modes = '"studio_mix_compatibility", "vocal_residual_test", "audio_separator_discovery"'
new_modes = '"studio_mix_compatibility", "reference_free_stem_auditor", "instrument_wireframe", "combined_recovery_planner", "vocal_residual_test", "audio_separator_discovery"'
if old_modes in text:
    text = text.replace(old_modes, new_modes)
elif '"combined_recovery_planner"' not in text:
    raise RuntimeError('Could not patch modes list')

check_anchor = '            "studio_mix_compatibility": ("studio_mix_compatibility", "build_studio_mix_compatibility"),\n'
check_insert = check_anchor + '            "reference_free_stem_auditor": ("reference_free_stem_auditor", "build_reference_free_stem_auditor"),\n            "instrument_wireframe": ("instrument_wireframe", "build_instrument_wireframe"),\n            "combined_recovery_planner": ("combined_recovery_planner", "build_combined_recovery_planner"),\n'
if '"combined_recovery_planner": ("combined_recovery_planner"' not in text:
    if check_anchor not in text:
        raise RuntimeError('Could not patch health checks')
    text = text.replace(check_anchor, check_insert)

route_anchor = '        if mode == "studio_mix_compatibility":\n            from studio_mix_compatibility import build_studio_mix_compatibility\n            return build_studio_mix_compatibility(payload, progress=progress)\n'
route_insert = route_anchor + '        if mode == "reference_free_stem_auditor":\n            from reference_free_stem_auditor import build_reference_free_stem_auditor\n            return build_reference_free_stem_auditor(payload, progress=progress)\n        if mode == "instrument_wireframe":\n            from instrument_wireframe import build_instrument_wireframe\n            return build_instrument_wireframe(payload, progress=progress)\n        if mode == "combined_recovery_planner":\n            from combined_recovery_planner import build_combined_recovery_planner\n            return build_combined_recovery_planner(payload, progress=progress)\n'
if 'if mode == "combined_recovery_planner"' not in text:
    if route_anchor not in text:
        raise RuntimeError('Could not patch handler routes')
    text = text.replace(route_anchor, route_insert)

path.write_text(text)
print('combined recovery planner handler patch applied')
