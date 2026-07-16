from pathlib import Path

path = Path('/app/handler.py')
text = path.read_text()

# Add the mode without depending on the exact state left by earlier patches.
if '"combined_recovery_planner"' not in text:
    mode_anchors = [
        '"instrument_wireframe", "vocal_residual_test"',
        '"reference_free_stem_auditor", "vocal_residual_test"',
        '"studio_mix_compatibility", "vocal_residual_test"',
    ]
    for anchor in mode_anchors:
        if anchor in text:
            text = text.replace(
                anchor,
                anchor.replace(', "vocal_residual_test"', ', "combined_recovery_planner", "vocal_residual_test"'),
                1,
            )
            break
    else:
        raise RuntimeError('Could not patch modes list')

check_anchor = '            "studio_mix_compatibility": ("studio_mix_compatibility", "build_studio_mix_compatibility"),\n'
check_line = '            "combined_recovery_planner": ("combined_recovery_planner", "build_combined_recovery_planner"),\n'
if '"combined_recovery_planner": ("combined_recovery_planner"' not in text:
    if check_anchor not in text:
        raise RuntimeError('Could not patch health checks')
    text = text.replace(check_anchor, check_anchor + check_line, 1)

route_anchor = (
    '        if mode == "studio_mix_compatibility":\n'
    '            from studio_mix_compatibility import build_studio_mix_compatibility\n'
    '            return build_studio_mix_compatibility(payload, progress=progress)\n'
)
route_insert = route_anchor + (
    '        if mode == "combined_recovery_planner":\n'
    '            from combined_recovery_planner import build_combined_recovery_planner\n'
    '            return build_combined_recovery_planner(payload, progress=progress)\n'
)
if 'if mode == "combined_recovery_planner"' not in text:
    if route_anchor not in text:
        raise RuntimeError('Could not patch handler routes')
    text = text.replace(route_anchor, route_insert, 1)

path.write_text(text)
print('combined recovery planner handler patch applied')
