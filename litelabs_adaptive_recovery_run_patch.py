from pathlib import Path

path = Path('/app/handler.py')
text = path.read_text()

if '"adaptive_recovery_run"' not in text:
    marker = '"track_profile"'
    if marker not in text:
        raise RuntimeError('Could not locate track_profile mode marker')
    text = text.replace(marker, marker + ', "adaptive_recovery_run"', 1)

check_anchor = '            "track_profile": ("track_profile", "build_track_profile"),\n'
check_line = '            "adaptive_recovery_run": ("adaptive_recovery_run", "build_adaptive_recovery_run"),\n'
if check_line not in text:
    if check_anchor not in text:
        raise RuntimeError('Could not patch adaptive recovery health check')
    text = text.replace(check_anchor, check_anchor + check_line)

route_anchor = '        if mode == "track_profile":\n            from track_profile import build_track_profile\n            return build_track_profile(payload, progress=progress)\n'
route_line = '        if mode == "adaptive_recovery_run":\n            from adaptive_recovery_run import build_adaptive_recovery_run\n            return build_adaptive_recovery_run(payload, progress=progress)\n'
if route_line not in text:
    if route_anchor not in text:
        raise RuntimeError('Could not patch adaptive recovery route')
    text = text.replace(route_anchor, route_anchor + route_line)

path.write_text(text)
print('adaptive recovery handler patch applied')
