from pathlib import Path

path = Path('/app/handler.py')
text = path.read_text()

if '"mss_candidate_lab"' not in text:
    marker = '"adaptive_recovery_run"'
    if marker not in text:
        raise RuntimeError('Could not locate adaptive_recovery_run mode marker')
    text = text.replace(marker, marker + ', "mss_candidate_lab"', 1)

check_anchor = '            "adaptive_recovery_run": ("adaptive_recovery_run", "build_adaptive_recovery_run"),\n'
check_line = '            "mss_candidate_lab": ("mss_candidate_lab", "build_mss_candidate_lab"),\n'
if check_line not in text:
    if check_anchor not in text:
        raise RuntimeError('Could not patch MSS candidate lab health check')
    text = text.replace(check_anchor, check_anchor + check_line)

route_anchor = '        if mode == "adaptive_recovery_run":\n            from adaptive_recovery_run import build_adaptive_recovery_run\n            return build_adaptive_recovery_run(payload, progress=progress)\n'
route_line = '        if mode == "mss_candidate_lab":\n            from mss_candidate_lab import build_mss_candidate_lab\n            return build_mss_candidate_lab(payload, progress=progress)\n'
if route_line not in text:
    if route_anchor not in text:
        raise RuntimeError('Could not patch MSS candidate lab route')
    text = text.replace(route_anchor, route_anchor + route_line)

path.write_text(text)
print('MSS candidate lab handler patch applied')
