from pathlib import Path

path = Path('/app/handler.py')
text = path.read_text()

# Add the mode without depending on a specific pre-patch mode-list ordering.
if '"track_profile"' not in text:
    marker = '    modes = ['
    start = text.find(marker)
    if start < 0:
        raise RuntimeError('Could not locate modes list')
    line_end = text.find('\n', start)
    line = text[start:line_end]
    if line.rstrip().endswith(']'):
        line = line.rstrip()[:-1] + ', "track_profile"]'
        text = text[:start] + line + text[line_end:]
    else:
        raise RuntimeError('Unexpected modes list format')

check_anchor = '            "studio_mix_compatibility": ("studio_mix_compatibility", "build_studio_mix_compatibility"),\n'
check_line = '            "track_profile": ("track_profile", "build_track_profile"),\n'
if check_line not in text:
    if check_anchor not in text:
        raise RuntimeError('Could not patch track_profile health check')
    text = text.replace(check_anchor, check_anchor + check_line, 1)

route_anchor = '        if mode == "studio_mix_compatibility":\n            from studio_mix_compatibility import build_studio_mix_compatibility\n            return build_studio_mix_compatibility(payload, progress=progress)\n'
route_block = route_anchor + '        if mode == "track_profile":\n            from track_profile import build_track_profile\n            return build_track_profile(payload, progress=progress)\n'
if 'if mode == "track_profile"' not in text:
    if route_anchor not in text:
        raise RuntimeError('Could not patch track_profile route')
    text = text.replace(route_anchor, route_block, 1)

path.write_text(text)
print('track profile handler patch applied')
