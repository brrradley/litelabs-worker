from pathlib import Path

path = Path('/app/wind_brass_decomposition_v2.py')
text = path.read_text(encoding='utf-8')

old = '''            if now >= next_heartbeat:\n                span = max(end_percent - start_percent, 1)\n                soft_fraction = min(elapsed / max(float(timeout), 1.0), 0.90)\n                pct = start_percent + int(span * soft_fraction)\n                _emit(progress, f'{stage_name} running — {elapsed:.0f}s elapsed', min(pct, end_percent - 1))\n                next_heartbeat = now + heartbeat_seconds\n'''
new = '''            if now >= next_heartbeat:\n                span = max(end_percent - start_percent, 1)\n                # UI progress should move on a realistic specialist-stage clock,\n                # not against the emergency 30-minute subprocess timeout.\n                expected_seconds = 60.0\n                if 'Mega53' in stage_name:\n                    expected_seconds = 30.0\n                elif 'DrumSep' in stage_name:\n                    expected_seconds = 55.0\n                elif 'Wind/Brass' in stage_name:\n                    expected_seconds = 60.0\n                elif 'Lead/Backing' in stage_name:\n                    expected_seconds = 60.0\n                elif 'Saxophone' in stage_name:\n                    expected_seconds = 60.0\n                soft_fraction = min(elapsed / expected_seconds, 0.90)\n                pct = min(start_percent + int(span * soft_fraction), end_percent - 1)\n                _emit(progress, f'{stage_name} — {pct}%', pct)\n                next_heartbeat = now + heartbeat_seconds\n'''
if old not in text:
    raise RuntimeError('Could not locate polled heartbeat block')
text = text.replace(old, new, 1)
path.write_text(text, encoding='utf-8')
print('LiteLABS specialist percentage progress patch applied')
