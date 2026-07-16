from pathlib import Path

path = Path('/app/handler.py')
text = path.read_text(encoding='utf-8')

old_modes = '"vocal_residual_test", "audio_separator_discovery"]'
new_modes = '"vocal_residual_test", "audio_separator_discovery", "mss_candidate_benchmark"]'
if old_modes in text:
    text = text.replace(old_modes, new_modes, 1)
elif '"mss_candidate_benchmark"' not in text:
    raise RuntimeError('Could not add mss_candidate_benchmark to modes')

marker = '''        if mode == "studio_mix_compatibility":
            from studio_mix_compatibility import build_studio_mix_compatibility
            return build_studio_mix_compatibility(payload, progress=progress)
'''
replacement = marker + '''        if mode == "mss_candidate_benchmark":
            from mss_candidate_benchmark import build_mss_candidate_benchmark
            return build_mss_candidate_benchmark(payload, progress=progress)
'''
if marker in text and 'from mss_candidate_benchmark import build_mss_candidate_benchmark' not in text:
    text = text.replace(marker, replacement, 1)
elif 'from mss_candidate_benchmark import build_mss_candidate_benchmark' not in text:
    raise RuntimeError('Could not wire mss_candidate_benchmark handler route')

path.write_text(text, encoding='utf-8')
print('MSS candidate benchmark handler patch applied')
