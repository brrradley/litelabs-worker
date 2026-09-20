from pathlib import Path

path = Path('/app/experimental_children_v1.py')
text = path.read_text(encoding='utf-8')

old = '["audio-separator", str(other_parent), "--model_filename", WIND_MODEL, "--model_file_dir", str(audio_separator_model_dir), "--output_dir", str(wind_out), "--output_format", "FLAC"]'
new = '["audio-separator", str(other_parent), "--model_filename", WIND_MODEL, "--model_file_dir", str(audio_separator_model_dir), "--output_dir", str(wind_out), "--output_format", "FLAC", "--vr_batch_size", "8", "--vr_window_size", "1024", "--use_autocast"]'
if old not in text:
    raise RuntimeError('Could not locate Wind/Brass audio-separator command')
text = text.replace(old, new, 1)

# Record the tuned execution profile in the report for benchmark comparisons.
needle = '"architecture": "Other parent -> routed Wind/Brass family + residual",\n'
replacement = needle + '                "inference_profile": {"vr_batch_size": 8, "vr_window_size": 1024, "autocast": True, "quality_note": "A4500-safe speed profile; batch 32 and 16 OOMed at 1024, so batch 8 retains the fast window with safer VRAM headroom"},\n'
if needle not in text:
    raise RuntimeError('Could not locate wind report architecture')
text = text.replace(needle, replacement, 1)

path.write_text(text, encoding='utf-8')
print('LiteLABS A4500-safe fast VR wind profile applied: batch=8 window=1024 autocast')
