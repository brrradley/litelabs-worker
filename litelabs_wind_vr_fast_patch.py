from pathlib import Path

path = Path('/app/experimental_children_v1.py')
text = path.read_text(encoding='utf-8')

old = '["audio-separator", str(other_parent), "--model_filename", WIND_MODEL, "--model_file_dir", str(audio_separator_model_dir), "--output_dir", str(wind_out), "--output_format", "FLAC"]'
new = '["audio-separator", str(other_parent), "--model_filename", WIND_MODEL, "--model_file_dir", str(audio_separator_model_dir), "--output_dir", str(wind_out), "--output_format", "FLAC", "--vr_batch_size", "16", "--vr_window_size", "1024", "--use_autocast"]'
if old not in text:
    raise RuntimeError('Could not locate Wind/Brass audio-separator command')
text = text.replace(old, new, 1)

# Record the tuned execution profile in the report for benchmark comparisons.
needle = '"architecture": "Other parent -> routed Wind/Brass family + residual",\n'
replacement = needle + '                "inference_profile": {"vr_batch_size": 16, "vr_window_size": 1024, "autocast": True, "quality_note": "GPU-safe speed profile; 1024 window retained, batch reduced after A4500 OOM at 32"},\n'
if needle not in text:
    raise RuntimeError('Could not locate wind report architecture')
text = text.replace(needle, replacement, 1)

path.write_text(text, encoding='utf-8')
print('LiteLABS GPU-safe fast VR wind profile applied: batch=16 window=1024 autocast')
