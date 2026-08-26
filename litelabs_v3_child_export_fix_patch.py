from pathlib import Path

path = Path('/app/experimental_children_v1.py')
text = path.read_text(encoding='utf-8')

# Demucs LocalRepo indexes .th files by Path.stem, not by the filename including
# the .th extension. Passing SAX_MODEL verbatim makes the CLI reject the model
# before inference starts.
old_sax_cmd = '[str(DEMUCS3_PYTHON), "-m", "demucs", "--repo", str(SAX_MODEL_DIR), "-n", SAX_MODEL, "-o", str(sax_out), str(sax_input)]'
new_sax_cmd = '[str(DEMUCS3_PYTHON), "-m", "demucs", "--repo", str(SAX_MODEL_DIR), "-n", Path(SAX_MODEL).stem, "-o", str(sax_out), str(sax_input)]'
if old_sax_cmd not in text:
    raise RuntimeError('Could not locate Demucs sax command')
text = text.replace(old_sax_cmd, new_sax_cmd, 1)

# audio-separator can return exit code 0 even after logging an internal model
# failure. Treat the stage as successful only when usable child audio was
# actually collected.
old_init = '''        wind_returncode = None\n        wind_stdout = ""\n'''
new_init = '''        wind_returncode = None\n        wind_stdout = ""\n        wind_output_ok = False\n'''
if old_init not in text:
    raise RuntimeError('Could not locate wind status initialisation')
text = text.replace(old_init, new_init, 1)

# Make output collection tolerant of the exact label punctuation emitted by the
# VR model. Prefer explicit Woodwind/No-Woodwind names, but if the model emits
# two audio files with alternate punctuation, use the non-"no" file as target
# and the "no" file as residual.
old_candidates = '''                candidates = [p for p in wind_out.rglob("*") if p.is_file() and p.suffix.lower() in {".wav", ".flac", ".mp3"}]\n                target_src = next((p for p in candidates if "woodwind" in p.name.lower() and "no_woodwind" not in p.name.lower() and "no woodwind" not in p.name.lower() and "no-woodwind" not in p.name.lower()), None)\n                residual_src = next((p for p in candidates if "no_woodwind" in p.name.lower() or "no woodwind" in p.name.lower() or "no-woodwind" in p.name.lower()), None)\n'''
new_candidates = '''                candidates = [p for p in wind_out.rglob("*") if p.is_file() and p.suffix.lower() in {".wav", ".flac", ".mp3"}]\n                def _norm_label(p):\n                    return p.name.lower().replace("_", " ").replace("-", " ").replace("(", " ").replace(")", " ")\n                target_src = next((p for p in candidates if "woodwind" in _norm_label(p) and "no woodwind" not in _norm_label(p)), None)\n                residual_src = next((p for p in candidates if "no woodwind" in _norm_label(p)), None)\n                if len(candidates) == 2 and (target_src is None or residual_src is None):\n                    residual_src = residual_src or next((p for p in candidates if " no " in f" {_norm_label(p)} "), None)\n                    target_src = target_src or next((p for p in candidates if p != residual_src), None)\n'''
if old_candidates not in text:
    raise RuntimeError('Could not locate wind output collector')
text = text.replace(old_candidates, new_candidates, 1)

# Set success after files have actually been copied.
old_after = '''                if residual_src:\n                    wind_residual_path = experimental / f"{track}_wind_brass_residual.flac"\n                    _copy_as_flac(residual_src, wind_residual_path)\n                    wind_files.append(wind_residual_path.name)\n'''
new_after = '''                if residual_src:\n                    wind_residual_path = experimental / f"{track}_wind_brass_residual.flac"\n                    _copy_as_flac(residual_src, wind_residual_path)\n                    wind_files.append(wind_residual_path.name)\n                wind_output_ok = bool(family_path and family_path.is_file() and wind_residual_path and wind_residual_path.is_file())\n'''
if old_after not in text:
    raise RuntimeError('Could not locate wind residual export block')
text = text.replace(old_after, new_after, 1)

old_report_ok = '"ok": wind_returncode == 0 if wind_returncode is not None else False,'
new_report_ok = '"ok": bool(wind_output_ok),'
if old_report_ok not in text:
    raise RuntimeError('Could not locate wind report ok field')
text = text.replace(old_report_ok, new_report_ok, 1)

# Preserve the process return code too, because audio-separator may mask an
# internal failure with shell exit 0.
needle = '                "ran": family_route != "none",\n'
replacement = needle + '                "process_returncode": wind_returncode,\n                "output_validated": bool(wind_output_ok),\n'
if needle not in text:
    raise RuntimeError('Could not locate wind report ran field')
text = text.replace(needle, replacement, 1)

path.write_text(text, encoding='utf-8')
print('LiteLABS wind export validation and Demucs sax model-name fix applied')
