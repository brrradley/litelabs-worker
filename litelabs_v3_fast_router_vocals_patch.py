from pathlib import Path

path = Path('/app/experimental_children_v1.py')
text = path.read_text(encoding='utf-8')

# Add a BS-RoFormer Karaoke specialist for lead/backing decomposition.
needle = 'DEMUCS3_PYTHON = Path("/opt/demucs3/bin/python")\n'
replacement = needle + 'KARAOKE_CONFIG = Path("/models/karaoke_bs_roformer/config.yaml")\nKARAOKE_CHECKPOINT = Path("/models/karaoke_bs_roformer/model.ckpt")\n'
if needle not in text:
    raise RuntimeError('Could not locate routed specialist constants')
text = text.replace(needle, replacement, 1)

old_required = 'required = [DRUM5_CONFIG, DRUM5_CHECKPOINT, SAX_MODEL_DIR / SAX_MODEL, audio_separator_model_dir / WIND_MODEL, MEGA53_CONFIG, MEGA53_CHECKPOINT, DEMUCS3_PYTHON]'
new_required = 'required = [DRUM5_CONFIG, DRUM5_CHECKPOINT, SAX_MODEL_DIR / SAX_MODEL, audio_separator_model_dir / WIND_MODEL, MEGA53_CONFIG, MEGA53_CHECKPOINT, DEMUCS3_PYTHON, KARAOKE_CONFIG, KARAOKE_CHECKPOINT]'
if old_required not in text:
    raise RuntimeError('Could not locate routed required-model list')
text = text.replace(old_required, new_required, 1)

# Add vocal specialist work dirs.
old_dirs = 'mega_in = root / "mega53_in"\n        sax_out = root / "sax"'
new_dirs = 'mega_in = root / "mega53_in"\n        vocal_in = root / "vocal_in"\n        vocal_out = root / "vocal_karaoke"\n        sax_out = root / "sax"'
if old_dirs not in text:
    raise RuntimeError('Could not locate Mega53 work dirs')
text = text.replace(old_dirs, new_dirs, 1)
text = text.replace(
    'for d in (srcdir, swout, drum_in, drum_out, wind_out, mega_out, mega_in, sax_out, final, experimental, logs):',
    'for d in (srcdir, swout, drum_in, drum_out, wind_out, mega_out, mega_in, vocal_in, vocal_out, sax_out, final, experimental, logs):',
    1,
)

# Broad parent names should describe instrument families, not overclaim one instrument.
old_parent = '''        for stem in SW_STEMS:\n            _copy_as_flac(stems[stem], final / f"{track}_{stem}.flac")\n'''
new_parent = '''        parent_labels = {\n            "vocals": "vocals",\n            "drums": "percussion",\n            "bass": "bass",\n            "guitar": "strings",\n            "piano": "keys",\n            "other": "other",\n        }\n        for stem in SW_STEMS:\n            _copy_as_flac(stems[stem], final / f"{track}_{parent_labels.get(stem, stem)}.flac")\n'''
if old_parent not in text:
    raise RuntimeError('Could not locate parent export loop')
text = text.replace(old_parent, new_parent, 1)

# Add lead/backing vocal decomposition immediately after DrumSep.
anchor = '        # Mega53 is the routing brain: identify whether the Other parent is\n'
vocal_block = '''        # Split the already-clean RoFormer vocal parent with a dedicated\n        # BS-RoFormer Karaoke model. On a vocals-only input its Vocals output\n        # is the lead side and its Instrumental output is the backing/harmony side.\n        emit("Running Lead/Backing Vocal Separation", 47)\n        _copy_as_flac(stems["vocals"], vocal_in / "vocals.flac")\n        rc_vocal, vocal_elapsed = _run_polled(\n            ["python", str(repo_dir / "inference.py"), "--model_type", "bs_roformer", "--config_path", str(KARAOKE_CONFIG), "--start_check_point", str(KARAOKE_CHECKPOINT), "--input_folder", str(vocal_in), "--store_dir", str(vocal_out), "--device_ids", "0", "--disable_detailed_pbar", "--filename_template", "{file_name}/{instr}"],\n            cwd=repo_dir, timeout=timeout, log_path=logs / "karaoke.log", progress=progress,\n            stage_name="Lead/Backing Vocal Separation", start_percent=47, end_percent=58, heartbeat_seconds=heartbeat,\n        )\n        timings["karaoke_bs_roformer"] = round(vocal_elapsed, 3)\n        vocal_files = []\n        vocal_report = {"ok": rc_vocal == 0, "files": []}\n        if rc_vocal == 0:\n            vocal_candidates = [p for p in vocal_out.rglob("*") if p.is_file() and p.suffix.lower() in {".wav", ".flac"}]\n            lead_src = next((p for p in vocal_candidates if p.stem.lower() == "vocals" or "_vocals" in p.stem.lower()), None)\n            backing_src = next((p for p in vocal_candidates if p.stem.lower() == "instrumental" or "instrumental" in p.stem.lower()), None)\n            if lead_src:\n                lead_dest = experimental / f"{track}_lead_vocals.flac"\n                _copy_as_flac(lead_src, lead_dest)\n                vocal_files.append(lead_dest.name)\n            if backing_src:\n                backing_dest = experimental / f"{track}_backing_vocals.flac"\n                _copy_as_flac(backing_src, backing_dest)\n                vocal_files.append(backing_dest.name)\n            vocal_report["files"] = vocal_files\n\n'''
if anchor not in text:
    raise RuntimeError('Could not locate Mega53 router block')
text = text.replace(anchor, vocal_block + anchor, 1)

# Mega53 inventory should inspect representative excerpts, not separate the whole track.
old_mega_input = '        _copy_as_flac(stems["other"], mega_in / "other.flac")\n        emit("Running Mega53 Instrument Inventory", 54)\n'
new_mega_input = '''        # Fast inventory: sample three 6-second windows across the Other parent.\n        # This preserves early/middle/late instrument evidence while capping the\n        # expensive 53-output inference to ~18 seconds of audio.\n        other_audio, other_sr = _read(other_parent)\n        segment_len = max(1, int(other_sr * 6.0))\n        total_len = len(other_audio)\n        centers = (0.18, 0.50, 0.82)\n        samples = []\n        for frac in centers:\n            centre = int(total_len * frac)\n            start = max(0, min(max(total_len - segment_len, 0), centre - segment_len // 2))\n            samples.append(other_audio[start:start + segment_len])\n        inventory_audio = np.concatenate([s for s in samples if len(s)], axis=0) if samples else other_audio[:segment_len]\n        _write_flac(mega_in / "other.flac", inventory_audio, other_sr)\n        emit("Running Mega53 Instrument Inventory", 60)\n'''
if old_mega_input not in text:
    raise RuntimeError('Could not locate Mega53 full-track input')
text = text.replace(old_mega_input, new_mega_input, 1)
text = text.replace('stage_name="Mega53 instrument inventory", start_percent=54, end_percent=68, heartbeat_seconds=heartbeat,', 'stage_name="Mega53 Instrument Inventory", start_percent=60, end_percent=70, heartbeat_seconds=heartbeat,', 1)
text = text.replace('cwd=repo_dir, timeout=timeout, log_path=logs / "mega53.log", progress=None,', 'cwd=repo_dir, timeout=timeout, log_path=logs / "mega53.log", progress=progress,', 1)
# Compare discovery outputs to the same sampled parent used by Mega53.
text = text.replace('            parent_audio, _ = _read(other_parent)\n            for name in ("saxophone",', '            parent_audio, _ = _read(mega_in / "other.flac")\n            for name in ("saxophone",', 1)

# Move routed family/sax percentages after the faster inventory.
text = text.replace('emit("Running Wind/Brass Family Separation", 70)', 'emit("Running Wind/Brass Family Separation", 72)', 1)
text = text.replace('emit("Running Saxophone Specialist Separation", 82)', 'emit("Running Saxophone Specialist Separation", 84)', 1)

# Use the polled runner for Wind and Sax too, so the widget receives real percentage updates.
old_wind_run = '''            t0 = time.monotonic()\n            wind_cmd = ["audio-separator", str(other_parent), "--model_filename", WIND_MODEL, "--model_file_dir", str(audio_separator_model_dir), "--output_dir", str(wind_out), "--output_format", "FLAC"]\n            wind = subprocess.run(wind_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)\n            wind_returncode = wind.returncode\n            wind_stdout = wind.stdout or ""\n            timings["wind_uvr"] = round(time.monotonic() - t0, 3)\n            if wind.returncode == 0:\n'''
new_wind_run = '''            wind_cmd = ["audio-separator", str(other_parent), "--model_filename", WIND_MODEL, "--model_file_dir", str(audio_separator_model_dir), "--output_dir", str(wind_out), "--output_format", "FLAC"]\n            wind_returncode, wind_elapsed = _run_polled(\n                wind_cmd, cwd=None, timeout=timeout, log_path=logs / "wind_uvr.log", progress=progress,\n                stage_name="Wind/Brass Family Separation", start_percent=72, end_percent=82, heartbeat_seconds=heartbeat,\n            )\n            wind_stdout = (logs / "wind_uvr.log").read_text(encoding="utf-8", errors="replace") if (logs / "wind_uvr.log").is_file() else ""\n            timings["wind_uvr"] = round(wind_elapsed, 3)\n            if wind_returncode == 0:\n'''
if old_wind_run not in text:
    raise RuntimeError('Could not locate Wind subprocess block')
text = text.replace(old_wind_run, new_wind_run, 1)

old_sax_run = '''            t0 = time.monotonic()\n            sax_input = woodwind_path if woodwind_path and woodwind_path.is_file() else (family_path if family_path and family_path.is_file() else other_parent)\n            sax_cmd = [str(DEMUCS3_PYTHON), "-m", "demucs", "--repo", str(SAX_MODEL_DIR), "-n", SAX_MODEL, "-o", str(sax_out), str(sax_input)]\n            sax = subprocess.run(sax_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)\n            sax_returncode = sax.returncode\n            sax_stdout = sax.stdout or ""\n            timings["sax_demucs"] = round(time.monotonic() - t0, 3)\n            sax_files = _copy_audio_tree(sax_out, experimental, f"{track}_sax_specialist") if sax.returncode == 0 else []\n'''
new_sax_run = '''            sax_input = woodwind_path if woodwind_path and woodwind_path.is_file() else (family_path if family_path and family_path.is_file() else other_parent)\n            sax_cmd = [str(DEMUCS3_PYTHON), "-m", "demucs", "--repo", str(SAX_MODEL_DIR), "-n", SAX_MODEL, "-o", str(sax_out), str(sax_input)]\n            sax_returncode, sax_elapsed = _run_polled(\n                sax_cmd, cwd=None, timeout=timeout, log_path=logs / "sax.log", progress=progress,\n                stage_name="Saxophone Specialist Separation", start_percent=84, end_percent=90, heartbeat_seconds=heartbeat,\n            )\n            sax_stdout = (logs / "sax.log").read_text(encoding="utf-8", errors="replace") if (logs / "sax.log").is_file() else ""\n            timings["sax_demucs"] = round(sax_elapsed, 3)\n            sax_files = _copy_audio_tree(sax_out, experimental, f"{track}_sax_specialist") if sax_returncode == 0 else []\n'''
if old_sax_run not in text:
    raise RuntimeError('Could not locate Sax subprocess block')
text = text.replace(old_sax_run, new_sax_run, 1)

# Add vocal specialist metadata/report.
text = text.replace('            "saxophone": SAX_MODEL,\n', '            "saxophone": SAX_MODEL,\n            "lead_backing_vocals": "BS-RoFormer Karaoke (frazer/becruily)",\n', 1)
text = text.replace('            "drums_5stem": drum_report,\n', '            "drums_5stem": drum_report,\n            "lead_backing_vocals": vocal_report,\n', 1)
text = text.replace('"schema_version": 4,', '"schema_version": 5,', 1)

# Add README timing label for the new vocal stage.
text = text.replace('        "mega53": "Mega53 separation",\n', '        "mega53": "Mega53 instrument inventory",\n        "karaoke_bs_roformer": "Lead/Backing vocal separation",\n', 1)

path.write_text(text, encoding='utf-8')
print('LiteLABS fast inventory, family progress, vocal children and parent labels applied')
