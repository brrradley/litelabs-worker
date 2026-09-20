from pathlib import Path

path = Path('/app/experimental_children_v1.py')
text = path.read_text(encoding='utf-8')

# 1) DrumSep was using the polled runner but suppressing the callback, so its
# percentages only reached worker logs. Forward them to the normal progress callback.
old = '''            cwd=repo_dir, timeout=timeout, log_path=logs / "drum5.log", progress=None,\n            stage_name="DrumSep 5-stem", start_percent=32, end_percent=52, heartbeat_seconds=heartbeat,\n'''
new = '''            cwd=repo_dir, timeout=timeout, log_path=logs / "drum5.log", progress=progress,\n            stage_name="DrumSep 5-stem", start_percent=32, end_percent=46, heartbeat_seconds=heartbeat,\n'''
if old not in text:
    raise RuntimeError('Could not locate DrumSep polled callback block')
text = text.replace(old, new, 1)

# 2) The karaoke checkpoint is target_instrument=Vocals, so MSS writes only the
# target (lead) output. Reconstruct the backing side as parent - lead so the two
# children close exactly back to the RoFormer Vocal parent.
old = '''            backing_src = next((p for p in vocal_candidates if p.stem.lower() == "instrumental" or "instrumental" in p.stem.lower()), None)\n            if lead_src:\n                lead_dest = experimental / f"{track}_lead_vocals.flac"\n                _copy_as_flac(lead_src, lead_dest)\n                vocal_files.append(lead_dest.name)\n            if backing_src:\n                backing_dest = experimental / f"{track}_backing_vocals.flac"\n                _copy_as_flac(backing_src, backing_dest)\n                vocal_files.append(backing_dest.name)\n            vocal_report["files"] = vocal_files\n'''
new = '''            backing_src = next((p for p in vocal_candidates if p.stem.lower() == "instrumental" or "instrumental" in p.stem.lower()), None)\n            if lead_src:\n                lead_dest = experimental / f"{track}_lead_vocals.flac"\n                _copy_as_flac(lead_src, lead_dest)\n                vocal_files.append(lead_dest.name)\n                # Prefer an explicit complement if the framework produced one;\n                # otherwise derive backing/harmony from the clean vocal parent.\n                backing_dest = experimental / f"{track}_backing_vocals.flac"\n                if backing_src:\n                    _copy_as_flac(backing_src, backing_dest)\n                else:\n                    vocal_parent_audio, vocal_sr = _read(stems["vocals"])\n                    lead_audio, _ = _read(lead_src)\n                    vn = min(len(vocal_parent_audio), len(lead_audio))\n                    _write_flac(backing_dest, vocal_parent_audio[:vn] - lead_audio[:vn], vocal_sr)\n                    vocal_report["backing_method"] = "vocal_parent_minus_lead"\n                vocal_files.append(backing_dest.name)\n                # Record exact reconstruction quality for the two vocal children.\n                lead_audio2, _ = _read(lead_dest)\n                backing_audio2, _ = _read(backing_dest)\n                parent_audio2, _ = _read(stems["vocals"])\n                vn2 = min(len(lead_audio2), len(backing_audio2), len(parent_audio2))\n                vocal_sum = lead_audio2[:vn2] + backing_audio2[:vn2]\n                vocal_report["parent_vs_children_sum_cosine"] = round(float(_cos(parent_audio2[:vn2], vocal_sum)), 6)\n            vocal_report["files"] = vocal_files\n'''
if old not in text:
    raise RuntimeError('Could not locate lead/backing export block')
text = text.replace(old, new, 1)

# 3) Add a second, short Mega53 inventory over the Wind/Brass residual. This is
# detection only: do not create more child stems until a trusted specialist exists.
anchor = '''        # The published sax model explicitly requires Demucs 3.0.6. Only run it\n'''
block = '''        residual_inventory = {}\n        residual_inventory_route = []\n        residual_inventory_elapsed = 0.0\n        if wind_residual_path and wind_residual_path.is_file():\n            emit("Analysing Residual Instruments", 83)\n            residual_audio, residual_sr = _read(wind_residual_path)\n            residual_segment_len = max(1, int(residual_sr * 6.0))\n            residual_total = len(residual_audio)\n            residual_samples = []\n            for frac in (0.18, 0.50, 0.82):\n                centre = int(residual_total * frac)\n                start = max(0, min(max(residual_total - residual_segment_len, 0), centre - residual_segment_len // 2))\n                residual_samples.append(residual_audio[start:start + residual_segment_len])\n            residual_sample = np.concatenate([s for s in residual_samples if len(s)], axis=0) if residual_samples else residual_audio[:residual_segment_len]\n            residual_in = root / "residual_inventory_in"\n            residual_out = root / "residual_inventory_out"\n            residual_in.mkdir(parents=True, exist_ok=True)\n            residual_out.mkdir(parents=True, exist_ok=True)\n            _write_flac(residual_in / "residual.flac", residual_sample, residual_sr)\n            rc_residual, residual_inventory_elapsed = _run_polled(\n                ["python", str(repo_dir / "inference.py"), "--model_type", "bs_roformer", "--config_path", str(MEGA53_CONFIG), "--start_check_point", str(MEGA53_CHECKPOINT), "--input_folder", str(residual_in), "--store_dir", str(residual_out), "--device_ids", "0", "--disable_detailed_pbar", "--filename_template", "{file_name}/{instr}"],\n                cwd=repo_dir, timeout=timeout, log_path=logs / "residual_inventory.log", progress=progress,\n                stage_name="Residual Instrument Inventory", start_percent=83, end_percent=89, heartbeat_seconds=heartbeat,\n            )\n            timings["residual_inventory"] = round(residual_inventory_elapsed, 3)\n            if rc_residual == 0:\n                residual_files = [p for p in residual_out.rglob("*") if p.is_file() and p.suffix.lower() in {".wav", ".flac"}]\n                residual_by_name = {p.stem.lower().replace("_", "-"): p for p in residual_files}\n                # Broad list on purpose: inventory informs which specialist family to add next.\n                residual_targets = ("synth", "keys", "piano", "digital-piano", "organ", "guitar", "electric-guitar", "strings", "violin", "brass", "trumpet", "saxophone", "percussion", "drums", "bass")\n                residual_parent_audio, _ = _read(residual_in / "residual.flac")\n                for name in residual_targets:\n                    p = residual_by_name.get(name)\n                    if not p:\n                        continue\n                    a, _ = _read(p)\n                    rn = min(len(a), len(residual_parent_audio))\n                    rms = float(np.sqrt(np.mean(a[:rn] * a[:rn]) + 1e-12))\n                    item = {"rms_dbfs": _db(rms), "parent_cosine": round(float(_cos(a[:rn], residual_parent_audio[:rn])), 6)}\n                    residual_inventory[name] = item\n                    if item["rms_dbfs"] >= -42.0 and abs(item["parent_cosine"]) >= 0.20:\n                        residual_inventory_route.append(name)\n\n'''
if anchor not in text:
    raise RuntimeError('Could not locate sax specialist anchor for residual inventory')
text = text.replace(anchor, block + anchor, 1)

# Move sax slightly later to leave UI range for residual detection.
text = text.replace('emit("Running Saxophone Specialist Separation", 84)', 'emit("Running Saxophone Specialist Separation", 90)', 1)
text = text.replace('stage_name="Saxophone Specialist Separation", start_percent=84, end_percent=90, heartbeat_seconds=heartbeat,', 'stage_name="Saxophone Specialist Separation", start_percent=90, end_percent=94, heartbeat_seconds=heartbeat,', 1)

# Include residual evidence in report and README timings.
old_report = '''            "mega53_inventory": {"ok": rc == 0, "route": family_route, "evidence": inventory},\n'''
new_report = '''            "mega53_inventory": {"ok": rc == 0, "route": family_route, "evidence": inventory},\n            "residual_inventory": {"ran": bool(wind_residual_path and wind_residual_path.is_file()), "detected": residual_inventory_route, "evidence": residual_inventory, "policy": "detection only; route to trusted specialists before exporting new children"},\n'''
if old_report not in text:
    raise RuntimeError('Could not locate Mega53 report entry')
text = text.replace(old_report, new_report, 1)
text = text.replace('        "karaoke_bs_roformer": "Lead/Backing vocal separation",\n', '        "karaoke_bs_roformer": "Lead/Backing vocal separation",\n        "residual_inventory": "Residual instrument inventory",\n', 1)
text = text.replace('"schema_version": 5,', '"schema_version": 6,', 1)

path.write_text(text, encoding='utf-8')
print('LiteLABS DrumSep widget progress, backing complement and residual inventory applied')
