from pathlib import Path

exp_path = Path('/app/experimental_children_v1.py')
preset_path = Path('/app/preset_pack.py')
text = exp_path.read_text(encoding='utf-8')

# ---------------------------------------------------------------------------
# Locked vocal benchmark v1 (Disturbia exact multitrack, 2026-09-17)
#   parent:  BS-RoFormer-SW vocals
#   lead:    25% SW + 75% MelBand Becruily vocal parent -> Becruily karaoke
#   backing: SW vocal parent - fast SW->Becruily karaoke lead
# The best lead and best backing intentionally come from different validated
# routes. Each route conserves its own parent; the two exported best stems are
# not asserted to be a single mass-conserving pair.
# ---------------------------------------------------------------------------
start_marker = '        # Split the already-clean RoFormer vocal parent with a dedicated\n'
end_marker = '        # Mega53 is the routing brain:'
start = text.find(start_marker)
end = text.find(end_marker, start)
if start < 0 or end < 0:
    raise RuntimeError('Could not locate existing vocal specialist block')

vocal_block = '''        # LiteLABS locked vocal benchmark v1. Parent vocals remain the\n        # untouched BS-RoFormer-SW output. Lead and backing use the strongest\n        # individually validated Disturbia recipes.\n        emit("LiteLABS-VX Vocal Separation", 47)\n        sw_vocals_audio, vocal_sr = _read(stems["vocals"])\n\n        alt_parent_out = root / "vocal_alt_parent"\n        quality_out = root / "vocal_quality_karaoke"\n        fast_out = root / "vocal_fast_karaoke"\n        for directory in (alt_parent_out, quality_out, fast_out):\n            directory.mkdir(parents=True, exist_ok=True)\n\n        def _vx_find_output(directory, role):
            candidates = [
                p for p in directory.rglob("*")
                if p.is_file() and p.suffix.lower() in {".wav", ".flac", ".mp3"}
            ]
            role = str(role).lower()
            if role == "vocals":
                preferred = [
                    p for p in candidates
                    if "(vocals)" in p.name.lower()
                    or p.stem.lower() == "vocals"
                    or "_vocals" in p.stem.lower()
                ]
            elif role == "secondary":
                # Locked Disturbia benchmark: for Becruily Karaoke the
                # secondary / nominal Instrumental output is the lead vocal.
                preferred = [
                    p for p in candidates
                    if "(instrumental)" in p.name.lower()
                    or p.stem.lower() == "instrumental"
                    or "_instrumental" in p.stem.lower()
                ]
            else:
                raise ValueError(f"Unknown LiteLABS-VX output role: {role}")
            return max(preferred, key=lambda p: p.stat().st_size) if preferred else None

        def _vx_run(model_name, input_path, output_dir, stage_name, start_percent, end_percent, output_role):
            cmd = [
                "audio-separator", str(input_path),
                "--model_filename", model_name,
                "--model_file_dir", str(audio_separator_model_dir),
                "--output_dir", str(output_dir),
                "--output_format", "FLAC",
                "--mdxc_segment_size", "256",
                "--mdxc_overlap", "8",
                "--mdxc_batch_size", "1",
                "--use_autocast",
            ]
            rc, elapsed = _run_polled(
                cmd, cwd=None, timeout=timeout, log_path=logs / f"{stage_name}.log", progress=progress,
                stage_name="LiteLABS-VX Vocal Separation", start_percent=start_percent, end_percent=end_percent,
                heartbeat_seconds=heartbeat,
            )
            if rc != 0:
                raise RuntimeError(f"LiteLABS-VX stage failed: {stage_name}")
            selected_path = _vx_find_output(output_dir, output_role)
            if selected_path is None:
                files = [p.name for p in output_dir.rglob("*") if p.is_file()]
                raise RuntimeError(
                    f"LiteLABS-VX stage produced no {output_role} output: {stage_name}; files={files}"
                )
            return selected_path, elapsed

        # Alternate vocal parent used only for the high-quality lead blend.\n        alt_path, alt_elapsed = _vx_run(\n            "mel_band_roformer_vocals_becruily.ckpt", source, alt_parent_out,\n            "vx_alt_parent", 48, 52, "vocals",\n        )\n        alt_vocals_audio, _ = _read(alt_path)\n        blend_n = min(len(sw_vocals_audio), len(alt_vocals_audio))\n        blended_parent = (\n            0.25 * np.asarray(sw_vocals_audio[:blend_n], dtype=np.float32)\n            + 0.75 * np.asarray(alt_vocals_audio[:blend_n], dtype=np.float32)\n        )\n        blended_parent_path = root / "vocal_parent_sw25_mel75.flac"\n        _write_flac(blended_parent_path, blended_parent, vocal_sr)\n\n        # Best lead benchmark.\n        quality_lead_path, quality_elapsed = _vx_run(\n            "mel_band_roformer_karaoke_becruily.ckpt", blended_parent_path, quality_out,\n            "vx_quality_lead", 52, 56, "secondary",\n        )\n        quality_lead, _ = _read(quality_lead_path)\n\n        # Best backing benchmark. Run the same karaoke model on the stronger SW\n        # parent and retain its exact residual as backing.\n        fast_lead_path, fast_elapsed = _vx_run(\n            "mel_band_roformer_karaoke_becruily.ckpt", stems["vocals"], fast_out,\n            "vx_fast_backing", 56, 59, "secondary",\n        )\n        fast_lead, _ = _read(fast_lead_path)\n        fast_n = min(len(sw_vocals_audio), len(fast_lead))\n        fast_parent = np.asarray(sw_vocals_audio[:fast_n], dtype=np.float32)\n        fast_lead = np.asarray(fast_lead[:fast_n], dtype=np.float32)\n        best_backing = fast_parent - fast_lead\n\n        lead_n = min(len(quality_lead), blend_n)\n        best_lead = np.asarray(quality_lead[:lead_n], dtype=np.float32)\n        lead_dest = experimental / f"{track}_lead_vocals.flac"\n        backing_dest = experimental / f"{track}_backing_vocals.flac"\n        _write_flac(lead_dest, best_lead, vocal_sr)\n        _write_flac(backing_dest, best_backing, vocal_sr)\n\n        fast_rebuilt = fast_lead + best_backing\n        fast_residual = fast_parent - fast_rebuilt\n        fast_parent_rms = float(np.sqrt(np.mean(fast_parent * fast_parent) + 1e-12))\n        fast_residual_rms = float(np.sqrt(np.mean(fast_residual * fast_residual) + 1e-12))\n\n        timings["vocal_alt_parent"] = round(alt_elapsed, 3)\n        timings["vocal_quality_karaoke"] = round(quality_elapsed, 3)\n        timings["vocal_fast_karaoke"] = round(fast_elapsed, 3)\n        timings["karaoke_bs_roformer"] = round(alt_elapsed + quality_elapsed + fast_elapsed, 3)\n        vocal_files = [lead_dest.name, backing_dest.name]\n        vocal_report = {\n            "ok": True,\n            "benchmark_id": "vocal_benchmark_v1",\n            "files": vocal_files,\n            "parent_recipe": "BS-RoFormer-SW vocals",\n            "lead_recipe": "25% SW + 75% MelBand Becruily vocal parent -> Becruily karaoke secondary/Instrumental",\n            "backing_recipe": "SW vocals - fast SW->Becruily karaoke secondary/Instrumental",\n            "best_stems_share_single_parent_pair": False,\n            "fast_backing_route_parent_reconstruction_cosine": round(float(_cos(fast_parent, fast_rebuilt)), 9),\n            "fast_backing_route_residual_relative_to_parent_db": _db(fast_residual_rms / max(fast_parent_rms, 1e-12)),\n        }\n\n'''
text = text[:start] + vocal_block + text[end:]

# Update internal model metadata if the historical label is present.
text = text.replace(
    '"lead_backing_vocals": "BS-RoFormer Karaoke (frazer/becruily)",',
    '"lead_backing_vocals": "LiteLABS locked vocal benchmark v1 (SW + Becruily MelBand)",',
    1,
)

# ---------------------------------------------------------------------------
# Drum output policy: DrumSep still predicts hh+cymbals in the same inference
# pass. Merge them in memory before final FLAC encoding; no extra model pass.
# ---------------------------------------------------------------------------
old_drum_export = '''                for name in DRUM5:\n                    dest = experimental / f"{track}_drums_5stem_{name}.flac"\n                    _write_flac(dest, refined[name], drum_sr)\n                    drum_report["files"].append(dest.name)\n'''
new_drum_export = '''                merge_started = time.monotonic()\n                final_children = {\n                    "kick": refined["kick"],\n                    "snare": refined["snare"],\n                    "toms": refined["toms"],\n                    "hats": refined["hh"] + refined["cymbals"],\n                }\n                merged_sum = np.sum(np.stack(list(final_children.values()), axis=0), axis=0)\n                merged_residual = parent - merged_sum\n                merged_residual_rms = float(np.sqrt(np.mean(merged_residual * merged_residual) + 1e-12))\n                drum_report["hats_merge"] = {\n                    "applied": True,\n                    "method": "in_memory_hh_plus_cymbals_before_final_encode",\n                    "source_children": ["hh", "cymbals"],\n                    "output_child": "hats",\n                    "parent_vs_final_children_sum_cosine": round(float(_cos(parent, merged_sum)), 6),\n                    "residual_relative_to_parent_db": _db(merged_residual_rms / max(parent_rms, 1e-12)),\n                }\n                for name, audio in final_children.items():\n                    dest = experimental / f"{track}_drums_5stem_{name}.flac"\n                    _write_flac(dest, audio, drum_sr)\n                    drum_report["files"].append(dest.name)\n                timings["drum_hats_merge_and_encode"] = round(time.monotonic() - merge_started, 3)\n'''
if old_drum_export not in text:
    if 'in_memory_hh_plus_cymbals_before_final_encode' not in text:
        raise RuntimeError('Could not locate DrumSep final child export loop')
else:
    text = text.replace(old_drum_export, new_drum_export, 1)

exp_path.write_text(text, encoding='utf-8')

# Keep the advertised experimental preset aligned with the actual output pack.
preset = preset_path.read_text(encoding='utf-8')
preset = preset.replace(
    '"lead_vocals", "backing_vocals", "kick", "snare", "toms", "hi_hats",\n        "cymbals", "bass",',
    '"lead_vocals", "backing_vocals", "kick", "snare", "toms", "hats", "bass",',
    1,
)
preset = preset.replace('    "hi_hats": "Hi-Hats",\n    "cymbals": "Cymbals",\n', '    "hats": "Hats",\n', 1)
if '"hats": "Hats"' not in preset:
    raise RuntimeError('Could not update experimental preset hats capability')
preset_path.write_text(preset, encoding='utf-8')

print('LiteLABS locked vocal benchmark v1 and in-memory hats output applied')
