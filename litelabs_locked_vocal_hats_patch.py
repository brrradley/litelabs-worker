from pathlib import Path

exp_path = Path('/app/experimental_children_v1.py')
preset_path = Path('/app/preset_pack.py')
text = exp_path.read_text(encoding='utf-8')

# ---------------------------------------------------------------------------
# LiteLABS-VX v2 production route.
# One karaoke pass operates only on the clean SW vocal parent. The model's
# native vocals output is lead and its instrumental output is backing.
# ---------------------------------------------------------------------------
start_marker = '        # Split the already-clean RoFormer vocal parent with a dedicated\n'
end_marker = '        # Mega53 is the routing brain:'
start = text.find(start_marker)
end = text.find(end_marker, start)
if start < 0 or end < 0:
    raise RuntimeError('Could not locate existing vocal specialist block')

vocal_block = '''        # LiteLABS-VX v2: one-pass lead/backing separation from the already-clean
        # BS-RoFormer vocal parent. Never feed the full mix into the karaoke model:
        # doing so was the source of music-heavy "lead vocal" failures.
        emit("LiteLABS-VX Vocal Separation", 47)
        sw_vocals_audio, vocal_sr = _read(stems["vocals"])

        vocal_out = root / "vocal_lead_backing"
        vocal_out.mkdir(parents=True, exist_ok=True)

        def _vx_find_output(directory, role):
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
            elif role == "instrumental":
                preferred = [
                    p for p in candidates
                    if "(instrumental)" in p.name.lower()
                    or p.stem.lower() == "instrumental"
                    or "_instrumental" in p.stem.lower()
                ]
            else:
                raise ValueError(f"Unknown LiteLABS-VX output role: {role}")
            return max(preferred, key=lambda p: p.stat().st_size) if preferred else None

        cmd = [
            "audio-separator", str(stems["vocals"]),
            "--model_filename", "mel_band_roformer_karaoke_becruily.ckpt",
            "--model_file_dir", str(audio_separator_model_dir),
            "--output_dir", str(vocal_out),
            "--output_format", "FLAC",
            "--mdxc_segment_size", "256",
            "--mdxc_overlap", "8",
            "--mdxc_batch_size", "1",
            "--use_autocast",
        ]
        rc_vx, vx_elapsed = _run_polled(
            cmd, cwd=None, timeout=timeout, log_path=logs / "vx_lead_backing.log",
            progress=progress, stage_name="LiteLABS-VX Vocal Separation",
            start_percent=48, end_percent=59, heartbeat_seconds=heartbeat,
        )
        if rc_vx != 0:
            raise RuntimeError("LiteLABS-VX lead/backing separation failed")

        lead_path = _vx_find_output(vocal_out, "vocals")
        backing_path = _vx_find_output(vocal_out, "instrumental")
        if lead_path is None or backing_path is None:
            files = [p.name for p in vocal_out.rglob("*") if p.is_file()]
            raise RuntimeError(
                f"LiteLABS-VX expected vocals+instrumental outputs; files={files}"
            )

        best_lead, _ = _read(lead_path)
        best_backing, _ = _read(backing_path)
        vocal_n = min(len(sw_vocals_audio), len(best_lead), len(best_backing))
        vocal_parent = np.asarray(sw_vocals_audio[:vocal_n], dtype=np.float32)
        best_lead = np.asarray(best_lead[:vocal_n], dtype=np.float32)
        best_backing = np.asarray(best_backing[:vocal_n], dtype=np.float32)

        lead_dest = experimental / f"{track}_lead_vocals.flac"
        backing_dest = experimental / f"{track}_backing_vocals.flac"
        _write_flac(lead_dest, best_lead, vocal_sr)
        _write_flac(backing_dest, best_backing, vocal_sr)

        rebuilt = best_lead + best_backing
        residual = vocal_parent - rebuilt
        parent_rms = float(np.sqrt(np.mean(vocal_parent * vocal_parent) + 1e-12))
        residual_rms = float(np.sqrt(np.mean(residual * residual) + 1e-12))

        timings["vocal_lead_backing"] = round(vx_elapsed, 3)
        timings["karaoke_bs_roformer"] = round(vx_elapsed, 3)
        vocal_files = [lead_dest.name, backing_dest.name]
        vocal_report = {
            "ok": True,
            "benchmark_id": "vocal_route_v2_single_pass",
            "files": vocal_files,
            "parent_recipe": "BS-RoFormer-SW vocals",
            "lead_recipe": "Becruily Karaoke vocals output from SW vocal parent",
            "backing_recipe": "Becruily Karaoke instrumental output from SW vocal parent",
            "best_stems_share_single_parent_pair": True,
            "parent_reconstruction_cosine": round(float(_cos(vocal_parent, rebuilt)), 9),
            "residual_relative_to_parent_db": _db(
                residual_rms / max(parent_rms, 1e-12)
            ),
        }

'''
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
old_drum_export = '''                for name in DRUM5:\n                    dest = experimental / f"{track}_{name}.flac"\n                    _write_flac(dest, refined[name], drum_sr)\n                    drum_report["files"].append(dest.name)\n'''
new_drum_export = '''                merge_started = time.monotonic()\n                final_children = {\n                    "kick": refined["kick"],\n                    "snare": refined["snare"],\n                    "toms": refined["toms"],\n                    "hats": refined["hh"] + refined["cymbals"],\n                }\n                merged_sum = np.sum(np.stack(list(final_children.values()), axis=0), axis=0)\n                merged_residual = parent - merged_sum\n                merged_residual_rms = float(np.sqrt(np.mean(merged_residual * merged_residual) + 1e-12))\n                drum_report["hats_merge"] = {\n                    "applied": True,\n                    "method": "in_memory_hh_plus_cymbals_before_final_encode",\n                    "source_children": ["hh", "cymbals"],\n                    "output_child": "hats",\n                    "parent_vs_final_children_sum_cosine": round(float(_cos(parent, merged_sum)), 6),\n                    "residual_relative_to_parent_db": _db(merged_residual_rms / max(parent_rms, 1e-12)),\n                }\n                for name, audio in final_children.items():\n                    dest = experimental / f"{track}_{name}.flac"\n                    _write_flac(dest, audio, drum_sr)\n                    drum_report["files"].append(dest.name)\n                timings["drum_hats_merge_and_encode"] = round(time.monotonic() - merge_started, 3)\n'''
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

print('LiteLABS-VX v2 single-pass vocals and in-memory hats output applied')
