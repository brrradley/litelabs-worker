from pathlib import Path

path = Path('/app/handler.py')
text = path.read_text()

text = text.replace('import tempfile\nimport zipfile', 'import tempfile\nimport zipfile\nimport shutil')

text = text.replace(
    'def run_audio_separator(input_file: Path, output_dir: Path, model_filename: str, output_format: str) -> list[Path]:',
    'def run_audio_separator(input_file: Path, output_dir: Path, model_filename: str, output_format: str, extra_args: list[str] | None = None) -> list[Path]:',
)
text = text.replace(
    '    print("LiteLABS extra vocals RUN:", " ".join(cmd), flush=True)',
    '    if extra_args:\n        cmd.extend(extra_args)\n    print("LiteLABS extra vocals RUN:", " ".join(cmd), flush=True)',
)

text = text.replace(
    '    dance_tempo = 118.0 <= tempo <= 136.0\n    if (dance_tempo and percussive >= 0.48 and bass >= 0.13) or (percussive >= 0.62 and bass >= 0.15):',
    '    rap_tempo = 75.0 <= tempo <= 115.0\n    if rap_tempo and 0.32 <= percussive <= 0.58 and bass >= 0.13:\n        reason = "source audio has rap/hip-hop-like rhythm profile"\n        if tempo:\n            reason += f" ({tempo:.0f} BPM, percussive {percussive:.2f})"\n        return "hip_hop_rap", reason\n    dance_tempo = 118.0 <= tempo <= 136.0\n    if (dance_tempo and percussive >= 0.48 and bass >= 0.13) or (percussive >= 0.62 and bass >= 0.15):',
)

text = text.replace(
    '        genre_override, genre_reason = source_genre_override(source_features)',
    '        genre_override, genre_reason = source_genre_override(source_features)\n        tempo = float(source_features.get("tempo", 0.0) or 0.0)\n        percussive = float(source_features.get("percussive_ratio", 0.0) or 0.0)\n        bass = float(source_features.get("bass_ratio", 0.0) or 0.0)\n        if 75.0 <= tempo <= 115.0 and 0.32 <= percussive <= 0.58 and bass >= 0.13:\n            genre_override = "hip_hop_rap"\n            genre_reason = f"source audio has rap/hip-hop-like rhythm profile ({tempo:.0f} BPM, percussive {percussive:.2f})"',
)

block_start = text.find('    try:\n        backing_outputs = run_audio_separator(vocal_file, temp_root / "backing", model_backing, output_format)')
block_end = text.find('    append_readme_notes(readme, included_notes, omitted_notes)', block_start)
if block_start == -1 or block_end == -1:
    raise SystemExit('extra vocal block not found')

new_block = '''    try:
        backing_args = ["--vr_aggression", os.getenv("LITELABS_BACKING_VR_AGGRESSION", "16"), "--vr_enable_post_process", "--vr_post_process_threshold", os.getenv("LITELABS_BACKING_POST_THRESHOLD", "0.10")]
        backing_outputs = run_audio_separator(vocal_file, temp_root / "backing", model_backing, output_format, backing_args)
        backing_candidate = next((p for p in backing_outputs if "(vocals)" in p.name.lower()), None)
        if not backing_candidate:
            backing_candidate = next((p for p in backing_outputs if "vocals" in p.name.lower() and "instrumental" not in p.name.lower()), None)

        if backing_candidate and backing_candidate.exists():
            useful, reason = is_useful_extra_vocal(backing_candidate, 0.20, 0.03)
            if useful:
                dest = master_dir / f"{output_index:02d}_{vocal_file.stem.replace('_vocals', '')}_backing_vocals.{output_format}"
                master_pack.copy_or_convert_audio(backing_candidate, dest, output_format)
                included_notes.append(f"{output_index:02d} Backing Vocals")
                changes.append("added Backing Vocals")
                output_index += 1
            else:
                omitted_notes.append(f"Backing Vocals — {reason}")
        else:
            omitted_notes.append("Backing Vocals — model did not produce a backing vocal file")
    except Exception as exc:
        print(f"LiteLABS backing vocal pass skipped: {exc}", flush=True)
        omitted_notes.append("Backing Vocals — experimental backing vocal pass failed for this track")
        changes.append("backing vocal pass skipped")

    omitted_notes.append("Dry Main Vocals — disabled while the dereverb model is reviewed")
    changes.append("dry vocal pass disabled")
    shutil.rmtree(temp_root, ignore_errors=True)
'''

text = text[:block_start] + new_block + text[block_end:]
path.write_text(text)
