from pathlib import Path

path = Path('/app/handler.py')
text = path.read_text()

text = text.replace('import tempfile\nimport zipfile', 'import tempfile\nimport zipfile\nimport shutil')
text = text.replace(
    'model_dry = os.getenv("LITELABS_DRY_MODEL", "deverb_bs_roformer_8_256dim_8depth.ckpt")',
    'model_dry = os.getenv("LITELABS_DRY_MODEL", "deverb_bs_roformer_8_384dim_10depth.ckpt")',
)
text = text.replace(
    'def run_audio_separator(input_file: Path, output_dir: Path, model_filename: str, output_format: str) -> list[Path]:',
    'def run_audio_separator(input_file: Path, output_dir: Path, model_filename: str, output_format: str, extra_args: list[str] | None = None) -> list[Path]:',
)
text = text.replace(
    '    print("LiteLABS extra vocals RUN:", " ".join(cmd), flush=True)',
    '    if extra_args:\n        cmd.extend(extra_args)\n    print("LiteLABS extra vocals RUN:", " ".join(cmd), flush=True)',
)
text = text.replace(
    '    if (dance_tempo and percussive >= 0.48 and bass >= 0.13) or (percussive >= 0.62 and bass >= 0.15):',
    '    rap_tempo = 75.0 <= tempo <= 115.0\n    if rap_tempo and 0.32 <= percussive <= 0.58 and bass >= 0.13:\n        reason = "source audio has rap/hip-hop-like rhythm profile"\n        if tempo:\n            reason += f" ({tempo:.0f} BPM, percussive {percussive:.2f})"\n        return "hip_hop_rap", reason\n    if (dance_tempo and percussive >= 0.48 and bass >= 0.13) or (percussive >= 0.62 and bass >= 0.15):',
)
text = text.replace(
    '        genre_override, genre_reason = source_genre_override(source_features)',
    '        genre_override, genre_reason = source_genre_override(source_features)\n        tempo = float(source_features.get("tempo", 0.0) or 0.0)\n        percussive = float(source_features.get("percussive_ratio", 0.0) or 0.0)\n        bass = float(source_features.get("bass_ratio", 0.0) or 0.0)\n        if 75.0 <= tempo <= 115.0 and 0.32 <= percussive <= 0.58 and bass >= 0.13:\n            genre_override = "hip_hop_rap"\n            genre_reason = f"source audio has rap/hip-hop-like rhythm profile ({tempo:.0f} BPM, percussive {percussive:.2f})"',
)

start = text.find('def classify_extra_vocal_outputs(files: list[Path]) -> dict[str, Path]:')
end = text.find('\ndef is_useful_extra_vocal', start)
if start != -1 and end != -1:
    text = text[:start] + '''def classify_extra_vocal_outputs(files: list[Path]) -> dict[str, Path]:
    classified: dict[str, Path] = {}
    for file in files:
        lower = file.name.lower()
        normalised = lower.replace("_", " ").replace("-", " ")
        if any(token in lower for token in ["backing", "backing_only", "back_vocal", "bv_vocal", "_bv", "-bv"]):
            classified.setdefault("backing", file)
        elif any(token in lower for token in ["lead", "lead_only", "main_vocal", "main vocals"]):
            classified.setdefault("lead", file)
        elif any(token in normalised for token in ["dry", "dereverb", "deverb", "no reverb", "no echo", "noreverb"]):
            classified.setdefault("dry", file)
        elif "reverb" in normalised or "echo" in normalised:
            continue
        elif "vocals" in lower or "vocal" in lower:
            classified.setdefault("dry", file)
    return classified


def pick_dry_vocal_output(files: list[Path]) -> Path | None:
    if not files:
        return None
    accepted = []
    for file in files:
        lower = file.name.lower()
        normalised = lower.replace("_", " ").replace("-", " ")
        is_reverb_residual = ("reverb" in normalised or "echo" in normalised) and not ("no reverb" in normalised or "no echo" in normalised)
        if is_reverb_residual or "instrumental" in lower:
            continue
        if "no reverb" in normalised or "no echo" in normalised:
            accepted.append((0, -file.stat().st_size, file))
        elif "dry" in normalised or "dereverb" in normalised or "deverb" in normalised:
            accepted.append((1, -file.stat().st_size, file))
    if not accepted:
        return None
    return sorted(accepted)[0][2]


def is_valid_dry_main_vocal(source: Path, dry: Path) -> tuple[bool, str]:
    try:
        import librosa
        import numpy as np
        src, sr = librosa.load(source, sr=22050, mono=True)
        out, _ = librosa.load(dry, sr=22050, mono=True)
        size = min(src.size, out.size)
        if size < sr:
            return False, "too short to validate"
        src = src[:size]
        out = out[:size]
        src_rms = float(np.sqrt(np.mean(src ** 2)) + 1e-9)
        out_rms = float(np.sqrt(np.mean(out ** 2)) + 1e-9)
        ratio = out_rms / src_rms
        corr = float(np.corrcoef(src, out)[0, 1]) if size > 2 else 0.0
        if ratio < 0.45:
            return False, "looked like residual/reverb rather than main vocal"
        if corr < 0.55:
            return False, "too different from main vocal / likely residual"
        return True, "validated"
    except Exception as exc:
        print(f"LiteLABS dry validation skipped: {exc}", flush=True)
        return True, "validation skipped"

''' + text[end+1:]

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

    try:
        dry_outputs = run_audio_separator(vocal_file, temp_root / "dry", model_dry, output_format)
        dry_candidate = pick_dry_vocal_output(dry_outputs)
        if dry_candidate and dry_candidate.exists():
            valid_dry, dry_reason = is_valid_dry_main_vocal(vocal_file, dry_candidate)
            useful, reason = is_useful_extra_vocal(dry_candidate, 0.24, 0.06)
            if useful and valid_dry:
                dest = master_dir / f"{output_index:02d}_{vocal_file.stem.replace('_vocals', '')}_dry_main_vocals.{output_format}"
                master_pack.copy_or_convert_audio(dry_candidate, dest, output_format)
                included_notes.append(f"{output_index:02d} Dry Main Vocals")
                changes.append("added Dry Main Vocals")
                output_index += 1
            else:
                omitted_notes.append(f"Dry Main Vocals — {dry_reason if not valid_dry else reason}")
        else:
            omitted_notes.append("Dry Main Vocals — dereverb model only produced reverb/residual, not a dry main vocal")
    except Exception as exc:
        print(f"LiteLABS dry vocal pass skipped: {exc}", flush=True)
        omitted_notes.append("Dry Main Vocals — experimental dry vocal pass failed for this track")
        changes.append("dry vocal pass skipped")

    shutil.rmtree(temp_root, ignore_errors=True)
'''
text = text[:block_start] + new_block + text[block_end:]
path.write_text(text)
