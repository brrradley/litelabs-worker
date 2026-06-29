FROM ghcr.io/brrradley/litelabs-worker:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV LITELABS_AUDIO_SEPARATOR_MODEL_DIR=/models/audio_separator

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libsamplerate0-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install audio-separator==0.44.2 onnxruntime-gpu==1.22.0
RUN mkdir -p /models/audio_separator

RUN python - <<'PY'
from pathlib import Path
path = Path('/app/handler.py')
text = path.read_text()
text = text.replace('import tempfile\nimport zipfile', 'import tempfile\nimport zipfile\nimport shutil')
text = text.replace('model_dry = os.getenv("LITELABS_DRY_MODEL", "deverb_bs_roformer_8_256dim_8depth.ckpt")', 'model_dry = os.getenv("LITELABS_DRY_MODEL", "deverb_bs_roformer_8_384dim_10depth.ckpt")')
old = '''    try:
        backing_outputs = run_audio_separator(vocal_file, temp_root / "backing", model_backing, output_format)
        classified = classify_extra_vocal_outputs(backing_outputs)
        lead_candidate = classified.get("lead")
        backing_candidate = classified.get("backing")

        if lead_candidate and lead_candidate.exists():
            useful, reason = is_useful_extra_vocal(lead_candidate, 0.28, 0.08)
            if useful:
                dest = master_dir / f"{output_index:02d}_{vocal_file.stem.replace('_vocals', '')}_lead_vocals.{output_format}"
                master_pack.copy_or_convert_audio(lead_candidate, dest, output_format)
                included_notes.append(f"{output_index:02d} Lead Vocals")
                changes.append("added Lead Vocals")
                output_index += 1
            else:
                omitted_notes.append(f"Lead Vocals — {reason}")

        if backing_candidate and backing_candidate.exists():
            useful, reason = is_useful_extra_vocal(backing_candidate, 0.24, 0.04)
            if useful:
                dest = master_dir / f"{output_index:02d}_{vocal_file.stem.replace('_vocals', '')}_backing_vocals.{output_format}"
                master_pack.copy_or_convert_audio(backing_candidate, dest, output_format)
                included_notes.append(f"{output_index:02d} Backing Vocals")
                changes.append("added Backing Vocals")
                output_index += 1
            else:
                omitted_notes.append(f"Backing Vocals — {reason}")
        else:
            omitted_notes.append("Backing Vocals — model did not produce a confident backing vocal file")

        dry_source = lead_candidate if lead_candidate and lead_candidate.exists() else vocal_file
        dry_outputs = run_audio_separator(dry_source, temp_root / "dry", model_dry, output_format)
        dry_candidate = classify_extra_vocal_outputs(dry_outputs).get("dry") or (dry_outputs[0] if dry_outputs else None)
        if dry_candidate and dry_candidate.exists():
            useful, reason = is_useful_extra_vocal(dry_candidate, 0.28, 0.08)
            if useful:
                dest = master_dir / f"{output_index:02d}_{vocal_file.stem.replace('_vocals', '')}_lead_vocals_dry.{output_format}"
                master_pack.copy_or_convert_audio(dry_candidate, dest, output_format)
                included_notes.append(f"{output_index:02d} Lead Vocals Dry")
                changes.append("added Lead Vocals Dry")
                output_index += 1
            else:
                omitted_notes.append(f"Lead Vocals Dry — {reason}")
        else:
            omitted_notes.append("Lead Vocals Dry — dereverb model did not produce a confident dry vocal file")
    except Exception as exc:
        print(f"LiteLABS extra vocal pass skipped: {exc}", flush=True)
        omitted_notes.append("Extra vocal stems — experimental local vocal pass failed for this track")
        changes.append("extra vocal pass skipped")

    append_readme_notes(readme, included_notes, omitted_notes)
'''
new = '''    try:
        backing_outputs = run_audio_separator(vocal_file, temp_root / "backing", model_backing, output_format)
        backing_candidate = next((p for p in backing_outputs if "(vocals)" in p.name.lower()), None)
        if not backing_candidate:
            backing_candidate = next((p for p in backing_outputs if "vocals" in p.name.lower()), None)

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
        dry_candidate = classify_extra_vocal_outputs(dry_outputs).get("dry") or next((p for p in dry_outputs if "(vocals)" in p.name.lower()), None) or (dry_outputs[0] if dry_outputs else None)
        if dry_candidate and dry_candidate.exists():
            useful, reason = is_useful_extra_vocal(dry_candidate, 0.24, 0.06)
            if useful:
                dest = master_dir / f"{output_index:02d}_{vocal_file.stem.replace('_vocals', '')}_lead_vocals_dry.{output_format}"
                master_pack.copy_or_convert_audio(dry_candidate, dest, output_format)
                included_notes.append(f"{output_index:02d} Lead Vocals Dry")
                changes.append("added Lead Vocals Dry")
                output_index += 1
            else:
                omitted_notes.append(f"Lead Vocals Dry — {reason}")
        else:
            omitted_notes.append("Lead Vocals Dry — dereverb model did not produce a dry vocal file")
    except Exception as exc:
        print(f"LiteLABS dry vocal pass skipped: {exc}", flush=True)
        omitted_notes.append("Lead Vocals Dry — experimental dry vocal pass failed for this track")
        changes.append("dry vocal pass skipped")

    shutil.rmtree(temp_root, ignore_errors=True)
    append_readme_notes(readme, included_notes, omitted_notes)
'''
if old not in text:
    raise SystemExit('expected extra vocal block not found')
text = text.replace(old, new)
path.write_text(text)
PY

CMD ["python", "-u", "/app/handler.py"]
