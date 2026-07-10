from pathlib import Path

path = Path('/app/research_tools.py')
text = path.read_text(encoding='utf-8')
start = text.index('def build_model_bakeoff(')
end = text.index('\ndef build_vocal_residual_test(', start)
replacement = r'''def build_model_bakeoff(input_path: Path, output_root: Path, filename: str, models: list | None = None, output_format: str = "flac", progress=None) -> dict:
    """Run a model bake-off and package a lean review archive.

    The first research version zipped the whole working directory, including raw
    intermediate WAV files and duplicate extracted/output folders. That made the
    first test archive nearly 2GB. This version keeps the working files for the
    duration of the job, but only packages the source, final stems, run_result
    files, README, and research_report.json.
    """
    track = safe_track_name(filename)
    output_format = (output_format or "flac").lower().strip()
    if output_format not in {"flac", "mp3"}:
        output_format = "flac"

    work_dir = output_root / f"{track}-model-bakeoff-work"
    review_dir = output_root / f"{track}-model-bakeoff-review"
    work_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, review_dir / f"00_source_{input_path.name}")

    specs = normalise_model_specs(models)
    runs: list[dict] = []
    total = max(1, len(specs))
    audio_exts = {".wav", ".flac", ".mp3", ".m4a"}

    def copy_review_outputs(run_dir: Path, review_run_dir: Path) -> list[str]:
        copied: list[str] = []
        candidates: list[Path] = []

        extracted = run_dir / "extracted"
        if extracted.exists():
            # Current LiteLABS extracts the final user-facing pack here.
            candidates = [p for p in sorted(extracted.rglob("*")) if p.is_file() and (p.suffix.lower() in audio_exts or p.name == "README.txt")]
        else:
            for folder_name in ["demucs_output", "audio_separator_output"]:
                folder = run_dir / folder_name
                if folder.exists():
                    candidates.extend([p for p in sorted(folder.rglob("*")) if p.is_file() and p.suffix.lower() in audio_exts])

        for source in candidates:
            if extracted.exists():
                relative = source.relative_to(extracted)
                # Strip the generated pack folder for readability when possible.
                if len(relative.parts) > 1:
                    relative = Path(*relative.parts[1:])
            else:
                relative = source.relative_to(run_dir)
            dest = review_run_dir / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            copied.append(str(dest.relative_to(review_dir)))
        return copied

    for index, spec in enumerate(specs, start=1):
        label = model_label(spec)
        if progress:
            progress(f"Running research model {index}/{total}: {label}", int(10 + (index - 1) * (75 / total)))
        run_dir = work_dir / f"{index:02d}_{safe_folder_name(label)}"
        review_run_dir = review_dir / f"{index:02d}_{safe_folder_name(label)}"
        run_dir.mkdir(parents=True, exist_ok=True)
        review_run_dir.mkdir(parents=True, exist_ok=True)

        result = run_model_spec(input_path, run_dir, spec, output_format, progress=progress)
        result["review_files"] = copy_review_outputs(run_dir, review_run_dir) if result.get("ok") else []
        result["metrics"] = collect_audio_metrics(review_run_dir) if review_run_dir.exists() else []
        write_json(review_run_dir / "run_result.json", result)
        runs.append(result)

    report = {
        "track": track,
        "source_file": input_path.name,
        "output_format": output_format,
        "archive_type": "lean_review_zip",
        "models_requested": specs,
        "runs": runs,
        "system_info": build_system_info(),
    }
    write_json(review_dir / "research_report.json", report)

    lines = [
        "LiteLABS research model bake-off",
        "",
        f"Track: {track}",
        f"Source: {input_path.name}",
        f"Models tested: {len(specs)}",
        "Archive: lean review ZIP (source + final stems + reports only)",
        "",
        "Runs:",
    ]
    for result in runs:
        status = "OK" if result.get("ok") else "FAILED"
        runtime = result.get("runtime_seconds", "?")
        review_count = len(result.get("review_files") or [])
        lines.append(f"- {result.get('label')}: {status} ({runtime}s, {review_count} review files)")
        if not result.get("ok"):
            lines.append(f"  Error: {result.get('error')}")
    lines.extend([
        "",
        "Use research_report.json for detailed metrics. The audio files are grouped by model folder.",
    ])
    write_text(review_dir / "README.txt", "\n".join(lines) + "\n")

    archive = output_root / f"{track}-model-bakeoff-review.zip"
    if progress:
        progress("Creating lean research ZIP", 92)
    zip_folder(review_dir, archive, review_dir.name)

    return {
        "track": track,
        "archive_path": str(archive),
        "archive_size_bytes": archive.stat().st_size,
        "files": sorted(str(p.relative_to(review_dir)) for p in review_dir.rglob("*") if p.is_file()),
        "runs": [{
            "label": r.get("label"),
            "ok": r.get("ok"),
            "runtime_seconds": r.get("runtime_seconds"),
            "error": r.get("error"),
            "review_files": r.get("review_files", []),
        } for r in runs],
    }
'''
path.write_text(text[:start] + replacement + text[end:], encoding='utf-8')
print('Applied lean research bakeoff archive patch')
