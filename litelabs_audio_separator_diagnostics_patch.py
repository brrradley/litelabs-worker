from pathlib import Path

path = Path('/app/research_tools.py')
text = path.read_text(encoding='utf-8')
start = text.index('def run_audio_separator_model(')
end = text.index('\n\ndef run_model_spec(', start)
replacement = r'''def run_audio_separator_model(input_path: Path, scratch_dir: Path, review_dir: Path, model_filename: str, output_format: str) -> dict:
    output_dir = scratch_dir / "audio_separator_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(os.getenv("LITELABS_AUDIO_SEPARATOR_MODEL_DIR", "/models/audio_separator"))
    model_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "audio-separator", str(input_path),
        "--model_filename", str(model_filename),
        "--model_file_dir", str(model_dir),
        "--output_dir", str(output_dir),
        "--output_format", output_format.upper(),
    ]
    print("\nRUN AUDIO-SEPARATOR:", " ".join(cmd), flush=True)
    completed = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=None,
    )
    command_output = completed.stdout or ""
    if command_output:
        print(command_output[-8000:], flush=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "audio-separator failed with exit code "
            f"{completed.returncode} for model {model_filename}. Output tail:\n"
            f"{command_output[-4000:]}"
        )

    review_files = copy_review_files(output_dir, review_dir / "audio_separator_output")
    metrics = collect_audio_metrics(review_dir)
    return {
        "model_filename": model_filename,
        "stems": [m["relative_path"] for m in metrics],
        "metrics": metrics,
        "review_files": review_files,
        "command_output_tail": command_output[-4000:],
    }
'''
path.write_text(text[:start] + replacement + text[end:], encoding='utf-8')
print('Applied audio-separator diagnostics patch')
