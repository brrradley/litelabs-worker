from pathlib import Path

path = Path('/app/mss_candidate_lab.py')
text = path.read_text(encoding='utf-8')

marker = '\n\ndef _benchmark_candidates(payload: dict, progress=None) -> dict:\n'
helper = '''\n\ndef _run_bs_baseline(audio_url: str, timeout_seconds: int = 1800) -> dict:
    model_dir = Path(os.getenv("STEMFORGE_MODEL_DIR", "/models/bs_roformer_sw"))
    config = model_dir / "BS-Roformer-SW.yaml"
    checkpoint = model_dir / "BS-Roformer-SW.ckpt"
    if not config.is_file() or not checkpoint.is_file():
        return {
            "ok": False,
            "error": "Current LiteLABS BS baseline files are missing",
            "config_path": str(config),
            "checkpoint_path": str(checkpoint),
        }
    with tempfile.TemporaryDirectory(prefix="litelabs_bs_baseline_") as temp:
        root = Path(temp)
        input_dir = root / "input"
        output_dir = root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        filename = unquote(Path(urlparse(audio_url).path).name) or "track.flac"
        downloaded_source = root / filename
        wav_source = input_dir / f"{Path(filename).stem}.wav"
        _download(audio_url, downloaded_source)
        conversion = subprocess.run(
            ["ffmpeg", "-y", "-i", str(downloaded_source), "-ar", "44100", "-ac", "2", str(wav_source)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
        )
        if conversion.returncode != 0 or not wav_source.is_file():
            return {
                "ok": False,
                "return_code": conversion.returncode,
                "error": "Failed to convert baseline input to WAV",
                "runtime_log": "\\n".join((conversion.stdout or "").splitlines()[-80:]),
                "output_files": [],
                "piano_metrics": None,
                "other_metrics": None,
            }
        command = [
            "bs-roformer-infer",
            "--config_path", str(config),
            "--model_path", str(checkpoint),
            "--input_folder", str(input_dir),
            "--store_dir", str(output_dir),
        ]
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
        )
        outputs = sorted(path for path in output_dir.rglob("*") if path.is_file())
        piano_matches = [path for path in outputs if "piano" in path.name.lower()]
        other_matches = [path for path in outputs if "other" in path.name.lower()]
        return {
            "ok": completed.returncode == 0 and bool(piano_matches) and bool(other_matches),
            "return_code": completed.returncode,
            "piano_metrics": _candidate_audio_metrics(piano_matches[0], wav_source) if piano_matches else None,
            "other_metrics": _candidate_audio_metrics(other_matches[0], wav_source) if other_matches else None,
            "runtime_log": "\\n".join((completed.stdout or "").splitlines()[-80:]),
            "output_files": [str(path.relative_to(output_dir)) for path in outputs],
        }
'''
if 'def _run_bs_baseline(' not in text:
    if marker not in text:
        raise RuntimeError('Could not locate candidate benchmark helper')
    text = text.replace(marker, helper + marker, 1)

old = '''    if progress:
        progress("Candidate benchmark complete", 100)
    return {
        "ok": True,
        "mode": "mss_candidate_lab",
        "schema_version": 3,
        "action": "benchmark",
        "track_url": audio_url,
'''
new = '''    if progress:
        progress("Running current LiteLABS baseline", 82)
    baseline = _run_bs_baseline(audio_url, timeout_seconds=int(payload.get("timeout_seconds") or 1800))
    if not baseline.get("ok"):
        return {"ok": False, "mode": "mss_candidate_lab", "action": "benchmark", "failed_stage": "baseline", "result": baseline}
    if progress:
        progress("Candidate benchmark complete", 100)
    return {
        "ok": True,
        "mode": "mss_candidate_lab",
        "schema_version": 4,
        "action": "benchmark",
        "track_url": audio_url,
'''
if old in text:
    text = text.replace(old, new, 1)
elif '"schema_version": 4' not in text:
    raise RuntimeError('Could not upgrade benchmark response')

old_tail = '''        "other": {
            "model_id": other["model"]["id"],
            "metrics": other.get("target_metrics"),
            "runtime_log": other.get("log_tail"),
            "auto_installed_on_worker": other.get("auto_installed_on_worker"),
        },
        "quality_warning": "These are reference-free sanity metrics, not proof of separation quality. Listening and compatible ground-truth tests remain required.",
        "next_action": "Compare these profiles with the current LiteLABS baseline and perform listening tests before promotion.",
'''
new_tail = '''        "other": {
            "model_id": other["model"]["id"],
            "metrics": other.get("target_metrics"),
            "runtime_log": other.get("log_tail"),
            "auto_installed_on_worker": other.get("auto_installed_on_worker"),
        },
        "baseline": {
            "model_id": "current-litelabs-bs-roformer-sw",
            "piano_metrics": baseline.get("piano_metrics"),
            "other_metrics": baseline.get("other_metrics"),
            "runtime_log": baseline.get("runtime_log"),
            "output_files": baseline.get("output_files"),
        },
        "comparison": {
            "piano_candidate_minus_baseline_rms_db": round(piano["target_metrics"]["rms_dbfs"] - baseline["piano_metrics"]["rms_dbfs"], 3),
            "other_candidate_minus_baseline_rms_db": round(other["target_metrics"]["rms_dbfs"] - baseline["other_metrics"]["rms_dbfs"], 3),
            "piano_candidate_minus_baseline_mixture_cosine": round(piano["target_metrics"]["mixture_cosine"] - baseline["piano_metrics"]["mixture_cosine"], 6),
            "other_candidate_minus_baseline_mixture_cosine": round(other["target_metrics"]["mixture_cosine"] - baseline["other_metrics"]["mixture_cosine"], 6),
        },
        "quality_warning": "These are reference-free sanity metrics, not proof of separation quality. Listening and compatible ground-truth tests remain required.",
        "next_action": "Use the direct baseline deltas to reject silent candidates and identify which surviving stems need listening tests.",
'''
if old_tail in text:
    text = text.replace(old_tail, new_tail, 1)
elif '"baseline": {' not in text:
    raise RuntimeError('Could not add baseline benchmark results')

path.write_text(text, encoding='utf-8')
print('MSS baseline comparison benchmark patch applied')
