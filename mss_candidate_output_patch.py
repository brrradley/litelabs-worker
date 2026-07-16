from pathlib import Path

path = Path('/app/mss_candidate_lab.py')
text = path.read_text(encoding='utf-8')

for import_line, anchor in [
    ('import zipfile\n', 'import time\n'),
    ('import numpy as np\n', 'import requests\n'),
    ('import soundfile as sf\n', 'import requests\n'),
]:
    if import_line not in text:
        text = text.replace(anchor, anchor + import_line, 1)

upload_marker = '\n\ndef _load_registry() -> list[dict]:\n'
upload_helper = '''\n\ndef _upload_put(url: str, source: Path) -> None:
    with source.open("rb") as handle:
        response = requests.put(url, data=handle, headers={"Content-Type": "application/zip"}, timeout=(30, 900))
    response.raise_for_status()


def _candidate_audio_metrics(path: Path, source_path: Path) -> dict:
    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    source, source_sr = sf.read(source_path, always_2d=True, dtype="float32")
    if sr != source_sr:
        raise RuntimeError(f"Sample-rate mismatch: {sr} vs {source_sr}")
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    if source.shape[1] == 1:
        source = np.repeat(source, 2, axis=1)
    length = min(len(audio), len(source))
    audio = audio[:length, :2].astype(np.float64)
    source = source[:length, :2].astype(np.float64)
    flat = audio.reshape(-1)
    mix = source.reshape(-1)
    rms = float(np.sqrt(np.mean(flat * flat) + 1e-12))
    peak = float(np.max(np.abs(flat)))
    active = float(np.mean(np.abs(flat) > max(rms * 0.1, 1e-5)))
    cosine = float(np.dot(flat, mix) / (np.linalg.norm(flat) * np.linalg.norm(mix) + 1e-12))
    return {
        "duration_seconds": round(length / sr, 3),
        "sample_rate": int(sr),
        "rms_dbfs": round(20.0 * np.log10(rms + 1e-12), 3),
        "peak_dbfs": round(20.0 * np.log10(peak + 1e-12), 3),
        "active_ratio": round(active, 6),
        "mixture_cosine": round(cosine, 6),
    }
'''
if 'def _upload_put(' not in text:
    if upload_marker not in text:
        raise RuntimeError('Could not locate registry marker for helpers')
    text = text.replace(upload_marker, upload_helper + upload_marker, 1)
elif 'def _candidate_audio_metrics(' not in text:
    insert_at = text.index(upload_marker)
    text = text[:insert_at] + upload_helper.split('\n\ndef _upload_put', 1)[0] + text[insert_at:]
    metric_code = upload_helper[upload_helper.index('\n\ndef _candidate_audio_metrics'):]
    text = text[:insert_at] + metric_code + text[insert_at:]

auto_old = '''    if not model["research_ready"]:
        return {"ok": False, "mode": "mss_candidate_lab", "action": "run", "error": "Model is not research-ready", "model": model}

    audio_url = str(payload.get("audio_url") or payload.get("source_url") or "").strip()
'''
auto_new = '''    auto_installed = False
    if not model["research_ready"]:
        missing_files_only = bool(model.get("validation_errors")) and all(
            str(error).startswith("missing config:") or str(error).startswith("missing checkpoint:")
            for error in model.get("validation_errors", [])
        )
        if missing_files_only:
            if progress:
                progress(f"Installing missing candidate files for {model_id}", 3)
            install_result = _install_candidate({"model_id": model_id}, progress=None)
            if not install_result.get("ok"):
                return {"ok": False, "mode": "mss_candidate_lab", "action": "run", "error": "Automatic candidate installation failed", "install_result": install_result}
            auto_installed = True
            inventory = _inventory()
            models = {item["id"]: item for item in inventory["registered_models"]}
            model = models.get(model_id)
        if not model or not model["research_ready"]:
            return {"ok": False, "mode": "mss_candidate_lab", "action": "run", "error": "Model is not research-ready", "model": model}

    audio_url = str(payload.get("audio_url") or payload.get("source_url") or "").strip()
'''
if auto_old in text:
    text = text.replace(auto_old, auto_new, 1)
elif 'auto_installed = False' not in text:
    raise RuntimeError('Could not locate MSS research-ready guard')

old = '''        files = sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file())
        return {
            "ok": completed.returncode == 0,
'''
new = '''        output_paths = sorted(path for path in output_dir.rglob("*") if path.is_file())
        files = [str(path.relative_to(output_dir)) for path in output_paths]
        succeeded = completed.returncode == 0 and bool(output_paths)
        archive = None
        uploaded = False
        target_matches = [path for path in output_paths if model["target_stem"] in path.name.lower()]
        target_metrics = _candidate_audio_metrics(target_matches[0], source) if succeeded and target_matches else None
        put_url = str(payload.get("result_put_url") or "").strip()
        if succeeded:
            archive = root / f"{model_id}-candidate-output.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
                for output_path in output_paths:
                    bundle.write(output_path, arcname=str(output_path.relative_to(output_dir)))
            if put_url:
                if progress:
                    progress(f"Uploading candidate {model_id}", 92)
                _upload_put(put_url, archive)
                uploaded = True
        return {
            "ok": succeeded,
'''
if old in text:
    text = text.replace(old, new, 1)
elif 'target_metrics = _candidate_audio_metrics' not in text:
    raise RuntimeError('Could not locate candidate output block')

old_tail = '''            "output_files": files,
            "log_tail": "\\n".join((completed.stdout or "").splitlines()[-80:]),
            "promotion_status": "candidate_only",
'''
new_tail = '''            "output_files": files,
            "target_metrics": target_metrics,
            "archive_name": archive.name if archive else None,
            "archive_size_bytes": archive.stat().st_size if archive else 0,
            "uploaded": uploaded,
            "result_url": payload.get("result_public_url"),
            "auto_installed_on_worker": auto_installed,
            "log_tail": "\\n".join((completed.stdout or "").splitlines()[-80:]),
            "promotion_status": "candidate_only",
'''
if old_tail in text:
    text = text.replace(old_tail, new_tail, 1)
elif '"target_metrics": target_metrics' not in text:
    raise RuntimeError('Could not locate candidate return block')

benchmark_marker = '\n\ndef build_mss_candidate_lab(payload: dict, progress=None) -> dict:\n'
benchmark_helper = '''\n\ndef _benchmark_candidates(payload: dict, progress=None) -> dict:
    audio_url = str(payload.get("audio_url") or payload.get("source_url") or "").strip()
    if not audio_url:
        return {"ok": False, "mode": "mss_candidate_lab", "action": "benchmark", "error": "audio_url is required"}
    base = dict(payload)
    base["action"] = "run"
    base.pop("model_id", None)
    if progress:
        progress("Running piano challenger", 5)
    piano = _run_candidate({**base, "model_id": "htdemucs6-piano-challenger"}, progress=None)
    if not piano.get("ok"):
        return {"ok": False, "mode": "mss_candidate_lab", "action": "benchmark", "failed_stage": "piano", "result": piano}
    if progress:
        progress("Running Other challenger", 55)
    other = _run_candidate({**base, "model_id": "viperx-bs-roformer-other-challenger"}, progress=None)
    if not other.get("ok"):
        return {"ok": False, "mode": "mss_candidate_lab", "action": "benchmark", "failed_stage": "other", "result": other}
    if progress:
        progress("Candidate benchmark complete", 100)
    return {
        "ok": True,
        "mode": "mss_candidate_lab",
        "schema_version": 3,
        "action": "benchmark",
        "track_url": audio_url,
        "piano": {
            "model_id": piano["model"]["id"],
            "metrics": piano.get("target_metrics"),
            "runtime_log": piano.get("log_tail"),
            "auto_installed_on_worker": piano.get("auto_installed_on_worker"),
        },
        "other": {
            "model_id": other["model"]["id"],
            "metrics": other.get("target_metrics"),
            "runtime_log": other.get("log_tail"),
            "auto_installed_on_worker": other.get("auto_installed_on_worker"),
        },
        "quality_warning": "These are reference-free sanity metrics, not proof of separation quality. Listening and compatible ground-truth tests remain required.",
        "next_action": "Compare these profiles with the current LiteLABS baseline and perform listening tests before promotion.",
    }
'''
if 'def _benchmark_candidates(' not in text:
    if benchmark_marker not in text:
        raise RuntimeError('Could not locate candidate lab builder')
    text = text.replace(benchmark_marker, benchmark_helper + benchmark_marker, 1)

run_route = '''    if action == "run":
        return _run_candidate(payload, progress=progress)
'''
benchmark_route = run_route + '''    if action == "benchmark":
        return _benchmark_candidates(payload, progress=progress)
'''
if run_route in text and 'if action == "benchmark"' not in text:
    text = text.replace(run_route, benchmark_route, 1)
elif 'if action == "benchmark"' not in text:
    raise RuntimeError('Could not add benchmark action route')

path.write_text(text, encoding='utf-8')
print('MSS candidate output, auto-install and benchmark patch applied')
