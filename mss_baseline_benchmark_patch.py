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
        filename = unquote(Path(urlparse(audio_url).path).name) or "track.flac"
        downloaded = root / filename
        _download(audio_url, downloaded)
        input_dir.mkdir(parents=True, exist_ok=True)
        source = input_dir / f"{Path(filename).stem}.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(downloaded), "-ar", "44100", "-ac", "2", str(source)
        ], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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
            "piano_metrics": _candidate_audio_metrics(piano_matches[0], source) if piano_matches else None,
            "other_metrics": _candidate_audio_metrics(other_matches[0], source) if other_matches else None,
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

campaign_marker = '\n\ndef build_mss_candidate_lab(payload: dict, progress=None) -> dict:\n'
campaign_helper = '''\n\ndef _campaign_gate(model: dict, metrics: dict | None, baseline_metrics: dict | None) -> dict:
    reasons = []
    if not metrics:
        return {"status": "rejected", "reasons": ["missing_metrics"], "score": -999.0}
    if metrics.get("rms_dbfs", -240.0) < -65.0:
        reasons.append("effectively_silent_rms")
    if metrics.get("peak_dbfs", -240.0) < -45.0:
        reasons.append("effectively_silent_peak")
    if metrics.get("active_ratio", 0.0) < 0.02:
        reasons.append("mostly_inactive")
    if model.get("target_stem") == "other" and metrics.get("mixture_cosine", 0.0) > 0.75:
        reasons.append("likely_instrumental_not_residual_other")
    delta_rms = round(metrics["rms_dbfs"] - baseline_metrics["rms_dbfs"], 3) if baseline_metrics else None
    delta_cosine = round(metrics["mixture_cosine"] - baseline_metrics["mixture_cosine"], 6) if baseline_metrics else None
    score = metrics.get("rms_dbfs", -240.0) + min(metrics.get("active_ratio", 0.0), 1.0) * 8.0 - abs(metrics.get("mixture_cosine", 0.0)) * 4.0
    if reasons:
        score -= 100.0
    return {
        "status": "rejected" if reasons else "listening_candidate",
        "reasons": reasons,
        "score": round(score, 3),
        "delta_vs_baseline_rms_db": delta_rms,
        "delta_vs_baseline_mixture_cosine": delta_cosine,
    }


def _run_campaign(payload: dict, progress=None) -> dict:
    audio_url = str(payload.get("audio_url") or payload.get("source_url") or "").strip()
    if not audio_url:
        return {"ok": False, "mode": "mss_candidate_lab", "action": "campaign", "error": "audio_url is required"}
    timeout_seconds = int(payload.get("timeout_seconds") or 1800)
    inventory = _inventory()
    enabled = [item for item in inventory["registered_models"] if item.get("enabled")]
    baseline = _run_bs_baseline(audio_url, timeout_seconds=timeout_seconds)
    if not baseline.get("ok"):
        return {"ok": False, "mode": "mss_candidate_lab", "action": "campaign", "failed_stage": "baseline", "result": baseline}
    results = []
    total = max(1, len(enabled))
    for index, model in enumerate(enabled, start=1):
        if progress:
            progress(f"Campaign {index}/{total}: {model['id']}", int(10 + (index - 1) * 75 / total))
        result = _run_candidate({"action": "run", "model_id": model["id"], "audio_url": audio_url, "timeout_seconds": timeout_seconds}, progress=None)
        metrics = result.get("target_metrics") if result.get("ok") else None
        gate = _campaign_gate(model, metrics, baseline.get(f"{model['target_stem']}_metrics"))
        results.append({
            "model_id": model["id"],
            "target_stem": model["target_stem"],
            "model_type": model["model_type"],
            "ok": bool(result.get("ok")),
            "metrics": metrics,
            "gate": gate,
            "auto_installed_on_worker": result.get("auto_installed_on_worker"),
            "runtime_log": result.get("log_tail"),
            "error": result.get("error"),
        })
    ranked = {}
    for stem in CANONICAL_STEMS:
        items = [item for item in results if item["target_stem"] == stem]
        items.sort(key=lambda item: item["gate"]["score"], reverse=True)
        if items:
            ranked[stem] = items
    finalists = {stem: [item["model_id"] for item in items if item["gate"]["status"] == "listening_candidate"][:2] for stem, items in ranked.items()}
    return {
        "ok": True,
        "mode": "mss_candidate_lab",
        "schema_version": 5,
        "action": "campaign",
        "track_url": audio_url,
        "baseline": {"model_id": "current-litelabs-bs-roformer-sw", "piano_metrics": baseline.get("piano_metrics"), "other_metrics": baseline.get("other_metrics")},
        "registered_candidates": len(enabled),
        "results": results,
        "ranked_by_stem": ranked,
        "listening_finalists": finalists,
        "automatically_rejected": [item["model_id"] for item in results if item["gate"]["status"] == "rejected"],
        "decision_policy": {"silent_rms_below_dbfs": -65.0, "silent_peak_below_dbfs": -45.0, "other_high_mixture_cosine_rejection": 0.75, "reference_free_metrics_are_not_ground_truth": True},
        "next_action": "Only listen to the listed finalists. Rejected candidates need no further manual testing on this track.",
    }
'''
if 'def _run_campaign(' not in text:
    if campaign_marker not in text:
        raise RuntimeError('Could not locate MSS lab builder')
    text = text.replace(campaign_marker, campaign_helper + campaign_marker, 1)

route = '''    if action == "benchmark":
        return _benchmark_candidates(payload, progress=progress)
'''
new_route = route + '''    if action == "campaign":
        return _run_campaign(payload, progress=progress)
'''
if route in text and 'if action == "campaign"' not in text:
    text = text.replace(route, new_route, 1)
elif 'if action == "campaign"' not in text:
    raise RuntimeError('Could not add campaign action route')

path.write_text(text, encoding='utf-8')
print('MSS baseline comparison and automated campaign patch applied')
