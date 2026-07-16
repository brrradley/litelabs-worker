from pathlib import Path

path = Path('/app/mss_candidate_lab.py')
text = path.read_text(encoding='utf-8')

marker = '\n\ndef build_mss_candidate_lab(payload: dict, progress=None) -> dict:\n'
helper = r'''

def _campaign_gate(model: dict, metrics: dict | None, baseline_metrics: dict | None) -> dict:
    reasons = []
    if not metrics:
        reasons.append("missing_metrics")
        return {"status": "rejected", "reasons": reasons, "score": -999.0}
    if metrics.get("rms_dbfs", -240.0) < -65.0:
        reasons.append("effectively_silent_rms")
    if metrics.get("peak_dbfs", -240.0) < -45.0:
        reasons.append("effectively_silent_peak")
    if metrics.get("active_ratio", 0.0) < 0.02:
        reasons.append("mostly_inactive")
    if model.get("target_stem") == "other" and model.get("model_type") in {"bs_roformer", "mel_band_roformer"}:
        notes = str(model.get("notes") or "").lower()
        if "dedicated other" in notes and metrics.get("mixture_cosine", 0.0) > 0.75:
            reasons.append("likely_instrumental_not_residual_other")
    delta_rms = None
    delta_cosine = None
    if baseline_metrics:
        delta_rms = round(metrics["rms_dbfs"] - baseline_metrics["rms_dbfs"], 3)
        delta_cosine = round(metrics["mixture_cosine"] - baseline_metrics["mixture_cosine"], 6)
    score = metrics.get("rms_dbfs", -240.0)
    score += min(metrics.get("active_ratio", 0.0), 1.0) * 8.0
    score -= abs(metrics.get("mixture_cosine", 0.0)) * 4.0
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
        result = _run_candidate({
            "action": "run",
            "model_id": model["id"],
            "audio_url": audio_url,
            "timeout_seconds": timeout_seconds,
        }, progress=None)
        metrics = result.get("target_metrics") if result.get("ok") else None
        baseline_metrics = baseline.get(f"{model['target_stem']}_metrics")
        gate = _campaign_gate(model, metrics, baseline_metrics)
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
        stem_results = [item for item in results if item["target_stem"] == stem]
        stem_results.sort(key=lambda item: item["gate"]["score"], reverse=True)
        if stem_results:
            ranked[stem] = stem_results
    finalists = {
        stem: [item["model_id"] for item in items if item["gate"]["status"] == "listening_candidate"][:2]
        for stem, items in ranked.items()
    }
    rejected = [item["model_id"] for item in results if item["gate"]["status"] == "rejected"]
    if progress:
        progress("Candidate campaign complete", 100)
    return {
        "ok": True,
        "mode": "mss_candidate_lab",
        "schema_version": 5,
        "action": "campaign",
        "track_url": audio_url,
        "baseline": {
            "model_id": "current-litelabs-bs-roformer-sw",
            "piano_metrics": baseline.get("piano_metrics"),
            "other_metrics": baseline.get("other_metrics"),
        },
        "registered_candidates": len(enabled),
        "results": results,
        "ranked_by_stem": ranked,
        "listening_finalists": finalists,
        "automatically_rejected": rejected,
        "decision_policy": {
            "silent_rms_below_dbfs": -65.0,
            "silent_peak_below_dbfs": -45.0,
            "other_high_mixture_cosine_rejection": 0.75,
            "reference_free_metrics_are_not_ground_truth": True,
        },
        "next_action": "Only listen to the listed finalists. Rejected candidates need no further manual testing on this track.",
    }
'''
if 'def _run_campaign(' not in text:
    if marker not in text:
        raise RuntimeError('Could not locate MSS lab builder')
    text = text.replace(marker, helper + marker, 1)

route = '''    if action == "benchmark":
        return _benchmark_candidates(payload, progress=progress)
'''
new_route = route + '''    if action == "campaign":
        return _run_campaign(payload, progress=progress)
'''
if route in text and 'if action == "campaign"' not in text:
    text = text.replace(route, new_route, 1)
elif 'if action == "campaign"' not in text:
    raise RuntimeError('Could not add campaign route')

path.write_text(text, encoding='utf-8')
print('MSS automated campaign patch applied')
