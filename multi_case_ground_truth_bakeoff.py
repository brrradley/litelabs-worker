from __future__ import annotations

from collections import defaultdict

from model_ground_truth_bakeoff import build_model_ground_truth_bakeoff


def build_multi_case_ground_truth_bakeoff(payload: dict, progress=None) -> dict:
    cases = payload.get("cases") or []
    if not isinstance(cases, list) or not cases:
        return {"ok": False, "mode": "multi_case_ground_truth_bakeoff", "error": "input.cases must be a non-empty array"}

    shared = {
        "models": payload.get("models"),
        "model_timeout_seconds": payload.get("model_timeout_seconds"),
        "mdxc_segment_size": payload.get("mdxc_segment_size"),
        "mdxc_overlap": payload.get("mdxc_overlap"),
        "mdxc_batch_size": payload.get("mdxc_batch_size"),
        "use_autocast": payload.get("use_autocast", True),
    }

    results = []
    grouped = defaultdict(lambda: defaultdict(list))

    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            results.append({"ok": False, "case_index": index, "error": "Case must be an object"})
            continue

        label = str(case.get("label") or f"case-{index + 1}")
        target_stem = str(case.get("target_stem") or "").strip().lower()
        case_payload = {
            **shared,
            "audio_url": case.get("audio_url"),
            "target_stem": target_stem,
            "references": case.get("references") or {},
        }
        if case.get("models"):
            case_payload["models"] = case.get("models")

        if progress:
            progress(f"Running batch case {index + 1}/{len(cases)}: {label}", int(3 + 94 * index / max(1, len(cases))))

        result = build_model_ground_truth_bakeoff(case_payload)
        result["case_index"] = index
        result["case_label"] = label
        results.append(result)

        for stem, rows in (result.get("leaderboards") or {}).items():
            for row in rows:
                grouped[stem][row["model"]].append({"case_label": label, **row})

    combined = {}
    for stem, models in grouped.items():
        rows = []
        for model, entries in models.items():
            quality = [float(item.get("quality_score") or 0) for item in entries]
            si_sdr = [float(item.get("si_sdr_db") or -120) for item in entries]
            runtime = [float(item.get("runtime_seconds") or 0) for item in entries]
            rows.append({
                "model": model,
                "cases_completed": len(entries),
                "average_quality_score": round(sum(quality) / len(quality), 2),
                "average_si_sdr_db": round(sum(si_sdr) / len(si_sdr), 3),
                "average_runtime_seconds": round(sum(runtime) / len(runtime), 3),
                "case_scores": entries,
            })
        combined[stem] = sorted(rows, key=lambda item: (-item["average_quality_score"], -item["average_si_sdr_db"]))

    return {
        "ok": all(bool(item.get("ok")) for item in results),
        "mode": "multi_case_ground_truth_bakeoff",
        "schema_version": 1,
        "case_count": len(cases),
        "models_requested": payload.get("models"),
        "results": results,
        "combined_leaderboards": combined,
        "no_audio_exported": True,
        "metric_note": "Each case is scored against its own genuine reference; combined leaderboards average quality and SI-SDR across completed cases for each stem.",
    }
