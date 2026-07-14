from __future__ import annotations

import math
import time
from collections import defaultdict

from model_ground_truth_bakeoff import build_model_ground_truth_bakeoff

DEFAULT_EXPERIMENTS = [
    {"id": "bs_base", "model": "BS-Roformer-SW.ckpt", "segment": 256, "overlap": 8, "autocast": True},
    {"id": "bs_overlap_12", "model": "BS-Roformer-SW.ckpt", "segment": 256, "overlap": 12, "autocast": True},
    {"id": "bs_overlap_16", "model": "BS-Roformer-SW.ckpt", "segment": 256, "overlap": 16, "autocast": True},
    {"id": "bs_segment_384", "model": "BS-Roformer-SW.ckpt", "segment": 384, "overlap": 8, "autocast": True},
    {"id": "bs_segment_512", "model": "BS-Roformer-SW.ckpt", "segment": 512, "overlap": 8, "autocast": True},
    {"id": "bs_segment_512_overlap_12", "model": "BS-Roformer-SW.ckpt", "segment": 512, "overlap": 12, "autocast": True},
    {"id": "bs_fp32", "model": "BS-Roformer-SW.ckpt", "segment": 256, "overlap": 8, "autocast": False},
]


def _normalise_experiments(value):
    source = value if isinstance(value, list) and value else DEFAULT_EXPERIMENTS
    result = []
    for index, item in enumerate(source):
        if not isinstance(item, dict):
            continue
        model = str(item.get("model") or item.get("model_filename") or "BS-Roformer-SW.ckpt")
        result.append({
            "id": str(item.get("id") or f"experiment_{index:02d}"),
            "model": model,
            "segment": max(32, min(4096, int(item.get("segment") or item.get("mdxc_segment_size") or 256))),
            "overlap": max(2, min(50, int(item.get("overlap") or item.get("mdxc_overlap") or 8))),
            "batch_size": max(1, min(16, int(item.get("batch_size") or item.get("mdxc_batch_size") or 1))),
            "autocast": bool(item.get("autocast", item.get("use_autocast", True))),
            "route": str(item.get("route") or "direct"),
        })
    return result or list(DEFAULT_EXPERIMENTS)


def _failure_tags(score):
    tags = []
    q = float(score.get("quality_score") or 0.0)
    sdr = float(score.get("si_sdr_db") or -999.0)
    corr = float(score.get("correlation") or 0.0)
    spectral = float(score.get("spectral_similarity") or 0.0)
    nrmse = float(score.get("normalised_rmse") or 999.0)
    gain = abs(float(score.get("estimated_gain_db") or 0.0))
    offset = abs(float(score.get("alignment_offset_ms") or 0.0))
    if q < 70: tags.append("low_overall_quality")
    if sdr < 0: tags.append("negative_si_sdr")
    elif sdr < 5: tags.append("weak_separation")
    if corr < 0.75: tags.append("low_waveform_similarity")
    if spectral < 0.85: tags.append("spectral_damage_or_wrong_content")
    if nrmse > 1.0: tags.append("high_error")
    if gain > 6: tags.append("level_mismatch")
    if offset > 20: tags.append("timing_or_reference_mismatch")
    return tags


def _aggregate(rows, baseline_id):
    grouped = defaultdict(list)
    case_baselines = {}
    for row in rows:
        if not row.get("ok"):
            continue
        score = row.get("score") or {}
        grouped[row["experiment_id"]].append(row)
        if row["experiment_id"] == baseline_id:
            case_baselines[row["case_id"]] = float(score.get("quality_score") or 0.0)

    leaderboard = []
    for experiment_id, items in grouped.items():
        qualities = [float(x["score"].get("quality_score") or 0.0) for x in items]
        sdrs = [float(x["score"].get("si_sdr_db") or -999.0) for x in items]
        runtimes = [float(x.get("runtime_seconds") or 0.0) for x in items]
        deltas = []
        wins = losses = ties = 0
        for item in items:
            baseline = case_baselines.get(item["case_id"])
            if baseline is None:
                continue
            delta = float(item["score"].get("quality_score") or 0.0) - baseline
            deltas.append(delta)
            if delta > 0.20: wins += 1
            elif delta < -0.20: losses += 1
            else: ties += 1
        leaderboard.append({
            "experiment_id": experiment_id,
            "cases_completed": len(items),
            "average_quality_score": round(sum(qualities) / len(qualities), 3),
            "average_si_sdr_db": round(sum(sdrs) / len(sdrs), 3),
            "average_runtime_seconds": round(sum(runtimes) / len(runtimes), 3),
            "average_delta_vs_baseline": round(sum(deltas) / len(deltas), 3) if deltas else None,
            "wins_vs_baseline": wins,
            "ties_vs_baseline": ties,
            "losses_vs_baseline": losses,
            "safe_promotion_candidate": bool(deltas) and wins >= 2 and losses == 0 and (sum(deltas) / len(deltas)) >= 0.25,
        })
    leaderboard.sort(key=lambda x: (-x["average_quality_score"], x["average_runtime_seconds"]))
    return leaderboard


def build_adaptive_research_campaign(payload: dict, progress=None) -> dict:
    cases = payload.get("cases") or []
    if not isinstance(cases, list) or not cases:
        return {"ok": False, "mode": "adaptive_research_campaign", "error": "input.cases must be a non-empty array"}
    experiments = _normalise_experiments(payload.get("experiments"))
    baseline_id = str(payload.get("baseline_experiment_id") or experiments[0]["id"])
    pairs = [(ci, ei) for ci in range(len(cases)) for ei in range(len(experiments))]
    cursor = max(0, int(payload.get("cursor") or 0))
    max_runs = max(1, min(100, int(payload.get("max_runs") or 12)))
    time_budget = max(60, min(3300, int(payload.get("time_budget_seconds") or 3000)))
    end = min(len(pairs), cursor + max_runs)
    started = time.monotonic()
    rows = []

    for position in range(cursor, end):
        if position > cursor and time.monotonic() - started >= time_budget:
            end = position
            break
        case_index, experiment_index = pairs[position]
        case = cases[case_index]
        exp = experiments[experiment_index]
        target = str(case.get("target_stem") or "vocals").lower().strip()
        case_id = str(case.get("id") or case.get("label") or f"case_{case_index:02d}")
        if progress:
            progress(f"{case_id}: {exp['id']}", int(3 + 92 * (position - cursor) / max(1, end - cursor)))
        request = {
            "audio_url": case.get("audio_url") or case.get("url"),
            "target_stem": target,
            "references": case.get("references") or {},
            "residual_stem": case.get("residual_stem"),
            "models": [exp["model"]],
            "model_timeout_seconds": int(payload.get("model_timeout_seconds") or 1800),
            "mdxc_segment_size": exp["segment"],
            "mdxc_overlap": exp["overlap"],
            "mdxc_batch_size": exp["batch_size"],
            "use_autocast": exp["autocast"],
        }
        result = build_model_ground_truth_bakeoff(request)
        run = (result.get("runs") or [{}])[0]
        score = ((run.get("scores") or {}).get(target) or {})
        rows.append({
            "pair_index": position,
            "case_index": case_index,
            "case_id": case_id,
            "case_label": case.get("label") or case_id,
            "traits": case.get("traits") or {},
            "target_stem": target,
            "experiment_id": exp["id"],
            "experiment": exp,
            "ok": bool(run.get("ok")),
            "runtime_seconds": run.get("runtime_seconds"),
            "peak_gpu_memory_mib": run.get("peak_gpu_memory_mib"),
            "score": score,
            "failure_tags": _failure_tags(score) if score else [],
            "error": run.get("error") or result.get("error"),
        })

    by_stem = {}
    for stem in sorted({row["target_stem"] for row in rows}):
        by_stem[stem] = _aggregate([row for row in rows if row["target_stem"] == stem], baseline_id)

    failures = defaultdict(int)
    for row in rows:
        for tag in row.get("failure_tags") or []:
            failures[tag] += 1

    next_cursor = end if end < len(pairs) else None
    return {
        "ok": True,
        "mode": "adaptive_research_campaign",
        "schema_version": 1,
        "baseline_experiment_id": baseline_id,
        "case_count": len(cases),
        "experiment_count": len(experiments),
        "total_runs": len(pairs),
        "cursor": cursor,
        "processed_runs": len(rows),
        "next_cursor": next_cursor,
        "complete": next_cursor is None,
        "experiments": experiments,
        "results": rows,
        "leaderboards_by_stem": by_stem,
        "failure_summary": dict(sorted(failures.items(), key=lambda item: (-item[1], item[0]))),
        "promotion_rule": "Promote only when an experiment beats the baseline by >0.20 on at least two cases, has no losses >0.20, and averages >=0.25 improvement.",
        "learning_note": "This learns routing and inference policy from scored outcomes. It does not update neural-network weights.",
        "no_audio_exported": True,
    }
