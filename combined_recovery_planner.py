from __future__ import annotations

from typing import Any

PRIMARY_STEMS = ("vocals", "drums", "bass", "guitar", "piano", "other")


def _route_for(stem: str) -> str:
    routes = {
        "piano": "recover_raw_piano_then_compare_alternate_piano_candidate",
        "other": "recover_raw_other_then_check_synth_and_keys_absorption",
        "vocals": "run_vocal_specialist_candidate_and_compare",
        "drums": "run_alternate_drum_candidate_and_compare_transients",
        "bass": "run_alternate_bass_candidate_and_compare_low_end",
        "guitar": "run_alternate_guitar_candidate_and_check_vocal_or_keys_bleed",
    }
    return routes.get(stem, "run_alternate_candidate_and_compare")


def _likely_absorption_targets(missing: list[str], audit: dict[str, Any]) -> list[dict[str, Any]]:
    found = set(audit.get("primary_stems_found") or [])
    pairwise = audit.get("pairwise_duplication") or []
    risk_by_role: dict[str, float] = {role: 0.0 for role in found}
    for row in pairwise:
        left = str(row.get("left") or "")
        right = str(row.get("right") or "")
        risk = float(row.get("duplication_risk") or 0.0)
        if left in risk_by_role:
            risk_by_role[left] = max(risk_by_role[left], risk)
        if right in risk_by_role:
            risk_by_role[right] = max(risk_by_role[right], risk)

    targets: list[dict[str, Any]] = []
    for stem in missing:
        if stem == "piano":
            candidates = [role for role in ("guitar", "other", "vocals") if role in found]
        elif stem == "other":
            candidates = [role for role in ("guitar", "drums", "vocals", "piano") if role in found]
        else:
            candidates = sorted(found)
        candidates.sort(key=lambda role: risk_by_role.get(role, 0.0), reverse=True)
        targets.append({
            "missing_stem": stem,
            "candidate_stems": candidates[:3],
            "basis": "Heuristic only: inspect surviving stems for absorbed material; ownership is not proven.",
        })
    return targets


def _classification(stem: str, expected: set[str], delivered: set[str], score: float) -> dict[str, Any]:
    if stem in expected and stem in delivered:
        state = "confirmed"
        action = "include"
        reason = "Detected in the source profile and present in the delivered pack."
    elif stem in expected:
        state = "uncertain"
        action = "recover_retry_and_include"
        reason = "Detected in the source profile but missing from the delivered pack."
    elif stem in delivered:
        state = "unexpected_output"
        action = "keep_for_review"
        reason = "Delivered by the separator but not confidently expected by the source profile."
    else:
        state = "not_expected"
        action = "omission_allowed_only_if_effectively_silent"
        reason = "Not confidently detected in the source and not present in the delivered pack."
    return {
        "stem": stem,
        "state": state,
        "action": action,
        "source_confidence": round(score, 6),
        "reason": reason,
    }


def build_combined_recovery_planner(payload: dict, progress=None) -> dict:
    source_url = str(payload.get("source_url") or payload.get("audio_url") or "").strip()
    stem_pack_url = str(payload.get("stem_pack_url") or payload.get("pack_url") or "").strip()
    if not source_url or not stem_pack_url:
        return {"ok": False, "mode": "combined_recovery_planner", "error": "source_url and stem_pack_url are required"}

    if progress:
        progress("Building track profile", 5)
    from track_profile import build_track_profile
    profile_payload = dict(payload)
    profile_payload["audio_url"] = source_url
    profile = build_track_profile(profile_payload, progress=None)
    if not profile.get("ok"):
        return {"ok": False, "mode": "combined_recovery_planner", "stage": "track_profile", "error": profile.get("error", "track profile failed")}

    if progress:
        progress("Auditing delivered stem pack", 55)
    from reference_free_stem_auditor import build_reference_free_stem_auditor
    audit_payload = dict(payload)
    audit_payload["source_url"] = source_url
    audit_payload["stem_pack_url"] = stem_pack_url
    audit = build_reference_free_stem_auditor(audit_payload, progress=None)
    if not audit.get("ok"):
        return {"ok": False, "mode": "combined_recovery_planner", "stage": "reference_free_auditor", "error": audit.get("error", "audit failed")}

    expected_set = set(profile.get("expected_primary_stems") or [])
    delivered_set = set(audit.get("primary_stems_found") or [])
    expected = [stem for stem in PRIMARY_STEMS if stem in expected_set]
    delivered = [stem for stem in PRIMARY_STEMS if stem in delivered_set]
    missing = [stem for stem in expected if stem not in delivered_set]
    unexpected = [stem for stem in delivered if stem not in expected_set]
    scores = profile.get("primary_stem_scores") or {}

    mandatory_retries = []
    for stem in missing:
        confidence = float(scores.get(stem) or 0.0)
        mandatory_retries.append({
            "stem": stem,
            "priority": 3,
            "confidence": round(confidence, 6),
            "reason": f"Detected in the source profile at confidence {confidence:.3f} but missing from the delivered pack.",
            "route": _route_for(stem),
            "must_export": True,
        })

    additional_retries = []
    for item in audit.get("retry_queue") or []:
        stem = str(item.get("stem") or "")
        if stem and stem not in missing:
            additional_retries.append(item)

    reconstruction = audit.get("reconstruction") or {}
    residual_db = float(reconstruction.get("residual_vs_source_db") or 0.0)
    likely_absorption = bool(missing and residual_db <= -25.0)
    recovery_plan = mandatory_retries + additional_retries
    recovery_plan.sort(key=lambda item: (-int(item.get("priority") or 0), str(item.get("stem") or "")))

    export_decisions = [
        _classification(stem, expected_set, delivered_set, float(scores.get(stem) or 0.0))
        for stem in PRIMARY_STEMS
    ]

    if mandatory_retries:
        overall_status = "mandatory_recovery_required"
    elif recovery_plan:
        overall_status = "targeted_retry_recommended"
    else:
        overall_status = "baseline_pack_consistent"

    include_details = bool(payload.get("include_details", False))
    result = {
        "ok": True,
        "mode": "combined_recovery_planner",
        "schema_version": 2,
        "overall_status": overall_status,
        "source_url": source_url,
        "stem_pack_url": stem_pack_url,
        "track_profile": {
            "primary_genre": profile.get("primary_genre"),
            "routing_family": profile.get("routing_family"),
            "styles": profile.get("styles") or [],
        },
        "expected_primary_stems": expected,
        "delivered_primary_stems": delivered,
        "missing_expected_stems": missing,
        "unexpected_delivered_stems": unexpected,
        "primary_stem_scores": scores,
        "export_decisions": export_decisions,
        "mandatory_retries": mandatory_retries,
        "additional_retries": additional_retries,
        "recovery_plan": recovery_plan,
        "likely_stem_absorption": likely_absorption,
        "likely_absorption_targets": _likely_absorption_targets(missing, audit) if likely_absorption else [],
        "pack_reconstruction": reconstruction,
        "export_policy": {
            "confirmed": "Detected in source and credible output: include.",
            "uncertain": "Detected in source but weak or missing output: retry and include; never silently omit.",
            "not_expected": "Not detected in source and effectively silent output: omission may be allowed.",
            "piano_keys": "Never omit when detected by the source profile.",
            "other": "Keep as the catch-all for synths, strings, brass, effects and unassigned material when detected.",
            "dry_main_vocals": "Disabled while the dereverb model is reviewed; normal vocals remain mandatory.",
            "instrumental_is_derived_output": True,
        },
        "final_pack_policy": {
            "always_include_when_expected": expected,
            "derived_outputs": ["instrumental"],
            "optional_outputs": ["backing_vocals"],
            "temporarily_disabled_outputs": ["dry_main_vocals"],
        },
        "next_action": "Recover missing raw outputs, compare alternate candidates, then rerun the planner." if mandatory_retries else "Review targeted retries and keep the baseline where no challenger wins.",
        "research_only": bool(profile.get("research_only", True)),
        "licensing_note": profile.get("licensing_note"),
        "limitations": [
            "Instrument presence does not prove which surviving stem absorbed missing material.",
            "The absorption target list is a prioritised listening and challenger-test heuristic, not ground truth.",
            "A challenger should replace a baseline stem only after reconstruction, contamination and listening checks improve.",
        ],
    }
    if include_details:
        result["profile_details"] = profile
        result["audit"] = audit
    else:
        result["profile_summary"] = {
            "instruments": profile.get("instruments") or [],
            "routing_hints": profile.get("routing_hints") or [],
            "broad_genres": profile.get("broad_genres") or [],
        }
        result["audit_summary"] = {
            "overall_status": audit.get("overall_status"),
            "stems": audit.get("stems") or [],
            "pairwise_duplication": audit.get("pairwise_duplication") or [],
        }

    if progress:
        progress("Recovery plan complete", 100)
    return result
