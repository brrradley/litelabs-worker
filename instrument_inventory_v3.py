from __future__ import annotations

from typing import Any


ALIASES = {
    "electricguitar": "electric-guitar",
    "acousticguitar": "acoustic-guitar",
    "classicalguitar": "classical-guitar",
    "electricpiano": "digital-piano",
    "keyboard": "keys",
    "synthesizer": "synth",
    "pipeorgan": "organ",
    "doublebass": "double-bass",
}

FAMILIES = {
    "drums": {"drums", "kick", "snare", "toms", "hh", "percussion", "congas", "tambourine", "timpani"},
    "guitar": {"guitar", "electric-guitar", "acoustic-guitar", "classical-guitar", "dobro", "mandolin", "banjo", "ukulele"},
    "keys": {"piano", "digital-piano", "keys", "organ", "harpsichord"},
    "wind-brass": {"wind", "brass", "woodwind", "saxophone", "trumpet", "trombone", "clarinet", "flute", "oboe", "bassoon", "french-horn", "tuba", "harmonica"},
    "strings": {"strings", "bowed-strings", "violin", "viola", "cello", "double-bass"},
    "electronic": {"synth"},
    "vocals": {"vocal", "lead-vocal", "back-vocal"},
    "bass": {"bass", "double-bass"},
}

# These labels only describe a family/container and should support specific identities,
# not become automatic final exports themselves. Drums and bass are deliberately
# excluded because they are actionable parent/canonical stems.
SUPPORT_ONLY_GENERIC_LABELS = {"guitar", "keys", "wind", "brass", "woodwind", "strings", "synth", "vocal"}


def _canon(name: str) -> str:
    value = str(name or "").strip().lower().replace("_", "-")
    return ALIASES.get(value, value)


def _family(name: str) -> str:
    canon = _canon(name)
    for family, members in FAMILIES.items():
        if canon in members:
            return family
    return "other"


def _mtg_index(mtg: dict) -> dict[str, dict]:
    output = {}
    for item in mtg.get("instrument_inventory") or mtg.get("instruments") or []:
        if not isinstance(item, dict):
            continue
        key = _canon(item.get("instrument"))
        if key:
            output[key] = item
    return output


def _mega_index(mega: dict) -> dict[str, dict]:
    output = {}
    for item in mega.get("all_outputs_ranked") or []:
        if not isinstance(item, dict):
            continue
        key = _canon(item.get("instrument"))
        if key:
            output[key] = item
    return output


def _mega_strength(item: dict | None) -> str:
    if not item:
        return "none"
    metrics = item.get("metrics") or {}
    rms = float(metrics.get("rms_dbfs", -120.0))
    cosine = float(metrics.get("mixture_cosine", 0.0))
    active = float(metrics.get("active_ratio", 0.0))
    if rms >= -35.0 and (cosine >= 0.18 or active >= 0.20):
        return "strong"
    if rms >= -50.0 and (cosine >= 0.10 or active >= 0.10):
        return "moderate"
    if rms >= -55.0:
        return "weak"
    return "absent"


def _mtg_strength(item: dict | None) -> str:
    if not item:
        return "none"
    peak = float(item.get("peak_confidence", 0.0))
    presence = float(item.get("window_presence_ratio", 0.0))
    if peak >= 0.55 and presence >= 0.25:
        return "strong"
    if peak >= 0.35 and presence >= 0.20:
        return "moderate"
    if peak >= 0.20:
        return "weak"
    return "none"


def _family_support(name: str, mega: dict[str, dict]) -> dict[str, Any]:
    family = _family(name)
    members = FAMILIES.get(family, set())
    evidence = []
    for member in members:
        if member == _canon(name):
            continue
        item = mega.get(member)
        strength = _mega_strength(item)
        if strength in {"strong", "moderate"}:
            evidence.append({"instrument": member, "strength": strength})
    return {"family": family, "support": evidence[:8], "supported": bool(evidence)}


def _verdict(name: str, mtg_item: dict | None, mega_item: dict | None, family_support: dict) -> tuple[str, str]:
    canon = _canon(name)
    family = _family(canon)
    mtg_strength = _mtg_strength(mtg_item)
    mega_strength = _mega_strength(mega_item)
    supported = bool(family_support.get("supported"))

    # Mega53 is a discovery model and is known to miss genuine canonical material.
    # Therefore its silence may veto a narrow subtype, but must not erase a strong
    # MTG detection for actionable parent/canonical stems such as bass or drums.
    if mega_strength == "absent" and mtg_strength in {"strong", "moderate"}:
        if canon in {"bass", "drums"}:
            return "likely", "Strong MTG evidence retained because Mega53 silence cannot veto an actionable canonical stem"
        return "rejected", "MTG detection contradicted by an effectively absent Mega53 output"

    # Guitar-family timbres are a known confusion case on this test material.
    # Two broad discovery systems agreeing is not sufficient to auto-confirm a
    # specific guitar subtype without a dedicated separation/listening check.
    if family == "guitar" and canon != "guitar":
        if mega_strength == "strong" and mtg_strength in {"strong", "moderate"}:
            return "likely", "Both discovery systems indicate this guitar subtype, but guitar identity requires specialist verification"

    if mega_strength == "strong" and mtg_strength in {"strong", "moderate"}:
        return "confirmed", "MTG and Mega53 independently agree"
    if mega_strength == "strong" and mtg_strength == "weak" and supported:
        return "confirmed", "Strong Mega53 output plus coherent family evidence confirms a weak MTG hint"
    if mega_strength == "strong" and mtg_strength in {"weak", "none"}:
        return "likely", "Strong Mega53 evidence without strong MTG agreement"
    if mega_strength == "moderate" and mtg_strength in {"strong", "moderate"}:
        return "likely", "Both systems provide useful evidence, but not strongly enough to confirm"
    if mega_strength == "moderate" and supported:
        return "likely", "Moderate Mega53 evidence reinforced by related instruments in the same family"
    if mega_strength in {"weak", "moderate"} or mtg_strength in {"weak", "moderate"}:
        return "uncertain", "Some evidence exists but is not reliable enough to route extraction automatically"
    return "rejected", "Neither analyser provides sufficient evidence"


def _route_action(name: str, verdict: str) -> str | None:
    canon = _canon(name)
    if verdict not in {"confirmed", "likely"}:
        return None
    if canon == "drums":
        return "decompose_drums"
    if canon == "bass":
        return "preserve_bass"
    if canon in {"kick", "snare", "toms", "hh"}:
        return "drum_child_evidence"

    # Automatic specialist extraction is intentionally conservative. A 'likely'
    # narrow instrument remains evidence only until a specialist pass validates it.
    if verdict != "confirmed":
        return None

    if canon in {"saxophone", "trumpet", "trombone", "clarinet", "flute", "oboe", "bassoon", "french-horn", "tuba", "harmonica"}:
        return f"extract_{canon}"
    if canon in {"piano", "digital-piano", "organ", "keys"}:
        return f"extract_{canon}"
    if canon in {"electric-guitar", "acoustic-guitar", "guitar"}:
        return f"extract_{canon}"
    if canon in {"strings", "violin", "viola", "cello"}:
        return f"extract_{canon}"
    if canon == "synth":
        return "preserve_as_residual_candidate"
    return None


def merge_instrument_evidence(mtg: dict, mega: dict) -> dict:
    mtg_by_name = _mtg_index(mtg)
    mega_by_name = _mega_index(mega)
    names = sorted(set(mtg_by_name) | set(mega_by_name))
    decisions = []

    for name in names:
        mtg_item = mtg_by_name.get(name)
        mega_item = mega_by_name.get(name)
        family_support = _family_support(name, mega_by_name)
        verdict, reason = _verdict(name, mtg_item, mega_item, family_support)
        action = _route_action(name, verdict)
        decisions.append({
            "instrument": name,
            "family": family_support["family"],
            "generic_family_label": name in SUPPORT_ONLY_GENERIC_LABELS,
            "verdict": verdict,
            "reason": reason,
            "route_action": action,
            "mtg": {
                "strength": _mtg_strength(mtg_item),
                "peak_confidence": mtg_item.get("peak_confidence") if mtg_item else None,
                "window_presence_ratio": mtg_item.get("window_presence_ratio") if mtg_item else None,
                "time_ranges": mtg_item.get("time_ranges") if mtg_item else None,
            },
            "mega53": {
                "strength": _mega_strength(mega_item),
                "metrics": mega_item.get("metrics") if mega_item else None,
                "status": mega_item.get("status") if mega_item else None,
            },
            "family_support": family_support["support"],
        })

    order = {"confirmed": 0, "likely": 1, "uncertain": 2, "rejected": 3}
    decisions.sort(key=lambda x: (order[x["verdict"]], x["generic_family_label"], x["instrument"]))

    specific_confirmed = [d for d in decisions if d["verdict"] == "confirmed" and not d["generic_family_label"]]
    likely = [d for d in decisions if d["verdict"] == "likely" and not d["generic_family_label"]]
    uncertain = [d for d in decisions if d["verdict"] == "uncertain" and not d["generic_family_label"]]
    rejected = [d for d in decisions if d["verdict"] == "rejected" and not d["generic_family_label"]]
    extraction_plan = [
        {"instrument": d["instrument"], "family": d["family"], "action": d["route_action"], "verdict": d["verdict"]}
        for d in decisions if d.get("route_action") and not d["generic_family_label"]
    ]

    return {
        "ok": True,
        "mode": "instrument_inventory_v3",
        "schema_version": 4,
        "research_only": True,
        "policy": {
            "mtg_is_evidence_not_truth": True,
            "mega53_is_evidence_not_ground_truth": True,
            "generic_family_labels_are_supporting_evidence_not_separate_exports": True,
            "narrow_specialist_auto_extraction_requires_confirmed_verdict": True,
            "actionable_parent_stems_may_route_when_likely": True,
            "parent_stem_removed_only_after_verified_child_reconstruction": True,
            "guitar_subtypes_require_specialist_verification": True,
        },
        "confirmed": specific_confirmed,
        "likely": likely,
        "uncertain": uncertain,
        "rejected": rejected,
        "all_decisions": decisions,
        "extraction_plan": extraction_plan,
        "source_summaries": {
            "mtg_ok": bool(mtg.get("ok")),
            "mega53_ok": bool(mega.get("ok")),
            "mega53_output_count": mega.get("output_count"),
        },
    }


def build_instrument_inventory_v3(payload: dict, progress=None) -> dict:
    audio_url = str(payload.get("audio_url") or payload.get("source_url") or "").strip()
    if not audio_url:
        return {"ok": False, "mode": "instrument_inventory_v3", "error": "audio_url is required"}

    from instrument_wireframe import build_instrument_wireframe
    from mss_candidate_lab import build_mss_candidate_lab

    if progress:
        progress("Running first-pass instrument recognition", 5)
    mtg_payload = dict(payload)
    mtg_payload["audio_url"] = audio_url
    mtg = build_instrument_wireframe(mtg_payload, progress=None)
    if not mtg.get("ok"):
        return {"ok": False, "mode": "instrument_inventory_v3", "failed_stage": "mtg", "result": mtg}

    if progress:
        progress("Cross-checking with 53-stem discovery", 35)
    mega = build_mss_candidate_lab({
        "action": "mega53_discovery",
        "audio_url": audio_url,
        "timeout_seconds": int(payload.get("timeout_seconds") or 1800),
        "audible_rms_floor_dbfs": float(payload.get("audible_rms_floor_dbfs") or -55.0),
        "active_ratio_floor": float(payload.get("active_ratio_floor") or 0.01),
    }, progress=None)
    if not mega.get("ok"):
        return {"ok": False, "mode": "instrument_inventory_v3", "failed_stage": "mega53", "result": mega}

    if progress:
        progress("Merging instrument evidence", 95)
    merged = merge_instrument_evidence(mtg, mega)
    merged["audio_url"] = audio_url
    merged["mtg_instrument_count"] = len(mtg.get("instrument_inventory") or mtg.get("instruments") or [])
    merged["mega53_output_count"] = mega.get("output_count")
    return merged
