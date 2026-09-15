from pathlib import Path

path = Path('/app/qa_research.py')
text = path.read_text(encoding='utf-8')

replacements = [
    ('QA_VERSION = 2', 'QA_VERSION = 3'),
    (
        '    leakage_corr: float,\n) -> tuple[float, dict, list[str]]:',
        '    leakage_corr: float,\n    stem_name: str,\n    qa_group: str,\n) -> tuple[float, dict, list[str], list[str]]:',
    ),
    (
        '    useful_signal = 0.65 * activity_score + 0.35 * level_score\n    technical_health = _clamp01(1.0 - min(1.0, clipping * 250.0))',
        '    useful_signal = 0.65 * activity_score + 0.35 * level_score\n'
        '    scoring_useful_signal = useful_signal\n'
        '    score_adjustments: list[str] = []\n\n'
        '    # Keys/strings are often arrangement-sparse but still excellent when present.\n'
        '    # Do not punish a clean tonal parent merely for not playing throughout the song.\n'
        '    # Near-empty stems still hit the existing hard caps below.\n'
        '    if qa_group == "parent" and stem_name in {"keys", "strings"} and 0.35 <= useful_signal < 0.70:\n'
        '        scoring_useful_signal = 0.70\n'
        '        score_adjustments.append("sparse tonal stem signal floor")\n\n'
        '    technical_health = _clamp01(1.0 - min(1.0, clipping * 250.0))',
    ),
    (
        '        + 0.25 * useful_signal\n',
        '        + 0.25 * scoring_useful_signal\n',
    ),
    (
        '        "useful_signal": round(useful_signal, 6),\n        "technical_health": round(technical_health, 6),',
        '        "useful_signal": round(useful_signal, 6),\n'
        '        "scoring_useful_signal": round(scoring_useful_signal, 6),\n'
        '        "technical_health": round(technical_health, 6),',
    ),
    (
        '    return round(_clamp01(score), 6), components, cap_reasons\n',
        '    return round(_clamp01(score), 6), components, cap_reasons, score_adjustments\n',
    ),
    (
        '        score, components, cap_reasons = _stem_score(\n            metrics[name], distinctness, reconstruction, leakage_corr\n        )',
        '        score, components, cap_reasons, score_adjustments = _stem_score(\n'
        '            metrics[name], distinctness, reconstruction, leakage_corr, name, qa_group\n'
        '        )\n\n'
        '        # The current backing-vocal export is a parent-minus-lead residual. It can\n'
        '        # measure extremely clean while retaining subtle cancellation/artefact texture,\n'
        '        # so apply a small confidence correction only to that derived method.\n'
        '        model_name = model_by_stem.get(name, "unknown")\n'
        '        if name == "backing_vocals" and "parent-minus-lead" in model_name.lower():\n'
        '            score = round(_clamp01(score * 0.93), 6)\n'
        '            score_adjustments.append("derived backing-vocal residual confidence")',
    ),
    (
        '            "model": model_by_stem.get(name, "unknown"),',
        '            "model": model_name,',
    ),
    (
        '            "score_cap_reasons": cap_reasons,\n',
        '            "score_cap_reasons": cap_reasons,\n            "score_adjustments": score_adjustments,\n',
    ),
    (
        '        "score_type": "heuristic_research_signal_v2",',
        '        "score_type": "heuristic_research_signal_v3",',
    ),
    (
        '            "hard_quality_caps": True,\n',
        '            "hard_quality_caps": True,\n'
        '            "sparse_tonal_signal_floor": 0.70,\n'
        '            "derived_backing_vocal_factor": 0.93,\n',
    ),
]

for old, new in replacements:
    if old not in text:
        raise RuntimeError(f'QA v3 patch anchor missing: {old[:100]!r}')
    text = text.replace(old, new, 1)

path.write_text(text, encoding='utf-8')
print('LiteLABS QA v3 calibration applied')
