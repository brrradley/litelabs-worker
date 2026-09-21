from pathlib import Path

qa_path = Path('/app/qa_research.py')
exp_path = Path('/app/experimental_children_v1.py')

qa = qa_path.read_text(encoding='utf-8')

# The f6d1 production image can contain the learning_observation field without
# the assignment block that creates it. That turns a post-delivery QA detail
# into a fatal UnboundLocalError after the expensive GPU work has completed.
source_anchor = '    source_audio, source_sr = _read(source)\n    source_mono = _mono(source_audio)\n'
if 'source_metrics = _signal_metrics(source_audio, source_sr)' not in qa:
    if source_anchor not in qa:
        raise RuntimeError('Could not locate QA source metrics anchor')
    qa = qa.replace(
        source_anchor,
        source_anchor + '    source_metrics = _signal_metrics(source_audio, source_sr)\n',
        1,
    )

learning_block = '''    model_families = sorted({\n        str(item.get("model") or "unknown")\n        for item in stem_records.values()\n    })\n    stem_scores = {\n        name: float(item.get("score", 0.0))\n        for name, item in stem_records.items()\n    }\n    scored_for_mean = [\n        float(item.get("score", 0.0))\n        for item in stem_records.values()\n        if str(item.get("semantic_role") or "") != "complement_residual"\n    ]\n    mean_stem_score = (\n        round(float(np.mean(scored_for_mean)), 6)\n        if scored_for_mean else None\n    )\n    stem_roles = {\n        name: str(item.get("semantic_role") or "unknown")\n        for name, item in stem_records.items()\n    }\n    timings = {}\n    if isinstance(extra, dict) and isinstance(extra.get("timings_seconds"), dict):\n        timings = {\n            str(key): round(float(value), 3)\n            for key, value in extra["timings_seconds"].items()\n            if isinstance(value, (int, float, np.number))\n        }\n    learning_observation = {\n        "schema_version": 1,\n        "phase": "post_delivery",\n        "purpose": "SET adaptive routing evidence; never block customer delivery",\n        "track_profile": {\n            "genre": genre,\n            "genre_reason": genre_reason,\n            "duration_seconds": round(len(source_mono) / max(source_sr, 1), 3),\n            "source_metrics": source_metrics,\n        },\n        "recipe": {\n            "preset": preset,\n            "pipeline_revision": pipeline_revision,\n            "models": model_families,\n        },\n        "outcomes": {\n            "mean_stem_score": mean_stem_score,\n            "stem_scores": stem_scores,\n            "stem_roles": stem_roles,\n            "mean_excludes_roles": ["complement_residual"],\n            "reconstruction_cosine": round(reconstruction_cosine, 6) if reconstruction_cosine is not None else None,\n            "group_reconstruction": {\n                key: round(value, 6) if value is not None else None\n                for key, value in group_reconstruction.items()\n            },\n        },\n        "timings_seconds": timings,\n    }\n\n'''
record_anchor = '    record = {\n'
if '    learning_observation = {\n' not in qa:
    if record_anchor not in qa:
        raise RuntimeError('Could not locate QA record anchor')
    qa = qa.replace(record_anchor, learning_block + record_anchor, 1)

# IMPORTANT: target the final record block, not the identically-named
# pipeline_revision field inside learning_observation["recipe"]. The previous
# broad replace inserted learning_observation into its own dict while it was
# still being assigned, causing the exact UnboundLocalError seen in production.
record_fields_anchor = (
    '        "preset": preset,\n'
    '        "pipeline_revision": pipeline_revision,\n'
    '        "reconstruction_cosine": round(reconstruction_cosine, 6) if reconstruction_cosine is not None else None,\n'
)
record_fields_replacement = (
    '        "preset": preset,\n'
    '        "pipeline_revision": pipeline_revision,\n'
    '        "source_metrics": source_metrics,\n'
    '        "learning_observation": learning_observation,\n'
    '        "reconstruction_cosine": round(reconstruction_cosine, 6) if reconstruction_cosine is not None else None,\n'
)
if '        "learning_observation": learning_observation,\n' not in qa:
    if record_fields_anchor not in qa:
        raise RuntimeError('Could not locate final QA record fields anchor')
    qa = qa.replace(record_fields_anchor, record_fields_replacement, 1)

# Promote detector/G400 metadata from the QA extra payload to stable top-level
# fields so LiteRECORDS can build its public README from the same evidence.
qa = qa.replace(
    '    if extra:\n        record["pipeline_metrics"] = extra\n',
    '    if extra:\n'
    '        record["pipeline_metrics"] = extra\n'
    '        for metadata_key in ("detected_instruments", "detected_by_family", "genre_top10", "genre_broad_families"):\n'
    '            if metadata_key in extra:\n'
    '                record[metadata_key] = extra[metadata_key]\n',
    1,
)

# Keep the silent QA hierarchy aligned with the production drum output. DrumSep
# still predicts hh+cymbals internally, but the customer-facing child is hats.
qa = qa.replace(
    '    "drum_children": ("kick", "snare", "toms", "hi_hats", "cymbals"),\n',
    '    "drum_children": ("kick", "snare", "toms", "hats"),\n',
    1,
)

qa_path.write_text(qa, encoding='utf-8')

# The experimental QA collector was written before hh+cymbals were merged.
# Teach it the new hats filename so it is scored in the drum child group.
exp = exp_path.read_text(encoding='utf-8')
old = '''            elif "drums_5stem_toms" in lower:\n                label, model = "toms", "MDX23C DrumSep 5-stem"\n            elif "drums_5stem_hh" in lower:\n                label, model = "hi_hats", "MDX23C DrumSep 5-stem"\n            elif "drums_5stem_cymbals" in lower:\n                label, model = "cymbals", "MDX23C DrumSep 5-stem"\n'''
new = '''            elif "drums_5stem_toms" in lower:\n                label, model = "toms", "MDX23C DrumSep 5-stem"\n            elif "drums_5stem_hats" in lower:\n                label, model = "hats", "MDX23C DrumSep 5-stem (hh+cymbals merged)"\n            elif "drums_5stem_hh" in lower:\n                label, model = "hi_hats", "MDX23C DrumSep 5-stem"\n            elif "drums_5stem_cymbals" in lower:\n                label, model = "cymbals", "MDX23C DrumSep 5-stem"\n'''
if '"drums_5stem_hats" in lower' not in exp:
    if old not in exp:
        raise RuntimeError('Could not locate experimental QA drum mapping')
    exp = exp.replace(old, new, 1)

exp_path.write_text(exp, encoding='utf-8')

# Static guarantees for the exact production regression we just saw.
qa_check = qa_path.read_text(encoding='utf-8')
exp_check = exp_path.read_text(encoding='utf-8')
assert 'learning_observation = {' in qa_check
assert '"learning_observation": learning_observation' in qa_check
assert qa_check.index('learning_observation = {') < qa_check.index('"learning_observation": learning_observation')
# Ensure the self-reference was NOT inserted inside the learning_observation
# recipe itself.
recipe_fragment = (
    '        "recipe": {\n'
    '            "preset": preset,\n'
    '            "pipeline_revision": pipeline_revision,\n'
    '            "models": model_families,\n'
    '        },\n'
)
assert recipe_fragment in qa_check
assert '"drum_children": ("kick", "snare", "toms", "hats")' in qa_check
assert '"drums_5stem_hats" in lower' in exp_check
print('LiteLABS QA learning observation + hats telemetry hotfix applied')
