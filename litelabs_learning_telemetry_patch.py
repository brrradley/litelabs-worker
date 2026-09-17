from pathlib import Path


QA_PATH = Path('/app/qa_research.py')
WIRING_PATH = Path('/app/litelabs_research_qa_patch.py')


def patch_qa() -> None:
    text = QA_PATH.read_text(encoding='utf-8')

    anchor = '    source_audio, source_sr = _read(source)\n    source_mono = _mono(source_audio)\n'
    replacement = anchor + '    source_metrics = _signal_metrics(source_audio, source_sr)\n'
    if 'source_metrics = _signal_metrics(source_audio, source_sr)' not in text:
        if anchor not in text:
            raise RuntimeError('Could not locate QA source-metrics anchor')
        text = text.replace(anchor, replacement, 1)

    record_anchor = '    record = {\n'
    learning_block = '''    model_families = sorted({\n        str(record.get("model") or "unknown")\n        for record in stem_records.values()\n    })\n    stem_scores = {\n        name: float(record.get("score", 0.0))\n        for name, record in stem_records.items()\n    }\n    mean_stem_score = (\n        round(float(np.mean(list(stem_scores.values()))), 6)\n        if stem_scores else None\n    )\n    timings = {}\n    if isinstance(extra, dict) and isinstance(extra.get("timings_seconds"), dict):\n        timings = {\n            str(key): round(float(value), 3)\n            for key, value in extra["timings_seconds"].items()\n            if isinstance(value, (int, float, np.number))\n        }\n    learning_observation = {\n        "schema_version": 1,\n        "phase": "post_delivery",\n        "purpose": "SET adaptive routing evidence; never block customer delivery",\n        "track_profile": {\n            "genre": genre,\n            "genre_reason": genre_reason,\n            "duration_seconds": round(len(source_mono) / max(source_sr, 1), 3),\n            "source_metrics": source_metrics,\n        },\n        "recipe": {\n            "preset": preset,\n            "pipeline_revision": pipeline_revision,\n            "models": model_families,\n        },\n        "outcomes": {\n            "mean_stem_score": mean_stem_score,\n            "stem_scores": stem_scores,\n            "reconstruction_cosine": round(reconstruction_cosine, 6) if reconstruction_cosine is not None else None,\n            "group_reconstruction": {\n                key: round(value, 6) if value is not None else None\n                for key, value in group_reconstruction.items()\n            },\n        },\n        "timings_seconds": timings,\n    }\n\n'''
    if '"phase": "post_delivery"' not in text:
        if record_anchor not in text:
            raise RuntimeError('Could not locate QA record anchor')
        text = text.replace(record_anchor, learning_block + record_anchor, 1)

    field_anchor = '        "pipeline_revision": pipeline_revision,\n'
    field_replacement = field_anchor + '        "source_metrics": source_metrics,\n        "learning_observation": learning_observation,\n'
    if '        "learning_observation": learning_observation,\n' not in text:
        if field_anchor not in text:
            raise RuntimeError('Could not locate QA learning-field anchor')
        text = text.replace(field_anchor, field_replacement, 1)

    QA_PATH.write_text(text, encoding='utf-8')


def patch_wiring() -> None:
    text = WIRING_PATH.read_text(encoding='utf-8')

    preset_anchor = '            pipeline_revision="rs-parent-v1",\\n            job_id=payload.get("progress_job_id") or payload.get("job_id"),\\n'
    preset_replacement = '            pipeline_revision="rs-parent-v1",\\n            job_id=payload.get("progress_job_id") or payload.get("job_id"),\\n            genre_reason=genre_reason,\\n            extra={"timings_seconds": dict(timings)},\\n'
    if 'extra={"timings_seconds": dict(timings)}' not in text:
        if preset_anchor not in text:
            raise RuntimeError('Could not locate preset QA wiring anchor')
        text = text.replace(preset_anchor, preset_replacement, 1)

    experimental_anchor = '        qa_pipeline_metrics = {}\\n        if isinstance(drum_report, dict):\\n'
    experimental_replacement = '        qa_pipeline_metrics = {"timings_seconds": dict(timings)}\\n        if isinstance(drum_report, dict):\\n'
    if 'qa_pipeline_metrics = {"timings_seconds": dict(timings)}' not in text:
        if experimental_anchor not in text:
            raise RuntimeError('Could not locate experimental timing wiring anchor')
        text = text.replace(experimental_anchor, experimental_replacement, 1)

    exp_call_anchor = '            job_id=payload.get("progress_job_id") or payload.get("job_id"),\\n            extra=qa_pipeline_metrics,\\n'
    exp_call_replacement = '            job_id=payload.get("progress_job_id") or payload.get("job_id"),\\n            extra=qa_pipeline_metrics,\\n            genre_reason=qa_genre_reason,\\n'
    if 'genre_reason=qa_genre_reason' not in text:
        if exp_call_anchor not in text:
            raise RuntimeError('Could not locate experimental QA genre-reason anchor')
        text = text.replace(exp_call_anchor, exp_call_replacement, 1)

    WIRING_PATH.write_text(text, encoding='utf-8')


patch_qa()
patch_wiring()
print('SET post-delivery learning telemetry patch applied')
