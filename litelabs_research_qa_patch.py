from pathlib import Path


def patch_qa_learning_telemetry() -> None:
    path = Path('/app/qa_research.py')
    text = path.read_text(encoding='utf-8')

    source_anchor = '    source_audio, source_sr = _read(source)\n    source_mono = _mono(source_audio)\n'
    if 'source_metrics = _signal_metrics(source_audio, source_sr)' not in text:
        if source_anchor not in text:
            raise RuntimeError('Could not locate QA source metrics anchor')
        text = text.replace(
            source_anchor,
            source_anchor + '    source_metrics = _signal_metrics(source_audio, source_sr)\n',
            1,
        )

    record_anchor = '    record = {\n'
    learning_block = '''    model_families = sorted({\n        str(item.get("model") or "unknown")\n        for item in stem_records.values()\n    })\n    stem_scores = {\n        name: float(item.get("score", 0.0))\n        for name, item in stem_records.items()\n    }\n    mean_stem_score = (\n        round(float(np.mean(list(stem_scores.values()))), 6)\n        if stem_scores else None\n    )\n    timings = {}\n    if isinstance(extra, dict) and isinstance(extra.get("timings_seconds"), dict):\n        timings = {\n            str(key): round(float(value), 3)\n            for key, value in extra["timings_seconds"].items()\n            if isinstance(value, (int, float, np.number))\n        }\n    learning_observation = {\n        "schema_version": 1,\n        "phase": "post_delivery",\n        "purpose": "SET adaptive routing evidence; never block customer delivery",\n        "track_profile": {\n            "genre": genre,\n            "genre_reason": genre_reason,\n            "duration_seconds": round(len(source_mono) / max(source_sr, 1), 3),\n            "source_metrics": source_metrics,\n        },\n        "recipe": {\n            "preset": preset,\n            "pipeline_revision": pipeline_revision,\n            "models": model_families,\n        },\n        "outcomes": {\n            "mean_stem_score": mean_stem_score,\n            "stem_scores": stem_scores,\n            "reconstruction_cosine": round(reconstruction_cosine, 6) if reconstruction_cosine is not None else None,\n            "group_reconstruction": {\n                key: round(value, 6) if value is not None else None\n                for key, value in group_reconstruction.items()\n            },\n        },\n        "timings_seconds": timings,\n    }\n\n'''
    if '"phase": "post_delivery"' not in text:
        if record_anchor not in text:
            raise RuntimeError('Could not locate QA learning record anchor')
        text = text.replace(record_anchor, learning_block + record_anchor, 1)

    field_anchor = '        "pipeline_revision": pipeline_revision,\n'
    if '        "learning_observation": learning_observation,\n' not in text:
        if field_anchor not in text:
            raise RuntimeError('Could not locate QA learning field anchor')
        text = text.replace(
            field_anchor,
            field_anchor + '        "source_metrics": source_metrics,\n        "learning_observation": learning_observation,\n',
            1,
        )

    path.write_text(text, encoding='utf-8')


def patch_preset_pack() -> None:
    path = Path('/app/preset_pack.py')
    text = path.read_text(encoding='utf-8')

    import_anchor = 'from wind_brass_decomposition_v2 import _run_polled\n'
    if 'from qa_research import build_research_qa' not in text:
        if import_anchor not in text:
            raise RuntimeError('Could not locate preset import anchor')
        text = text.replace(import_anchor, import_anchor + 'from qa_research import build_research_qa\n', 1)

    qa_anchor = '        execution_seconds = time.monotonic() - started\n'
    qa_insert = '''        qa_stems: dict[str, Path] = {}\n        for stem_file in final.glob(f"{track}_*.flac"):\n            logical = stem_file.stem[len(track) + 1:] if stem_file.stem.startswith(track + "_") else stem_file.stem\n            qa_stems[logical] = stem_file\n        qa_models = {name: "BS-RoFormer-SW" for name in qa_stems}\n        if "instrumental" in qa_models:\n            qa_models["instrumental"] = "BS-RoFormer-SW (derived instrumental)"\n        research_qa = build_research_qa(\n            source=source,\n            stems=qa_stems,\n            model_by_stem=qa_models,\n            filename=supplied_filename,\n            input_size_bytes=downloaded.stat().st_size,\n            input_format=Path(supplied_filename).suffix.lstrip(".") or Path(raw_name).suffix.lstrip("."),\n            genre=genre,\n            preset=preset,\n            pipeline_revision="rs-parent-v1",\n            job_id=payload.get("progress_job_id") or payload.get("job_id"),\n            genre_reason=genre_reason,\n            extra={"timings_seconds": dict(timings)},\n        )\n        print(f"LiteLABS silent research QA complete for {len(qa_stems)} stems", flush=True)\n\n'''
    if 'research_qa = build_research_qa(' not in text:
        if qa_anchor not in text:
            raise RuntimeError('Could not locate preset QA anchor')
        text = text.replace(qa_anchor, qa_insert + qa_anchor, 1)

    return_anchor = '            "specialist_separators_run": [],\n            "timings_seconds": timings,\n        }\n'
    if '            "research_qa": research_qa,\n            "timings_seconds": timings,\n        }\n' not in text:
        idx = text.rfind(return_anchor)
        if idx < 0:
            raise RuntimeError('Could not locate preset return anchor')
        replacement = '            "specialist_separators_run": [],\n            "research_qa": research_qa,\n            "timings_seconds": timings,\n        }\n'
        text = text[:idx] + replacement + text[idx + len(return_anchor):]

    path.write_text(text, encoding='utf-8')


def patch_experimental() -> None:
    path = Path('/app/experimental_children_v1.py')
    text = path.read_text(encoding='utf-8')

    import_anchor = 'from wind_brass_decomposition_v2 import _cos, _run_polled\n'
    if 'from qa_research import build_research_qa' not in text:
        if import_anchor not in text:
            raise RuntimeError('Could not locate experimental import anchor')
        text = text.replace(import_anchor, import_anchor + 'from qa_research import build_research_qa\n', 1)

    report_anchor = '        report = {\n'
    qa_insert = '''        try:\n            from preset_pack import _detect_parent_genre\n            qa_genre, qa_genre_reason = _detect_parent_genre(stems, source)\n        except Exception as exc:\n            print(f"LiteLABS experimental QA genre fallback: {exc}", flush=True)\n            qa_genre, qa_genre_reason = "mixed_or_unknown", "genre analysis unavailable"\n\n        qa_stems: dict[str, Path] = {}\n        qa_models: dict[str, str] = {}\n\n        def add_qa_stem(label: str, candidate: Path, model: str) -> None:\n            if candidate.is_file() and label not in qa_stems:\n                qa_stems[label] = candidate\n                qa_models[label] = model\n\n        for candidate in final.glob("*.flac"):\n            lower = candidate.name.lower()\n            for label in ("vocals", "percussion", "bass", "strings", "keys", "other", "instrumental"):\n                if lower.endswith(f"_{label}.flac"):\n                    model = "BS-RoFormer-SW (derived instrumental)" if label == "instrumental" else "BS-RoFormer-SW"\n                    add_qa_stem(label, candidate, model)\n                    break\n\n        for candidate in experimental.glob("*.flac"):\n            lower = candidate.name.lower()\n            model = "experimental-specialist"\n            label = candidate.stem\n            if "lead_vocals" in lower:\n                label, model = "lead_vocals", "Becruily Karaoke (25% SW + 75% MelBand parent)"\n            elif "backing_vocals" in lower:\n                label, model = "backing_vocals", "Becruily Karaoke (SW parent-minus-fast-lead)"\n            elif "drums_5stem_kick" in lower:\n                label, model = "kick", "MDX23C DrumSep 5-stem"\n            elif "drums_5stem_snare" in lower:\n                label, model = "snare", "MDX23C DrumSep 5-stem"\n            elif "drums_5stem_toms" in lower:\n                label, model = "toms", "MDX23C DrumSep 5-stem"\n            elif "drums_5stem_hh" in lower:\n                label, model = "hi_hats", "MDX23C DrumSep 5-stem"\n            elif "drums_5stem_cymbals" in lower:\n                label, model = "cymbals", "MDX23C DrumSep 5-stem"\n            elif "sax_specialist" in lower and ("_sax" in lower or lower.endswith("sax.flac")):\n                label, model = "saxophone", "filosax_demucs_v3_14.22_SDR.th"\n            elif "sax_specialist" in lower:\n                label, model = "sax_residual", "filosax_demucs_v3_14.22_SDR.th"\n            elif "wind_brass_residual" in lower:\n                label, model = "wind_brass_residual", "17_HP-Wind_Inst-UVR.pth"\n            elif "woodwind" in lower or "wind_brass" in lower:\n                label, model = "wind_brass", "17_HP-Wind_Inst-UVR.pth"\n            add_qa_stem(label, candidate, model)\n\n        qa_pipeline_metrics = {"timings_seconds": dict(timings)}\n        if isinstance(vocal_report, dict):\n            qa_pipeline_metrics["lead_backing_vocals"] = dict(vocal_report)\n        if isinstance(drum_report, dict):\n            qa_pipeline_metrics["drums_5stem"] = {\n                key: drum_report.get(key)\n                for key in ("parent_vs_children_sum_cosine", "residual_relative_to_parent_db", "common_export_gain", "raw_max_peak")\n                if key in drum_report\n            }\n        research_qa = build_research_qa(\n            source=source,\n            stems=qa_stems,\n            model_by_stem=qa_models,\n            filename=str(payload.get("filename") or raw_name),\n            input_size_bytes=downloaded.stat().st_size,\n            input_format=Path(str(payload.get("filename") or raw_name)).suffix.lstrip("."),\n            genre=qa_genre,\n            preset="experimental",\n            pipeline_revision="rs1-dr1-vx1-ir1-wb1-sx1",\n            job_id=payload.get("progress_job_id") or payload.get("job_id"),\n            extra=qa_pipeline_metrics,\n            genre_reason=qa_genre_reason,\n        )\n        print(f"LiteLABS silent research QA complete for {len(qa_stems)} stems", flush=True)\n\n'''
    if 'qa_pipeline_metrics = {"timings_seconds": dict(timings)}' not in text:
        if report_anchor not in text:
            raise RuntimeError('Could not locate experimental report anchor')
        text = text.replace(report_anchor, qa_insert + report_anchor, 1)

    # Keep QA out of the downloadable report; expose it only in the worker result so
    # LiteRECORDS can persist it in admin-only research logs.
    return_anchor = '            "report": report,\n        })'
    if '            "research_qa": research_qa,\n            "report": report,\n        })' not in text:
        if return_anchor not in text:
            raise RuntimeError('Could not locate experimental return anchor')
        text = text.replace(return_anchor, '            "research_qa": research_qa,\n            "report": report,\n        })', 1)

    path.write_text(text, encoding='utf-8')


patch_qa_learning_telemetry()
patch_preset_pack()
patch_experimental()
print('LiteLABS silent research QA + SET post-delivery learning telemetry wiring applied')
