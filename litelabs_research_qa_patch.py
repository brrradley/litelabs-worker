from pathlib import Path


def patch_preset_pack() -> None:
    path = Path('/app/preset_pack.py')
    text = path.read_text(encoding='utf-8')

    import_anchor = 'from wind_brass_decomposition_v2 import _run_polled\n'
    if 'from qa_research import build_research_qa' not in text:
        if import_anchor not in text:
            raise RuntimeError('Could not locate preset import anchor')
        text = text.replace(import_anchor, import_anchor + 'from qa_research import build_research_qa\n', 1)

    qa_anchor = '        execution_seconds = time.monotonic() - started\n'
    qa_insert = '''        qa_stems: dict[str, Path] = {}\n        for stem_file in final.glob(f"{track}_*.flac"):\n            logical = stem_file.stem[len(track) + 1:] if stem_file.stem.startswith(track + "_") else stem_file.stem\n            qa_stems[logical] = stem_file\n        qa_models = {name: "BS-RoFormer-SW" for name in qa_stems}\n        if "instrumental" in qa_models:\n            qa_models["instrumental"] = "BS-RoFormer-SW (derived instrumental)"\n        research_qa = build_research_qa(\n            source=source,\n            stems=qa_stems,\n            model_by_stem=qa_models,\n            filename=supplied_filename,\n            input_size_bytes=downloaded.stat().st_size,\n            input_format=Path(supplied_filename).suffix.lstrip(".") or Path(raw_name).suffix.lstrip("."),\n            genre=genre,\n            preset=preset,\n            pipeline_revision="rs-parent-v1",\n            job_id=payload.get("progress_job_id") or payload.get("job_id"),\n        )\n        print(f"LiteLABS silent research QA complete for {len(qa_stems)} stems", flush=True)\n\n'''
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
    qa_insert = '''        try:\n            from preset_pack import _detect_parent_genre\n            qa_genre, qa_genre_reason = _detect_parent_genre(stems, source)\n        except Exception as exc:\n            print(f"LiteLABS experimental QA genre fallback: {exc}", flush=True)\n            qa_genre, qa_genre_reason = "mixed_or_unknown", "genre analysis unavailable"\n\n        qa_stems: dict[str, Path] = {}\n        qa_models: dict[str, str] = {}\n\n        def add_qa_stem(label: str, candidate: Path, model: str) -> None:\n            if candidate.is_file() and label not in qa_stems:\n                qa_stems[label] = candidate\n                qa_models[label] = model\n\n        for candidate in final.glob("*.flac"):\n            lower = candidate.name.lower()\n            for label in ("vocals", "percussion", "bass", "strings", "keys", "other", "instrumental"):\n                if lower.endswith(f"_{label}.flac"):\n                    model = "BS-RoFormer-SW (derived instrumental)" if label == "instrumental" else "BS-RoFormer-SW"\n                    add_qa_stem(label, candidate, model)\n                    break\n\n        for candidate in experimental.glob("*.flac"):\n            lower = candidate.name.lower()\n            model = "experimental-specialist"\n            label = candidate.stem\n            if "lead_vocals" in lower:\n                label, model = "lead_vocals", "BS-RoFormer Karaoke"\n            elif "backing_vocals" in lower:\n                label, model = "backing_vocals", "BS-RoFormer Karaoke (parent-minus-lead)"\n            elif "drums_5stem_kick" in lower:\n                label, model = "kick", "MDX23C DrumSep 5-stem"\n            elif "drums_5stem_snare" in lower:\n                label, model = "snare", "MDX23C DrumSep 5-stem"\n            elif "drums_5stem_toms" in lower:\n                label, model = "toms", "MDX23C DrumSep 5-stem"\n            elif "drums_5stem_hh" in lower:\n                label, model = "hi_hats", "MDX23C DrumSep 5-stem"\n            elif "drums_5stem_cymbals" in lower:\n                label, model = "cymbals", "MDX23C DrumSep 5-stem"\n            elif "sax_specialist" in lower and ("_sax" in lower or lower.endswith("sax.flac")):\n                label, model = "saxophone", "filosax_demucs_v3_14.22_SDR.th"\n            elif "sax_specialist" in lower:\n                label, model = "sax_residual", "filosax_demucs_v3_14.22_SDR.th"\n            elif "wind_brass_residual" in lower:\n                label, model = "wind_brass_residual", "17_HP-Wind_Inst-UVR.pth"\n            elif "woodwind" in lower or "wind_brass" in lower:\n                label, model = "wind_brass", "17_HP-Wind_Inst-UVR.pth"\n            add_qa_stem(label, candidate, model)\n\n        qa_pipeline_metrics = {}\n        if isinstance(drum_report, dict):\n            qa_pipeline_metrics["drums_5stem"] = {\n                key: drum_report.get(key)\n                for key in ("parent_vs_children_sum_cosine", "residual_relative_to_parent_db", "common_export_gain", "raw_max_peak")\n                if key in drum_report\n            }\n        research_qa = build_research_qa(\n            source=source,\n            stems=qa_stems,\n            model_by_stem=qa_models,\n            filename=str(payload.get("filename") or raw_name),\n            input_size_bytes=downloaded.stat().st_size,\n            input_format=Path(str(payload.get("filename") or raw_name)).suffix.lstrip("."),\n            genre=qa_genre,\n            preset="experimental",\n            pipeline_revision="rs1-dr1-vx1-ir1-wb1-sx1",\n            job_id=payload.get("progress_job_id") or payload.get("job_id"),\n            extra=qa_pipeline_metrics,\n        )\n        print(f"LiteLABS silent research QA complete for {len(qa_stems)} stems", flush=True)\n\n'''
    if 'qa_pipeline_metrics = {}' not in text:
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


patch_preset_pack()
patch_experimental()
print('LiteLABS silent research QA wiring applied')
