from pathlib import Path

path = Path('/app/preset_pack.py')
text = path.read_text(encoding='utf-8')

if 'parent_analysis_v3' not in text:
    helper_anchor = 'def _format_execution_time(seconds: float) -> str:\n'
    if helper_anchor not in text:
        raise RuntimeError('Could not locate parent preset helper anchor')

    helpers = '''# parent_analysis_v3\ndef _litelabs_probe_result(proc, timeout: int = 120) -> dict:\n    try:\n        stdout, stderr = proc.communicate(timeout=timeout)\n    except Exception:\n        try:\n            proc.kill()\n        except Exception:\n            pass\n        return {}\n    if proc.returncode != 0:\n        if stderr:\n            print(stderr[-2000:], flush=True)\n        return {}\n    for line in reversed([line.strip() for line in (stdout or "").splitlines() if line.strip()]):\n        try:\n            parsed = json.loads(line)\n        except Exception:\n            continue\n        if isinstance(parsed, dict) and parsed.get("ok"):\n            return parsed\n    return {}\n\ndef _litelabs_clean_instruments(report: dict) -> tuple[list[str], list[str]]:\n    raw: list[str] = []\n    public: list[str] = []\n    aliases = {\n        "electricguitar": "Electric Guitar",\n        "doublebass": "Double Bass",\n        "drummachine": "Drum Machine",\n        "synthesizer": "Synthesizer",\n    }\n    suppressed = {"computer", "voice"}\n    for item in report.get("detected_instruments") or []:\n        if not isinstance(item, dict):\n            continue\n        label = str(item.get("label") or "").strip().lower()\n        if not label or label in suppressed or label in raw:\n            continue\n        raw.append(label)\n        public.append(aliases.get(label, label.replace("_", " ").title()))\n    return raw, public\n\ndef _litelabs_mmss(seconds: float) -> str:\n    total = max(0, int(round(seconds)))\n    minutes, secs = divmod(total, 60)\n    return f"{minutes:02d}:{secs:02d}"\n\n'''
    text = text.replace(helper_anchor, helpers + helper_anchor, 1)

    conversion_anchor = '''        if conv.returncode != 0:\n            return {"ok": False, "preset": preset, "failed_stage": "convert"}\n\n        emit("Running BS-RoFormer Parent Separation", 12)\n'''
    conversion_new = '''        if conv.returncode != 0:\n            return {"ok": False, "preset": preset, "failed_stage": "convert"}\n\n        analysis_started = time.monotonic()\n        analysis_env = {\n            **__import__("os").environ,\n            "CUDA_VISIBLE_DEVICES": "-1",\n            "TF_CPP_MIN_LOG_LEVEL": "2",\n        }\n        genre_proc = subprocess.Popen(\n            ["python", "-u", "/app/genre_probe.py", str(source)],\n            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=analysis_env,\n        )\n        instrument_proc = subprocess.Popen(\n            ["python", "-u", "/app/instrument_probe.py", str(source)],\n            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=analysis_env,\n        )\n\n        emit("Running BS-RoFormer Parent Separation", 12)\n'''
    if conversion_anchor not in text:
        raise RuntimeError('Could not locate parent conversion/separation anchor')
    text = text.replace(conversion_anchor, conversion_new, 1)

    old_analysis = '''        genre, genre_reason = _detect_parent_genre(stems, source)\n        print(f"LiteLABS detected genre: {genre} ({genre_reason})", flush=True)\n\n        exported: list[str] = []\n'''
    new_analysis = '''        genre_analysis = _litelabs_probe_result(genre_proc)\n        instrument_analysis = _litelabs_probe_result(instrument_proc)\n        timings["source_analysis"] = round(time.monotonic() - analysis_started, 3)\n\n        genre = str(genre_analysis.get("genre") or "").strip()\n        if genre:\n            genre_reason = str(genre_analysis.get("raw_genre") or "G400 validated detector")\n        else:\n            genre, genre_reason = _detect_parent_genre(stems, source)\n            genre_analysis = {"ok": False, "genre": genre, "fallback": True}\n\n        instrument_raw, instruments = _litelabs_clean_instruments(instrument_analysis)\n        instrument_inventory = {\n            "ran": bool(instrument_analysis),\n            "source": "inst_mtg" if instrument_analysis else "unavailable",\n            "detected": instrument_raw,\n        }\n        print(f"LiteLABS detected genre: {genre}", flush=True)\n        print(f"LiteLABS detected instruments: {instruments}", flush=True)\n\n        exported: list[str] = []\n'''
    if old_analysis not in text:
        raise RuntimeError('Could not locate parent legacy analysis block')
    text = text.replace(old_analysis, new_analysis, 1)

    readme_tail = '''        _write_readme(\n            final / "README.txt",\n            filename=supplied_filename,\n            preset=preset,\n            exported=exported,\n            genre=genre,\n            execution_seconds=execution_seconds,\n        )\n\n        report = {\n'''
    readme_tail_new = '''        _write_readme(\n            final / "README.txt",\n            filename=supplied_filename,\n            preset=preset,\n            exported=exported,\n            genre=genre,\n            execution_seconds=execution_seconds,\n        )\n\n        # Standardise Basic/Core README metadata with the validated Experimental\n        # analysis block while leaving the established parent-pack body intact.\n        readme_path = final / "README.txt"\n        readme_text = readme_path.read_text(encoding="utf-8")\n        readme_lines = [\n            line for line in readme_text.splitlines()\n            if not line.startswith("Detected genre: ")\n            and not line.startswith("Execution time: ")\n            and not line.startswith("Duration: ")\n        ]\n        readme_text = "\\n".join(readme_lines).rstrip() + "\\n"\n        analysis_block = (\n            f"Duration: {_litelabs_mmss(execution_seconds)}\\n\\n"\n            "ANALYSIS\\n"\n            "--------\\n"\n            f"Detected genre: {genre}\\n"\n            f"Detected instruments: {', '.join(instruments) if instruments else 'None detected'}\\n\\n"\n        )\n        included_marker = "INCLUDED STEMS\\n--------------\\n"\n        if included_marker in readme_text:\n            readme_text = readme_text.replace(included_marker, analysis_block + included_marker, 1)\n        else:\n            readme_text += "\\n" + analysis_block\n        readme_path.write_text(readme_text, encoding="utf-8")\n\n        report = {\n'''
    if readme_tail not in text:
        raise RuntimeError('Could not locate parent README call/report boundary')
    text = text.replace(readme_tail, readme_tail_new, 1)

    report_anchor = '''            "detected_genre": genre,\n            "genre_reason": genre_reason,\n            "execution_seconds": round(execution_seconds, 3),\n'''
    report_new = '''            "detected_genre": genre,\n            "detected_instruments": instruments,\n            "genre_reason": genre_reason,\n            "genre_analysis": genre_analysis,\n            "instrument_analysis": instrument_analysis,\n            "instrument_inventory": instrument_inventory,\n            "duration": _litelabs_mmss(execution_seconds),\n            "execution_seconds": round(execution_seconds, 3),\n'''
    if report_anchor not in text:
        raise RuntimeError('Could not locate parent report fields')
    text = text.replace(report_anchor, report_new, 1)

    return_anchor = '''            "files": sorted(exported),\n            "detected_genre": genre,\n            "execution_seconds": round(execution_seconds, 3),\n'''
    return_new = '''            "files": sorted(exported),\n            "detected_genre": genre,\n            "detected_instruments": instruments,\n            "genre_analysis": genre_analysis,\n            "instrument_analysis": instrument_analysis,\n            "instrument_inventory": instrument_inventory,\n            "report": report,\n            "duration": _litelabs_mmss(execution_seconds),\n            "execution_seconds": round(execution_seconds, 3),\n'''
    if return_anchor not in text:
        raise RuntimeError('Could not locate parent result fields')
    text = text.replace(return_anchor, return_new, 1)

    path.write_text(text, encoding='utf-8')

check = path.read_text(encoding='utf-8')
compile(check, str(path), 'exec')
assert 'parent_analysis_v3' in check
assert '/app/genre_probe.py' in check and '/app/instrument_probe.py' in check
assert 'Detected instruments:' in check
assert '"instrument_inventory": instrument_inventory' in check
assert '_litelabs_mmss(execution_seconds)' in check
print('LiteLABS Basic/Core validated G400 + Inst-MTG analysis patch applied')
