from pathlib import Path

path = Path('/app/handler.py')
text = path.read_text(encoding='utf-8')

marker = 'print("LiteLABS handler ready", flush=True)\nrunpod.serverless.start({"handler": handler})\n'
if marker not in text:
    raise RuntimeError('Could not locate RunPod handler startup marker')

replacement = '''# Attach the exact image build revision to every dictionary response. This wraps\n# all existing branches, including health/capabilities requests injected by older\n# production layers, without changing their behaviour.\n_BUILD_SHA = os.getenv("LITELABS_BUILD_SHA", "unknown").strip() or "unknown"\n_original_handler = handler\n\ndef handler(job: dict) -> dict:\n    result = _original_handler(job)\n    if isinstance(result, dict):\n        result.setdefault("build_sha", _BUILD_SHA)\n    return result\n\nprint(f"LiteLABS handler ready (build {_BUILD_SHA})", flush=True)\nrunpod.serverless.start({"handler": handler})\n'''

text = text.replace(marker, replacement, 1)
path.write_text(text, encoding='utf-8')
print('LiteLABS build identity response wrapper applied')

# Basic/Core use the same validated G400 + Inst-MTG preflight analysis as
# Experimental. Patch the inherited preset worker late in the image build so the
# behaviour is identical regardless of the historical base-image revision.
preset_path = Path('/app/preset_pack.py')
preset = preset_path.read_text(encoding='utf-8')

if 'parent_analysis_v2' not in preset:
    helper_anchor = 'def _format_execution_time(seconds: float) -> str:\n'
    if helper_anchor not in preset:
        raise RuntimeError('Could not locate parent preset duration helper')

    helpers = '''# parent_analysis_v2\ndef _litelabs_probe_result(proc, timeout: int = 120) -> dict:\n    try:\n        stdout, stderr = proc.communicate(timeout=timeout)\n    except Exception:\n        try:\n            proc.kill()\n        except Exception:\n            pass\n        return {}\n    if proc.returncode != 0:\n        if stderr:\n            print(stderr[-2000:], flush=True)\n        return {}\n    for line in reversed([line.strip() for line in (stdout or "").splitlines() if line.strip()]):\n        try:\n            parsed = json.loads(line)\n        except Exception:\n            continue\n        if isinstance(parsed, dict) and parsed.get("ok"):\n            return parsed\n    return {}\n\ndef _litelabs_clean_instruments(report: dict) -> tuple[list[str], list[str]]:\n    raw: list[str] = []\n    public: list[str] = []\n    aliases = {\n        "electricguitar": "Electric Guitar",\n        "doublebass": "Double Bass",\n        "drummachine": "Drum Machine",\n        "synthesizer": "Synthesizer",\n    }\n    suppressed = {"computer", "voice"}\n    for item in report.get("detected_instruments") or []:\n        if not isinstance(item, dict):\n            continue\n        label = str(item.get("label") or "").strip().lower()\n        if not label or label in suppressed or label in raw:\n            continue\n        raw.append(label)\n        public.append(aliases.get(label, label.replace("_", " ").title()))\n    return raw, public\n\n'''
    preset = preset.replace(helper_anchor, helpers + helper_anchor, 1)

    old_time = '''def _format_execution_time(seconds: float) -> str:\n    seconds = max(0, int(round(seconds)))\n    minutes, seconds = divmod(seconds, 60)\n    hours, minutes = divmod(minutes, 60)\n    if hours:\n        return f"{hours}h {minutes}m {seconds}s"\n    if minutes:\n        return f"{minutes}m {seconds}s"\n    return f"{seconds}s"\n'''
    new_time = '''def _format_execution_time(seconds: float) -> str:\n    total_seconds = max(0, int(round(seconds)))\n    minutes, seconds = divmod(total_seconds, 60)\n    return f"{minutes:02d}:{seconds:02d}"\n'''
    if old_time not in preset:
        raise RuntimeError('Could not locate old parent preset duration formatter')
    preset = preset.replace(old_time, new_time, 1)

    signature = '''    exported: list[str],\n    genre: str,\n    execution_seconds: float,\n) -> None:\n'''
    signature_new = '''    exported: list[str],\n    genre: str,\n    instruments: list[str],\n    execution_seconds: float,\n) -> None:\n'''
    if signature not in preset:
        raise RuntimeError('Could not locate parent README signature')
    preset = preset.replace(signature, signature_new, 1)

    readme_old = '''        "Output format: FLAC\\n"\n        f"Detected genre: {genre}\\n"\n        f"Execution time: {_format_execution_time(execution_seconds)}\\n\\n"\n        "INCLUDED STEMS\\n"\n'''
    readme_new = '''        "Output format: FLAC\\n"\n        f"Duration: {_format_execution_time(execution_seconds)}\\n\\n"\n        "ANALYSIS\\n"\n        "--------\\n"\n        f"Detected genre: {genre}\\n"\n        f"Detected instruments: {', '.join(instruments) if instruments else 'None detected'}\\n\\n"\n        "INCLUDED STEMS\\n"\n'''
    if readme_old not in preset:
        raise RuntimeError('Could not locate parent README analysis lines')
    preset = preset.replace(readme_old, readme_new, 1)

    conversion_anchor = '''        if conv.returncode != 0:\n            return {"ok": False, "preset": preset, "failed_stage": "convert"}\n\n        emit("Running BS-RoFormer Parent Separation", 12)\n'''
    conversion_new = '''        if conv.returncode != 0:\n            return {"ok": False, "preset": preset, "failed_stage": "convert"}\n\n        analysis_started = time.monotonic()\n        analysis_env = {\n            **__import__("os").environ,\n            "CUDA_VISIBLE_DEVICES": "-1",\n            "TF_CPP_MIN_LOG_LEVEL": "2",\n        }\n        genre_proc = subprocess.Popen(\n            ["python", "-u", "/app/genre_probe.py", str(source)],\n            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=analysis_env,\n        )\n        instrument_proc = subprocess.Popen(\n            ["python", "-u", "/app/instrument_probe.py", str(source)],\n            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=analysis_env,\n        )\n\n        emit("Running BS-RoFormer Parent Separation", 12)\n'''
    if conversion_anchor not in preset:
        raise RuntimeError('Could not locate parent post-conversion anchor')
    preset = preset.replace(conversion_anchor, conversion_new, 1)

    old_analysis = '''        genre, genre_reason = _detect_parent_genre(stems, source)\n        print(f"LiteLABS detected genre: {genre} ({genre_reason})", flush=True)\n\n        exported: list[str] = []\n'''
    new_analysis = '''        genre_analysis = _litelabs_probe_result(genre_proc)\n        instrument_analysis = _litelabs_probe_result(instrument_proc)\n        timings["source_analysis"] = round(time.monotonic() - analysis_started, 3)\n\n        genre = str(genre_analysis.get("genre") or "").strip()\n        if genre:\n            genre_reason = str(genre_analysis.get("raw_genre") or "G400 validated detector")\n        else:\n            genre, genre_reason = _detect_parent_genre(stems, source)\n            genre_analysis = {"ok": False, "genre": genre, "fallback": True}\n\n        instrument_raw, instruments = _litelabs_clean_instruments(instrument_analysis)\n        instrument_inventory = {\n            "ran": bool(instrument_analysis),\n            "source": "inst_mtg" if instrument_analysis else "unavailable",\n            "detected": instrument_raw,\n        }\n        print(f"LiteLABS detected genre: {genre}", flush=True)\n        print(f"LiteLABS detected instruments: {instruments}", flush=True)\n\n        exported: list[str] = []\n'''
    if old_analysis not in preset:
        raise RuntimeError('Could not locate parent legacy genre block')
    preset = preset.replace(old_analysis, new_analysis, 1)

    readme_call = '''            exported=exported,\n            genre=genre,\n            execution_seconds=execution_seconds,\n'''
    readme_call_new = '''            exported=exported,\n            genre=genre,\n            instruments=instruments,\n            execution_seconds=execution_seconds,\n'''
    if readme_call not in preset:
        raise RuntimeError('Could not locate parent README call')
    preset = preset.replace(readme_call, readme_call_new, 1)

    report_anchor = '''            "detected_genre": genre,\n            "genre_reason": genre_reason,\n            "execution_seconds": round(execution_seconds, 3),\n'''
    report_new = '''            "detected_genre": genre,\n            "detected_instruments": instruments,\n            "genre_reason": genre_reason,\n            "genre_analysis": genre_analysis,\n            "instrument_analysis": instrument_analysis,\n            "instrument_inventory": instrument_inventory,\n            "execution_seconds": round(execution_seconds, 3),\n'''
    if report_anchor not in preset:
        raise RuntimeError('Could not locate parent report analysis fields')
    preset = preset.replace(report_anchor, report_new, 1)

    return_anchor = '''            "files": sorted(exported),\n            "detected_genre": genre,\n            "execution_seconds": round(execution_seconds, 3),\n'''
    return_new = '''            "files": sorted(exported),\n            "detected_genre": genre,\n            "detected_instruments": instruments,\n            "genre_analysis": genre_analysis,\n            "instrument_analysis": instrument_analysis,\n            "instrument_inventory": instrument_inventory,\n            "report": report,\n            "duration": _format_execution_time(execution_seconds),\n            "execution_seconds": round(execution_seconds, 3),\n'''
    if return_anchor not in preset:
        raise RuntimeError('Could not locate parent return analysis fields')
    preset = preset.replace(return_anchor, return_new, 1)

    preset_path.write_text(preset, encoding='utf-8')

check = preset_path.read_text(encoding='utf-8')
compile(check, str(preset_path), 'exec')
assert 'parent_analysis_v2' in check
assert 'genre_probe.py' in check and 'instrument_probe.py' in check
assert 'Detected instruments:' in check
assert 'return f"{minutes:02d}:{seconds:02d}"' in check
assert '"instrument_inventory": instrument_inventory' in check
print('LiteLABS Basic/Core validated analysis + MM:SS patch applied')
