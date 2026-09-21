from pathlib import Path

path = Path('/app/experimental_children_v1.py')
text = path.read_text(encoding='utf-8')

# validated_analysis_pre_stem_v2
# Launch both validated Essentia detectors immediately after download. They run
# concurrently, and overlap with the existing ffmpeg source preparation, so the
# analysis wall time is approximately the slower probe rather than their sum.
download_anchor = '        timings["download"] = round(time.monotonic() - t0, 3)\n\n'
if 'validated_analysis_pre_stem_v2' not in text:
    if download_anchor not in text:
        raise RuntimeError('Could not locate post-download analysis anchor')

    launch_block = '''        # validated_analysis_pre_stem_v2
        analysis_started = time.monotonic()
        analysis_env = __import__("os").environ.copy()
        analysis_env["CUDA_VISIBLE_DEVICES"] = "-1"
        analysis_env["TF_CPP_MIN_LOG_LEVEL"] = "2"

        genre_probe_process = subprocess.Popen(
            ["python", "-u", "/app/genre_probe.py", str(downloaded)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=analysis_env,
        )
        instrument_probe_process = subprocess.Popen(
            ["python", "-u", "/app/instrument_probe.py", str(downloaded)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=analysis_env,
        )

        genre_report = {
            "ok": False,
            "mode": "genre_probe",
            "genre": "Unverified",
            "raw_genre": "",
            "error": "not_completed",
        }
        instrument_report = {
            "ok": False,
            "mode": "instrument_probe",
            "detected_instruments": [],
            "error": "not_completed",
        }

'''
    text = text.replace(download_anchor, download_anchor + launch_block, 1)

wait_anchor = '        rc, elapsed = _run_polled(\n'
if 'validated_analysis_complete_before_stems_v2' not in text:
    if wait_anchor not in text:
        raise RuntimeError('Could not locate parent-separation command anchor')

    wait_block = '''        # validated_analysis_complete_before_stems_v2
        analysis_timeout = max(30, min(timeout, 300))

        def _collect_probe(process, label, fallback):
            try:
                probe_stdout, probe_stderr = process.communicate(
                    timeout=analysis_timeout
                )
                if process.returncode != 0:
                    result = dict(fallback)
                    result.update({
                        "error": f"{label} failed",
                        "returncode": process.returncode,
                        "stderr_tail": (probe_stderr or "")[-2000:],
                    })
                    return result

                probe_lines = [
                    line.strip()
                    for line in (probe_stdout or "").splitlines()
                    if line.strip()
                ]
                if not probe_lines:
                    raise RuntimeError(f"{label} returned no JSON")
                return json.loads(probe_lines[-1])
            except Exception as exc:
                try:
                    process.kill()
                except Exception:
                    pass
                result = dict(fallback)
                result.update({
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                })
                return result

        genre_report = _collect_probe(
            genre_probe_process,
            "G400 genre probe",
            {
                "ok": False,
                "mode": "genre_probe",
                "genre": "Unverified",
                "raw_genre": "",
            },
        )
        instrument_report = _collect_probe(
            instrument_probe_process,
            "Inst-MTG instrument probe",
            {
                "ok": False,
                "mode": "instrument_probe",
                "detected_instruments": [],
            },
        )
        timings["source_analysis"] = round(
            time.monotonic() - analysis_started,
            3,
        )
        if genre_report.get("runtime_seconds") is not None:
            timings["genre_probe"] = genre_report.get("runtime_seconds")
        if instrument_report.get("runtime_seconds") is not None:
            timings["instrument_probe"] = instrument_report.get(
                "runtime_seconds"
            )

'''
    text = text.replace(wait_anchor, wait_block + wait_anchor, 1)

# Keep both validated detector outputs in the technical report. These are
# observational results only at this stage; this patch does not alter routing.
if '"genre_analysis": genre_report' not in text:
    report_anchor = '        report = {\n'
    if report_anchor not in text:
        raise RuntimeError('Could not locate experimental report dict')
    text = text.replace(
        report_anchor,
        report_anchor + '            "genre_analysis": genre_report,\n',
        1,
    )

if '"instrument_analysis": instrument_report' not in text:
    report_anchor = '        report = {\n'
    if report_anchor not in text:
        raise RuntimeError('Could not locate experimental report dict')
    text = text.replace(
        report_anchor,
        report_anchor + '            "instrument_analysis": instrument_report,\n',
        1,
    )

path.write_text(text, encoding='utf-8')

check = path.read_text(encoding='utf-8')
compile(check, str(path), 'exec')
assert 'validated_analysis_pre_stem_v2' in check
assert 'validated_analysis_complete_before_stems_v2' in check
assert '["python", "-u", "/app/genre_probe.py", str(downloaded)]' in check
assert '["python", "-u", "/app/instrument_probe.py", str(downloaded)]' in check
assert check.index('validated_analysis_complete_before_stems_v2') < check.index(
    'rc, elapsed = _run_polled('
)
assert '"genre_analysis": genre_report' in check
assert '"instrument_analysis": instrument_report' in check
print('LiteLABS concurrent pre-stem genre + instrument analysis applied')
