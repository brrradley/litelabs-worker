from pathlib import Path

path = Path('/app/experimental_children_v1.py')
text = path.read_text(encoding='utf-8')

# genre_probe_pre_stem_v1
# Run the already-validated G400 probe before any stem model starts. Launch it
# as soon as the source download is complete and overlap it with ffmpeg source
# preparation so its ~3-4 second runtime adds as little wall time as possible.
download_anchor = '        timings["download"] = round(time.monotonic() - t0, 3)\n\n'
if 'genre_probe_pre_stem_v1' not in text:
    if download_anchor not in text:
        raise RuntimeError('Could not locate post-download genre probe anchor')
    launch_block = '''        # genre_probe_pre_stem_v1
        genre_probe_started = time.monotonic()
        genre_probe_env = __import__("os").environ.copy()
        genre_probe_env["CUDA_VISIBLE_DEVICES"] = "-1"
        genre_probe_env["TF_CPP_MIN_LOG_LEVEL"] = "2"
        genre_probe_process = subprocess.Popen(
            ["python", "-u", "/app/genre_probe.py", str(downloaded)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=genre_probe_env,
        )
        genre_report = {
            "ok": False,
            "mode": "genre_probe",
            "genre": "Unverified",
            "raw_genre": "",
            "error": "not_completed",
        }

'''
    text = text.replace(download_anchor, download_anchor + launch_block, 1)

wait_anchor = '        rc, elapsed = _run_polled(\n'
if 'genre_probe_complete_before_stems_v1' not in text:
    if wait_anchor not in text:
        raise RuntimeError('Could not locate parent-separation command anchor')
    wait_block = '''        # genre_probe_complete_before_stems_v1
        try:
            probe_stdout, probe_stderr = genre_probe_process.communicate(
                timeout=max(30, min(timeout, 300))
            )
            timings["genre_probe"] = round(
                time.monotonic() - genre_probe_started,
                3,
            )
            if genre_probe_process.returncode == 0:
                probe_lines = [
                    line.strip()
                    for line in (probe_stdout or "").splitlines()
                    if line.strip()
                ]
                if not probe_lines:
                    raise RuntimeError("G400 genre probe returned no JSON")
                genre_report = json.loads(probe_lines[-1])
            else:
                genre_report = {
                    "ok": False,
                    "mode": "genre_probe",
                    "genre": "Unverified",
                    "raw_genre": "",
                    "error": "G400 genre probe failed",
                    "returncode": genre_probe_process.returncode,
                    "stderr_tail": (probe_stderr or "")[-2000:],
                }
        except Exception as exc:
            try:
                genre_probe_process.kill()
            except Exception:
                pass
            timings["genre_probe"] = round(
                time.monotonic() - genre_probe_started,
                3,
            )
            genre_report = {
                "ok": False,
                "mode": "genre_probe",
                "genre": "Unverified",
                "raw_genre": "",
                "error": str(exc),
                "error_type": exc.__class__.__name__,
            }

'''
    text = text.replace(wait_anchor, wait_block + wait_anchor, 1)

# Preserve the canonical genre analysis in the technical report as well.
if '"genre_analysis": genre_report' not in text:
    report_anchor = '        report = {\n'
    if report_anchor not in text:
        raise RuntimeError('Could not locate experimental report dict')
    text = text.replace(
        report_anchor,
        report_anchor + '            "genre_analysis": genre_report,\n',
        1,
    )

path.write_text(text, encoding='utf-8')

check = path.read_text(encoding='utf-8')
compile(check, str(path), 'exec')
assert 'genre_probe_pre_stem_v1' in check
assert 'genre_probe_complete_before_stems_v1' in check
assert '["python", "-u", "/app/genre_probe.py", str(downloaded)]' in check
assert check.index('genre_probe_complete_before_stems_v1') < check.index('rc, elapsed = _run_polled(')
assert '"genre_analysis": genre_report' in check
print('LiteLABS pre-stem G400 genre integration applied')
