from pathlib import Path

path = Path('/app/handler.py')
text = path.read_text(encoding='utf-8')

marker = '    payload = job.get("input") or {}\n'
if marker not in text:
    raise RuntimeError(
        'Could not locate handler payload anchor for instrument probe'
    )

if 'instrument_probe_handler_v1' not in text:
    block = '''    # instrument_probe_handler_v1
    if str(payload.get("mode") or "").strip() == "instrument_probe":
        import json as _json
        import os as _os
        import subprocess as _subprocess
        import tempfile as _tempfile
        from pathlib import Path as _Path
        from urllib.parse import urlparse as _urlparse

        import requests as _requests

        audio_url = str(payload.get("audio_url") or "").strip()
        local_audio_path = str(
            payload.get("audio_path") or payload.get("local_audio_path") or ""
        ).strip()
        if not audio_url and not local_audio_path:
            return {
                "ok": False,
                "mode": "instrument_probe",
                "error": (
                    "instrument_probe requires input.audio_url "
                    "or input.audio_path"
                ),
            }

        try:
            timeout_seconds = max(
                30,
                int(payload.get("instrument_timeout_seconds") or 300),
            )
            with _tempfile.TemporaryDirectory(
                prefix="litelabs_instrument_probe_"
            ) as temp_dir:
                if local_audio_path:
                    input_path = _Path(local_audio_path).expanduser().resolve()
                    allowed_root = _Path("/tmp").resolve()
                    if allowed_root not in input_path.parents:
                        return {
                            "ok": False,
                            "mode": "instrument_probe",
                            "error": (
                                "instrument_probe local audio_path "
                                "must be inside /tmp"
                            ),
                        }
                    if not input_path.is_file():
                        return {
                            "ok": False,
                            "mode": "instrument_probe",
                            "error": f"Local audio file not found: {input_path}",
                        }
                    filename = str(
                        payload.get("filename") or input_path.name
                    )
                else:
                    filename = str(
                        payload.get("filename")
                        or _Path(_urlparse(audio_url).path).name
                        or "track.audio"
                    )
                    input_path = _Path(temp_dir) / filename
                    with _requests.get(
                        audio_url,
                        stream=True,
                        timeout=(30, 300),
                    ) as response:
                        response.raise_for_status()
                        with input_path.open("wb") as handle:
                            for chunk in response.iter_content(
                                chunk_size=1024 * 1024
                            ):
                                if chunk:
                                    handle.write(chunk)

                env = _os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = "-1"
                env["TF_CPP_MIN_LOG_LEVEL"] = "2"
                completed = _subprocess.run(
                    [
                        "python",
                        "-u",
                        "/app/instrument_probe.py",
                        str(input_path),
                    ],
                    stdout=_subprocess.PIPE,
                    stderr=_subprocess.PIPE,
                    text=True,
                    env=env,
                    timeout=timeout_seconds,
                    check=False,
                )
                if completed.returncode != 0:
                    return {
                        "ok": False,
                        "mode": "instrument_probe",
                        "error": "Inst-MTG instrument probe failed",
                        "returncode": completed.returncode,
                        "stderr_tail": (completed.stderr or "")[-4000:],
                        "stdout_tail": (completed.stdout or "")[-2000:],
                    }

                output_lines = [
                    line.strip()
                    for line in (completed.stdout or "").splitlines()
                    if line.strip()
                ]
                if not output_lines:
                    raise RuntimeError(
                        "Inst-MTG instrument probe returned no JSON output"
                    )
                result = _json.loads(output_lines[-1])
                result["filename"] = filename
                return result
        except Exception as exc:
            return {
                "ok": False,
                "mode": "instrument_probe",
                "error": str(exc),
                "error_type": exc.__class__.__name__,
            }

'''
    text = text.replace(marker, marker + '\n' + block, 1)

path.write_text(text, encoding='utf-8')

check = path.read_text(encoding='utf-8')
compile(check, str(path), 'exec')
assert 'instrument_probe_handler_v1' in check
assert '"/app/instrument_probe.py"' in check
print('LiteLABS isolated instrument probe handler route applied')
