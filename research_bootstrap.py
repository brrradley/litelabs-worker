from __future__ import annotations

import os
import py_compile
import runpy
import sys
import traceback
from pathlib import Path


def log(message: str) -> None:
    print(f"[LiteLABS research bootstrap] {message}", flush=True)


def main() -> None:
    log("starting")
    log(f"python: {sys.version}")
    log(f"cwd: {Path.cwd()}")
    log(f"argv: {sys.argv}")
    log(f"PYTHONPATH: {os.getenv('PYTHONPATH', '')}")
    log(f"build: {os.getenv('LITELABS_RESEARCH_BUILD', 'unknown')}")

    app_dir = Path('/app')
    log("/app listing:")
    try:
        for item in sorted(app_dir.iterdir()):
            kind = "dir" if item.is_dir() else "file"
            size = item.stat().st_size if item.is_file() else 0
            log(f" - {item.name} [{kind}] {size} bytes")
    except Exception as exc:
        log(f"could not list /app: {exc!r}")

    # Apply the multitrack metadata guard again at container startup. This is
    # intentionally redundant with the Docker build step: it makes the runtime
    # fail-safe even if an inherited/cached layer ever exposes the unpatched
    # campaign source. The patch is idempotent.
    macos_patch = app_dir / 'litelabs_multitrack_macos_patch.py'
    if macos_patch.exists():
        log("applying multitrack macOS metadata guard at runtime")
        try:
            runpy.run_path(str(macos_patch), run_name='__main__')
            log("multitrack macOS metadata guard applied")
        except Exception:
            log("multitrack macOS metadata guard FAILED")
            traceback.print_exc()
            raise

    for filename in ['handler.py', 'research_tools.py', 'master_pack.py', 'multitrack_ground_truth_campaign.py']:
        path = app_dir / filename
        log(f"checking {path}: exists={path.exists()}")
        if not path.exists():
            continue
        try:
            py_compile.compile(str(path), doraise=True)
            log(f"py_compile OK: {filename}")
        except Exception:
            log(f"py_compile FAILED: {filename}")
            traceback.print_exc()
            raise

    # Runtime assertion: never start a worker that would treat AppleDouble
    # placeholders as real audio.
    try:
        sys.path.insert(0, str(app_dir))
        import multitrack_ground_truth_campaign as mt
        fake = Path('/tmp/__MACOSX/RIHANNA/._arp_01.L.wav')
        assert hasattr(mt, '_is_real_audio_file')
        assert not mt._is_real_audio_file(fake, {'.wav'})
        log("runtime multitrack metadata self-test OK")
    except Exception:
        log("runtime multitrack metadata self-test FAILED")
        traceback.print_exc()
        raise

    log("starting /app/handler.py via runpy")
    try:
        runpy.run_path('/app/handler.py', run_name='__main__')
    except Exception:
        log("handler crashed during startup")
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
