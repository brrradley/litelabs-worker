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

    app_dir = Path('/app')
    log("/app listing:")
    try:
        for item in sorted(app_dir.iterdir()):
            kind = "dir" if item.is_dir() else "file"
            size = item.stat().st_size if item.is_file() else 0
            log(f" - {item.name} [{kind}] {size} bytes")
    except Exception as exc:
        log(f"could not list /app: {exc!r}")

    for filename in ['handler.py', 'research_tools.py', 'master_pack.py']:
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

    log("starting /app/handler.py via runpy")
    try:
        runpy.run_path('/app/handler.py', run_name='__main__')
    except Exception:
        log("handler crashed during startup")
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
