from __future__ import annotations

import json
import os
import py_compile
import runpy
import sys
import time
import traceback
from pathlib import Path


def log(message: str) -> None:
    print(f"[LiteLABS research bootstrap] {message}", flush=True)


def _truthy(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _pod_progress(message: str, percent: int) -> None:
    print(f"[LiteLABS research pod] {percent:3d}% {message}", flush=True)


def _run_pod_campaign() -> None:
    from multitrack_ground_truth_campaign import build_multitrack_ground_truth_campaign

    zip_url = str(os.getenv("LITELABS_BENCHMARK_ZIP_URL", "")).strip()
    if not zip_url:
        raise RuntimeError("LITELABS_BENCHMARK_ZIP_URL is required in Pod mode")

    output = Path(os.getenv("LITELABS_BENCHMARK_OUTPUT", "/workspace/litelabs-research/disturbia_campaign.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    max_runs = max(1, min(100, int(os.getenv("LITELABS_BENCHMARK_MAX_RUNS", "100"))))
    time_budget = max(120, min(3300, int(os.getenv("LITELABS_BENCHMARK_TIME_BUDGET", "3300"))))
    model_timeout = max(300, min(3300, int(os.getenv("LITELABS_BENCHMARK_MODEL_TIMEOUT", "1800"))))

    cursor = 0
    passes = []
    all_results = []
    started = time.monotonic()

    log(f"Pod campaign source: {zip_url}")
    log(f"Pod campaign output: {output}")

    while True:
        payload = {
            "zip_url": zip_url,
            "cursor": cursor,
            "max_runs": max_runs,
            "time_budget_seconds": time_budget,
            "model_timeout_seconds": model_timeout,
        }
        log(f"Pod campaign pass starting at cursor {cursor}")
        result = build_multitrack_ground_truth_campaign(payload, progress=_pod_progress)
        passes.append(result)
        all_results.extend(result.get("results") or [])

        snapshot = {
            "ok": bool(result.get("ok")),
            "mode": "multitrack_ground_truth_campaign_pod",
            "source_zip_url": zip_url,
            "last_cursor": cursor,
            "next_cursor": result.get("next_cursor"),
            "complete": bool(result.get("complete")),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "passes": passes,
            "results": all_results,
        }
        output.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        log(f"Pod campaign checkpoint written: {output}")

        if not result.get("ok"):
            raise RuntimeError(str(result.get("error") or "Pod campaign failed"))
        next_cursor = result.get("next_cursor")
        if next_cursor is None:
            log(f"Pod campaign complete in {snapshot['elapsed_seconds']}s")
            return
        cursor = int(next_cursor)


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

    if _truthy('LITELABS_RESEARCH_POD_MODE'):
        log("dedicated Pod mode enabled")
        try:
            _run_pod_campaign()
        except Exception:
            log("research Pod campaign crashed")
            traceback.print_exc()
            raise
        return

    log("starting /app/handler.py via runpy")
    try:
        runpy.run_path('/app/handler.py', run_name='__main__')
    except Exception:
        log("handler crashed during startup")
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
