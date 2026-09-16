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


def _ensure_multitrack_metadata_guard(app_dir: Path) -> None:
    """Make the AppleDouble filter part of the runtime source before import.

    This is deliberately self-contained rather than trusting the historical patch
    helper. Pod launches can inherit a source layer that predates the helper even
    when the image tag itself is current. We patch the campaign source, compile it,
    evict any stale module, and then import the exact file we just verified.
    """
    path = app_dir / "multitrack_ground_truth_campaign.py"
    text = path.read_text(encoding="utf-8")

    helper_anchor = "def _track_category(name: str) -> str:\n"
    helper = '''def _is_real_audio_file(path: Path, extensions: set[str]) -> bool:\n    # Ignore macOS AppleDouble resource-fork placeholders. They often carry\n    # audio-looking extensions but are metadata, not playable media.\n    if path.suffix.lower() not in extensions:\n        return False\n    if path.name.startswith("._"):\n        return False\n    if any(part == "__MACOSX" for part in path.parts):\n        return False\n    return True\n\n\n'''

    if "def _is_real_audio_file(" not in text:
        if helper_anchor not in text:
            raise RuntimeError("Could not locate multitrack helper insertion point")
        text = text.replace(helper_anchor, helper + helper_anchor, 1)

    old_wavs = '    wavs = [p for p in extracted.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"]\n'
    new_wavs = '    wavs = [p for p in extracted.rglob("*") if p.is_file() and _is_real_audio_file(p, {".wav"})]\n'
    if old_wavs in text:
        text = text.replace(old_wavs, new_wavs, 1)
    elif new_wavs not in text:
        raise RuntimeError("Could not locate multitrack WAV discovery")

    old_masters = '        masters = [p for p in extracted.rglob("*") if p.is_file() and p.suffix.lower() in {".aif", ".aiff"}]\n'
    new_masters = '        masters = [p for p in extracted.rglob("*") if p.is_file() and _is_real_audio_file(p, {".aif", ".aiff"})]\n'
    if old_masters in text:
        text = text.replace(old_masters, new_masters, 1)
    elif new_masters not in text:
        raise RuntimeError("Could not locate master AIF discovery")

    path.write_text(text, encoding="utf-8")
    py_compile.compile(str(path), doraise=True)
    sys.modules.pop("multitrack_ground_truth_campaign", None)
    log("runtime multitrack source guard written and compiled")


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


def _hold_completed_pod() -> None:
    log("campaign complete; holding Pod open for result retrieval (it will NOT rerun)")
    log("retrieve /workspace/litelabs-research/disturbia_campaign.json before stopping the Pod")
    while True:
        time.sleep(3600)


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

    try:
        _ensure_multitrack_metadata_guard(app_dir)
    except Exception:
        log("runtime multitrack metadata guard FAILED")
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
        sys.modules.pop('multitrack_ground_truth_campaign', None)
        import multitrack_ground_truth_campaign as mt
        fake = Path('/tmp/__MACOSX/RIHANNA/._arp_01.L.wav')
        assert hasattr(mt, '_is_real_audio_file')
        assert not mt._is_real_audio_file(fake, {'.wav'})
        assert mt._is_real_audio_file(Path('/tmp/real.wav'), {'.wav'})
        log("runtime multitrack metadata self-test OK")
    except Exception:
        log("runtime multitrack metadata self-test FAILED")
        traceback.print_exc()
        raise

    if _truthy('LITELABS_RESEARCH_POD_MODE'):
        log("dedicated Pod mode enabled")
        try:
            _run_pod_campaign()
            _hold_completed_pod()
        except Exception:
            log("research Pod campaign crashed")
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
