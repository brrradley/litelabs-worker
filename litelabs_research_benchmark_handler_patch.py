from pathlib import Path

# Upgrade the benchmark module in-image so one serverless request can exercise
# the full five-track suite while retaining the already-tested single-track code.
benchmark_path = Path('/app/research_benchmark_v2.py')
benchmark = benchmark_path.read_text(encoding='utf-8')

if '# litelabs_research_suite_v1' not in benchmark:
    # Add the strong public Kimberley MelBand vocal-parent challenger alongside
    # the requested Viperx 12.9755 checkpoint.
    leadback_anchor = '        emit("Running current lead/back baseline", 36)\n'
    if leadback_anchor not in benchmark:
        raise RuntimeError('Could not locate vocal-parent benchmark insertion point')
    kimberley_block = '''        emit("Testing Kimberley MelBand vocal-parent challenger", 31)\n        kimberley_dir = root / "vocal_kimberley"\n        kimberley = _run_separator(source, kimberley_dir, "vocals_mel_band_roformer.ckpt", [], timeout)\n        report["tests"]["vocal_parent_kimberley"] = kimberley\n        if kimberley.get("returncode") == 0:\n            kimberley_vocals = Path(kimberley["primary"])\n            kimberley_inst = Path(kimberley["secondary"])\n            kimberley["pair_metrics"] = _pair_metrics(source, kimberley_vocals, kimberley_inst)\n            kimberley["vs_current_sw_vocals"] = _similarity(sw_vocals, kimberley_vocals)\n            _copy_named(kimberley_vocals, outputs / "02b_kimberley_vocals.flac")\n            _copy_named(kimberley_inst, outputs / "02b_kimberley_instrumental.flac")\n\n'''
    benchmark = benchmark.replace(leadback_anchor, kimberley_block + leadback_anchor, 1)

    # Preserve the original implementation as the single-track worker.
    function_anchor = 'def run_research_benchmark(payload: dict, progress=None) -> dict:\n'
    if function_anchor not in benchmark:
        raise RuntimeError('Could not locate single-track research benchmark function')
    benchmark = benchmark.replace(
        function_anchor,
        'def _run_research_benchmark_single(payload: dict, progress=None) -> dict:\n',
        1,
    )

    benchmark += r'''

# litelabs_research_suite_v1
def _suite_inputs(payload: dict) -> list[dict]:
    supplied = payload.get("audio_urls") or payload.get("tracks") or []
    if not supplied:
        url = str(payload.get("audio_url") or payload.get("source_url") or "").strip()
        if not url:
            return []
        return [{"url": url, "filename": payload.get("filename") or ""}]

    tracks: list[dict] = []
    for index, item in enumerate(supplied):
        if isinstance(item, str):
            url = item.strip()
            meta = {"url": url, "filename": "", "label": f"track_{index + 1}"}
        elif isinstance(item, dict):
            url = str(item.get("url") or item.get("audio_url") or item.get("source_url") or "").strip()
            meta = {
                "url": url,
                "filename": str(item.get("filename") or ""),
                "label": str(item.get("label") or item.get("name") or f"track_{index + 1}"),
            }
        else:
            continue
        if url:
            tracks.append(meta)
    return tracks[:8]


def _aggregate_suite(reports: list[dict]) -> dict:
    buckets: dict[str, list[float]] = {}
    successes: dict[str, int] = {}
    for report in reports:
        for name, test in (report.get("tests") or {}).items():
            if not isinstance(test, dict):
                continue
            if test.get("returncode") == 0 or "returncode" not in test:
                successes[name] = successes.get(name, 0) + 1
            runtime = test.get("runtime_seconds")
            if isinstance(runtime, (int, float)):
                buckets.setdefault(name, []).append(float(runtime))
    aggregate = {}
    for name, values in buckets.items():
        values = sorted(values)
        middle = len(values) // 2
        median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2.0
        aggregate[name] = {
            "runs": len(values),
            "successful_tracks": successes.get(name, 0),
            "mean_runtime_seconds": round(sum(values) / len(values), 3),
            "median_runtime_seconds": round(median, 3),
            "min_runtime_seconds": round(min(values), 3),
            "max_runtime_seconds": round(max(values), 3),
        }
    return aggregate


def run_research_benchmark(payload: dict, progress=None) -> dict:
    tracks = _suite_inputs(payload)
    if not tracks:
        return {"ok": False, "mode": MODE, "error": "audio_url or audio_urls is required"}
    if len(tracks) == 1 and not (payload.get("audio_urls") or payload.get("tracks")):
        return _run_research_benchmark_single(payload, progress=progress)

    suite_started = time.monotonic()
    suite_id = uuid.uuid4().hex[:10]
    suite_archive = Path('/tmp') / f'litelabs-research-suite-{suite_id}.zip'
    reports: list[dict] = []
    failures: list[dict] = []

    with tempfile.TemporaryDirectory(prefix='litelabs_research_suite_') as temp:
        suite_root = Path(temp)
        for index, item in enumerate(tracks):
            percent = max(1, int((index / max(1, len(tracks))) * 90))
            label = item.get('label') or f'track_{index + 1}'
            if progress:
                progress(f'Research suite {index + 1}/{len(tracks)}: {label}', percent)

            child_payload = dict(payload)
            for key in ('audio_urls', 'tracks', 'result_put_url', 'result_public_url'):
                child_payload.pop(key, None)
            child_payload['audio_url'] = item['url']
            if item.get('filename'):
                child_payload['filename'] = item['filename']

            result = _run_research_benchmark_single(child_payload, progress=None)
            if not result.get('ok'):
                failures.append({"index": index, "label": label, "url": item['url'], "result": result})
                continue

            report = result.get('report') or {}
            report['suite_label'] = label
            reports.append(report)
            archive_name = str(result.get('archive_name') or '')
            single_archive = Path('/tmp') / archive_name
            track_name = str(result.get('track') or f'track_{index + 1}')
            destination = suite_root / f'{index + 1:02d}_{track_name}'
            destination.mkdir(parents=True, exist_ok=True)
            if single_archive.is_file():
                with zipfile.ZipFile(single_archive, 'r') as bundle:
                    bundle.extractall(destination)
                single_archive.unlink(missing_ok=True)

        suite_report = {
            "schema_version": 1,
            "mode": "research_benchmark_suite_v1",
            "build_sha": os.getenv("LITELABS_BUILD_SHA", "unknown"),
            "track_count_requested": len(tracks),
            "track_count_completed": len(reports),
            "failures": failures,
            "published_reference_metrics": PUBLISHED,
            "aggregate_runtime": _aggregate_suite(reports),
            "tracks": reports,
            "total_runtime_seconds": round(time.monotonic() - suite_started, 3),
        }
        (suite_root / 'SUITE_REPORT.json').write_text(json.dumps(_json_safe(suite_report), indent=2), encoding='utf-8')

        with zipfile.ZipFile(suite_archive, 'w', compression=zipfile.ZIP_STORED) as bundle:
            for path in sorted(suite_root.rglob('*')):
                if path.is_file():
                    bundle.write(path, arcname=str(path.relative_to(suite_root)))

    archive_size = suite_archive.stat().st_size
    uploaded = False
    put_url = str(payload.get('result_put_url') or '').strip()
    if put_url:
        if progress:
            progress('Uploading consolidated research suite', 96)
        with suite_archive.open('rb') as handle:
            response = requests.put(put_url, data=handle, headers={"Content-Type": "application/zip"}, timeout=(30, 1800))
        response.raise_for_status()
        uploaded = True

    result = {
        "ok": len(reports) > 0,
        "mode": "research_benchmark_suite_v1",
        "build_sha": os.getenv("LITELABS_BUILD_SHA", "unknown"),
        "archive_name": suite_archive.name,
        "archive_size_bytes": archive_size,
        "uploaded": uploaded,
        "result_url": payload.get('result_public_url'),
        "report": suite_report,
    }
    if uploaded:
        suite_archive.unlink(missing_ok=True)
    if progress:
        progress('Research suite complete', 100)
    return _json_safe(result)
'''

    benchmark_path.write_text(benchmark, encoding='utf-8')

benchmark_check = benchmark_path.read_text(encoding='utf-8')
compile(benchmark_check, str(benchmark_path), 'exec')
assert '# litelabs_research_suite_v1' in benchmark_check
assert 'vocal_parent_kimberley' in benchmark_check
assert 'audio_urls' in benchmark_check

path = Path('/app/handler.py')
text = path.read_text(encoding='utf-8')
marker = '# litelabs_research_benchmark_v2\n'
if marker not in text:
    anchor = '    if mode == "routed_extraction_v1":\n'
    if anchor not in text:
        raise RuntimeError('Could not locate routed_extraction_v1 handler anchor')
    block = '''    # litelabs_research_benchmark_v2
    if mode == "research_benchmark_v2":
        try:
            from research_benchmark_v2 import run_research_benchmark
            result = run_research_benchmark(payload, progress=progress)
            result.setdefault("research_only", True)
            return result
        except Exception as exc:
            post_progress(progress_url, progress_token, progress_job_id, f"Research benchmark error: {exc}", 100)
            return {"ok": False, "mode": mode, "error": str(exc), "error_type": exc.__class__.__name__, "research_only": True}

'''
    text = text.replace(anchor, block + anchor, 1)
    path.write_text(text, encoding='utf-8')

check = path.read_text(encoding='utf-8')
compile(check, str(path), 'exec')
assert '# litelabs_research_benchmark_v2' in check
assert 'from research_benchmark_v2 import run_research_benchmark' in check
print('LiteLABS serverless research benchmark + batch suite route applied')
