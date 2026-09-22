from pathlib import Path

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
print('LiteLABS serverless research benchmark route applied')
