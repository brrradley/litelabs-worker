from __future__ import annotations

import requests

PATCHES = [
    "https://raw.githubusercontent.com/brrradley/litelabs-worker/d4d19d097662eee2d1b130ef572bc1d92fa70766/mss_baseline_benchmark_patch.py",
    "https://raw.githubusercontent.com/brrradley/litelabs-worker/08e427abee708bf99f532e6e25a227a22cb61e87/mss_batch_candidates_patch.py",
]

for url in PATCHES:
    response = requests.get(url, timeout=(30, 120))
    response.raise_for_status()
    source = response.text
    exec(compile(source, url, "exec"), {"__name__": "__main__", "__file__": url})

print("MSS baseline, campaign and batch-candidate patches applied")
