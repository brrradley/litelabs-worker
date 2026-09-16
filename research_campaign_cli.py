from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from multitrack_ground_truth_campaign import build_multitrack_ground_truth_campaign


def _progress(message: str, percent: int) -> None:
    print(f"[LiteLABS research pod] {percent:3d}% {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LiteLABS multitrack ground-truth research on a dedicated Pod")
    parser.add_argument("--zip-url", default=os.getenv("LITELABS_BENCHMARK_ZIP_URL", "").strip())
    parser.add_argument("--output", default=os.getenv("LITELABS_BENCHMARK_OUTPUT", "/workspace/litelabs-research/disturbia_campaign.json"))
    parser.add_argument("--max-runs-per-pass", type=int, default=int(os.getenv("LITELABS_BENCHMARK_MAX_RUNS", "100")))
    parser.add_argument("--time-budget-seconds", type=int, default=int(os.getenv("LITELABS_BENCHMARK_TIME_BUDGET", "3300")))
    parser.add_argument("--model-timeout-seconds", type=int, default=int(os.getenv("LITELABS_BENCHMARK_MODEL_TIMEOUT", "1800")))
    args = parser.parse_args()

    if not args.zip_url:
        parser.error("--zip-url or LITELABS_BENCHMARK_ZIP_URL is required")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cursor = 0
    passes = []
    all_results = []
    started = time.monotonic()

    print(f"[LiteLABS research pod] source: {args.zip_url}", flush=True)
    print(f"[LiteLABS research pod] output: {output_path}", flush=True)

    while True:
        payload = {
            "zip_url": args.zip_url,
            "cursor": cursor,
            "max_runs": max(1, min(100, args.max_runs_per_pass)),
            "time_budget_seconds": max(120, min(3300, args.time_budget_seconds)),
            "model_timeout_seconds": max(300, min(3300, args.model_timeout_seconds)),
        }
        print(f"[LiteLABS research pod] starting pass at cursor {cursor}", flush=True)
        result = build_multitrack_ground_truth_campaign(payload, progress=_progress)
        passes.append(result)
        all_results.extend(result.get("results") or [])

        snapshot = {
            "ok": bool(result.get("ok")),
            "mode": "multitrack_ground_truth_campaign_pod",
            "source_zip_url": args.zip_url,
            "started_cursor": 0,
            "last_cursor": cursor,
            "next_cursor": result.get("next_cursor"),
            "complete": bool(result.get("complete")),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "passes": passes,
            "results": all_results,
        }
        output_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        print(f"[LiteLABS research pod] checkpoint written: {output_path}", flush=True)

        if not result.get("ok"):
            print("[LiteLABS research pod] campaign failed", file=sys.stderr, flush=True)
            return 1
        next_cursor = result.get("next_cursor")
        if next_cursor is None:
            print(f"[LiteLABS research pod] campaign complete in {snapshot['elapsed_seconds']}s", flush=True)
            return 0
        cursor = int(next_cursor)


if __name__ == "__main__":
    raise SystemExit(main())
