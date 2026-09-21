from __future__ import annotations

import argparse
import json
from pathlib import Path

from experimental_children_v1 import build_experimental_children_v1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LiteLABS Experimental research mode from an interactive pod")
    parser.add_argument("audio_url")
    parser.add_argument("--filename", default="")
    parser.add_argument("--output-dir", default="/workspace/litelabs-results")
    parser.add_argument("--timeout", type=int, default=2400)
    parser.add_argument("--no-multi-lead", action="store_true")
    args = parser.parse_args()

    filename = args.filename.strip() or Path(args.audio_url).name
    payload = {
        "audio_url": args.audio_url,
        "filename": filename,
        "mode": "experimental_children_v1",
        "output_format": "flac",
        "research_multi_lead": not args.no_multi_lead,
        "research_output_dir": args.output_dir,
        "timeout_seconds": args.timeout,
    }

    def progress(message: str, percent: int) -> None:
        print(f"[{percent:3d}%] {message}", flush=True)

    result = build_experimental_children_v1(payload, progress=progress)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
