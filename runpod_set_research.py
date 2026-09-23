#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests


TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}


def parse_track(value: str) -> dict:
    # url|filename|label ; filename/label are optional.
    parts = value.split("|", 2)
    url = parts[0].strip()
    if not url:
        raise argparse.ArgumentTypeError("track URL cannot be empty")
    result = {"url": url}
    if len(parts) > 1 and parts[1].strip():
        result["filename"] = parts[1].strip()
    if len(parts) > 2 and parts[2].strip():
        result["label"] = parts[2].strip()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Submit the complete SET targeted research suite as one RunPod serverless request."
    )
    parser.add_argument(
        "--endpoint-id",
        default=os.getenv("RUNPOD_ENDPOINT_ID", ""),
        help="RunPod endpoint ID (or RUNPOD_ENDPOINT_ID)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("RUNPOD_API_KEY", ""),
        help="RunPod API key (or RUNPOD_API_KEY)",
    )
    parser.add_argument(
        "--track",
        action="append",
        type=parse_track,
        default=[],
        metavar="URL|FILENAME|LABEL",
        help="Track to test. Repeat for a multi-track suite; still submits only one RunPod job.",
    )
    parser.add_argument("--audio-url", default="", help="Convenience single-track URL")
    parser.add_argument("--filename", default="", help="Filename for --audio-url")
    parser.add_argument("--label", default="", help="Label for --audio-url")
    parser.add_argument(
        "--references-json",
        default="",
        help=(
            "Optional JSON file mapping reference names to public URLs. "
            "Supported names: vocals, instrumental, lead, backing, kick, snare, toms, cymbals, hh."
        ),
    )
    parser.add_argument("--result-put-url", default="", help="Optional LiteRECORDS/chunk receiver URL")
    parser.add_argument("--result-public-url", default="", help="Optional public URL for the uploaded ZIP")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--save-json", default="set-research-runpod-result.json")
    parser.add_argument("--no-wait", action="store_true", help="Submit and print the job id without polling")
    args = parser.parse_args()

    if not args.endpoint_id:
        parser.error("--endpoint-id or RUNPOD_ENDPOINT_ID is required")
    if not args.api_key:
        parser.error("--api-key or RUNPOD_API_KEY is required")

    tracks = list(args.track)
    if args.audio_url:
        item = {"url": args.audio_url}
        if args.filename:
            item["filename"] = args.filename
        if args.label:
            item["label"] = args.label
        tracks.append(item)

    if not tracks:
        parser.error("at least one --track or --audio-url is required")

    references = {}
    if args.references_json:
        references = json.loads(Path(args.references_json).read_text(encoding="utf-8"))
        if not isinstance(references, dict):
            parser.error("--references-json must contain a JSON object")

    payload: dict = {
        "mode": "research_benchmark_v2",
        "timeout_seconds": max(300, args.timeout_seconds),
        "output_format": "flac",
    }
    if len(tracks) == 1:
        payload["audio_url"] = tracks[0]["url"]
        if tracks[0].get("filename"):
            payload["filename"] = tracks[0]["filename"]
    else:
        payload["tracks"] = tracks

    if references:
        if len(tracks) > 1:
            parser.error(
                "references currently apply to a single source track; run a one-track request when supplying --references-json"
            )
        payload["references"] = references
    if args.result_put_url:
        payload["result_put_url"] = args.result_put_url
    if args.result_public_url:
        payload["result_public_url"] = args.result_public_url

    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json",
    }
    base = f"https://api.runpod.ai/v2/{args.endpoint_id}"
    submit = requests.post(
        f"{base}/run",
        headers=headers,
        json={"input": payload},
        timeout=60,
    )
    submit.raise_for_status()
    submitted = submit.json()
    job_id = submitted.get("id")
    if not job_id:
        print(json.dumps(submitted, indent=2))
        raise RuntimeError("RunPod did not return a job id")

    print(f"submitted RunPod research job: {job_id}", flush=True)
    print(
        "suite: SW parent -> discovered 124-band -> DrumSep5/new drums -> "
        "lead/back matrix -> Anvuew 22.50 dereverb",
        flush=True,
    )

    if args.no_wait:
        print(json.dumps(submitted, indent=2))
        return 0

    while True:
        status_response = requests.get(
            f"{base}/status/{job_id}",
            headers=headers,
            timeout=60,
        )
        status_response.raise_for_status()
        state = status_response.json()
        status = str(state.get("status") or "").upper()
        print(f"RunPod {job_id}: {status or 'UNKNOWN'}", flush=True)

        if status in TERMINAL:
            Path(args.save_json).write_text(json.dumps(state, indent=2), encoding="utf-8")
            print(f"saved result JSON: {args.save_json}", flush=True)
            output = state.get("output")
            if output:
                print(json.dumps(output, indent=2))
            return 0 if status == "COMPLETED" else 1

        time.sleep(max(1.0, args.poll_seconds))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except requests.HTTPError as exc:
        body = ""
        if exc.response is not None:
            body = exc.response.text[-4000:]
        print(f"RunPod HTTP error: {exc}\n{body}", file=sys.stderr)
        raise
