from __future__ import annotations

import json
import subprocess
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".aif", ".aiff", ".ogg"}


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with destination.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)


def _probe(path: Path) -> dict:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "format=duration:stream=sample_rate,channels,channel_layout,codec_name,bits_per_sample",
        "-of", "json", str(path),
    ]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=60)
    if completed.returncode != 0:
        return {"ok": False, "error": (completed.stderr or completed.stdout or "ffprobe failed")[-1000:]}
    try:
        data = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": str(exc)}
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}
    return {
        "ok": True,
        "duration_seconds": round(float(fmt.get("duration") or 0.0), 3),
        "sample_rate": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
        "channel_layout": stream.get("channel_layout"),
        "codec": stream.get("codec_name"),
        "bits_per_sample": int(stream.get("bits_per_sample") or 0),
    }


def _guess_role(name: str) -> str:
    lower = name.lower()
    rules = [
        ("vocals", ["vocal", "vox", "lead vox", "backing vox", "bv", "choir"]),
        ("drums", ["drum", "kick", "snare", "hat", "cymbal", "tom", "overhead", "room", "perc"]),
        ("bass", ["bass"]),
        ("guitar", ["guitar", "gtr", "acoustic", "electric"]),
        ("piano", ["piano", "keys", "keyboard", "organ", "rhodes", "wurlitzer"]),
        ("other", ["synth", "string", "brass", "horn", "fx", "effect", "tambourine", "shaker"]),
        ("instrumental", ["instrumental"]),
    ]
    for role, tokens in rules:
        if any(token in lower for token in tokens):
            return role
    return "unclassified"


def build_stem_pack_inventory(payload: dict, progress=None) -> dict:
    packs = payload.get("packs") or []
    if not isinstance(packs, list) or not packs:
        return {"ok": False, "mode": "stem_pack_inventory", "error": "input.packs must be a non-empty array"}

    results = []
    with tempfile.TemporaryDirectory(prefix="litelabs_pack_inventory_") as temp:
        root = Path(temp)
        for index, pack in enumerate(packs):
            label = str(pack.get("label") or f"pack-{index + 1}")
            url = str(pack.get("url") or "").strip()
            if not url:
                results.append({"label": label, "ok": False, "error": "Missing pack URL"})
                continue
            if progress:
                progress(f"Downloading {label}", int(5 + index * 80 / max(1, len(packs))))
            zip_path = root / f"{index:02d}_{Path(urlparse(url).path).name or 'pack.zip'}"
            extract_dir = root / f"extract_{index:02d}"
            try:
                _download(url, zip_path)
                if not zipfile.is_zipfile(zip_path):
                    raise ValueError("Downloaded file is not a valid ZIP")
                with zipfile.ZipFile(zip_path, "r") as archive:
                    archive.extractall(extract_dir)
                    members = archive.infolist()
                files = []
                role_counts: dict[str, int] = {}
                durations = []
                for path in sorted(p for p in extract_dir.rglob("*") if p.is_file()):
                    relative = str(path.relative_to(extract_dir))
                    role = _guess_role(relative)
                    role_counts[role] = role_counts.get(role, 0) + 1
                    item = {
                        "path": relative,
                        "size_bytes": path.stat().st_size,
                        "extension": path.suffix.lower(),
                        "guessed_role": role,
                    }
                    if path.suffix.lower() in AUDIO_EXTENSIONS:
                        item["audio"] = _probe(path)
                        if item["audio"].get("ok"):
                            durations.append(float(item["audio"].get("duration_seconds") or 0.0))
                    files.append(item)
                results.append({
                    "label": label,
                    "ok": True,
                    "zip_size_bytes": zip_path.stat().st_size,
                    "member_count": len(members),
                    "audio_file_count": sum(1 for item in files if "audio" in item),
                    "role_counts": role_counts,
                    "duration_range_seconds": [round(min(durations), 3), round(max(durations), 3)] if durations else None,
                    "files": files,
                })
            except Exception as exc:
                results.append({"label": label, "ok": False, "error": str(exc), "error_type": exc.__class__.__name__})

    return {
        "ok": all(item.get("ok") for item in results),
        "mode": "stem_pack_inventory",
        "schema_version": 1,
        "packs": results,
        "next_step": "Use this inventory to confirm official-stem grouping and alignment before objective comparison.",
    }
