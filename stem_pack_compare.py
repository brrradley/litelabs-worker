from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests

from ground_truth_benchmark import _load, _normalise_audio, _score

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a"}


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with destination.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)


def _safe_extract(zip_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination.resolve() not in target.parents and target != destination.resolve():
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        archive.extractall(destination)


def _audio_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in AUDIO_EXTENSIONS
        and "__MACOSX" not in path.parts
        and not path.name.startswith("._")
    )


def _find_by_tokens(files: list[Path], required: list[str], excluded: list[str] | None = None) -> Path | None:
    excluded = excluded or []
    ranked: list[tuple[int, Path]] = []
    for path in files:
        lower = path.name.lower()
        if all(token.lower() in lower for token in required) and not any(token.lower() in lower for token in excluded):
            ranked.append((len(path.name), path))
    return min(ranked, default=(0, None), key=lambda item: item[0])[1]


def _resolve_mapping(files: list[Path], mapping: dict[str, str] | None, defaults: dict[str, tuple[list[str], list[str]]]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    mapping = mapping or {}
    for stem, explicit in mapping.items():
        explicit_lower = str(explicit).lower()
        match = next((path for path in files if str(path).lower().endswith(explicit_lower) or path.name.lower() == explicit_lower), None)
        if match:
            result[stem] = match
    for stem, (required, excluded) in defaults.items():
        if stem not in result:
            found = _find_by_tokens(files, required, excluded)
            if found:
                result[stem] = found
    return result


def build_stem_pack_compare(payload: dict, progress=None) -> dict:
    official_url = str(payload.get("official_url") or "").strip()
    litelabs_url = str(payload.get("litelabs_url") or "").strip()
    if not official_url or not litelabs_url:
        return {"ok": False, "mode": "stem_pack_compare", "error": "official_url and litelabs_url are required"}

    official_defaults = {
        "vocals": (["lead", "vocal"], []),
        "drums": (["drum"], []),
        "bass": (["bass"], []),
        "guitar": (["guitar"], []),
        "piano": (["piano"], []),
    }
    litelabs_defaults = {
        "vocals": (["_vocals"], ["backing"]),
        "drums": (["_drums"], []),
        "bass": (["_bass"], []),
        "guitar": (["_guitar"], []),
        "piano": (["_piano"], []),
    }

    with tempfile.TemporaryDirectory(prefix="litelabs_pack_compare_") as temp:
        root = Path(temp)
        official_zip = root / (Path(urlparse(official_url).path).name or "official.zip")
        litelabs_zip = root / (Path(urlparse(litelabs_url).path).name or "litelabs.zip")
        if progress:
            progress("Downloading official studio pack", 5)
        _download(official_url, official_zip)
        if progress:
            progress("Downloading LiteLABS pack", 12)
        _download(litelabs_url, litelabs_zip)

        official_root = root / "official"
        litelabs_root = root / "litelabs"
        official_root.mkdir()
        litelabs_root.mkdir()
        _safe_extract(official_zip, official_root)
        _safe_extract(litelabs_zip, litelabs_root)

        official_files = _audio_files(official_root)
        litelabs_files = _audio_files(litelabs_root)
        official = _resolve_mapping(official_files, payload.get("official_mapping"), official_defaults)
        litelabs = _resolve_mapping(litelabs_files, payload.get("litelabs_mapping"), litelabs_defaults)

        stems = [str(stem).lower() for stem in (payload.get("stems") or ["vocals", "drums", "bass", "guitar", "piano"])]
        rows: list[dict] = []
        for index, stem in enumerate(stems):
            row = {"stem": stem, "ok": False}
            reference = official.get(stem)
            estimate = litelabs.get(stem)
            row["official_file"] = str(reference.relative_to(official_root)) if reference else None
            row["litelabs_file"] = str(estimate.relative_to(litelabs_root)) if estimate else None
            if not reference or not estimate:
                row["error"] = "Missing official or LiteLABS stem"
                row["failure_type"] = "missing_output" if reference and not estimate else "missing_reference"
                rows.append(row)
                continue
            if progress:
                progress(f"Scoring {stem}", int(20 + 70 * index / max(1, len(stems))))
            try:
                ref_wav = root / "normalised" / f"official_{stem}.wav"
                est_wav = root / "normalised" / f"litelabs_{stem}.wav"
                _normalise_audio(reference, ref_wav)
                _normalise_audio(estimate, est_wav)
                ref_audio, ref_rate = _load(ref_wav)
                est_audio, est_rate = _load(est_wav)
                if ref_rate != est_rate:
                    raise ValueError("Sample-rate mismatch after normalisation")
                score = _score(ref_audio, est_audio, ref_rate)
                row.update({"ok": True, "score": score})
            except Exception as exc:
                row.update({"error": str(exc), "error_type": exc.__class__.__name__})
            rows.append(row)

        valid = [row for row in rows if row.get("ok")]
        average_quality = round(sum(float(row["score"].get("quality_score") or 0) for row in valid) / len(valid), 2) if valid else None
        return {
            "ok": True,
            "mode": "stem_pack_compare",
            "schema_version": 1,
            "official_url": official_url,
            "litelabs_url": litelabs_url,
            "official_files": [str(path.relative_to(official_root)) for path in official_files],
            "litelabs_files": [str(path.relative_to(litelabs_root)) for path in litelabs_files],
            "results": rows,
            "valid_comparisons": len(valid),
            "average_quality_score": average_quality,
            "notes": [
                "Drum OH is treated as the official drums reference because it is the only drum-labelled studio stem in this pack.",
                "A missing LiteLABS piano result is recorded as a product failure rather than silently excluded.",
                "Direct reference scoring assumes both packs originate from the same aligned master; reported alignment offsets reveal mismatches.",
            ],
            "no_audio_exported": True,
        }
