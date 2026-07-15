from __future__ import annotations

import math
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
import soundfile as sf

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
PRIMARY_STEMS = ("vocals", "drums", "bass", "guitar", "piano", "other")


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with destination.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)


def _safe_extract(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive, "r") as zipped:
        root = destination.resolve()
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
            if member.filename.startswith("__MACOSX/") or "/._" in member.filename or Path(member.filename).name.startswith("._"):
                continue
            zipped.extract(member, destination)


def _find_audio(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS)


def _role(path: Path) -> str | None:
    name = path.stem.lower().replace("-", "_").replace(" ", "_")
    if "instrumental" in name or "karaoke" in name:
        return "instrumental"
    if "backing_vocal" in name or "background_vocal" in name or "bvox" in name:
        return "backing_vocals"
    if "lead_vocal" in name or "main_vocal" in name:
        return "vocals"
    if "vocal" in name or "vox" in name:
        return "vocals"
    if "drum" in name or "percussion" in name:
        return "drums"
    if "bass" in name:
        return "bass"
    if "guitar" in name or "gtr" in name:
        return "guitar"
    if "piano" in name or "keys" in name or "keyboard" in name:
        return "piano"
    if "other" in name or "synth" in name or "strings" in name:
        return "other"
    return None


def _load(path: Path, target_rate: int = 44100) -> tuple[np.ndarray, int]:
    audio, rate = sf.read(path, always_2d=True, dtype="float32")
    if rate != target_rate:
        # Keep the worker dependency-light; ffmpeg is always available in the image.
        import subprocess
        converted = path.with_name(path.stem + f"_{target_rate}.wav")
        subprocess.run(["ffmpeg", "-y", "-i", str(path), "-ar", str(target_rate), "-ac", "2", str(converted)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        audio, rate = sf.read(converted, always_2d=True, dtype="float32")
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    return audio.astype(np.float32, copy=False), rate


def _mono(audio: np.ndarray) -> np.ndarray:
    return np.mean(audio, axis=1, dtype=np.float64).astype(np.float32)


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)) + 1e-12))


def _db(value: float) -> float:
    return 20.0 * math.log10(max(value, 1e-12))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    length = min(len(a), len(b))
    if length < 2:
        return 0.0
    a = a[:length].astype(np.float64)
    b = b[:length].astype(np.float64)
    a -= np.mean(a)
    b -= np.mean(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def _spectral_vector(audio: np.ndarray, rate: int) -> np.ndarray:
    mono = _mono(audio)
    if len(mono) > rate * 90:
        indexes = np.linspace(0, len(mono) - 1, rate * 90, dtype=np.int64)
        mono = mono[indexes]
    window = np.hanning(len(mono)) if len(mono) > 1 else np.ones(len(mono))
    spectrum = np.abs(np.fft.rfft(mono * window)) + 1e-12
    bands = np.geomspace(20, rate / 2, 65)
    freqs = np.fft.rfftfreq(len(mono), 1.0 / rate)
    values = []
    for low, high in zip(bands[:-1], bands[1:]):
        mask = (freqs >= low) & (freqs < high)
        values.append(float(np.mean(spectrum[mask])) if np.any(mask) else 0.0)
    vector = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-12 else vector


def _spectral_similarity(a: np.ndarray, b: np.ndarray, rate: int) -> float:
    av = _spectral_vector(a, rate)
    bv = _spectral_vector(b, rate)
    return float(np.dot(av, bv)) if np.any(av) and np.any(bv) else 0.0


def _band_ratios(audio: np.ndarray, rate: int) -> dict:
    mono = _mono(audio)
    if len(mono) > rate * 60:
        indexes = np.linspace(0, len(mono) - 1, rate * 60, dtype=np.int64)
        mono = mono[indexes]
    spectrum = np.square(np.abs(np.fft.rfft(mono)), dtype=np.float64)
    freqs = np.fft.rfftfreq(len(mono), 1.0 / rate)
    total = float(np.sum(spectrum) + 1e-12)
    def band(low: float, high: float) -> float:
        return float(np.sum(spectrum[(freqs >= low) & (freqs < high)]) / total)
    return {
        "sub_bass_ratio": round(band(20, 90), 6),
        "low_mid_ratio": round(band(90, 400), 6),
        "mid_ratio": round(band(400, 2500), 6),
        "presence_ratio": round(band(2500, 7000), 6),
        "air_ratio": round(band(7000, rate / 2), 6),
    }


def _stem_stats(audio: np.ndarray, rate: int) -> dict:
    mono = _mono(audio)
    rms = _rms(audio)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    crest = peak / max(rms, 1e-12)
    active = float(np.mean(np.abs(mono) > max(rms * 0.12, 1e-4))) if len(mono) else 0.0
    diff_rms = _rms(np.diff(mono)) if len(mono) > 1 else 0.0
    transient_ratio = diff_rms / max(_rms(mono), 1e-12)
    return {
        "duration_seconds": round(len(audio) / rate, 3),
        "rms_dbfs": round(_db(rms), 3),
        "peak_dbfs": round(_db(peak), 3),
        "crest_factor": round(crest, 4),
        "active_ratio": round(active, 6),
        "transient_ratio": round(transient_ratio, 6),
        "clipped_sample_ratio": round(float(np.mean(np.abs(audio) >= 0.999)), 8),
        **_band_ratios(audio, rate),
    }


def _recommend(role: str, stats: dict, max_dup: float, reconstruction_db: float) -> tuple[str, list[str]]:
    reasons: list[str] = []
    severity = "pass"
    if stats["active_ratio"] < 0.025:
        reasons.append("very low activity; keep the stem but mark it uncertain")
        severity = "review"
    if max_dup >= 0.82:
        reasons.append("very high similarity to another stem suggests duplication or heavy bleed")
        severity = "retry"
    elif max_dup >= 0.68:
        reasons.append("elevated similarity to another stem suggests possible bleed")
        severity = "review" if severity == "pass" else severity
    if stats["clipped_sample_ratio"] > 0.0001:
        reasons.append("clipping detected")
        severity = "retry"
    if role == "drums" and stats["transient_ratio"] < 0.12:
        reasons.append("weak transient profile for drums; compare a specialist or alternate pass")
        severity = "review" if severity == "pass" else severity
    if role == "bass" and stats["sub_bass_ratio"] + stats["low_mid_ratio"] < 0.28:
        reasons.append("unusually little low-frequency energy for bass")
        severity = "review" if severity == "pass" else severity
    if role == "vocals" and stats["mid_ratio"] + stats["presence_ratio"] < 0.36:
        reasons.append("weak vocal-band concentration")
        severity = "review" if severity == "pass" else severity
    if reconstruction_db > -25:
        reasons.append("whole-pack reconstruction error is high; stem-level judgement is less reliable")
        severity = "review" if severity == "pass" else severity
    if not reasons:
        reasons.append("no strong reference-free warning detected")
    return severity, reasons


def build_reference_free_stem_auditor(payload: dict, progress=None) -> dict:
    source_url = str(payload.get("source_url") or payload.get("audio_url") or "").strip()
    stem_pack_url = str(payload.get("stem_pack_url") or payload.get("pack_url") or "").strip()
    if not source_url or not stem_pack_url:
        return {"ok": False, "mode": "reference_free_stem_auditor", "error": "source_url and stem_pack_url are required"}

    with tempfile.TemporaryDirectory(prefix="litelabs_rf_audit_") as temp:
        root = Path(temp)
        source_path = root / (Path(urlparse(source_url).path).name or "source.flac")
        pack_path = root / "stem_pack.zip"
        extract_root = root / "pack"
        if progress:
            progress("Downloading source and stem pack", 5)
        _download(source_url, source_path)
        _download(stem_pack_url, pack_path)
        _safe_extract(pack_path, extract_root)

        source, rate = _load(source_path)
        files = _find_audio(extract_root)
        mapped: dict[str, Path] = {}
        ignored: list[str] = []
        for path in files:
            role = _role(path)
            if role in PRIMARY_STEMS and role not in mapped:
                mapped[role] = path
            else:
                ignored.append(str(path.relative_to(extract_root)))

        if progress:
            progress("Loading primary stems", 20)
        loaded: dict[str, np.ndarray] = {}
        for index, role in enumerate(PRIMARY_STEMS):
            path = mapped.get(role)
            if path:
                loaded[role], _ = _load(path, rate)
            if progress:
                progress(f"Inspecting {role}", 25 + index * 7)

        if not loaded:
            return {"ok": False, "mode": "reference_free_stem_auditor", "error": "No primary stems could be identified"}

        length = min([len(source)] + [len(audio) for audio in loaded.values()])
        source = source[:length]
        loaded = {role: audio[:length] for role, audio in loaded.items()}
        summed = np.sum(np.stack(list(loaded.values()), axis=0), axis=0)
        residual = source - summed
        source_rms = _rms(source)
        residual_rms = _rms(residual)
        reconstruction_error_db = _db(residual_rms / max(source_rms, 1e-12))
        reconstruction_corr = _corr(_mono(source), _mono(summed))
        reconstruction_spectral = _spectral_similarity(source, summed, rate)

        pairwise = []
        max_dup_by_role = {role: 0.0 for role in loaded}
        roles = list(loaded)
        for i, left in enumerate(roles):
            for right in roles[i + 1:]:
                corr = abs(_corr(_mono(loaded[left]), _mono(loaded[right])))
                spectral = _spectral_similarity(loaded[left], loaded[right], rate)
                duplication = 0.55 * corr + 0.45 * spectral
                max_dup_by_role[left] = max(max_dup_by_role[left], duplication)
                max_dup_by_role[right] = max(max_dup_by_role[right], duplication)
                pairwise.append({
                    "left": left,
                    "right": right,
                    "absolute_correlation": round(corr, 6),
                    "spectral_similarity": round(spectral, 6),
                    "duplication_risk": round(duplication, 6),
                })

        stems = []
        retry_queue = []
        for role in roles:
            stats = _stem_stats(loaded[role], rate)
            severity, reasons = _recommend(role, stats, max_dup_by_role[role], reconstruction_error_db)
            suggested_route = "keep_baseline"
            if severity == "retry":
                suggested_route = "run_alternate_parameter_pass_and_specialist_candidate"
            elif severity == "review":
                suggested_route = "compare_alternate_parameter_pass"
            item = {
                "stem": role,
                "file": str(mapped[role].relative_to(extract_root)),
                "status": severity,
                "suggested_route": suggested_route,
                "max_duplication_risk": round(max_dup_by_role[role], 6),
                "reasons": reasons,
                "statistics": stats,
            }
            stems.append(item)
            if severity in {"review", "retry"}:
                retry_queue.append({"stem": role, "priority": 2 if severity == "retry" else 1, "suggested_route": suggested_route, "reasons": reasons})

        retry_queue.sort(key=lambda item: (-item["priority"], item["stem"]))
        missing = [role for role in PRIMARY_STEMS if role not in loaded]
        overall_status = "pass"
        if any(item["status"] == "retry" for item in stems) or reconstruction_error_db > -15:
            overall_status = "retry_recommended"
        elif retry_queue or missing or reconstruction_error_db > -25:
            overall_status = "review_recommended"

        return {
            "ok": True,
            "mode": "reference_free_stem_auditor",
            "schema_version": 1,
            "source_url": source_url,
            "stem_pack_url": stem_pack_url,
            "overall_status": overall_status,
            "primary_stems_found": roles,
            "missing_primary_stems": missing,
            "reconstruction": {
                "duration_seconds": round(length / rate, 3),
                "residual_vs_source_db": round(reconstruction_error_db, 3),
                "waveform_correlation": round(reconstruction_corr, 6),
                "spectral_similarity": round(reconstruction_spectral, 6),
                "interpretation": "More-negative residual dB is better. This is a consistency check, not studio-reference quality scoring.",
            },
            "stems": stems,
            "pairwise_duplication": sorted(pairwise, key=lambda row: -row["duplication_risk"]),
            "retry_queue": retry_queue,
            "ignored_audio_files": ignored,
            "limitations": [
                "Reference-free warnings identify inconsistency and likely contamination, not absolute studio-stem correctness.",
                "Two wrong models can agree, and a clean-looking stem can still omit wanted material.",
                "The auditor should guide alternate passes and listening tests rather than automatically replace stems yet.",
            ],
        }
