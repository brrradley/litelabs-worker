from __future__ import annotations

import json
import os
import time
import tempfile
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

import multitrack_ground_truth_campaign as mt
import synthetic_ground_truth_campaign as sgt
from ground_truth_benchmark import _score

MODE = "experimental_vocal_cascade_bakeoff_v1"
CURRENT_KARAOKE = "bs_roformer_karaoke_frazer_becruily.ckpt"
BECRUILY_KARAOKE = "mel_band_roformer_karaoke_becruily.ckpt"
SW_MODEL = "BS-Roformer-SW.ckpt"


def log(message: str) -> None:
    print(f"[LiteLABS vocal bakeoff] {message}", flush=True)


def progress(message: str, percent: int) -> None:
    print(f"[LiteLABS vocal bakeoff] {percent:3d}% {message}", flush=True)


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2) + 1e-12))


def _db(value: float) -> float:
    return float(20.0 * np.log10(max(float(value), 1e-12)))


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n <= 0:
        return 0.0
    aa = np.asarray(a[:n], dtype=np.float64).reshape(-1)
    bb = np.asarray(b[:n], dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / denom) if denom > 1e-12 else 0.0


def _sum(items: list[np.ndarray]) -> np.ndarray:
    items = [x for x in items if x is not None and len(x)]
    if not items:
        raise ValueError("No audio arrays to sum")
    n = min(len(x) for x in items)
    return np.sum(np.stack([x[:n] for x in items], axis=0), axis=0)


def _safe_name(name: str) -> str:
    return " ".join(str(name).strip().lower().replace("_", " ").replace("-", " ").split())


def _vocal_refs(tracks: dict[str, np.ndarray], work: Path, sr: int) -> tuple[dict[str, Path], dict]:
    by_name = {_safe_name(name): (name, audio) for name, audio in tracks.items()}
    lead_entry = by_name.get("lead vocal")
    background_entry = by_name.get("background vocal")
    if not lead_entry or not background_entry:
        raise RuntimeError(
            "Disturbia pack is missing exact lead/background vocal references; "
            f"available vocal-like names: {[name for name in tracks if sgt.track_category(name)[0] == 'vocals']}"
        )

    vocal_members: list[tuple[str, np.ndarray]] = []
    for name, audio in tracks.items():
        category, _warning = sgt.track_category(name)
        if category == "vocals":
            vocal_members.append((name, audio))

    lead_name, lead = lead_entry
    backing_name, backing_strict = background_entry
    nonlead = [(name, audio) for name, audio in vocal_members if name != lead_name]
    backing_inclusive = _sum([audio for _name, audio in nonlead])
    total_vocals = _sum([audio for _name, audio in vocal_members])
    core_pair = _sum([lead, backing_strict])

    ref_dir = work / "vocal_refs"
    ref_dir.mkdir(parents=True, exist_ok=True)
    refs = {
        "lead": ref_dir / "lead.wav",
        "backing_strict": ref_dir / "backing_strict.wav",
        "backing_inclusive": ref_dir / "backing_inclusive.wav",
        "core_pair": ref_dir / "core_pair.wav",
        "total_vocals": ref_dir / "total_vocals.wav",
    }
    mt._write(refs["lead"], lead, sr)
    mt._write(refs["backing_strict"], backing_strict, sr)
    mt._write(refs["backing_inclusive"], backing_inclusive, sr)
    mt._write(refs["core_pair"], core_pair, sr)
    mt._write(refs["total_vocals"], total_vocals, sr)

    meta = {
        "lead_members": [lead_name],
        "backing_strict_members": [backing_name],
        "backing_inclusive_members": [name for name, _audio in nonlead],
        "all_vocal_members": [name for name, _audio in vocal_members],
        "note": (
            "Strict backing scores only the studio 'background vocal' track. Inclusive backing scores every "
            "non-lead vocal/FX track, which is fairer for a residual child that must contain all non-lead vocal material."
        ),
    }
    return refs, meta


def _find_output(directory: Path, token: str) -> Path | None:
    token = token.lower()
    files = [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in {".wav", ".flac", ".mp3"}]
    exact = [p for p in files if p.stem.lower() == token]
    if exact:
        return max(exact, key=lambda p: p.stat().st_size)
    matches = [p for p in files if token in p.stem.lower()]
    if matches:
        return max(matches, key=lambda p: p.stat().st_size)
    return None


def _run_separator(model: str, source: Path, output_dir: Path, model_dir: Path, timeout: int) -> tuple[np.ndarray, np.ndarray | None, float, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "audio-separator", str(source),
        "--model_filename", model,
        "--model_file_dir", str(model_dir),
        "--output_dir", str(output_dir),
        "--output_format", "FLAC",
        "--mdxc_segment_size", "256",
        "--mdxc_overlap", "8",
        "--mdxc_batch_size", "1",
        "--use_autocast",
    ]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    elapsed = time.monotonic() - t0
    tail = "\n".join((proc.stdout or "").splitlines()[-40:])
    if proc.returncode != 0:
        raise RuntimeError(f"{model} failed: {tail}")
    lead_path = _find_output(output_dir, "vocals")
    native_backing_path = _find_output(output_dir, "instrumental")
    if lead_path is None:
        raise RuntimeError(f"{model} produced no vocals output. Files: {[p.name for p in output_dir.rglob('*') if p.is_file()]}")
    lead, _sr = mt._load(lead_path)
    native_backing = mt._load(native_backing_path)[0] if native_backing_path else None
    return lead, native_backing, elapsed, tail


def _pair_metrics(
    *,
    name: str,
    input_audio: np.ndarray,
    lead: np.ndarray,
    backing: np.ndarray | None,
    refs_audio: dict[str, np.ndarray],
    sr: int,
    runtime_seconds: float,
    model: str,
    route: str,
) -> dict:
    n = min(len(input_audio), len(lead), *( [len(backing)] if backing is not None else [] ))
    input_audio = input_audio[:n]
    lead = lead[:n]
    result = {
        "name": name,
        "model": model,
        "route": route,
        "runtime_seconds": round(runtime_seconds, 3),
        "lead": _score(lead, refs_audio["lead"][: min(n, len(refs_audio["lead"]))], sr),
    }
    if backing is None:
        result["backing_available"] = False
        return result

    backing = backing[:n]
    pair_sum = lead + backing
    residual = input_audio - pair_sum
    result.update({
        "backing_available": True,
        "backing_strict": _score(backing, refs_audio["backing_strict"][: min(n, len(refs_audio["backing_strict"]))], sr),
        "backing_inclusive": _score(backing, refs_audio["backing_inclusive"][: min(n, len(refs_audio["backing_inclusive"]))], sr),
        "core_pair": _score(pair_sum, refs_audio["core_pair"][: min(n, len(refs_audio["core_pair"]))], sr),
        "input_reconstruction_cosine": round(_cos(input_audio, pair_sum), 9),
        "input_reconstruction_residual_db": round(_db(_rms(residual) / max(_rms(input_audio), 1e-12)), 3),
        "lead_backing_correlation": round(abs(_cos(lead, backing)), 6),
    })
    lead_q = float(result["lead"].get("quality_score", 0.0))
    backing_q = float(result["backing_inclusive"].get("quality_score", 0.0))
    result["lead_backing_composite"] = round(0.60 * lead_q + 0.40 * backing_q, 4)
    return result


def run() -> dict:
    zip_url = str(os.getenv("LITELABS_BENCHMARK_ZIP_URL", "https://literecords.com/tmp/disturbia_test.zip")).strip()
    output = Path(os.getenv("LITELABS_BENCHMARK_OUTPUT", "/workspace/litelabs-research/disturbia_vocal_cascade_bakeoff_v1.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    timeout = max(600, min(5400, int(os.getenv("LITELABS_BENCHMARK_MODEL_TIMEOUT", "3600"))))
    model_dir = Path(os.getenv("LITELABS_AUDIO_SEPARATOR_MODEL_DIR", "/models/audio_separator"))
    model_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="litelabs_vocal_bakeoff_") as temp:
        root = Path(temp)
        archive = root / (Path(urlparse(zip_url).path).name or "disturbia.zip")
        extracted = root / "extracted"
        work = root / "work"

        progress("Downloading and preparing Disturbia multitracks", 2)
        mt._download(zip_url, archive)
        mt._safe_extract(archive, extracted)
        sgt._remove_macos_metadata(extracted)
        mt._track_category = sgt.track_category
        tracks, sr, inventory = mt._build_tracks(extracted, work)
        source_path, parent_refs, scale = mt._build_references(tracks, sr, work)
        integrity = sgt._reference_integrity(source_path, parent_refs, sr)
        if not integrity["exact_partition"]:
            raise RuntimeError(f"Synthetic source integrity failed: {integrity}")

        refs, vocal_ref_meta = _vocal_refs(tracks, work, sr)
        refs_audio = {name: mt._load(path)[0] for name, path in refs.items()}
        source_audio, _ = mt._load(source_path)

        progress("Extracting the fixed BS-RoFormer-SW vocal parent", 12)
        sw_out = root / "sw"
        cmd = [
            "audio-separator", str(source_path),
            "--model_filename", SW_MODEL,
            "--model_file_dir", str(model_dir),
            "--output_dir", str(sw_out),
            "--output_format", "FLAC",
            "--mdxc_segment_size", "256",
            "--mdxc_overlap", "8",
            "--mdxc_batch_size", "1",
            "--use_autocast",
        ]
        t0 = time.monotonic()
        sw_proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        sw_elapsed = time.monotonic() - t0
        if sw_proc.returncode != 0:
            raise RuntimeError("BS-RoFormer-SW failed: " + " | ".join((sw_proc.stdout or "").splitlines()[-40:]))
        sw_vocals_path = _find_output(sw_out, "vocals")
        if sw_vocals_path is None:
            raise RuntimeError("BS-RoFormer-SW vocals parent was not found")
        sw_vocals, _ = mt._load(sw_vocals_path)

        candidates = []

        progress("Testing current SET BS-RoFormer Karaoke on SW vocals", 30)
        set_lead, set_native_backing, set_elapsed, set_tail = _run_separator(
            CURRENT_KARAOKE, sw_vocals_path, root / "set_karaoke", model_dir, timeout
        )
        set_n = min(len(sw_vocals), len(set_lead))
        set_residual = sw_vocals[:set_n] - set_lead[:set_n]
        candidates.append(_pair_metrics(
            name="set_current_native_pair",
            input_audio=sw_vocals,
            lead=set_lead,
            backing=set_native_backing,
            refs_audio=refs_audio,
            sr=sr,
            runtime_seconds=set_elapsed,
            model=CURRENT_KARAOKE,
            route="BS-RoFormer-SW vocals -> current BS-RoFormer Karaoke native outputs",
        ))
        candidates.append(_pair_metrics(
            name="set_current_residual_pair",
            input_audio=sw_vocals,
            lead=set_lead,
            backing=set_residual,
            refs_audio=refs_audio,
            sr=sr,
            runtime_seconds=set_elapsed,
            model=CURRENT_KARAOKE,
            route="BS-RoFormer-SW vocals -> current BS-RoFormer Karaoke lead; backing = parent - lead",
        ))

        progress("Testing Becruily MelBand Karaoke on SW vocals", 55)
        bec_parent_lead, bec_parent_native, bec_parent_elapsed, bec_parent_tail = _run_separator(
            BECRUILY_KARAOKE, sw_vocals_path, root / "becruily_parent", model_dir, timeout
        )
        bec_n = min(len(sw_vocals), len(bec_parent_lead))
        bec_parent_residual = sw_vocals[:bec_n] - bec_parent_lead[:bec_n]
        candidates.append(_pair_metrics(
            name="becruily_parent_native_pair",
            input_audio=sw_vocals,
            lead=bec_parent_lead,
            backing=bec_parent_native,
            refs_audio=refs_audio,
            sr=sr,
            runtime_seconds=bec_parent_elapsed,
            model=BECRUILY_KARAOKE,
            route="BS-RoFormer-SW vocals -> Becruily MelBand Karaoke native outputs",
        ))
        candidates.append(_pair_metrics(
            name="becruily_parent_residual_pair",
            input_audio=sw_vocals,
            lead=bec_parent_lead,
            backing=bec_parent_residual,
            refs_audio=refs_audio,
            sr=sr,
            runtime_seconds=bec_parent_elapsed,
            model=BECRUILY_KARAOKE,
            route="BS-RoFormer-SW vocals -> Becruily MelBand Karaoke lead; backing = parent - lead",
        ))

        progress("Testing Chas Lead v1: Becruily directly on original mix", 78)
        bec_direct_lead, bec_direct_native, bec_direct_elapsed, bec_direct_tail = _run_separator(
            BECRUILY_KARAOKE, source_path, root / "becruily_direct", model_dir, timeout
        )
        candidates.append(_pair_metrics(
            name="becruily_direct_lead_v1",
            input_audio=source_audio,
            lead=bec_direct_lead,
            backing=None,
            refs_audio=refs_audio,
            sr=sr,
            runtime_seconds=bec_direct_elapsed,
            model=BECRUILY_KARAOKE,
            route="Original mix -> Becruily MelBand Karaoke (Lead v1)",
        ))

        paired = [c for c in candidates if c.get("backing_available")]
        paired_rank = sorted(
            [
                {
                    "name": c["name"],
                    "lead_quality": c["lead"].get("quality_score"),
                    "backing_inclusive_quality": c.get("backing_inclusive", {}).get("quality_score"),
                    "composite": c.get("lead_backing_composite"),
                    "reconstruction_cosine": c.get("input_reconstruction_cosine"),
                    "runtime_seconds": c.get("runtime_seconds"),
                }
                for c in paired
            ],
            key=lambda item: float(item.get("composite") or -1.0),
            reverse=True,
        )
        lead_rank = sorted(
            [
                {
                    "name": c["name"],
                    "lead_quality": c["lead"].get("quality_score"),
                    "lead_si_sdr_db": c["lead"].get("si_sdr_db"),
                    "lead_corr": c["lead"].get("correlation"),
                    "runtime_seconds": c.get("runtime_seconds"),
                }
                for c in candidates
            ],
            key=lambda item: float(item.get("lead_quality") or -1.0),
            reverse=True,
        )

        result = {
            "ok": True,
            "mode": MODE,
            "source_zip_url": zip_url,
            "sample_rate": sr,
            "global_scale": scale,
            "reference_integrity": integrity,
            "inventory": inventory,
            "vocal_reference_definition": vocal_ref_meta,
            "models": {
                "parent": SW_MODEL,
                "current_set_karaoke": CURRENT_KARAOKE,
                "chas_becruily_karaoke": BECRUILY_KARAOKE,
            },
            "timings_seconds": {
                "bs_roformer_sw_parent": round(sw_elapsed, 3),
                "current_set_karaoke": round(set_elapsed, 3),
                "becruily_on_sw_parent": round(bec_parent_elapsed, 3),
                "becruily_direct_mix": round(bec_direct_elapsed, 3),
            },
            "candidates": candidates,
            "paired_ranking": paired_rank,
            "lead_ranking": lead_rank,
            "runtime_tails": {
                "current_set": set_tail,
                "becruily_parent": bec_parent_tail,
                "becruily_direct": bec_direct_tail,
            },
            "next_phase_if_promising": (
                "If Becruily wins or is close, test Chas Lead v2 exactly: build an averaged/ensemble vocal parent, "
                "apply Becruily to that parent, and define backing as the exact parent-minus-lead residual."
            ),
            "complete": True,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        progress("Vocal cascade bakeoff complete", 100)
        log(f"retrieve {output}")
        return result


if __name__ == "__main__":
    result = run()
    while True:
        time.sleep(3600)
