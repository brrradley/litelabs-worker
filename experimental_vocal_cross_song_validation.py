from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

import experimental_vocal_cascade_bakeoff as vb
import multitrack_ground_truth_campaign as mt
import synthetic_ground_truth_campaign as sgt
from ground_truth_benchmark import _score

MODE = "experimental_vocal_cross_song_validation_v7"
DEFAULT_ZIP = "https://literecords.com/tmp/tears-for-fears-shout.zip"
DEFAULT_OUTPUT = "/workspace/litelabs-research/shout_vocal_cross_song_validation_v7.json"


def _norm(name: str) -> str:
    return " ".join(str(name).strip().lower().replace("_", " ").replace("-", " ").split())


def _looks_vocal(name: str) -> bool:
    n = _norm(name)
    try:
        if sgt.track_category(name) == "vocals":
            return True
    except Exception:
        pass
    tokens = (
        "vocal", "vox", "voice", "bgv", "backing", "background", "harmony",
        "choir", "adlib", "ad lib", "breath", "scream", "robot", "pitched voc",
    )
    return any(token in n for token in tokens)


def _lead_rank(name: str) -> tuple[int, int, str]:
    n = _norm(name)
    if "lead vocal" in n or "lead vox" in n:
        return (0, len(n), n)
    if "main vocal" in n or "main vox" in n:
        return (1, len(n), n)
    if n in {"vocal", "vocals", "vox", "voice"}:
        return (2, len(n), n)
    if "lead" in n and ("vocal" in n or "vox" in n or "voice" in n):
        return (3, len(n), n)
    if "lead" in n:
        return (4, len(n), n)
    return (50, len(n), n)


def _is_explicit_backing(name: str) -> bool:
    n = _norm(name)
    return any(token in n for token in ("backing", "background", "bgv", "bg vocal", "harmony", "choir"))


def _sum_audio(items: list[np.ndarray]) -> np.ndarray:
    if not items:
        raise RuntimeError("No audio arrays to sum")
    n = min(len(x) for x in items)
    return np.sum(np.stack([np.asarray(x[:n], dtype=np.float32) for x in items], axis=0), axis=0)


def _build_vocal_refs(tracks: dict[str, np.ndarray], work: Path, sr: int) -> tuple[dict[str, np.ndarray], dict]:
    vocal_like = [(name, audio) for name, audio in tracks.items() if _looks_vocal(name)]
    if not vocal_like:
        raise RuntimeError(f"No vocal-like tracks found. Inventory: {sorted(tracks)}")

    ranked = sorted(vocal_like, key=lambda item: _lead_rank(item[0]))
    if _lead_rank(ranked[0][0])[0] >= 50:
        raise RuntimeError(
            "Could not identify a lead-vocal reference safely before GPU inference. "
            f"Vocal-like tracks: {[name for name, _ in vocal_like]}"
        )

    lead_name, lead = ranked[0]
    nonlead = [(name, audio) for name, audio in vocal_like if name != lead_name]
    if not nonlead:
        raise RuntimeError(
            "Cross-song validation needs at least one non-lead vocal reference to score backing. "
            f"Lead={lead_name}; vocal-like={[name for name, _ in vocal_like]}"
        )

    explicit_backing = [(name, audio) for name, audio in nonlead if _is_explicit_backing(name)]
    backing_inclusive = _sum_audio([audio for _name, audio in nonlead])
    backing_strict = (
        _sum_audio([audio for _name, audio in explicit_backing])
        if explicit_backing else backing_inclusive.copy()
    )
    total_vocals = _sum_audio([lead] + [audio for _name, audio in nonlead])
    core_pair = lead[: min(len(lead), len(backing_strict))] + backing_strict[: min(len(lead), len(backing_strict))]

    refs = {
        "lead": np.asarray(lead, dtype=np.float32),
        "backing_strict": np.asarray(backing_strict, dtype=np.float32),
        "backing_inclusive": np.asarray(backing_inclusive, dtype=np.float32),
        "core_pair": np.asarray(core_pair, dtype=np.float32),
        "total_vocals": np.asarray(total_vocals, dtype=np.float32),
    }
    meta = {
        "lead_members": [lead_name],
        "backing_strict_members": [name for name, _audio in explicit_backing] or [name for name, _audio in nonlead],
        "backing_inclusive_members": [name for name, _audio in nonlead],
        "all_vocal_members": [name for name, _audio in vocal_like],
        "strict_backing_fell_back_to_inclusive": not bool(explicit_backing),
        "note": (
            "Lead selection is conservative and must match a lead/main-vocal style name before GPU inference. "
            "Inclusive backing contains every non-lead vocal-like track."
        ),
    }
    return refs, meta


def _write_temp_audio(path: Path, audio: np.ndarray, sr: int) -> None:
    mt._write(path, np.asarray(audio, dtype=np.float32), sr)


def _run_sw_parent(source_path: Path, output_dir: Path, model_dir: Path, timeout: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "audio-separator", str(source_path),
        "--model_filename", vb.SW_MODEL,
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
        raise RuntimeError(f"BS-RoFormer-SW failed: {tail}")
    vocals_path = vb._find_output(output_dir, "vocals")
    if vocals_path is None:
        raise RuntimeError("BS-RoFormer-SW produced no vocals output")
    return vocals_path, mt._load(vocals_path)[0], elapsed, tail


def _brief(candidate: dict) -> dict:
    return {
        "name": candidate["name"],
        "lead_quality": candidate["lead"].get("quality_score"),
        "backing_inclusive_quality": candidate.get("backing_inclusive", {}).get("quality_score"),
        "backing_strict_quality": candidate.get("backing_strict", {}).get("quality_score"),
        "composite": candidate.get("lead_backing_composite"),
        "runtime_seconds": candidate.get("runtime_seconds"),
        "reconstruction_cosine": candidate.get("input_reconstruction_cosine"),
    }


def run() -> dict:
    zip_url = str(os.getenv("LITELABS_BENCHMARK_ZIP_URL", DEFAULT_ZIP)).strip()
    output = Path(os.getenv("LITELABS_BENCHMARK_OUTPUT", DEFAULT_OUTPUT))
    output.parent.mkdir(parents=True, exist_ok=True)
    timeout = max(600, min(5400, int(os.getenv("LITELABS_BENCHMARK_MODEL_TIMEOUT", "3600"))))
    model_dir = Path(os.getenv("LITELABS_AUDIO_SEPARATOR_MODEL_DIR", "/models/audio_separator"))
    model_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="litelabs_vocal_cross_song_") as temp:
        root = Path(temp)
        archive = root / (Path(urlparse(zip_url).path).name or "benchmark.zip")
        extracted = root / "extracted"
        work = root / "work"

        print(f"[{MODE}] preparing exact-ground-truth cross-song pack", flush=True)
        mt._download(zip_url, archive)
        mt._safe_extract(archive, extracted)
        sgt._remove_macos_metadata(extracted)
        mt._track_category = sgt.track_category
        tracks, sr, inventory = mt._build_tracks(extracted, work)
        source_path, parent_refs, scale = mt._build_references(tracks, sr, work)
        integrity = sgt._reference_integrity(source_path, parent_refs, sr)
        if not integrity["exact_partition"]:
            raise RuntimeError(f"Synthetic source integrity failed: {integrity}")

        refs_audio, vocal_meta = _build_vocal_refs(tracks, work, sr)

        print(f"[{MODE}] reference mapping accepted before GPU work: {vocal_meta}", flush=True)
        sw_path, sw_vocals, sw_elapsed, sw_tail = _run_sw_parent(source_path, root / "sw", model_dir, timeout)

        parent_quality = {
            "bs_roformer_sw": _score(sw_vocals[: min(len(sw_vocals), len(refs_audio["total_vocals"]))], refs_audio["total_vocals"][: min(len(sw_vocals), len(refs_audio["total_vocals"]))], sr),
        }

        candidates = []

        # Current production child-vocal route, using the orientation established on Disturbia:
        # the nominal Vocals output behaves as backing, so lead is the exact parent residual.
        current_nominal, _current_secondary, current_elapsed, current_tail = vb._run_current_set_karaoke(
            sw_path, root / "current_set", timeout
        )
        n = min(len(sw_vocals), len(current_nominal))
        current_backing = current_nominal[:n]
        current_lead = sw_vocals[:n] - current_backing
        candidates.append(vb._pair_metrics(
            name="current_set_residual_orientation",
            input_audio=sw_vocals,
            lead=current_lead,
            backing=current_backing,
            refs_audio=refs_audio,
            sr=sr,
            runtime_seconds=current_elapsed,
            model=vb.CURRENT_KARAOKE,
            route="SW vocals -> current SET karaoke nominal output as backing; lead = SW parent - backing",
        ))

        # Fast candidate: SW parent -> Becruily; empirically strong secondary output is lead.
        _fast_nominal, fast_secondary, fast_elapsed, fast_tail = vb._run_separator(
            vb.BECRUILY_KARAOKE, sw_path, root / "fast_becruily", model_dir, timeout
        )
        if fast_secondary is None:
            raise RuntimeError("Becruily produced no secondary output for fast SW-parent route")
        n = min(len(sw_vocals), len(fast_secondary))
        fast_lead = fast_secondary[:n]
        fast_backing = sw_vocals[:n] - fast_lead
        candidates.append(vb._pair_metrics(
            name="fast_sw_to_becruily_residual_pair",
            input_audio=sw_vocals,
            lead=fast_lead,
            backing=fast_backing,
            refs_audio=refs_audio,
            sr=sr,
            runtime_seconds=fast_elapsed,
            model=vb.BECRUILY_KARAOKE,
            route="SW vocals -> Becruily secondary output as lead; backing = SW parent - lead",
        ))

        # Disturbia quality candidate: 25% SW + 75% alternate MelBand vocal parent -> Becruily.
        alt_vocals, _alt_other, alt_elapsed, alt_tail = vb._run_separator(
            vb.ALT_VOCAL_PARENT, source_path, root / "alt_parent", model_dir, timeout
        )
        n_parent = min(len(sw_vocals), len(alt_vocals))
        blend = (
            0.25 * np.asarray(sw_vocals[:n_parent], dtype=np.float32)
            + 0.75 * np.asarray(alt_vocals[:n_parent], dtype=np.float32)
        )
        blend_path = root / "vocal_parent_sw25_mel75.wav"
        _write_temp_audio(blend_path, blend, sr)
        parent_quality["melband_becruily"] = _score(alt_vocals[: min(len(alt_vocals), len(refs_audio["total_vocals"]))], refs_audio["total_vocals"][: min(len(alt_vocals), len(refs_audio["total_vocals"]))], sr)
        parent_quality["blend_sw25_mel75"] = _score(blend[: min(len(blend), len(refs_audio["total_vocals"]))], refs_audio["total_vocals"][: min(len(blend), len(refs_audio["total_vocals"]))], sr)

        _blend_nominal, blend_secondary, blend_elapsed, blend_tail = vb._run_separator(
            vb.BECRUILY_KARAOKE, blend_path, root / "blend_becruily", model_dir, timeout
        )
        if blend_secondary is None:
            raise RuntimeError("Becruily produced no secondary output for 25/75 blend route")
        n = min(len(blend), len(blend_secondary))
        blend_lead = blend_secondary[:n]
        blend_backing = blend[:n] - blend_lead
        blend_candidate = vb._pair_metrics(
            name="quality_sw25_mel75_to_becruily_residual_pair",
            input_audio=blend,
            lead=blend_lead,
            backing=blend_backing,
            refs_audio=refs_audio,
            sr=sr,
            runtime_seconds=alt_elapsed + blend_elapsed,
            model=f"{vb.ALT_VOCAL_PARENT} + {vb.BECRUILY_KARAOKE}",
            route="25% SW + 75% MelBand vocal parent -> Becruily secondary output as lead; backing = blend parent - lead",
        )
        blend_candidate["runtime_components_seconds"] = {
            "alternate_vocal_parent": round(alt_elapsed, 3),
            "becruily_on_blend": round(blend_elapsed, 3),
            "incremental_beyond_existing_sw_parent": round(alt_elapsed + blend_elapsed, 3),
        }
        candidates.append(blend_candidate)

        rows = [_brief(c) for c in candidates]
        result = {
            "ok": True,
            "mode": MODE,
            "source_zip_url": zip_url,
            "sample_rate": sr,
            "global_scale": scale,
            "reference_integrity": integrity,
            "inventory": inventory,
            "vocal_reference_definition": vocal_meta,
            "models": {
                "parent": vb.SW_MODEL,
                "current_set_karaoke": vb.CURRENT_KARAOKE,
                "becruily_karaoke": vb.BECRUILY_KARAOKE,
                "alternate_vocal_parent": vb.ALT_VOCAL_PARENT,
            },
            "vocal_parent_quality": parent_quality,
            "candidates": candidates,
            "comparison": {
                "rows": rows,
                "by_lead_quality": sorted(rows, key=lambda x: float(x.get("lead_quality") or -1.0), reverse=True),
                "by_backing_inclusive_quality": sorted(rows, key=lambda x: float(x.get("backing_inclusive_quality") or -1.0), reverse=True),
                "by_composite": sorted(rows, key=lambda x: float(x.get("composite") or -1.0), reverse=True),
                "note": (
                    "This is the final cross-song validation: current SET baseline, fast SW->Becruily residual, "
                    "and the Disturbia quality winner 25/75 SW/MelBand->Becruily. No further blend sweep is run."
                ),
            },
            "timings_seconds": {
                "bs_roformer_sw_parent": round(sw_elapsed, 3),
                "current_set_karaoke": round(current_elapsed, 3),
                "fast_becruily_on_sw": round(fast_elapsed, 3),
                "alternate_melband_parent": round(alt_elapsed, 3),
                "becruily_on_25_75_blend": round(blend_elapsed, 3),
            },
            "runtime_tails": {
                "sw": sw_tail,
                "current_set": current_tail,
                "fast_becruily": fast_tail,
                "alternate_parent": alt_tail,
                "blend_becruily": blend_tail,
            },
            "complete": True,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[{MODE}] wrote {output}", flush=True)
        return result


if __name__ == "__main__":
    run()
