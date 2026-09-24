#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

AUDIO_EXTS = {".wav", ".flac", ".aif", ".aiff"}

ALIASES = {
    "kick": ("kick", "kick_drum", "bass_drum"),
    "snare": ("snare",),
    "high_tom": ("high_tom", "hightom", "tom_high"),
    "mid_tom": ("low_mid_tom", "mid_tom", "midlow_tom", "tom_mid"),
    "floor_tom": ("floor_tom", "high_floor_tom", "low_tom", "tom_floor"),
    "closed_hh": ("closed_hi_hat", "closed_hihat", "closed_hh", "hihat_closed"),
    "open_hh": ("open_hi_hat", "open_hihat", "open_hh", "hihat_open"),
    "crash": ("crash_cymbal", "crash"),
    "ride": ("ride_cymbal", "ride"),
}


def norm(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")


def find_stem(folder: Path, aliases: tuple[str, ...]) -> Path | None:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    for p in files:
        n = norm(p.stem)
        if any(a in n for a in aliases):
            return p
    return None


def load(path: Path):
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    return audio, int(sr)


def sum_stems(paths: list[Path], sample_rate: int) -> np.ndarray:
    arrays = []
    for p in paths:
        audio, sr = load(p)
        if sr != sample_rate:
            raise RuntimeError(f"sample-rate mismatch {p}: {sr} != {sample_rate}")
        arrays.append(audio)
    n = min(len(a) for a in arrays)
    return np.sum(np.stack([a[:n] for a in arrays], axis=0), axis=0).astype(np.float32)


def candidate_dirs(root: Path):
    for folder in sorted([p for p in root.rglob("*") if p.is_dir()]):
        found = sum(1 for aliases in ALIASES.values() if find_stem(folder, aliases))
        if found >= 6:
            yield folder


def main() -> int:
    ap = argparse.ArgumentParser(description="Collapse StemGMD's nine drum pieces into SET 4-stem MSST data.")
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--valid-ratio", type=float, default=0.05)
    args = ap.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    tracks = []
    for folder in candidate_dirs(source):
        mapped = {k: find_stem(folder, v) for k, v in ALIASES.items()}
        if all(mapped.values()):
            tracks.append((folder, mapped))

    if not tracks:
        raise SystemExit("No complete nine-piece StemGMD performances found; inspect extracted layout/filenames.")

    split_at = max(1, int(len(tracks) * (1.0 - max(0.01, min(0.25, args.valid_ratio)))))
    for idx, (folder, m) in enumerate(tracks):
        split = "train" if idx < split_at else "valid"
        dest = output / split / f"{idx:06d}_{norm(folder.name)}"
        dest.mkdir(parents=True, exist_ok=True)

        kick, sr = load(m["kick"])
        snare, sr2 = load(m["snare"])
        if sr != sr2:
            raise RuntimeError(f"sample-rate mismatch in {folder}")

        toms = sum_stems([m["high_tom"], m["mid_tom"], m["floor_tom"]], sr)
        cymbals = sum_stems([m["closed_hh"], m["open_hh"], m["crash"], m["ride"]], sr)

        n = min(len(kick), len(snare), len(toms), len(cymbals))
        kick = kick[:n]
        snare = snare[:n]
        toms = toms[:n]
        cymbals = cymbals[:n]
        mixture = kick + snare + toms + cymbals

        sf.write(dest / "kick.wav", kick, sr, subtype="FLOAT")
        sf.write(dest / "snare.wav", snare, sr, subtype="FLOAT")
        sf.write(dest / "toms.wav", toms, sr, subtype="FLOAT")
        sf.write(dest / "cymbals.wav", cymbals, sr, subtype="FLOAT")
        sf.write(dest / "mixture.wav", mixture, sr, subtype="FLOAT")

    print(f"prepared {len(tracks)} performances")
    print(f"train: {split_at}; valid: {len(tracks)-split_at}")
    print(f"output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
