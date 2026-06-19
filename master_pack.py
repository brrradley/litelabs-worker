from __future__ import annotations

import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path


def safe_track_name(filename: str) -> str:
    stem = Path(filename).stem or "track"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem or "track"


def run(cmd: list[str | Path]) -> None:
    print("\nRUN:", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def ffmpeg_to_flac(src: Path, dest: Path) -> None:
    run(["ffmpeg", "-y", "-i", src, "-c:a", "flac", dest])


def build_master_pack(input_audio: Path, work_root: Path, model_dir: Path, output_root: Path) -> dict:
    input_audio = input_audio.resolve()
    work_root = work_root.resolve()
    model_dir = model_dir.resolve()
    output_root = output_root.resolve()

    require_file(input_audio, "input audio")
    require_file(model_dir / "BS-Roformer-SW.ckpt", "BS-RoFormer-SW checkpoint")
    require_file(model_dir / "BS-Roformer-SW.yaml", "BS-RoFormer-SW config")

    track = safe_track_name(input_audio.name)
    job_root = work_root / track
    song_dir = job_root / "song"
    bs_out = job_root / "bs_roformer_sw"
    dem_out = job_root / "demucs6s"
    dem_stems = dem_out / "htdemucs_6s" / track
    master = output_root / f"master_pack_{track}"

    for folder in (song_dir, bs_out, dem_out, master):
        folder.mkdir(parents=True, exist_ok=True)

    wav_file = song_dir / f"{track}.wav"
    run(["ffmpeg", "-y", "-i", input_audio, wav_file])

    run([
        "bs-roformer-infer",
        "--config_path", model_dir / "BS-Roformer-SW.yaml",
        "--model_path", model_dir / "BS-Roformer-SW.ckpt",
        "--input_folder", song_dir,
        "--store_dir", bs_out,
    ])

    run(["demucs", "-n", "htdemucs_6s", "-d", "cuda", "--flac", "-o", dem_out, wav_file])

    bs_vocals = bs_out / f"{track}_vocals.wav"
    bs_drums = bs_out / f"{track}_drums.wav"
    bs_guitar = bs_out / f"{track}_guitar.wav"
    bs_piano = bs_out / f"{track}_piano.wav"
    bs_other = bs_out / f"{track}_other.wav"

    for label, path in {
        "BS vocals": bs_vocals,
        "BS drums": bs_drums,
        "BS guitar": bs_guitar,
        "BS piano": bs_piano,
        "BS other": bs_other,
        "Demucs bass": dem_stems / "bass.flac",
        "Demucs drums": dem_stems / "drums.flac",
        "Demucs guitar": dem_stems / "guitar.flac",
        "Demucs piano": dem_stems / "piano.flac",
        "Demucs other": dem_stems / "other.flac",
    }.items():
        require_file(path, label)

    ffmpeg_to_flac(bs_vocals, master / f"01_{track}_vocals_bs_roformer_sw.flac")
    ffmpeg_to_flac(bs_drums, master / f"02_{track}_drums_bs_roformer_sw.flac")
    shutil.copy2(dem_stems / "bass.flac", master / f"03_{track}_bass_htdemucs_6s.flac")
    ffmpeg_to_flac(bs_guitar, master / f"04_{track}_guitar_bs_roformer_sw.flac")
    ffmpeg_to_flac(bs_piano, master / f"05_{track}_piano_keys_bs_roformer_sw.flac")
    ffmpeg_to_flac(bs_other, master / f"06_{track}_synth_strings_other_bs_roformer_sw.flac")

    run([
        "ffmpeg", "-y",
        "-i", dem_stems / "bass.flac",
        "-i", dem_stems / "drums.flac",
        "-i", dem_stems / "guitar.flac",
        "-i", dem_stems / "piano.flac",
        "-i", dem_stems / "other.flac",
        "-filter_complex", "amix=inputs=5:duration=longest:normalize=0",
        "-c:a", "flac",
        master / f"07_{track}_instrumental_clean_htdemucs_6s.flac",
    ])

    (master / "README.txt").write_text(
        f"LiteRECORDS StemForge master pack\n\nTrack: {track}\n\n"
        "01 vocals: BS-RoFormer-SW\n"
        "02 drums: BS-RoFormer-SW\n"
        "03 bass: htdemucs_6s\n"
        "04 guitar: BS-RoFormer-SW\n"
        "05 piano_keys: BS-RoFormer-SW\n"
        "06 synth_strings_other: BS-RoFormer-SW other\n"
        "07 instrumental_clean: htdemucs_6s non-vocal stems\n",
        encoding="utf-8",
    )

    archive = output_root / f"{track}_master_pack.tar.gz"
    if archive.exists():
        archive.unlink()
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(master, arcname=master.name)

    return {
        "track": track,
        "archive_path": str(archive),
        "stems": sorted(p.name for p in master.iterdir() if p.is_file()),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_audio", type=Path)
    parser.add_argument("--work-root", type=Path, default=Path("/tmp/stemforge/work"))
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/stemforge/output"))
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("STEMFORGE_MODEL_DIR", "/models/bs_roformer_sw")))
    args = parser.parse_args()

    result = build_master_pack(args.input_audio, args.work_root, args.model_dir, args.output_root)
    print(result)


if __name__ == "__main__":
    main()
