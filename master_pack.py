from __future__ import annotations

import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Callable

ProgressCallback = Callable[[str, int], None]


def safe_track_name(filename: str) -> str:
    stem = Path(filename).stem or "track"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem or "track"


def safe_output_format(value: str | None) -> str:
    value = (value or "flac").lower().strip()
    return value if value in {"mp3", "flac"} else "flac"


def run(cmd: list[str | Path]) -> None:
    print("\nRUN:", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)


def notify(progress: ProgressCallback | None, message: str, percent: int) -> None:
    print(f"LiteLABS progress {percent}%: {message}", flush=True)
    if progress:
        progress(message, percent)


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def convert_audio(src: Path, dest: Path, output_format: str) -> None:
    output_format = safe_output_format(output_format)
    if output_format == "mp3":
        run(["ffmpeg", "-y", "-i", src, "-codec:a", "libmp3lame", "-q:a", "2", dest])
    else:
        run(["ffmpeg", "-y", "-i", src, "-c:a", "flac", dest])


def copy_or_convert_audio(src: Path, dest: Path, output_format: str) -> None:
    output_format = safe_output_format(output_format)
    if output_format == "flac" and src.suffix.lower() == ".flac":
        shutil.copy2(src, dest)
    else:
        convert_audio(src, dest, output_format)


def write_litelabs_readme(path: Path, track: str, output_format: str) -> None:
    ext = safe_output_format(output_format).upper()
    path.write_text(
        f"LiteLABS by LiteRECORDS\n\n"
        f"Track: {track}\n"
        f"Output format: {ext}\n\n"
        "Created by LiteRECORDS\n"
        "https://literecords.com\n\n"
        "This stem pack was generated using LiteLABS, an experimental music tool created by "
        "LiteRECORDS to support musicians, producers, remixers, DJs, and learners. LiteLABS "
        "is designed for educational, creative, and restoration purposes, helping users study "
        "arrangements, practise production techniques, prepare remix ideas, and better understand "
        "how tracks are built.\n\n"
        "This service is not intended to support piracy, unauthorised redistribution, or misuse "
        "of copyrighted material. Please only process music that you own, control, have permission "
        "to use, or are legally allowed to study or transform. LiteRECORDS is committed to supporting "
        "musicians and encouraging responsible, creative use of music technology.\n\n"
        "Included stems:\n\n"
        "01 Vocals\n"
        "02 Drums\n"
        "03 Bass\n"
        "04 Guitar\n"
        "05 Piano / Keys\n"
        "06 Synths / Strings / Other\n"
        "07 Clean Instrumental\n\n"
        "Generated with care by LiteLABS.\n",
        encoding="utf-8",
    )


def make_zip_archive(source_dir: Path, archive: Path, archive_root: str, progress: ProgressCallback | None = None) -> None:
    notify(progress, "Creating ZIP stem pack", 88)
    if archive.exists():
        archive.unlink()

    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zip_file:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                zip_file.write(path, arcname=str(Path(archive_root) / path.relative_to(source_dir)))

    size_mb = archive.stat().st_size / (1024 * 1024)
    print(f"LiteLABS archive created: {archive} ({size_mb:.2f} MB)", flush=True)
    notify(progress, "ZIP stem pack ready", 92)


def build_master_pack(
    input_audio: Path,
    work_root: Path,
    model_dir: Path,
    output_root: Path,
    progress: ProgressCallback | None = None,
    output_format: str = "flac",
) -> dict:
    input_audio = input_audio.resolve()
    work_root = work_root.resolve()
    model_dir = model_dir.resolve()
    output_root = output_root.resolve()
    output_format = safe_output_format(output_format)
    ext = output_format

    notify(progress, "Checking worker files", 18)
    require_file(input_audio, "input audio")
    require_file(model_dir / "BS-Roformer-SW.ckpt", "BS-RoFormer-SW checkpoint")
    require_file(model_dir / "BS-Roformer-SW.yaml", "BS-RoFormer-SW config")

    track = safe_track_name(input_audio.name)
    job_root = work_root / track
    song_dir = job_root / "song"
    bs_out = job_root / "bs_roformer_sw"
    dem_out = job_root / "demucs6s"
    dem_stems = dem_out / "htdemucs_6s" / track
    master = output_root / f"{track}-litelabs-{output_format}-stem-pack"

    for folder in (song_dir, bs_out, dem_out, master):
        folder.mkdir(parents=True, exist_ok=True)

    notify(progress, "Preparing audio", 22)
    wav_file = song_dir / f"{track}.wav"
    run(["ffmpeg", "-y", "-i", input_audio, wav_file])

    notify(progress, "Separating main stems", 30)
    run([
        "bs-roformer-infer",
        "--config_path", model_dir / "BS-Roformer-SW.yaml",
        "--model_path", model_dir / "BS-Roformer-SW.ckpt",
        "--input_folder", song_dir,
        "--store_dir", bs_out,
    ])

    notify(progress, "Separating supporting stems", 52)
    run(["demucs", "-n", "htdemucs_6s", "-d", "cuda", "--flac", "-o", dem_out, wav_file])

    bs_vocals = bs_out / f"{track}_vocals.wav"
    bs_drums = bs_out / f"{track}_drums.wav"
    bs_guitar = bs_out / f"{track}_guitar.wav"
    bs_piano = bs_out / f"{track}_piano.wav"
    bs_other = bs_out / f"{track}_other.wav"

    notify(progress, "Checking generated stems", 66)
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

    notify(progress, f"Building vocals.{ext}", 68)
    convert_audio(bs_vocals, master / f"01_{track}_vocals.{ext}", output_format)
    notify(progress, f"Building drums.{ext}", 71)
    convert_audio(bs_drums, master / f"02_{track}_drums.{ext}", output_format)
    notify(progress, f"Building bass.{ext}", 74)
    copy_or_convert_audio(dem_stems / "bass.flac", master / f"03_{track}_bass.{ext}", output_format)
    notify(progress, f"Building guitar.{ext}", 77)
    convert_audio(bs_guitar, master / f"04_{track}_guitar.{ext}", output_format)
    notify(progress, f"Building piano_keys.{ext}", 80)
    convert_audio(bs_piano, master / f"05_{track}_piano_keys.{ext}", output_format)
    notify(progress, f"Building synth_strings_other.{ext}", 83)
    convert_audio(bs_other, master / f"06_{track}_synth_strings_other.{ext}", output_format)

    notify(progress, f"Building clean instrumental.{ext}", 86)
    instrumental_wav = job_root / f"{track}_instrumental_clean.wav"
    run([
        "ffmpeg", "-y",
        "-i", dem_stems / "bass.flac",
        "-i", dem_stems / "drums.flac",
        "-i", dem_stems / "guitar.flac",
        "-i", dem_stems / "piano.flac",
        "-i", dem_stems / "other.flac",
        "-filter_complex", "amix=inputs=5:duration=longest:normalize=0",
        instrumental_wav,
    ])
    convert_audio(instrumental_wav, master / f"07_{track}_instrumental_clean.{ext}", output_format)

    notify(progress, "Writing README", 87)
    write_litelabs_readme(master / "README.txt", track, output_format)

    archive = output_root / f"{track}-litelabs-{output_format}-stem-pack.zip"
    make_zip_archive(master, archive, master.name, progress)

    return {
        "track": track,
        "archive_path": str(archive),
        "output_format": output_format,
        "stems": sorted(p.name for p in master.iterdir() if p.is_file()),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_audio", type=Path)
    parser.add_argument("--work-root", type=Path, default=Path("/tmp/stemforge/work"))
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/stemforge/output"))
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("STEMFORGE_MODEL_DIR", "/models/bs_roformer_sw")))
    parser.add_argument("--output-format", default="flac", choices=["mp3", "flac"])
    args = parser.parse_args()

    result = build_master_pack(args.input_audio, args.work_root, args.model_dir, args.output_root, output_format=args.output_format)
    print(result)


if __name__ == "__main__":
    main()
