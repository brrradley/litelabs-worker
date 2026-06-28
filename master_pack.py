from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ProgressCallback = Callable[[str, int], None]


@dataclass
class StemDecision:
    label: str
    source: Path
    include: bool
    reason: str
    score: float
    active_ratio: float
    mean_db: float
    max_db: float


def safe_track_name(filename: str) -> str:
    stem = Path(filename).stem or "track"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem or "track"


def safe_output_format(value: str | None) -> str:
    value = (value or "flac").lower().strip()
    return value if value in {"mp3", "flac"} else "flac"


def format_bytes(num_bytes: int) -> str:
    if num_bytes >= 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB"
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.2f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.2f} KB"
    return f"{num_bytes} bytes"


def format_elapsed(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} hours, {minutes} minutes, {secs} seconds"
    if minutes:
        return f"{minutes} minutes, {secs} seconds"
    return f"{secs} seconds"


def run(cmd: list[str | Path]) -> None:
    print("\nRUN:", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)


def run_capture(cmd: list[str | Path]) -> str:
    completed = subprocess.run(
        [str(x) for x in cmd],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout or ""


def notify(progress: ProgressCallback | None, message: str, percent: int) -> None:
    print(f"LiteLABS progress {percent}%: {message}", flush=True)
    if progress:
        progress(message, percent)


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def probe_duration(path: Path) -> float:
    output = run_capture([
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]).strip()
    try:
        return float(output)
    except ValueError:
        return 0.0


def parse_float(pattern: str, text: str, default: float) -> float:
    match = re.search(pattern, text)
    if not match:
        return default
    try:
        return float(match.group(1))
    except ValueError:
        return default


def analyse_audio(path: Path) -> dict:
    duration = probe_duration(path)

    vol = run_capture([
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i", path,
        "-af", "volumedetect",
        "-f", "null",
        "-",
    ])
    mean_db = parse_float(r"mean_volume:\s*(-?\d+(?:\.\d+)?) dB", vol, -99.0)
    max_db = parse_float(r"max_volume:\s*(-?\d+(?:\.\d+)?) dB", vol, -99.0)

    silence = run_capture([
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i", path,
        "-af", "silencedetect=noise=-45dB:d=0.30",
        "-f", "null",
        "-",
    ])
    silence_total = 0.0
    for value in re.findall(r"silence_duration:\s*(\d+(?:\.\d+)?)", silence):
        try:
            silence_total += float(value)
        except ValueError:
            pass

    if duration > 0:
        active_ratio = max(0.0, min(1.0, (duration - silence_total) / duration))
    else:
        active_ratio = 0.0

    loudness_score = max(0.0, min(1.0, (mean_db + 45.0) / 30.0))
    peak_score = max(0.0, min(1.0, (max_db + 35.0) / 25.0))
    score = round((active_ratio * 0.55) + (loudness_score * 0.30) + (peak_score * 0.15), 3)

    return {
        "duration": duration,
        "mean_db": mean_db,
        "max_db": max_db,
        "active_ratio": active_ratio,
        "score": score,
    }


def optional_stem_decision(label: str, source: Path) -> StemDecision:
    stats = analyse_audio(source)
    score = float(stats["score"])
    active_ratio = float(stats["active_ratio"])
    mean_db = float(stats["mean_db"])
    max_db = float(stats["max_db"])

    if max_db <= -45.0:
        return StemDecision(label, source, False, "not confidently detected", score, active_ratio, mean_db, max_db)
    if active_ratio < 0.08:
        return StemDecision(label, source, False, "low activity", score, active_ratio, mean_db, max_db)
    if mean_db < -38.0 and score < 0.42:
        return StemDecision(label, source, False, "low confidence / mostly bleed", score, active_ratio, mean_db, max_db)
    if score < 0.34:
        return StemDecision(label, source, False, "low confidence", score, active_ratio, mean_db, max_db)

    return StemDecision(label, source, True, "confidently detected", score, active_ratio, mean_db, max_db)


def score_of(stats: dict[str, dict], label: str) -> float:
    return float(stats.get(label, {}).get("score", 0.0))


def active_of(stats: dict[str, dict], label: str) -> float:
    return float(stats.get(label, {}).get("active_ratio", 0.0))


def detect_genre_from_audio(decisions: list[StemDecision], core_stats: dict[str, dict], original_stats: dict) -> tuple[str, str]:
    """Return a practical audio-derived genre/track-type label plus a short explanation.

    This deliberately does not use file metadata. It uses the separated stem activity as the first
    reliable signal, because that is what we can validate and tune from real LiteRECORDS uploads.
    """
    included = {d.label for d in decisions if d.include}
    optional_scores = {d.label: d.score for d in decisions}

    vocals = score_of(core_stats, "Vocals")
    drums = score_of(core_stats, "Drums")
    bass = score_of(core_stats, "Bass")
    guitar = optional_scores.get("Guitar", 0.0)
    piano = optional_scores.get("Piano / Keys", 0.0)
    synth_other = optional_scores.get("Synths / Strings / Other", 0.0)
    original_active = float(original_stats.get("active_ratio", 0.0))

    strong_rhythm = drums >= 0.46 and bass >= 0.34
    strong_vocal = vocals >= 0.45
    strong_guitar = "Guitar" in included and guitar >= 0.42
    strong_piano = "Piano / Keys" in included and piano >= 0.42
    strong_synth = "Synths / Strings / Other" in included and synth_other >= 0.42

    if strong_rhythm and strong_synth and not strong_guitar:
        return "electronic_dance", "strong drums/bass with active synth/other and no confident guitar"
    if strong_rhythm and strong_guitar:
        return "rock_band", "strong drums with confident guitar activity"
    if strong_piano and strong_vocal and drums < 0.42:
        return "piano_vocal_or_pop_ballad", "confident piano/keys with strong vocal and lighter drums"
    if strong_vocal and drums >= 0.35 and bass >= 0.25 and not strong_guitar:
        return "vocal_pop", "strong vocal with moderate rhythm section and no confident guitar"
    if strong_vocal and original_active > 0.35 and drums < 0.30 and bass < 0.30:
        return "acoustic_or_sparse", "strong vocal with low drum/bass activity"
    if strong_rhythm and not strong_vocal:
        return "instrumental_or_dance", "strong drums/bass with weaker vocal presence"
    return "mixed_or_unknown", "audio features did not strongly match a known route"


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


def write_litelabs_readme(
    path: Path,
    track: str,
    output_format: str,
    pack_size: str,
    elapsed_time: str,
    detected_genre: str,
    genre_reason: str,
    included_stems: list[str],
    omitted_stems: list[tuple[str, str]],
) -> None:
    ext = safe_output_format(output_format).upper()

    included_lines = "\n".join(included_stems) if included_stems else "None"
    if omitted_stems:
        omitted_lines = "\n".join(f"{label} — {reason}" for label, reason in omitted_stems)
        omitted_section = f"\n\nOmitted stems:\n\n{omitted_lines}"
    else:
        omitted_section = ""

    path.write_text(
        "Stem Extraction Tools by LiteLABS\n\n"
        f"Track: {track}\n"
        f"Output format: {ext}\n"
        f"Stem pack size: {pack_size}\n"
        f"Elapsed time: {elapsed_time}\n"
        f"Detected genre: {detected_genre}\n"
        f"Genre reason: {genre_reason}\n\n"
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
        f"{included_lines}"
        f"{omitted_section}\n\n"
        "Generated with care by LiteLABS.\n"
        "https://literecords.com\n",
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
    started_at = time.monotonic()

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
    master = output_root / f"{track}-stem-extraction-tools-{output_format}-pack"

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

    notify(progress, "Analysing audio and stem confidence", 67)
    optional_decisions = [
        optional_stem_decision("Guitar", bs_guitar),
        optional_stem_decision("Piano / Keys", bs_piano),
        optional_stem_decision("Synths / Strings / Other", bs_other),
    ]
    core_stats = {
        "Vocals": analyse_audio(bs_vocals),
        "Drums": analyse_audio(bs_drums),
        "Bass": analyse_audio(dem_stems / "bass.flac"),
    }
    original_stats = analyse_audio(wav_file)
    detected_genre, genre_reason = detect_genre_from_audio(optional_decisions, core_stats, original_stats)
    print(f"LiteLABS detected genre: {detected_genre} ({genre_reason})", flush=True)

    included_readme: list[str] = []
    omitted_readme: list[tuple[str, str]] = []
    output_index = 1

    def add_stem(label: str, src: Path, slug: str, percent: int) -> None:
        nonlocal output_index
        display_name = f"{output_index:02d} {label}"
        notify(progress, f"Building {display_name}.{ext}", percent)
        copy_or_convert_audio(src, master / f"{output_index:02d}_{track}_{slug}.{ext}", output_format)
        included_readme.append(display_name)
        output_index += 1

    add_stem("Vocals", bs_vocals, "vocals", 68)
    add_stem("Drums", bs_drums, "drums", 71)
    add_stem("Bass", dem_stems / "bass.flac", "bass", 74)

    for decision, slug, percent in [
        (optional_decisions[0], "guitar", 77),
        (optional_decisions[1], "piano_keys", 80),
        (optional_decisions[2], "synth_strings_other", 83),
    ]:
        if decision.include:
            add_stem(decision.label, decision.source, slug, percent)
        else:
            notify(progress, f"Omitting {decision.label}: {decision.reason}", percent)
            omitted_readme.append((decision.label, decision.reason))

    notify(progress, f"Building Clean Instrumental.{ext}", 86)
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
    copy_or_convert_audio(instrumental_wav, master / f"{output_index:02d}_{track}_instrumental_clean.{ext}", output_format)
    included_readme.append(f"{output_index:02d} Clean Instrumental")

    archive = output_root / f"{track}-stem-extraction-tools-{output_format}-pack.zip"

    notify(progress, "Writing README", 87)
    write_litelabs_readme(
        master / "README.txt",
        track,
        output_format,
        "calculating",
        format_elapsed(time.monotonic() - started_at),
        detected_genre,
        genre_reason,
        included_readme,
        omitted_readme,
    )

    make_zip_archive(master, archive, master.name, progress)
    final_size = format_bytes(archive.stat().st_size)

    notify(progress, "Finalising README", 91)
    write_litelabs_readme(
        master / "README.txt",
        track,
        output_format,
        final_size,
        format_elapsed(time.monotonic() - started_at),
        detected_genre,
        genre_reason,
        included_readme,
        omitted_readme,
    )
    make_zip_archive(master, archive, master.name, progress)

    return {
        "track": track,
        "archive_path": str(archive),
        "output_format": output_format,
        "detected_genre": detected_genre,
        "genre_reason": genre_reason,
        "included_stems": included_readme,
        "omitted_stems": [{"label": label, "reason": reason} for label, reason in omitted_readme],
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
