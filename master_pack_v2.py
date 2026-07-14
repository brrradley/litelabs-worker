from __future__ import annotations

import time
from pathlib import Path

from legacy_master_pack import *  # noqa: F401,F403


ENGINE_NAME = "BS-RoFormer-SW"
ENGINE_VERSION = "LiteLABS 2.0 baseline"


def build_master_pack(
    input_audio: Path,
    work_root: Path,
    model_dir: Path,
    output_root: Path,
    progress: ProgressCallback | None = None,
    output_format: str = "flac",
) -> dict:
    """Build the LiteLABS v2 stem pack using BS-RoFormer-SW for every primary stem."""
    started_at = time.monotonic()

    input_audio = input_audio.resolve()
    work_root = work_root.resolve()
    model_dir = model_dir.resolve()
    output_root = output_root.resolve()
    output_format = safe_output_format(output_format)
    ext = output_format

    notify(progress, "Checking LiteLABS v2 worker files", 18)
    require_file(input_audio, "input audio")
    require_file(model_dir / "BS-Roformer-SW.ckpt", "BS-RoFormer-SW checkpoint")
    require_file(model_dir / "BS-Roformer-SW.yaml", "BS-RoFormer-SW config")

    track = safe_track_name(input_audio.name)
    job_root = work_root / track
    song_dir = job_root / "song"
    bs_out = job_root / "bs_roformer_sw"
    master = output_root / f"{track}-stem-extraction-tools-{output_format}-pack"

    for folder in (song_dir, bs_out, master):
        folder.mkdir(parents=True, exist_ok=True)

    notify(progress, "Preparing audio", 22)
    wav_file = song_dir / f"{track}.wav"
    run(["ffmpeg", "-y", "-i", input_audio, wav_file])

    notify(progress, "Separating studio-style stems with BS-RoFormer-SW", 30)
    run([
        "bs-roformer-infer",
        "--config_path", model_dir / "BS-Roformer-SW.yaml",
        "--model_path", model_dir / "BS-Roformer-SW.ckpt",
        "--input_folder", song_dir,
        "--store_dir", bs_out,
    ])

    stems = {
        "Vocals": bs_out / f"{track}_vocals.wav",
        "Drums": bs_out / f"{track}_drums.wav",
        "Bass": bs_out / f"{track}_bass.wav",
        "Guitar": bs_out / f"{track}_guitar.wav",
        "Piano / Keys": bs_out / f"{track}_piano.wav",
        "Synths / Strings / Other": bs_out / f"{track}_other.wav",
    }

    notify(progress, "Checking generated stems", 62)
    for label, path in stems.items():
        require_file(path, f"BS-RoFormer-SW {label}")

    notify(progress, "Analysing track and stem confidence", 66)
    optional_decisions = [
        optional_stem_decision("Guitar", stems["Guitar"]),
        optional_stem_decision("Piano / Keys", stems["Piano / Keys"]),
        optional_stem_decision("Synths / Strings / Other", stems["Synths / Strings / Other"]),
    ]
    core_stats = {
        "Vocals": analyse_audio(stems["Vocals"]),
        "Drums": analyse_audio(stems["Drums"]),
        "Bass": analyse_audio(stems["Bass"]),
    }
    original_stats = analyse_audio(wav_file)
    detected_genre, genre_reason = detect_genre_from_audio(optional_decisions, core_stats, original_stats)
    print(f"LiteLABS v2 detected genre: {detected_genre} ({genre_reason})", flush=True)

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

    add_stem("Vocals", stems["Vocals"], "vocals", 68)
    add_stem("Drums", stems["Drums"], "drums", 71)
    add_stem("Bass", stems["Bass"], "bass", 74)

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
        "-i", stems["Bass"],
        "-i", stems["Drums"],
        "-i", stems["Guitar"],
        "-i", stems["Piano / Keys"],
        "-i", stems["Synths / Strings / Other"],
        "-filter_complex", "amix=inputs=5:duration=longest:normalize=0",
        instrumental_wav,
    ])
    copy_or_convert_audio(
        instrumental_wav,
        master / f"{output_index:02d}_{track}_instrumental_clean.{ext}",
        output_format,
    )
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
        f"{genre_reason}; engine: {ENGINE_NAME} ({ENGINE_VERSION})",
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
        f"{genre_reason}; engine: {ENGINE_NAME} ({ENGINE_VERSION})",
        included_readme,
        omitted_readme,
    )
    make_zip_archive(master, archive, master.name, progress)

    return {
        "track": track,
        "archive_path": str(archive),
        "output_format": output_format,
        "engine": ENGINE_NAME,
        "engine_version": ENGINE_VERSION,
        "detected_genre": detected_genre,
        "genre_reason": genre_reason,
        "included_stems": included_readme,
        "omitted_stems": [{"label": label, "reason": reason} for label, reason in omitted_readme],
        "stems": sorted(path.name for path in master.iterdir() if path.is_file()),
    }
