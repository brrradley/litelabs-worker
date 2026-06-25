from __future__ import annotations

import re
import subprocess
import zipfile
from pathlib import Path


def safe_track_name(filename: str) -> str:
    stem = Path(filename).stem or "track"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem or "track"


def run(cmd: list[str | Path]) -> None:
    print("\nRUN:", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def zip_folder(source_dir: Path, archive: Path, archive_root: str) -> None:
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zip_file:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                zip_file.write(path, arcname=str(Path(archive_root) / path.relative_to(source_dir)))


def build_vocal_residual_test(vocals_path: Path, lead_path: Path, output_root: Path, filename: str) -> dict:
    """Create lead/backing/check files from an existing full vocal stem and lead vocal stem.

    This does not run a vocal model. It is for testing Chas's residual method:

    backing vocals = full vocals - lead vocals
    check file = lead vocals + backing vocals

    If the method is behaving, the check file should sound very close to the original full vocals stem.
    """

    if not vocals_path.exists():
        raise FileNotFoundError(f"Missing full vocals input: {vocals_path}")
    if not lead_path.exists():
        raise FileNotFoundError(f"Missing lead vocals input: {lead_path}")

    track = safe_track_name(filename)
    pack_dir = output_root / f"{track}-vocal-residual-test"
    pack_dir.mkdir(parents=True, exist_ok=True)

    lead_out = pack_dir / f"01_{track}_lead_vocals.flac"
    backing_out = pack_dir / f"02_{track}_backing_vocals_residual.flac"
    check_out = pack_dir / f"03_{track}_lead_plus_backing_check.flac"
    full_copy = pack_dir / f"00_{track}_full_vocals_input.flac"

    print("Creating normalised full vocal reference", flush=True)
    run(["ffmpeg", "-y", "-i", vocals_path, "-c:a", "flac", full_copy])

    print("Creating lead vocal reference", flush=True)
    run(["ffmpeg", "-y", "-i", lead_path, "-c:a", "flac", lead_out])

    print("Creating backing vocal residual", flush=True)
    run([
        "ffmpeg", "-y",
        "-i", vocals_path,
        "-i", lead_path,
        "-filter_complex", "[1:a]volume=-1[invlead];[0:a][invlead]amix=inputs=2:duration=first:normalize=0[out]",
        "-map", "[out]",
        "-c:a", "flac",
        backing_out,
    ])

    print("Creating lead plus backing check file", flush=True)
    run([
        "ffmpeg", "-y",
        "-i", lead_out,
        "-i", backing_out,
        "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first:normalize=0[out]",
        "-map", "[out]",
        "-c:a", "flac",
        check_out,
    ])

    readme = pack_dir / "README.txt"
    write_text(
        readme,
        "LiteLABS research vocal residual test\n\n"
        f"Track: {track}\n\n"
        "Files included:\n"
        f"00_{track}_full_vocals_input.flac\n"
        f"01_{track}_lead_vocals.flac\n"
        f"02_{track}_backing_vocals_residual.flac\n"
        f"03_{track}_lead_plus_backing_check.flac\n\n"
        "Method:\n"
        "backing_vocals_residual = full_vocals_input - lead_vocals\n"
        "lead_plus_backing_check = lead_vocals + backing_vocals_residual\n\n"
        "The check file should sound very close to the original full vocal input if the lead/backing split is mathematically tidy.\n",
    )

    archive = output_root / f"{track}-vocal-residual-test.zip"
    zip_folder(pack_dir, archive, pack_dir.name)

    return {
        "track": track,
        "archive_path": str(archive),
        "files": sorted(p.name for p in pack_dir.iterdir() if p.is_file()),
    }
