from pathlib import Path

path = Path('/app/preset_pack.py')
text = path.read_text(encoding='utf-8')

old_sig = '''def _write_readme(
    path: Path,
    *,
    filename: str,
    preset: str,
    exported: list[str],
    genre: str,
    execution_seconds: float,
) -> None:
'''
new_sig = '''def _write_readme(
    path: Path,
    *,
    track: str,
    preset: str,
    exported: list[str],
    genre: str,
    genre_reason: str,
    execution_seconds: float,
    stem_pack_size_bytes: int = 0,
) -> None:
'''
if old_sig not in text:
    raise RuntimeError('Could not locate preset README signature')
text = text.replace(old_sig, new_sig, 1)

old_info = '''        f"Track: {filename}\\n"
        f"Pack: {PRESET_LABELS[preset]}\\n"
        "Output format: FLAC\\n"
        f"Detected genre: {genre}\\n"
        f"Execution time: {_format_execution_time(execution_seconds)}\\n\\n"
'''
new_info = '''        f"Track: {track}\\n"
        f"Pack: {PRESET_LABELS[preset]}\\n"
        "Output format: FLAC\\n"
        f"Stem pack size: {stem_pack_size_bytes / (1024 * 1024):.2f} MB\\n"
        f"Elapsed time: {max(0, int(round(execution_seconds)))} seconds\\n"
        f"Detected genre: {genre}\\n"
        f"Genre reason: {genre_reason}\\n\\n"
'''
if old_info not in text:
    raise RuntimeError('Could not locate preset README track information block')
text = text.replace(old_info, new_info, 1)

old_call = '''        _write_readme(
            final / "README.txt",
            filename=supplied_filename,
            preset=preset,
            exported=exported,
            genre=genre,
            execution_seconds=execution_seconds,
        )
'''
new_call = '''        _write_readme(
            final / "README.txt",
            track=track,
            preset=preset,
            exported=exported,
            genre=genre,
            genre_reason=genre_reason,
            execution_seconds=execution_seconds,
            stem_pack_size_bytes=0,
        )
'''
if old_call not in text:
    raise RuntimeError('Could not locate preset README call')
text = text.replace(old_call, new_call, 1)

old_pack = '''        emit("Packaging Stem Pack", 90)
        archive = root / f"{track}_{preset}_stem_pack.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
            for file in sorted(final.rglob("*")):
                if file.is_file():
                    bundle.write(file, arcname=str(file.relative_to(final)))

        uploaded = False
'''
new_pack = '''        emit("Packaging Stem Pack", 90)
        archive = root / f"{track}_{preset}_stem_pack.zip"

        def rebuild_archive() -> None:
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
                for file in sorted(final.rglob("*")):
                    if file.is_file():
                        bundle.write(file, arcname=str(file.relative_to(final)))

        # Build once to learn the real archive size, then rewrite the README and
        # rebuild until the displayed two-decimal MB value is stable.
        rebuild_archive()
        displayed_size = -1.0
        for _ in range(3):
            archive_size = archive.stat().st_size
            rounded_mb = round(archive_size / (1024 * 1024), 2)
            _write_readme(
                final / "README.txt",
                track=track,
                preset=preset,
                exported=exported,
                genre=genre,
                genre_reason=genre_reason,
                execution_seconds=execution_seconds,
                stem_pack_size_bytes=archive_size,
            )
            rebuild_archive()
            if rounded_mb == displayed_size:
                break
            displayed_size = rounded_mb

        uploaded = False
'''
if old_pack not in text:
    raise RuntimeError('Could not locate preset packaging block')
text = text.replace(old_pack, new_pack, 1)

path.write_text(text, encoding='utf-8')
print('LiteLABS preset README full metadata applied')
