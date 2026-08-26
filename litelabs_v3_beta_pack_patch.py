from pathlib import Path

path = Path('/app/experimental_children_v1.py')
text = path.read_text(encoding='utf-8')

text = text.replace(
    'def _write_readme(final: Path, track: str, timings: dict[str, float], models: dict[str, str]) -> None:',
    'def _write_experimental_readme(experimental: Path, track: str, timings: dict[str, float], models: dict[str, str]) -> None:',
    1,
)
text = text.replace(
    '(final / "README.txt").write_text("\\n".join(lines), encoding="utf-8")',
    '(experimental / "README.txt").write_text("\\n".join(lines), encoding="utf-8")',
    1,
)

old = '''        _write_readme(final, track, timings, models)\n        (final / f"{track}_EXPERIMENTAL_REPORT.json").write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")\n\n        emit("Packaging Parent and Experimental Stems", 92)\n        archive = root / f"{track}_parent_plus_experimental.zip"\n        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:\n            for p in sorted(final.rglob("*")):\n                if p.is_file():\n                    bundle.write(p, arcname=str(p.relative_to(final)))\n'''

new = '''        # Keep the established LiteLABS README format at the pack root.\n        import master_pack\n        optional_decisions = [\n            master_pack.optional_stem_decision("Guitar", stems["guitar"]),\n            master_pack.optional_stem_decision("Piano / Keys", stems["piano"]),\n            master_pack.optional_stem_decision("Synths / Strings / Other", stems["other"]),\n        ]\n        core_stats = {\n            "Vocals": master_pack.analyse_audio(stems["vocals"]),\n            "Drums": master_pack.analyse_audio(stems["drums"]),\n            "Bass": master_pack.analyse_audio(stems["bass"]),\n        }\n        original_stats = master_pack.analyse_audio(source)\n        detected_genre, genre_reason = master_pack.detect_genre_from_audio(\n            optional_decisions, core_stats, original_stats\n        )\n        parent_readme_stems = [\n            "01 Vocals",\n            "02 Drums",\n            "03 Bass",\n            "04 Guitar",\n            "05 Piano / Keys",\n            "06 Synths / Strings / Other",\n            "07 Clean Instrumental",\n        ]\n        elapsed_label = master_pack.format_elapsed(time.monotonic() - started)\n        master_pack.write_litelabs_readme(\n            final / "README.txt",\n            track,\n            "flac",\n            "calculating",\n            elapsed_label,\n            detected_genre,\n            genre_reason,\n            parent_readme_stems,\n            [],\n        )\n\n        # Experimental metadata belongs with the experimental stems, not the parents.\n        _write_experimental_readme(experimental, track, timings, models)\n        (experimental / f"{track}_EXPERIMENTAL_REPORT.json").write_text(\n            json.dumps(_json_safe(report), indent=2), encoding="utf-8"\n        )\n\n        emit("Packaging Parent and Experimental Stems", 92)\n        archive = root / f"{track}_parent_plus_experimental.zip"\n        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:\n            for p in sorted(final.rglob("*")):\n                if p.is_file():\n                    bundle.write(p, arcname=str(p.relative_to(final)))\n\n        # Match the legacy pack behaviour: update the root README with the real\n        # archive size, then rebuild the archive once.\n        master_pack.write_litelabs_readme(\n            final / "README.txt",\n            track,\n            "flac",\n            master_pack.format_bytes(archive.stat().st_size),\n            elapsed_label,\n            detected_genre,\n            genre_reason,\n            parent_readme_stems,\n            [],\n        )\n        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:\n            for p in sorted(final.rglob("*")):\n                if p.is_file():\n                    bundle.write(p, arcname=str(p.relative_to(final)))\n'''

if old not in text:
    raise RuntimeError('Could not locate experimental README/report packaging block')
text = text.replace(old, new, 1)

path.write_text(text, encoding='utf-8')
print('LiteLABS v3 beta pack layout patch applied')
