from pathlib import Path


def patch_file(path: Path, replacements: list[tuple[str, str]]) -> None:
    text = path.read_text(encoding="utf-8")
    for old, new in replacements:
        if old not in text:
            raise RuntimeError(f"Could not locate QA metadata anchor in {path}: {old[:80]!r}")
        text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")


patch_file(
    Path('/app/preset_pack.py'),
    [
        (
            '            genre=genre,\n            preset=preset,\n',
            '            genre=genre,\n            genre_reason=genre_reason,\n            preset=preset,\n',
        ),
        (
            '        print(f"LiteLABS silent research QA complete for {len(qa_stems)} stems", flush=True)\n',
            '        print(f"LiteLABS silent research QA complete for {len(qa_stems)} stems", flush=True)\n'
            '        print("LiteLABS QA scores: " + " ".join(f"{name}={float(record.get(\'score\', 0)):.3f}" for name, record in research_qa.get("stems", {}).items()), flush=True)\n',
        ),
    ],
)

patch_file(
    Path('/app/experimental_children_v1.py'),
    [
        (
            '            genre=qa_genre,\n            preset="experimental",\n',
            '            genre=qa_genre,\n            genre_reason=qa_genre_reason,\n            preset="experimental",\n',
        ),
        (
            '        print(f"LiteLABS silent research QA complete for {len(qa_stems)} stems", flush=True)\n',
            '        print(f"LiteLABS silent research QA complete for {len(qa_stems)} stems", flush=True)\n'
            '        print("LiteLABS QA scores: " + " ".join(f"{name}={float(record.get(\'score\', 0)):.3f}" for name, record in research_qa.get("stems", {}).items()), flush=True)\n',
        ),
    ],
)

print('LiteLABS QA genre reason and score-summary metadata applied')
