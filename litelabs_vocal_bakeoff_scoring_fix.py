from pathlib import Path

path = Path('/app/experimental_vocal_cascade_bakeoff.py')
text = path.read_text(encoding='utf-8')

# Bump the benchmark/result so corrected scores cannot be confused with v1.
text = text.replace('MODE = "experimental_vocal_cascade_bakeoff_v1"', 'MODE = "experimental_vocal_cascade_bakeoff_v2"', 1)
text = text.replace('disturbia_vocal_cascade_bakeoff_v1.json', 'disturbia_vocal_cascade_bakeoff_v2.json', 1)

# We need regex support to identify the final stem marker in audio-separator filenames.
if '\nimport re\n' not in text:
    text = text.replace('import os\n', 'import os\nimport re\n', 1)

# Disturbia contains vocal FX which the six-parent mapper intentionally classifies as "other".
# Include those in the inclusive non-lead vocal reference used by this child-vocal benchmark.
anchor = '\ndef _vocal_refs(tracks: dict[str, np.ndarray], work: Path, sr: int) -> tuple[dict[str, Path], dict]:\n'
helper = '''\ndef _is_extended_vocal_reference(name: str) -> bool:\n    n = _safe_name(name)\n    if sgt.track_category(name) == "vocals":\n        return True\n    return any(token in n for token in (\n        "pitched voc", "robot", "scream", "breath"\n    ))\n\n'''
if '_is_extended_vocal_reference' not in text:
    if anchor not in text:
        raise RuntimeError('Could not locate _vocal_refs anchor')
    text = text.replace(anchor, helper + anchor, 1)

old_loop = '''    vocal_members: list[tuple[str, np.ndarray]] = []\n    for name, audio in tracks.items():\n        category = sgt.track_category(name)\n        if category == "vocals":\n            vocal_members.append((name, audio))\n'''
new_loop = '''    vocal_members: list[tuple[str, np.ndarray]] = []\n    for name, audio in tracks.items():\n        if _is_extended_vocal_reference(name):\n            vocal_members.append((name, audio))\n'''
if old_loop not in text:
    raise RuntimeError('Could not locate patched vocal-members loop')
text = text.replace(old_loop, new_loop, 1)

# The old substring matcher can select the Instrumental output as "vocals" when the INPUT
# filename itself contains "(vocals)". Prefer an exact stem name, then the FINAL parenthetical
# stem marker written by audio-separator, then a stem-named directory, and only then substring.
start = text.index('def _find_output(directory: Path, token: str) -> Path | None:\n')
end = text.index('\n\ndef _run_separator(', start)
new_finder = '''def _find_output(directory: Path, token: str) -> Path | None:\n    token = token.lower()\n    files = [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in {".wav", ".flac", ".mp3"}]\n\n    exact = [p for p in files if p.stem.lower() == token]\n    if exact:\n        return max(exact, key=lambda p: p.stat().st_size)\n\n    marked = []\n    for p in files:\n        groups = re.findall(r"\\(([^()]*)\\)", p.stem)\n        if groups and groups[-1].strip().lower() == token:\n            marked.append(p)\n    if marked:\n        return max(marked, key=lambda p: p.stat().st_size)\n\n    parent_named = [p for p in files if p.parent.name.lower() == token]\n    if parent_named:\n        return max(parent_named, key=lambda p: p.stat().st_size)\n\n    matches = [p for p in files if token in p.stem.lower()]\n    if matches:\n        return max(matches, key=lambda p: p.stat().st_size)\n    return None\n'''
text = text[:start] + new_finder + text[end:]

# Add a native-output assignment matrix whenever both model outputs exist. This catches
# reversed/ambiguous karaoke labels rather than silently trusting Vocals/Instrumental names.
needle = '''    result.update({\n        "backing_available": True,\n'''
replacement = '''    result["native_assignment_matrix"] = {\n        "lead_output_vs_true_lead": _score(lead, refs_audio["lead"][: min(n, len(refs_audio["lead"]))], sr),\n        "lead_output_vs_true_backing": _score(lead, refs_audio["backing_inclusive"][: min(n, len(refs_audio["backing_inclusive"]))], sr),\n        "backing_output_vs_true_lead": _score(backing, refs_audio["lead"][: min(n, len(refs_audio["lead"]))], sr),\n        "backing_output_vs_true_backing": _score(backing, refs_audio["backing_inclusive"][: min(n, len(refs_audio["backing_inclusive"]))], sr),\n    }\n    result.update({\n        "backing_available": True,\n'''
if needle not in text:
    raise RuntimeError('Could not locate pair metrics result.update')
text = text.replace(needle, replacement, 1)

# Record which concrete files were selected for each separator run, making future filename
# mistakes visible in the JSON itself.
needle2 = '''    lead, _sr = mt._load(lead_path)\n    native_backing = mt._load(native_backing_path)[0] if native_backing_path else None\n    return lead, native_backing, elapsed, tail\n'''
replacement2 = '''    lead, _sr = mt._load(lead_path)\n    native_backing = mt._load(native_backing_path)[0] if native_backing_path else None\n    tail += f"\\nSELECTED_LEAD_FILE={lead_path.name}"\n    tail += f"\\nSELECTED_BACKING_FILE={native_backing_path.name if native_backing_path else 'NONE'}"\n    return lead, native_backing, elapsed, tail\n'''
if needle2 not in text:
    raise RuntimeError('Could not locate separator return block')
text = text.replace(needle2, replacement2, 1)

path.write_text(text, encoding='utf-8')
print('LiteLABS vocal bakeoff v2 scoring/reference fix applied')
