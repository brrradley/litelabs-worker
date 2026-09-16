from pathlib import Path

path = Path('/app/multitrack_ground_truth_campaign.py')
text = path.read_text(encoding='utf-8')

helper_anchor = '''def _track_category(name: str) -> str:\n'''
helper = '''def _is_real_audio_file(path: Path, extensions: set[str]) -> bool:\n    # macOS ZIPs commonly contain AppleDouble resource-fork placeholders under\n    # __MACOSX and files prefixed with ._. They have audio-looking extensions\n    # but are not audio and ffmpeg quite correctly rejects them.\n    if path.suffix.lower() not in extensions:\n        return False\n    if path.name.startswith("._"):\n        return False\n    if any(part == "__MACOSX" for part in path.parts):\n        return False\n    return True\n\n\n'''
if '_is_real_audio_file' not in text:
    if helper_anchor not in text:
        raise RuntimeError('Could not locate track-category anchor')
    text = text.replace(helper_anchor, helper + helper_anchor, 1)

old_wavs = '    wavs = [p for p in extracted.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"]\n'
new_wavs = '    wavs = [p for p in extracted.rglob("*") if p.is_file() and _is_real_audio_file(p, {".wav"})]\n'
if old_wavs in text:
    text = text.replace(old_wavs, new_wavs, 1)
elif new_wavs not in text:
    raise RuntimeError('Could not locate multitrack WAV discovery')

old_masters = '        masters = [p for p in extracted.rglob("*") if p.is_file() and p.suffix.lower() in {".aif", ".aiff"}]\n'
new_masters = '        masters = [p for p in extracted.rglob("*") if p.is_file() and _is_real_audio_file(p, {".aif", ".aiff"})]\n'
if old_masters in text:
    text = text.replace(old_masters, new_masters, 1)
elif new_masters not in text:
    raise RuntimeError('Could not locate master AIF discovery')

path.write_text(text, encoding='utf-8')
print('LiteLABS multitrack macOS metadata filter applied')
