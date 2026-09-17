from pathlib import Path
import re

path = Path('/app/experimental_children_v1.py')
text = path.read_text(encoding='utf-8')

if 'in_memory_hh_plus_cymbals_before_final_encode' in text:
    print('Drum hats merge already present; compatibility patch skipped')
else:
    pattern = re.compile(
        r'(?P<indent>[ \t]*)for name in DRUM5:\n'
        r'(?P=indent)[ \t]+dest = experimental / f"\{track\}_drums_5stem_\{name\}\.flac"\n'
        r'(?P=indent)[ \t]+_write_flac\(dest, (?P<audio>[^\n]+), drum_sr\)\n'
        r'(?P=indent)[ \t]+drum_report\["files"\]\.append\(dest\.name\)\n'
    )
    match = pattern.search(text)
    if not match:
        raise RuntimeError('Could not locate DrumSep final child export loop (compat)')

    indent = match.group('indent')
    audio_expr = match.group('audio').strip()

    if audio_expr == 'refined[name]':
        expr = lambda name: f'refined["{name}"]'
    elif audio_expr == 'loaded[name][:dn]':
        expr = lambda name: f'loaded["{name}"][:dn]'
    else:
        raise RuntimeError(f'Unsupported DrumSep child export expression: {audio_expr}')

    i1 = indent + '    '
    i2 = indent + '        '
    replacement = (
        f'{indent}# DrumSep still predicts hh and cymbals in one inference pass. Merge them\n'
        f'{indent}# in memory before final encoding so the public child is simply hats.\n'
        f'{indent}merge_started = time.monotonic()\n'
        f'{indent}final_children = {{\n'
        f'{i1}"kick": {expr("kick")},\n'
        f'{i1}"snare": {expr("snare")},\n'
        f'{i1}"toms": {expr("toms")},\n'
        f'{i1}"hats": {expr("hh")} + {expr("cymbals")},\n'
        f'{indent}}}\n'
        f'{indent}merged_sum = np.sum(np.stack(list(final_children.values()), axis=0), axis=0)\n'
        f'{indent}merged_residual = parent - merged_sum\n'
        f'{indent}merged_residual_rms = float(np.sqrt(np.mean(merged_residual * merged_residual) + 1e-12))\n'
        f'{indent}drum_report["hats_merge"] = {{\n'
        f'{i1}"applied": True,\n'
        f'{i1}"method": "in_memory_hh_plus_cymbals_before_final_encode",\n'
        f'{i1}"source_children": ["hh", "cymbals"],\n'
        f'{i1}"output_child": "hats",\n'
        f'{i1}"parent_vs_final_children_sum_cosine": round(float(_cos(parent, merged_sum)), 6),\n'
        f'{i1}"residual_relative_to_parent_db": _db(merged_residual_rms / max(parent_rms, 1e-12)),\n'
        f'{indent}}}\n'
        f'{indent}for name, audio in final_children.items():\n'
        f'{i1}dest = experimental / f"{{track}}_drums_5stem_{{name}}.flac"\n'
        f'{i1}_write_flac(dest, audio, drum_sr)\n'
        f'{i1}drum_report["files"].append(dest.name)\n'
        f'{indent}timings["drum_hats_merge_and_encode"] = round(time.monotonic() - merge_started, 3)\n'
    )
    text = text[:match.start()] + replacement + text[match.end():]
    path.write_text(text, encoding='utf-8')
    print(f'Drum hats merge compatibility patch applied to export expression: {audio_expr}')
