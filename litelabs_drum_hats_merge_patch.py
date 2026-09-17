from pathlib import Path

path = Path('/app/experimental_children_v1.py')
text = path.read_text(encoding='utf-8')

old = '''                for name in DRUM5:\n                    dest = experimental / f"{track}_drums_5stem_{name}.flac"\n                    _write_flac(dest, refined[name], drum_sr)\n                    drum_report["files"].append(dest.name)\n'''
new = '''                # DrumSep still predicts hh and cymbals in one 5-stem inference pass, so there is\n                # no extra neural inference cost to combining them. Merge in memory before final\n                # FLAC encoding so the user-facing child set is kick/snare/toms/hats and we avoid\n                # writing two separate high-frequency files only to combine them later.\n                merge_started = time.monotonic()\n                final_children = {\n                    "kick": refined["kick"],\n                    "snare": refined["snare"],\n                    "toms": refined["toms"],\n                    "hats": refined["hh"] + refined["cymbals"],\n                }\n                merged_sum = np.sum(np.stack(list(final_children.values()), axis=0), axis=0)\n                merged_residual = parent - merged_sum\n                merged_residual_rms = float(np.sqrt(np.mean(merged_residual * merged_residual) + 1e-12))\n                drum_report["hats_merge"] = {\n                    "applied": True,\n                    "method": "in_memory_hh_plus_cymbals_before_final_encode",\n                    "source_children": ["hh", "cymbals"],\n                    "output_child": "hats",\n                    "parent_vs_final_children_sum_cosine": round(float(_cos(parent, merged_sum)), 6),\n                    "residual_relative_to_parent_db": _db(merged_residual_rms / max(parent_rms, 1e-12)),\n                }\n                for name, audio in final_children.items():\n                    dest = experimental / f"{track}_drums_5stem_{name}.flac"\n                    _write_flac(dest, audio, drum_sr)\n                    drum_report["files"].append(dest.name)\n                timings["drum_hats_merge_and_encode"] = round(time.monotonic() - merge_started, 3)\n'''

if old not in text:
    if 'in_memory_hh_plus_cymbals_before_final_encode' not in text:
        raise RuntimeError('Could not locate final DrumSep child encoding loop')
else:
    text = text.replace(old, new, 1)

text = text.replace(
    '"Children remain experimental; drums use the validated 5-stem path with kick-only mild Wiener refinement and mass-conserving compensation."',
    '"Children remain experimental; DrumSep predicts five internal children, kick gets the validated mild Wiener refinement, and hh+cymbals are merged in memory to one hats stem before final encoding."',
    1,
)

path.write_text(text, encoding='utf-8')
print('LiteLABS drum output now merges hh+cymbals in memory as hats before final encoding')
