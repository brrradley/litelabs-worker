from pathlib import Path

path = Path('/app/experimental_vocal_cascade_bakeoff.py')
text = path.read_text(encoding='utf-8')

text = text.replace('MODE = "experimental_vocal_cascade_bakeoff_v2"', 'MODE = "experimental_vocal_cascade_bakeoff_v3"', 1)
text = text.replace('disturbia_vocal_cascade_bakeoff_v2.json', 'disturbia_vocal_cascade_bakeoff_v3.json', 1)

# Add an explicit current-SET flipped interpretation. The current MSS route can expose
# only one concrete karaoke child; its exact parent residual is therefore a valid second
# child. v2 showed that this residual aligns much more strongly with the true lead than
# the output labelled Vocals, so benchmark that orientation instead of trusting labels.
needle = '''        candidates.append(_pair_metrics(\n            name="set_current_residual_pair",\n            input_audio=sw_vocals,\n            lead=set_lead,\n            backing=set_residual,\n            refs_audio=refs_audio,\n            sr=sr,\n            runtime_seconds=set_elapsed,\n            model=CURRENT_KARAOKE,\n            route="BS-RoFormer-SW vocals -> current BS-RoFormer Karaoke lead; backing = parent - lead",\n        ))\n'''
replacement = needle + '''        candidates.append(_pair_metrics(\n            name="set_current_flipped_residual_pair",\n            input_audio=sw_vocals,\n            lead=set_residual,\n            backing=set_lead,\n            refs_audio=refs_audio,\n            sr=sr,\n            runtime_seconds=set_elapsed,\n            model=CURRENT_KARAOKE,\n            route="BS-RoFormer-SW vocals -> current karaoke output treated as backing; lead = parent - output",\n        ))\n'''
if needle not in text:
    raise RuntimeError('Could not locate current SET residual candidate')
text = text.replace(needle, replacement, 1)

# For Becruily, v2's assignment matrix showed the output named Instrumental aligns far
# better with the true lead than the output named Vocals. Add that native orientation
# explicitly while keeping the label-trusting candidates for auditability.
needle2 = '''        candidates.append(_pair_metrics(\n            name="becruily_parent_native_pair",\n            input_audio=sw_vocals,\n            lead=bec_parent_lead,\n            backing=bec_parent_native,\n            refs_audio=refs_audio,\n            sr=sr,\n            runtime_seconds=bec_parent_elapsed,\n            model=BECRUILY_KARAOKE,\n            route="BS-RoFormer-SW vocals -> Becruily MelBand Karaoke native outputs",\n        ))\n'''
replacement2 = needle2 + '''        if bec_parent_native is not None:\n            candidates.append(_pair_metrics(\n                name="becruily_parent_flipped_native_pair",\n                input_audio=sw_vocals,\n                lead=bec_parent_native,\n                backing=bec_parent_lead,\n                refs_audio=refs_audio,\n                sr=sr,\n                runtime_seconds=bec_parent_elapsed,\n                model=BECRUILY_KARAOKE,\n                route="BS-RoFormer-SW vocals -> Becruily MelBand Karaoke with Instrumental output treated as lead",\n            ))\n'''
if needle2 not in text:
    raise RuntimeError('Could not locate Becruily native candidate')
text = text.replace(needle2, replacement2, 1)

# Chas's Lead v1 must not be identified by a stem filename alone. Score BOTH Becruily
# outputs from the original mix so the benchmark tells us which output actually behaves
# like lead vocals on exact ground truth.
needle3 = '''        candidates.append(_pair_metrics(\n            name="becruily_direct_lead_v1",\n            input_audio=source_audio,\n            lead=bec_direct_lead,\n            backing=None,\n            refs_audio=refs_audio,\n            sr=sr,\n            runtime_seconds=bec_direct_elapsed,\n            model=BECRUILY_KARAOKE,\n            route="Original mix -> Becruily MelBand Karaoke (Lead v1)",\n        ))\n'''
replacement3 = needle3 + '''        if bec_direct_native is not None:\n            candidates.append(_pair_metrics(\n                name="becruily_direct_secondary_as_lead_v1",\n                input_audio=source_audio,\n                lead=bec_direct_native,\n                backing=None,\n                refs_audio=refs_audio,\n                sr=sr,\n                runtime_seconds=bec_direct_elapsed,\n                model=BECRUILY_KARAOKE,\n                route="Original mix -> Becruily MelBand Karaoke secondary/Instrumental output scored as Lead v1",\n            ))\n'''
if needle3 not in text:
    raise RuntimeError('Could not locate direct Becruily candidate')
text = text.replace(needle3, replacement3, 1)

# Make the purpose of this run explicit in the result.
needle4 = '''            "next_phase_if_promising": (\n'''
replacement4 = '''            "orientation_note": (\n                "v3 does not trust karaoke stem labels. It scores label-trusting and flipped interpretations, "\n                "including both direct-mix outputs, because v2 showed the nominal Instrumental/residual child "\n                "aligned much more strongly with true lead vocals."\n            ),\n            "next_phase_if_promising": (\n'''
if needle4 not in text:
    raise RuntimeError('Could not locate result next-phase field')
text = text.replace(needle4, replacement4, 1)

path.write_text(text, encoding='utf-8')
print('LiteLABS vocal bakeoff v3 orientation-aware scoring applied')
