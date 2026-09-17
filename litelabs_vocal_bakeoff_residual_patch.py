from pathlib import Path

path = Path('/app/experimental_vocal_cascade_bakeoff.py')
text = path.read_text(encoding='utf-8')

text = text.replace('MODE = "experimental_vocal_cascade_bakeoff_v3"', 'MODE = "experimental_vocal_cascade_bakeoff_v4"', 1)
text = text.replace('disturbia_vocal_cascade_bakeoff_v3.json', 'disturbia_vocal_cascade_bakeoff_v4.json', 1)

# v3 established that Becruily's secondary/Instrumental output is the useful lead estimate
# on this track. Test Chas's mass-conserving construction exactly: lead = that strong child,
# backing = the fixed SW vocal parent minus lead. This guarantees lead + backing == parent.
needle = '''        if bec_parent_native is not None:\n            candidates.append(_pair_metrics(\n                name="becruily_parent_flipped_native_pair",\n                input_audio=sw_vocals,\n                lead=bec_parent_native,\n                backing=bec_parent_lead,\n                refs_audio=refs_audio,\n                sr=sr,\n                runtime_seconds=bec_parent_elapsed,\n                model=BECRUILY_KARAOKE,\n                route="BS-RoFormer-SW vocals -> Becruily MelBand Karaoke with Instrumental output treated as lead",\n            ))\n'''
replacement = needle + '''        if bec_parent_native is not None:\n            bec_strong_n = min(len(sw_vocals), len(bec_parent_native))\n            bec_strong_lead = bec_parent_native[:bec_strong_n]\n            bec_strong_residual_backing = sw_vocals[:bec_strong_n] - bec_strong_lead\n            candidates.append(_pair_metrics(\n                name="becruily_parent_flipped_residual_pair",\n                input_audio=sw_vocals,\n                lead=bec_strong_lead,\n                backing=bec_strong_residual_backing,\n                refs_audio=refs_audio,\n                sr=sr,\n                runtime_seconds=bec_parent_elapsed,\n                model=BECRUILY_KARAOKE,\n                route="BS-RoFormer-SW vocals -> Becruily secondary output as lead; backing = parent - lead",\n            ))\n'''
if needle not in text:
    raise RuntimeError('Could not locate Becruily flipped-native candidate for v4 residual test')
text = text.replace(needle, replacement, 1)

# Add a focused production-candidate comparison to the result. This avoids reading a broad
# ranking where label-trusting/debug candidates can obscure the actual apples-to-apples test.
needle2 = '''        result = {\n'''
comparison_setup = '''        by_name = {c["name"]: c for c in candidates}\n        current_prod = by_name.get("set_current_flipped_residual_pair")\n        becruily_prod = by_name.get("becruily_parent_flipped_residual_pair")\n        production_candidate_comparison = None\n        if current_prod and becruily_prod:\n            current_runtime = float(current_prod.get("runtime_seconds") or 0.0)\n            becruily_runtime = float(becruily_prod.get("runtime_seconds") or 0.0)\n            production_candidate_comparison = {\n                "current_set": {\n                    "name": current_prod["name"],\n                    "lead_quality": current_prod["lead"].get("quality_score"),\n                    "backing_inclusive_quality": current_prod.get("backing_inclusive", {}).get("quality_score"),\n                    "composite": current_prod.get("lead_backing_composite"),\n                    "reconstruction_cosine": current_prod.get("input_reconstruction_cosine"),\n                    "runtime_seconds": current_runtime,\n                },\n                "becruily_residual": {\n                    "name": becruily_prod["name"],\n                    "lead_quality": becruily_prod["lead"].get("quality_score"),\n                    "backing_inclusive_quality": becruily_prod.get("backing_inclusive", {}).get("quality_score"),\n                    "composite": becruily_prod.get("lead_backing_composite"),\n                    "reconstruction_cosine": becruily_prod.get("input_reconstruction_cosine"),\n                    "runtime_seconds": becruily_runtime,\n                },\n                "delta_becruily_minus_current": {\n                    "lead_quality": round(float(becruily_prod["lead"].get("quality_score") or 0.0) - float(current_prod["lead"].get("quality_score") or 0.0), 4),\n                    "backing_inclusive_quality": round(float(becruily_prod.get("backing_inclusive", {}).get("quality_score") or 0.0) - float(current_prod.get("backing_inclusive", {}).get("quality_score") or 0.0), 4),\n                    "composite": round(float(becruily_prod.get("lead_backing_composite") or 0.0) - float(current_prod.get("lead_backing_composite") or 0.0), 4),\n                    "runtime_seconds": round(becruily_runtime - current_runtime, 3),\n                    "runtime_percent": round(((becruily_runtime / current_runtime) - 1.0) * 100.0, 2) if current_runtime > 0 else None,\n                },\n            }\n\n        result = {\n'''
if needle2 not in text:
    raise RuntimeError('Could not locate result construction for v4 comparison')
text = text.replace(needle2, comparison_setup, 1)

needle3 = '''            "orientation_note": (\n'''
replacement3 = '''            "production_candidate_comparison": production_candidate_comparison,\n            "v4_note": (\n                "v4 tests the exact mass-conserving Chas-style candidate suggested by v3: use Becruily's "\n                "strong secondary output as lead and derive backing as SW vocal parent minus that lead."\n            ),\n            "orientation_note": (\n'''
if needle3 not in text:
    raise RuntimeError('Could not locate orientation note for v4 result metadata')
text = text.replace(needle3, replacement3, 1)

path.write_text(text, encoding='utf-8')
print('LiteLABS vocal bakeoff v4 mass-conserving Becruily residual test applied')
