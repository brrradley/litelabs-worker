from pathlib import Path

path = Path('/app/experimental_vocal_cascade_bakeoff.py')
text = path.read_text(encoding='utf-8')

text = text.replace('MODE = "experimental_vocal_cascade_bakeoff_v4"', 'MODE = "experimental_vocal_cascade_bakeoff_v5"', 1)
text = text.replace('disturbia_vocal_cascade_bakeoff_v4.json', 'disturbia_vocal_cascade_bakeoff_v5.json', 1)

# Chas described Lead v2 as Becruily karaoke applied to an averaged vocals track.
# Use a genuinely different vocal extractor for the second parent so this tests an ensemble,
# rather than averaging two copies of the same family.
needle_const = 'BECRUILY_KARAOKE = "mel_band_roformer_karaoke_becruily.ckpt"\n'
if needle_const not in text:
    raise RuntimeError('Could not locate Becruily karaoke constant')
text = text.replace(
    needle_const,
    needle_const + 'ALT_VOCAL_PARENT = "mel_band_roformer_vocals_becruily.ckpt"\n',
    1,
)

# Build the second vocal parent and an exact 50/50 averaged parent before child-vocal tests.
needle_parent = '''        sw_vocals, _ = mt._load(sw_vocals_path)\n\n        candidates = []\n'''
replacement_parent = '''        sw_vocals, _ = mt._load(sw_vocals_path)\n\n        progress("Extracting alternate MelBand vocal parent for Chas Lead v2", 22)\n        alt_parent_vocals, _alt_parent_other, alt_parent_elapsed, alt_parent_tail = _run_separator(\n            ALT_VOCAL_PARENT, source_path, root / "alt_vocal_parent", model_dir, timeout\n        )\n        avg_n = min(len(sw_vocals), len(alt_parent_vocals))\n        averaged_vocal_parent = 0.5 * (\n            np.asarray(sw_vocals[:avg_n], dtype=np.float32)\n            + np.asarray(alt_parent_vocals[:avg_n], dtype=np.float32)\n        )\n        averaged_vocal_parent_path = root / "averaged_vocal_parent.wav"\n        mt._write(averaged_vocal_parent_path, averaged_vocal_parent, sr)\n\n        parent_quality = {\n            "bs_roformer_sw": _score(\n                sw_vocals[: min(len(sw_vocals), len(refs_audio["total_vocals"]))],\n                refs_audio["total_vocals"][: min(len(sw_vocals), len(refs_audio["total_vocals"]))],\n                sr,\n            ),\n            "melband_becruily": _score(\n                alt_parent_vocals[: min(len(alt_parent_vocals), len(refs_audio["total_vocals"]))],\n                refs_audio["total_vocals"][: min(len(alt_parent_vocals), len(refs_audio["total_vocals"]))],\n                sr,\n            ),\n            "average_50_50": _score(\n                averaged_vocal_parent[: min(len(averaged_vocal_parent), len(refs_audio["total_vocals"]))],\n                refs_audio["total_vocals"][: min(len(averaged_vocal_parent), len(refs_audio["total_vocals"]))],\n                sr,\n            ),\n        }\n\n        candidates = []\n'''
if needle_parent not in text:
    raise RuntimeError('Could not locate SW parent/candidate anchor')
text = text.replace(needle_parent, replacement_parent, 1)

# Run the exact Chas Lead-v2 concept: averaged vocal parent -> Becruily; use the output
# orientation established by v3/v4, then derive backing as the exact parent-minus-lead residual.
needle_direct = '''        progress("Testing Chas Lead v1: Becruily directly on original mix", 78)\n'''
ensemble_block = '''        progress("Testing Chas Lead v2: Becruily on averaged vocal parent", 72)\n        avg_bec_nominal, avg_bec_secondary, avg_bec_elapsed, avg_bec_tail = _run_separator(\n            BECRUILY_KARAOKE, averaged_vocal_parent_path, root / "becruily_averaged_parent", model_dir, timeout\n        )\n        if avg_bec_secondary is None:\n            raise RuntimeError("Becruily produced no secondary output for averaged-parent Lead v2 test")\n        avg_bec_n = min(len(averaged_vocal_parent), len(avg_bec_secondary))\n        avg_bec_lead = avg_bec_secondary[:avg_bec_n]\n        avg_bec_backing = averaged_vocal_parent[:avg_bec_n] - avg_bec_lead\n        avg_candidate = _pair_metrics(\n            name="chas_lead_v2_averaged_parent_residual_pair",\n            input_audio=averaged_vocal_parent,\n            lead=avg_bec_lead,\n            backing=avg_bec_backing,\n            refs_audio=refs_audio,\n            sr=sr,\n            runtime_seconds=alt_parent_elapsed + avg_bec_elapsed,\n            model=f"{ALT_VOCAL_PARENT} + {BECRUILY_KARAOKE}",\n            route=(\n                "50/50 BS-RoFormer-SW + MelBand Becruily vocal parents -> Becruily secondary output as lead; "\n                "backing = averaged parent - lead"\n            ),\n        )\n        avg_candidate["runtime_components_seconds"] = {\n            "alternate_vocal_parent": round(alt_parent_elapsed, 3),\n            "becruily_on_averaged_parent": round(avg_bec_elapsed, 3),\n            "combined_incremental_beyond_existing_sw_parent": round(alt_parent_elapsed + avg_bec_elapsed, 3),\n        }\n        candidates.append(avg_candidate)\n\n''' + needle_direct
if needle_direct not in text:
    raise RuntimeError('Could not locate direct Lead-v1 anchor')
text = text.replace(needle_direct, ensemble_block, 1)

# Add a focused three-way production/research comparison after v4's comparison setup.
needle_compare = '''        result = {\n'''
compare_setup = '''        by_name_v5 = {c["name"]: c for c in candidates}\n        current_v5 = by_name_v5.get("set_current_flipped_residual_pair")\n        fast_v5 = by_name_v5.get("becruily_parent_flipped_residual_pair")\n        ensemble_v5 = by_name_v5.get("chas_lead_v2_averaged_parent_residual_pair")\n        chas_v2_comparison = None\n        if current_v5 and fast_v5 and ensemble_v5:\n            def _brief(c):\n                return {\n                    "name": c["name"],\n                    "lead_quality": c["lead"].get("quality_score"),\n                    "backing_inclusive_quality": c.get("backing_inclusive", {}).get("quality_score"),\n                    "composite": c.get("lead_backing_composite"),\n                    "reconstruction_cosine": c.get("input_reconstruction_cosine"),\n                    "runtime_seconds": c.get("runtime_seconds"),\n                }\n            chas_v2_comparison = {\n                "current_set": _brief(current_v5),\n                "fast_sw_to_becruily": _brief(fast_v5),\n                "chas_lead_v2_averaged_parent": _brief(ensemble_v5),\n                "ensemble_minus_fast": {\n                    "lead_quality": round(float(ensemble_v5["lead"].get("quality_score") or 0.0) - float(fast_v5["lead"].get("quality_score") or 0.0), 4),\n                    "backing_inclusive_quality": round(float(ensemble_v5.get("backing_inclusive", {}).get("quality_score") or 0.0) - float(fast_v5.get("backing_inclusive", {}).get("quality_score") or 0.0), 4),\n                    "composite": round(float(ensemble_v5.get("lead_backing_composite") or 0.0) - float(fast_v5.get("lead_backing_composite") or 0.0), 4),\n                    "runtime_seconds": round(float(ensemble_v5.get("runtime_seconds") or 0.0) - float(fast_v5.get("runtime_seconds") or 0.0), 3),\n                },\n            }\n\n        result = {\n'''
if needle_compare not in text:
    raise RuntimeError('Could not locate result construction for v5 comparison')
text = text.replace(needle_compare, compare_setup, 1)

# Extend result metadata with parent-quality evidence, new timings and the Lead-v2 comparison.
needle_models = '''                "chas_becruily_karaoke": BECRUILY_KARAOKE,\n'''
if needle_models not in text:
    raise RuntimeError('Could not locate models result block')
text = text.replace(
    needle_models,
    needle_models + '                "alternate_vocal_parent": ALT_VOCAL_PARENT,\n',
    1,
)

needle_timing = '''                "becruily_direct_mix": round(bec_direct_elapsed, 3),\n'''
if needle_timing not in text:
    raise RuntimeError('Could not locate timing result block')
text = text.replace(
    needle_timing,
    needle_timing
    + '                "alternate_melband_vocal_parent": round(alt_parent_elapsed, 3),\n'
    + '                "becruily_on_averaged_parent": round(avg_bec_elapsed, 3),\n'
    + '                "chas_lead_v2_incremental_total": round(alt_parent_elapsed + avg_bec_elapsed, 3),\n',
    1,
)

needle_candidates = '''            "candidates": candidates,\n'''
if needle_candidates not in text:
    raise RuntimeError('Could not locate candidate result field')
text = text.replace(
    needle_candidates,
    '            "vocal_parent_quality": parent_quality,\n'
    + needle_candidates
    + '            "chas_lead_v2_comparison": chas_v2_comparison,\n',
    1,
)

needle_tails = '''                "becruily_direct": bec_direct_tail,\n'''
if needle_tails not in text:
    raise RuntimeError('Could not locate runtime tails')
text = text.replace(
    needle_tails,
    needle_tails
    + '                "alternate_vocal_parent": alt_parent_tail,\n'
    + '                "becruily_averaged_parent": avg_bec_tail,\n',
    1,
)

needle_note = '''            "v4_note": (\n'''
if needle_note not in text:
    raise RuntimeError('Could not locate v4 note')
text = text.replace(
    needle_note,
    '            "v5_note": (\n'
    '                "v5 tests Chas Lead v2 literally: average two independently extracted vocal parents 50/50, "\n'
    '                "run Becruily karaoke on that averaged parent, take the empirically strong secondary output "\n'
    '                "as lead, and derive backing as the exact averaged-parent residual."\n'
    '            ),\n'
    + needle_note,
    1,
)

text = text.replace(
    '"If Becruily wins or is close, test Chas Lead v2 exactly: build an averaged/ensemble vocal parent, "\n                "apply Becruily to that parent, and define backing as the exact parent-minus-lead residual."',
    '"v5 now tests Chas Lead v2. If the averaged parent materially improves lead/backing quality enough to justify "\n                "its extra runtime, validate it on another suitable exact-ground-truth vocal multitrack; otherwise keep "\n                "the faster SW -> Becruily residual route as the leading production candidate."',
    1,
)

path.write_text(text, encoding='utf-8')
print('LiteLABS vocal bakeoff v5 averaged-parent Lead v2 test applied')
