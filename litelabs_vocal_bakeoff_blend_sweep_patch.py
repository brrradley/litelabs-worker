from pathlib import Path

path = Path('/app/experimental_vocal_cascade_bakeoff.py')
text = path.read_text(encoding='utf-8')

text = text.replace('MODE = "experimental_vocal_cascade_bakeoff_v5"', 'MODE = "experimental_vocal_cascade_bakeoff_v6"', 1)
text = text.replace('disturbia_vocal_cascade_bakeoff_v5.json', 'disturbia_vocal_cascade_bakeoff_v6.json', 1)

needle = '''        candidates.append(avg_candidate)\n\n        progress("Testing Chas Lead v1: Becruily directly on original mix", 78)\n'''
block = '''        candidates.append(avg_candidate)\n\n        # v5 showed the 50/50 averaged parent materially improves the final lead even though\n        # the raw averaged parent scores below SW alone. Probe either side of 50/50 without\n        # paying for another parent extraction: 75/25 and 25/75 reuse the same two parents.\n        blend_sweep_tails = {}\n        blend_sweep_elapsed = {}\n        for sw_weight, tag, label in (\n            (0.75, "sw75_mel25", "75/25 SW/MelBand"),\n            (0.25, "sw25_mel75", "25/75 SW/MelBand"),\n        ):\n            mel_weight = 1.0 - sw_weight\n            blend_parent = (\n                sw_weight * np.asarray(sw_vocals[:avg_n], dtype=np.float32)\n                + mel_weight * np.asarray(alt_parent_vocals[:avg_n], dtype=np.float32)\n            )\n            blend_path = root / f"vocal_parent_{tag}.wav"\n            mt._write(blend_path, blend_parent, sr)\n            parent_quality[f"blend_{tag}"] = _score(\n                blend_parent[: min(len(blend_parent), len(refs_audio["total_vocals"]))],\n                refs_audio["total_vocals"][: min(len(blend_parent), len(refs_audio["total_vocals"]))],\n                sr,\n            )\n\n            progress(f"Testing Chas Lead v2 blend {label}", 75)\n            _nominal, secondary, elapsed, tail = _run_separator(\n                BECRUILY_KARAOKE, blend_path, root / f"becruily_{tag}", model_dir, timeout\n            )\n            if secondary is None:\n                raise RuntimeError(f"Becruily produced no secondary output for {label} blend")\n            n_blend = min(len(blend_parent), len(secondary))\n            lead = secondary[:n_blend]\n            backing = blend_parent[:n_blend] - lead\n            candidate = _pair_metrics(\n                name=f"chas_lead_v2_blend_{tag}_residual_pair",\n                input_audio=blend_parent,\n                lead=lead,\n                backing=backing,\n                refs_audio=refs_audio,\n                sr=sr,\n                runtime_seconds=alt_parent_elapsed + elapsed,\n                model=f"{ALT_VOCAL_PARENT} + {BECRUILY_KARAOKE}",\n                route=f"{label} vocal parent -> Becruily secondary output as lead; backing = parent - lead",\n            )\n            candidate["blend_weights"] = {"bs_roformer_sw": sw_weight, "melband_becruily": mel_weight}\n            candidate["runtime_components_seconds"] = {\n                "alternate_vocal_parent": round(alt_parent_elapsed, 3),\n                "becruily_on_blended_parent": round(elapsed, 3),\n                "combined_incremental_beyond_existing_sw_parent": round(alt_parent_elapsed + elapsed, 3),\n            }\n            candidates.append(candidate)\n            blend_sweep_tails[tag] = tail\n            blend_sweep_elapsed[tag] = round(elapsed, 3)\n\n        progress("Testing Chas Lead v1: Becruily directly on original mix", 82)\n'''
if needle not in text:
    raise RuntimeError('Could not locate v5 averaged candidate insertion point')
text = text.replace(needle, block, 1)

needle2 = '''        result = {\n'''
setup = '''        by_name_v6 = {c["name"]: c for c in candidates}\n        blend_names = [\n            "chas_lead_v2_blend_sw75_mel25_residual_pair",\n            "chas_lead_v2_averaged_parent_residual_pair",\n            "chas_lead_v2_blend_sw25_mel75_residual_pair",\n        ]\n        blend_rows = []\n        for blend_name in blend_names:\n            c = by_name_v6.get(blend_name)\n            if not c:\n                continue\n            blend_rows.append({\n                "name": c["name"],\n                "lead_quality": c["lead"].get("quality_score"),\n                "backing_inclusive_quality": c.get("backing_inclusive", {}).get("quality_score"),\n                "composite": c.get("lead_backing_composite"),\n                "runtime_seconds": c.get("runtime_seconds"),\n                "reconstruction_cosine": c.get("input_reconstruction_cosine"),\n            })\n        blend_sweep_comparison = {\n            "by_composite": sorted(blend_rows, key=lambda x: float(x.get("composite") or -1.0), reverse=True),\n            "by_lead_quality": sorted(blend_rows, key=lambda x: float(x.get("lead_quality") or -1.0), reverse=True),\n            "note": "All blend candidates reuse the same SW and MelBand parent extractions; their production runtime includes the one alternate-parent extraction plus one Becruily karaoke pass.",\n        }\n\n        result = {\n'''
if needle2 not in text:
    raise RuntimeError('Could not locate v5 result construction')
text = text.replace(needle2, setup, 1)

needle3 = '''            "chas_lead_v2_comparison": chas_v2_comparison,\n'''
if needle3 not in text:
    raise RuntimeError('Could not locate Chas v2 comparison result field')
text = text.replace(needle3, needle3 + '            "blend_sweep_comparison": blend_sweep_comparison,\n', 1)

needle4 = '''                "becruily_averaged_parent": avg_bec_tail,\n'''
if needle4 not in text:
    raise RuntimeError('Could not locate averaged-parent runtime tail')
text = text.replace(needle4, needle4 + '                "blend_sweep": blend_sweep_tails,\n', 1)

needle5 = '''            "v5_note": (\n'''
if needle5 not in text:
    raise RuntimeError('Could not locate v5 note')
text = text.replace(
    needle5,
    '            "v6_note": (\n'
    '                "v6 probes the parent blend around Chas Lead v2 at 75/25, 50/50 and 25/75 using the same "\n'
    '                "two extracted parents, so we can see whether 50/50 is near a useful optimum before spending on "\n'
    '                "a second-song validation."\n'
    '            ),\n'
    + needle5,
    1,
)

text = text.replace(
    '"v5 now tests Chas Lead v2. If the averaged parent materially improves lead/backing quality enough to justify "\n                "its extra runtime, validate it on another suitable exact-ground-truth vocal multitrack; otherwise keep "\n                "the faster SW -> Becruily residual route as the leading production candidate."',
    '"v6 tunes the Chas Lead-v2 parent blend. Use the strongest stable blend as the ensemble candidate, then validate "\n                "both it and the faster SW -> Becruily residual route on another exact-ground-truth song before any production change."',
    1,
)

path.write_text(text, encoding='utf-8')
print('LiteLABS vocal bakeoff v6 blend sweep applied')
