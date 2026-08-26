from pathlib import Path

path = Path('/app/experimental_children_v1.py')
text = path.read_text(encoding='utf-8')

# Keep the same ~18 seconds of Mega53 audio, but distribute it across more of
# the track so intermittent horns/winds are much less likely to be missed.
old = '''        segment_len = max(1, int(other_sr * 6.0))\n        total_len = len(other_audio)\n        centers = (0.18, 0.50, 0.82)\n'''
new = '''        segment_len = max(1, int(other_sr * 3.0))\n        total_len = len(other_audio)\n        centers = (0.08, 0.25, 0.42, 0.58, 0.75, 0.92)\n'''
if old not in text:
    raise RuntimeError('Could not locate fast Mega53 sampling profile')
text = text.replace(old, new, 1)

# Family routing should favour recall. Individual outputs still require a
# specialist separator, so a modestly lower inventory threshold is safer than
# silently omitting an obvious horn family.
old_score = '''        def inv_score(name: str) -> bool:\n            item = inventory.get(name) or {}\n            return float(item.get("rms_dbfs", -99.0)) >= -42.0 and abs(float(item.get("parent_cosine", 0.0))) >= 0.20\n'''
new_score = '''        def inv_score(name: str) -> bool:\n            item = inventory.get(name) or {}\n            return float(item.get("rms_dbfs", -99.0)) >= -45.0 and abs(float(item.get("parent_cosine", 0.0))) >= 0.12\n'''
if old_score not in text:
    raise RuntimeError('Could not locate Mega53 inventory gate')
text = text.replace(old_score, new_score, 1)

# Sax specialist is cheap. If the family router sees woodwind/mixed evidence,
# try it even when the specific Mega53 sax label falls just below threshold.
old_sax = '        sax_detected = inv_score("saxophone")\n        family_route = "mixed_wind_brass" if brass_detected and woodwind_detected else ("brass" if brass_detected else ("woodwind" if woodwind_detected else "none"))\n'
new_sax = '        sax_detected = inv_score("saxophone")\n        family_route = "mixed_wind_brass" if brass_detected and woodwind_detected else ("brass" if brass_detected else ("woodwind" if woodwind_detected else "none"))\n        run_sax_specialist = sax_detected or family_route in {"woodwind", "mixed_wind_brass"}\n'
if old_sax not in text:
    raise RuntimeError('Could not locate sax/family routing decision')
text = text.replace(old_sax, new_sax, 1)

text = text.replace('        if sax_detected:\n            emit("Running Saxophone Specialist Separation", 84)', '        if run_sax_specialist:\n            emit("Running Saxophone Specialist Separation", 84)', 1)
text = text.replace('                "ran": sax_detected,\n', '                "ran": run_sax_specialist,\n', 1)
text = text.replace('                "router_detected_sax": sax_detected,\n', '                "router_detected_sax": sax_detected,\n                "router_family_triggered_sax": bool(run_sax_specialist and not sax_detected),\n', 1)

path.write_text(text, encoding='utf-8')
print('LiteLABS fast inventory temporal recall and sax fallback patch applied')
