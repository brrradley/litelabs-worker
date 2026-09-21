from pathlib import Path

path = Path('/app/experimental_children_v1.py')
text = path.read_text(encoding='utf-8')

# Inst-MTG is now the validated, fast pre-stem detector. Replace the old
# ~50-60 second Mega53 inventory pass with the already-computed instrument_report
# and use it as the routing source for downstream specialists.
anchor = '''        _write_flac(final / f"{track}_instrumental.flac", mixture[:n] - vocals[:n], mix_sr)\n\n'''
if 'validated_instrument_router_v2' not in text:
    if anchor not in text:
        raise RuntimeError('Could not locate instrumental write anchor for fast instrument router')
    block = '''        # validated_instrument_router_v2
        detected_instruments = []
        instrument_evidence = {}
        if instrument_report.get("ok"):
            for item in instrument_report.get("detected_instruments") or []:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("label") or "").strip().lower()
                if not name:
                    continue
                detected_instruments.append(name)
                instrument_evidence[name] = {
                    "confidence": str(item.get("confidence") or ""),
                    "mean": float(item.get("mean") or 0.0),
                    "p90": float(item.get("p90") or 0.0),
                    "max": float(item.get("max") or 0.0),
                }

        family_map = {
            "Percussion": {"drums", "percussion", "drummachine"},
            "Bass": {"bass", "doublebass", "acousticbassguitar"},
            "Strings / Guitars": {"strings", "violin", "viola", "cello", "doublebass", "guitar", "acousticguitar", "electricguitar", "classicalguitar", "harp"},
            "Keys": {"keyboard", "piano", "electricpiano", "organ", "pipeorgan"},
            "Wind": {"saxophone", "trumpet", "trombone", "frenchhorn", "tuba", "flute", "clarinet", "oboe", "bassoon", "harmonica"},
            "Electronic": {"synthesizer", "sampler", "drummachine"},
        }
        detected_by_family = {}
        for family, members in family_map.items():
            found = [name for name in detected_instruments if name in members]
            if found:
                detected_by_family[family] = found

        drum_triggers = {"drums", "percussion", "drummachine"}
        run_drumsep_from_inventory = bool(drum_triggers.intersection(detected_instruments))

        wind_names = {"trumpet", "trombone", "frenchhorn", "tuba", "flute", "clarinet", "oboe", "bassoon", "harmonica", "saxophone"}
        brass_names = {"trumpet", "trombone", "frenchhorn", "tuba"}
        woodwind_names = {"flute", "clarinet", "oboe", "bassoon", "saxophone"}
        brass_detected = bool(brass_names.intersection(detected_instruments))
        woodwind_detected = bool(woodwind_names.intersection(detected_instruments))
        sax_detected = "saxophone" in detected_instruments
        family_route = (
            "mixed_wind_brass" if brass_detected and woodwind_detected
            else "brass" if brass_detected
            else "woodwind" if woodwind_detected
            else "none"
        )
        # Sax specialist is expensive and should only run on explicit sax evidence,
        # not merely because some other woodwind was detected.
        run_sax_specialist = sax_detected

        global_inventory = instrument_evidence
        rc_global_inventory = 0
        global_inventory_elapsed = float(instrument_report.get("runtime_seconds") or 0.0)
        timings["instrument_inventory"] = 0.0
        timings["mega53"] = 0.0
        global_inventory_report = {
            "schema_version": 2,
            "model": "LiteLABS Inst-MTG",
            "role": "validated_pre_stem_detection_and_routing",
            "runtime_seconds": global_inventory_elapsed,
            "detected": sorted(set(detected_instruments)),
            "detected_by_family": detected_by_family,
            "evidence": instrument_evidence,
            "drumsep_route": {
                "run": run_drumsep_from_inventory,
                "policy": "run DrumSep only when drums/percussion evidence is present",
            },
            "wind_route": {
                "family_route": family_route,
                "run_sax_specialist": run_sax_specialist,
            },
        }

'''
    text = text.replace(anchor, anchor + block, 1)

# Replace the old Mega53 router with the variables already populated above.
router_start_marker = '        # Mega53 is the routing brain:'
router_end_marker = '        wind_files: list[str] = []\n'
if 'validated_instrument_router_reused_v2' not in text:
    start = text.find(router_start_marker)
    end = text.find(router_end_marker, start)
    if start < 0 or end < 0:
        raise RuntimeError('Could not locate legacy Mega53 routing block')
    replacement = '''        # validated_instrument_router_reused_v2
        other_parent = root / "other_parent.flac"
        _copy_as_flac(stems["other"], other_parent)
        inventory = global_inventory
        rc = 0
        elapsed = 0.0

        def inv_score(name: str) -> bool:
            return name in detected_instruments

'''
    text = text[:start] + replacement + text[end:]

# The old residual Mega53 inventory was another ~20s pass. It no longer adds
# routing information that Inst-MTG did not already provide before separation.
residual_start = '        emit("Analysing Residual Instruments", 83)\n'
residual_end = '        emit("Running LiteLABS-SX Saxophone Separation", 90)\n'
if 'residual_inventory_fast_skip_v2' not in text:
    start = text.find(residual_start)
    end = text.find(residual_end, start)
    if start >= 0 and end >= 0:
        replacement = '''        # residual_inventory_fast_skip_v2
        residual_inventory = {
            "ran": False,
            "detected": [],
            "evidence": {},
            "policy": "skipped: validated pre-stem Inst-MTG inventory is authoritative",
        }
        timings["residual_inventory"] = 0.0

'''
        text = text[:start] + replacement + text[end:]

# Keep the fast detector evidence in the technical report under the established
# field name used by QA/admin tooling.
if '"instrument_inventory": global_inventory_report' not in text:
    report_anchor = '        report = {\n'
    if report_anchor not in text:
        raise RuntimeError('Could not locate experimental report dict')
    text = text.replace(
        report_anchor,
        report_anchor + '            "instrument_inventory": global_inventory_report,\n',
        1,
    )

path.write_text(text, encoding='utf-8')

check = path.read_text(encoding='utf-8')
compile(check, str(path), 'exec')
assert 'validated_instrument_router_v2' in check
assert 'validated_instrument_router_reused_v2' in check
assert '"instrument_inventory": global_inventory_report' in check
assert 'MVSep Mega53' not in check
assert 'Analysing Instrument Inventory' not in check
print('LiteLABS validated Inst-MTG fast router applied')
