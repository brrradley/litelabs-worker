from pathlib import Path

path = Path('/app/experimental_children_v1.py')
text = path.read_text(encoding='utf-8')

# Research Inventory v1:
# - run Mega53 once, early, against representative Instrumental excerpts
# - use it as a detector/router rather than as a customer-facing separator
# - skip DrumSep when no kit components are detected
# - reuse the same inventory for wind/sax routing
# - add human-readable detected instruments to the README/report

if 'global_instrument_inventory_v1' not in text:
    anchor = '''        _write_flac(final / f"{track}_instrumental.flac", mixture[:n] - vocals[:n], mix_sr)\n\n'''
    block = '''        # global_instrument_inventory_v1
        # Detect instrumentation before expensive specialists. Mega53 is used
        # as a routing/inventory model only; its raw 53 outputs are not exported.
        inventory_parent = root / "inventory_parent.flac"
        _write_flac(inventory_parent, mixture[:n] - vocals[:n], mix_sr)
        global_inventory_in = root / "global_inventory_in"
        global_inventory_out = root / "global_inventory_out"
        global_inventory_in.mkdir(parents=True, exist_ok=True)
        global_inventory_out.mkdir(parents=True, exist_ok=True)

        inventory_audio, inventory_sr = _read(inventory_parent)
        inv_segment_len = max(1, int(inventory_sr * 3.0))
        inv_total = len(inventory_audio)
        inv_centers = (0.06, 0.18, 0.31, 0.44, 0.57, 0.70, 0.83, 0.95)
        inv_samples = []
        for frac in inv_centers:
            centre = int(inv_total * frac)
            start = max(0, min(max(inv_total - inv_segment_len, 0), centre - inv_segment_len // 2))
            piece = inventory_audio[start:start + inv_segment_len]
            if len(piece):
                inv_samples.append(piece)
        inventory_sample = np.concatenate(inv_samples, axis=0) if inv_samples else inventory_audio[:inv_segment_len]
        _write_flac(global_inventory_in / "instrumental.flac", inventory_sample, inventory_sr)

        emit("Analysing Instrument Inventory", 29)
        global_inventory_started = time.monotonic()
        rc_global_inventory, global_inventory_elapsed = _run_polled(
            ["python", str(repo_dir / "inference.py"), "--model_type", "bs_roformer",
             "--config_path", str(MEGA53_CONFIG), "--start_check_point", str(MEGA53_CHECKPOINT),
             "--input_folder", str(global_inventory_in), "--store_dir", str(global_inventory_out),
             "--device_ids", "0", "--disable_detailed_pbar", "--filename_template", "{file_name}/{instr}"],
            cwd=repo_dir, timeout=timeout, log_path=logs / "global_inventory.log", progress=progress,
            stage_name="Instrument Inventory", start_percent=29, end_percent=39, heartbeat_seconds=heartbeat,
        )
        timings["instrument_inventory"] = round(global_inventory_elapsed, 3)

        INVENTORY_NAMES = (
            "accordion", "acoustic-guitar", "back-vocal", "banjo", "bass", "bassoon",
            "bells", "bowed-strings", "brass", "cello", "clarinet", "congas",
            "digital-piano", "dobro", "double-bass", "drums", "electric-guitar",
            "flute", "french-horn", "glockenspiel", "guitar", "harmonica", "harp",
            "harpsichord", "hh", "keys", "kick", "lead-vocal", "mandolin", "marimba",
            "oboe", "organ", "percussion", "piano", "saxophone", "sitar", "snare",
            "strings", "synth", "tambourine", "timpani", "toms", "triangle",
            "trombone", "trumpet", "tuba", "ukulele", "viola", "violin", "vocals",
            "wind", "wind-chimes", "woodwind",
        )
        global_inventory = {}
        detected_instruments = []
        if rc_global_inventory == 0:
            inv_files = [
                p for p in global_inventory_out.rglob("*")
                if p.is_file() and p.suffix.lower() in {".wav", ".flac"}
            ]
            inv_by_name = {}
            for p in inv_files:
                key = p.stem.lower().replace("_", "-")
                inv_by_name[key] = p

            inventory_parent_sample, _ = _read(global_inventory_in / "instrumental.flac")
            for name in INVENTORY_NAMES:
                p = inv_by_name.get(name)
                if not p:
                    continue
                audio, _ = _read(p)
                nn = min(len(audio), len(inventory_parent_sample))
                if nn <= 0:
                    continue
                rms = float(np.sqrt(np.mean(audio[:nn] * audio[:nn]) + 1e-12))
                cosine = abs(float(_cos(audio[:nn], inventory_parent_sample[:nn])))
                rms_dbfs = _db(rms)
                if rms_dbfs >= -34.0 and cosine >= 0.18:
                    confidence = "strong"
                elif rms_dbfs >= -45.0 and cosine >= 0.10:
                    confidence = "likely"
                else:
                    confidence = "weak"
                global_inventory[name] = {
                    "rms_dbfs": round(rms_dbfs, 3),
                    "parent_cosine": round(cosine, 6),
                    "confidence": confidence,
                }
                if confidence in {"strong", "likely"}:
                    detected_instruments.append(name)

        family_map = {
            "Vocals": {"lead-vocal", "back-vocal", "vocals"},
            "Percussion": {"drums", "kick", "snare", "toms", "hh", "percussion", "congas", "tambourine", "timpani", "triangle", "bells", "glockenspiel", "marimba", "wind-chimes"},
            "Bass": {"bass", "double-bass"},
            "Strings / Guitars": {"strings", "bowed-strings", "violin", "viola", "cello", "double-bass", "guitar", "acoustic-guitar", "electric-guitar", "classical-guitar", "dobro", "banjo", "mandolin", "ukulele", "harp", "sitar"},
            "Keys": {"keys", "piano", "digital-piano", "organ", "harpsichord"},
            "Wind": {"wind", "woodwind", "brass", "saxophone", "trumpet", "trombone", "french-horn", "tuba", "flute", "clarinet", "oboe", "bassoon", "harmonica"},
            "Electronic": {"synth"},
        }
        detected_by_family = {}
        for family, members in family_map.items():
            found = [name for name in detected_instruments if name in members]
            if found:
                detected_by_family[family] = found

        kit_parts = {"kick", "snare", "toms", "hh"}
        detected_kit_parts = sorted(kit_parts.intersection(detected_instruments))
        run_drumsep_from_inventory = bool(detected_kit_parts)
        global_inventory_report = {
            "schema_version": 1,
            "model": "MVSep Mega53",
            "role": "detection_and_routing_only",
            "sampled_seconds": round(len(inventory_sample) / max(inventory_sr, 1), 3),
            "detected": sorted(detected_instruments),
            "detected_by_family": detected_by_family,
            "evidence": global_inventory,
            "drumsep_route": {
                "run": run_drumsep_from_inventory,
                "detected_kit_parts": detected_kit_parts,
                "policy": "run DrumSep only when specific kit children are detected",
            },
        }

'''
    if anchor not in text:
        raise RuntimeError('Could not locate instrumental write anchor for global inventory')
    text = text.replace(anchor, anchor + block, 1)

# Keep DrumSep observational in this first inventory benchmark. We record whether
# the inventory would have skipped it, but do not suppress the pass until the
# detector has been validated against several known tracks. This avoids turning
# a new detector threshold into a destructive routing decision.
# The report's drumsep_route.run is the candidate policy for the next step.

# Reuse the global inventory for the existing wind/sax family router instead of
# paying for a second Mega53 pass over the Other parent.
router_start_marker = '        # Mega53 is the routing brain:'
router_end_marker = '        wind_files: list[str] = []\n'
if 'global_inventory_reused_for_family_router' not in text:
    start = text.find(router_start_marker)
    end = text.find(router_end_marker, start)
    if start >= 0 and end >= 0:
        replacement = '''        # global_inventory_reused_for_family_router
        # The early whole-instrumental inventory is broader than the old
        # Other-only inventory and already paid the Mega53 cost.
        other_parent = root / "other_parent.flac"
        _copy_as_flac(stems["other"], other_parent)
        inventory = global_inventory
        rc = rc_global_inventory
        elapsed = global_inventory_elapsed
        timings["mega53"] = round(global_inventory_elapsed, 3)

        def inv_score(name: str) -> bool:
            item = inventory.get(name) or {}
            return str(item.get("confidence") or "") in {"strong", "likely"}

        brass_detected = any(inv_score(x) for x in ("trumpet", "brass", "trombone", "french-horn", "tuba"))
        woodwind_detected = any(inv_score(x) for x in ("saxophone", "wind", "woodwind", "clarinet", "flute", "oboe", "bassoon"))
        sax_detected = inv_score("saxophone")
        family_route = "mixed_wind_brass" if brass_detected and woodwind_detected else ("brass" if brass_detected else ("woodwind" if woodwind_detected else "none"))
        run_sax_specialist = sax_detected or family_route in {"woodwind", "mixed_wind_brass"}

'''
        text = text[:start] + replacement + text[end:]

# Put the detector evidence into the technical report.
if '"instrument_inventory": global_inventory_report' not in text:
    report_anchor = '        report = {\n'
    if report_anchor not in text:
        raise RuntimeError('Could not locate experimental report dict')
    text = text.replace(
        report_anchor,
        report_anchor + '            "instrument_inventory": global_inventory_report,\n',
        1,
    )

# Add a human-readable inventory to README before packaging.
if 'DETECTED INSTRUMENTS' not in text:
    packaging_marker = '        emit("Packaging Parent and Experimental Stems", 92)\n'
    readme_block = '''        # Surface the inventory to users. This is detection evidence, not a
        # claim that every detected instrument has a clean specialist stem.
        inventory_readme = final / "README.txt"
        if inventory_readme.is_file():
            readme_text = inventory_readme.read_text(encoding="utf-8", errors="replace")
            display_names = {
                "hh": "Hi-hat",
                "double-bass": "Double Bass",
                "french-horn": "French Horn",
                "acoustic-guitar": "Acoustic Guitar",
                "electric-guitar": "Electric Guitar",
                "digital-piano": "Digital Piano",
                "bowed-strings": "Bowed Strings",
                "wind-chimes": "Wind Chimes",
            }
            lines = ["DETECTED INSTRUMENTS", "--------------------"]
            if detected_by_family:
                for family, names in detected_by_family.items():
                    pretty = [display_names.get(name, name.replace("-", " ").title()) for name in names]
                    lines.append(f"{family}: {', '.join(pretty)}")
            else:
                lines.append("No individual instruments reached the current confidence threshold.")
            lines.extend([
                "",
                "Detection is used to route specialist models and avoid unnecessary processing.",
                "A detected instrument does not guarantee that an individual specialist stem was exported.",
                "",
            ])
            section = "\\n".join(lines)
            if "INCLUDED STEMS\\n--------------" in readme_text:
                readme_text = readme_text.replace("INCLUDED STEMS\\n--------------", section + "\\nINCLUDED STEMS\\n--------------", 1)
            elif "ABOUT THIS PACK" in readme_text:
                readme_text = readme_text.replace("ABOUT THIS PACK", section + "\\nABOUT THIS PACK", 1)
            else:
                readme_text += "\\n\\n" + section
            inventory_readme.write_text(readme_text, encoding="utf-8")

'''
    if packaging_marker not in text:
        raise RuntimeError('Could not locate packaging marker for inventory README')
    text = text.replace(packaging_marker, readme_block + packaging_marker, 1)

path.write_text(text, encoding='utf-8')

check = path.read_text(encoding='utf-8')
assert 'global_instrument_inventory_v1' in check
assert 'global_inventory_reused_for_family_router' in check
assert '"instrument_inventory": global_inventory_report' in check
assert 'DETECTED INSTRUMENTS' in check
print('LiteLABS Research Instrument Inventory v1 applied')
