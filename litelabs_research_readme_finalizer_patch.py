from pathlib import Path

path = Path('/app/experimental_children_v1.py')
text = path.read_text(encoding='utf-8')

# Own the final public README composition directly at the last pre-packaging
# point. Do not depend on older research patches having inserted a README block.
packaging_marker = '        emit("Packaging Parent and Experimental Stems", 92)\n'
if 'public_readme_inventory_final_v2' not in text:
    if packaging_marker not in text:
        raise RuntimeError('Could not locate final pre-packaging README anchor')

    clean_block = '''        # public_readme_inventory_final_v2
        inventory_readme = final / "README.txt"
        if inventory_readme.is_file():
            readme_text = inventory_readme.read_text(
                encoding="utf-8",
                errors="replace",
            )

            instrument_display_names = {
                "acousticbassguitar": "Acoustic Bass Guitar",
                "acousticguitar": "Acoustic Guitar",
                "classicalguitar": "Classical Guitar",
                "doublebass": "Double Bass",
                "drummachine": "Drum Machine",
                "electricguitar": "Electric Guitar",
                "electricpiano": "Electric Piano",
                "frenchhorn": "French Horn",
                "pipeorgan": "Pipe Organ",
                "synthesizer": "Synthesizer",
            }
            suppressed_public_instruments = {"computer", "voice"}

            # Remove all earlier heuristic analysis lines/sections that could
            # conflict with the validated detector output.
            filtered_lines = []
            skip_legacy_detection = False
            for line in readme_text.splitlines():
                stripped = line.strip()
                if stripped in {"GENRE", "DETECTED INSTRUMENTS", "ANALYSIS"}:
                    skip_legacy_detection = True
                    continue
                if skip_legacy_detection and (
                    stripped.startswith("----")
                    or stripped.startswith("Detected genre:")
                    or stripped.startswith("Genre reason:")
                    or stripped.startswith("Detected instruments:")
                    or not stripped
                ):
                    continue
                if skip_legacy_detection:
                    skip_legacy_detection = False
                if line.startswith("Detected genre:"):
                    continue
                if line.startswith("Genre reason:"):
                    continue
                if line.startswith("Detected instruments:"):
                    continue
                filtered_lines.append(line)
            readme_text = "\\n".join(filtered_lines).strip()

            detected_genre = str(
                genre_report.get("genre") or "Unverified"
            )

            public_instruments = []
            if instrument_report.get("ok"):
                for item in (
                    instrument_report.get("detected_instruments") or []
                ):
                    raw_name = str(
                        (item or {}).get("label") or ""
                    ).strip().lower()
                    if (
                        not raw_name
                        or raw_name in suppressed_public_instruments
                    ):
                        continue
                    pretty_name = instrument_display_names.get(
                        raw_name,
                        raw_name.replace("_", " ")
                        .replace("-", " ")
                        .title(),
                    )
                    if pretty_name not in public_instruments:
                        public_instruments.append(pretty_name)

            analysis_lines = [
                "ANALYSIS",
                "--------",
                f"Detected genre: {detected_genre}",
                "Detected instruments: "
                + (
                    ", ".join(public_instruments)
                    if public_instruments
                    else "None reached the current confidence threshold"
                ),
                "",
            ]
            analysis_section = "\\n".join(analysis_lines)

            included_marker = "INCLUDED STEMS\\n--------------"
            about_marker = "ABOUT THIS PACK"
            if included_marker in readme_text:
                readme_text = readme_text.replace(
                    included_marker,
                    analysis_section + "\\n" + included_marker,
                    1,
                )
            elif about_marker in readme_text:
                readme_text = readme_text.replace(
                    about_marker,
                    analysis_section + "\\n" + about_marker,
                    1,
                )
            else:
                readme_text = (
                    readme_text
                    + "\\n\\n"
                    + analysis_section
                )

            inventory_readme.write_text(
                readme_text.rstrip() + "\\n",
                encoding="utf-8",
            )

'''
    text = text.replace(
        packaging_marker,
        clean_block + packaging_marker,
        1,
    )

# Public model labels should never expose third-party implementation names.
old_loop = '''    for family, model in models.items():
        lines.append(f"{family.title()}: {model}")
'''
new_loop = '''    # public_readme_model_redaction_final_v2
    public_model_labels = {
        "parent": "LiteLABS Core",
        "drums": "LiteLABS Perc-5",
        "wind": "LiteLABS Wind",
        "saxophone": "LiteLABS Sax",
        "lead_backing_vocals": "LiteLABS VX",
        "instrument_inventory": "LiteLABS Instrument Analysis",
        "genre": "LiteLABS Genre Analysis",
    }
    for family in models:
        lines.append(
            f"{family.title()}: "
            f"{public_model_labels.get(family, 'LiteLABS Specialist')}"
        )
'''
if old_loop in text:
    text = text.replace(old_loop, new_loop, 1)

text = text.replace(
    '# public_readme_model_redaction_v1\n',
    '# public_readme_model_redaction_final_v2\n',
)

path.write_text(text, encoding='utf-8')

check = path.read_text(encoding='utf-8')
try:
    compile(check, str(path), 'exec')
except (IndentationError, SyntaxError) as exc:
    lines = check.splitlines()
    lineno = int(getattr(exc, 'lineno', 0) or 0)
    lo = max(1, lineno - 12)
    hi = min(len(lines), lineno + 12)
    print(
        '----- generated experimental_children_v1.py context -----',
        flush=True,
    )
    for number in range(lo, hi + 1):
        print(
            f"{number:04d}: {lines[number - 1]!r}",
            flush=True,
        )
    print('----- end generated source context -----', flush=True)
    raise

assert 'public_readme_inventory_final_v2' in check
assert 'ANALYSIS' in check
assert 'Detected genre: {detected_genre}' in check
assert 'genre_report.get("genre")' in check
assert 'instrument_report.get("detected_instruments")' in check
assert '"computer", "voice"' in check
assert '"electricguitar": "Electric Guitar"' in check
assert '"doublebass": "Double Bass"' in check
assert 'LITELABS G400 GENRE CANDIDATES' not in check
assert 'ESSENTIA GENRE CANDIDATES' not in check
print('LiteLABS final validated README composition verified')
