from pathlib import Path

path = Path('/app/experimental_children_v1.py')
text = path.read_text(encoding='utf-8')

# Rewrite the complete public README augmentation as one deterministic block.
# Earlier research patches build this area incrementally; replacing it here avoids
# indentation/escaping drift from patch-on-patch composition.
start_marker = '        # Surface the inventory to users. This is detection evidence, not a\n'
end_marker = '        emit("Packaging Parent and Experimental Stems", 92)\n'
start = text.find(start_marker)
end = text.find(end_marker, start)
if start < 0 or end < 0:
    raise RuntimeError('Could not locate public README augmentation block')

clean_block = '''        # public_readme_inventory_final_v1
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

            public_lines = ["DETECTED INSTRUMENTS", "--------------------"]
            if detected_by_family:
                for family, names in detected_by_family.items():
                    pretty = [
                        display_names.get(name, name.replace("-", " ").title())
                        for name in names
                    ]
                    public_lines.append(f"{family}: {', '.join(pretty)}")
            else:
                public_lines.append(
                    "No individual instruments reached the current confidence threshold."
                )

            public_lines.extend([
                "",
                "Detection is used to route specialist models and avoid unnecessary processing.",
                "A detected instrument does not guarantee that an individual specialist stem was exported.",
                "",
            ])

            if essentia_report.get("ok"):
                public_lines.extend([
                    "LITELABS G400 GENRE CANDIDATES",
                    "-----------------------------",
                ])
                for item in (essentia_report.get("genre_top10") or [])[:5]:
                    public_lines.append(
                        f"{item.get('label')}: mean {float(item.get('mean', 0.0)):.3f}"
                    )
                broad = essentia_report.get("genre_broad_families") or []
                if broad:
                    public_lines.append("")
                    public_lines.append(
                        "Broad families: " + ", ".join(
                            f"{item.get('family')} {float(item.get('score', 0.0)):.3f}"
                            for item in broad[:4]
                        )
                    )
                public_lines.append("")

            section = "\\n".join(public_lines)
            included_marker = "INCLUDED STEMS\\n--------------"
            if included_marker in readme_text:
                readme_text = readme_text.replace(
                    included_marker,
                    section + "\\n" + included_marker,
                    1,
                )
            elif "ABOUT THIS PACK" in readme_text:
                readme_text = readme_text.replace(
                    "ABOUT THIS PACK",
                    section + "\\nABOUT THIS PACK",
                    1,
                )
            else:
                readme_text += "\\n\\n" + section

            inventory_readme.write_text(readme_text, encoding="utf-8")

'''
text = text[:start] + clean_block + text[end:]

# Ensure the technical experimental README never prints third-party model names.
old_loop = '''    for family, model in models.items():
        lines.append(f"{family.title()}: {model}")
'''
new_loop = '''    # public_readme_model_redaction_final_v1
    public_model_labels = {
        "parent": "LiteLABS Core",
        "drums": "LiteLABS Perc-5",
        "wind": "LiteLABS Wind",
        "saxophone": "LiteLABS Sax",
        "lead_backing_vocals": "LiteLABS VX",
        "instrument_inventory": "LiteLABS Inst-MTG",
        "genre": "LiteLABS G400",
    }
    for family in models:
        lines.append(
            f"{family.title()}: {public_model_labels.get(family, 'LiteLABS Specialist')}"
        )
'''
if old_loop in text:
    text = text.replace(old_loop, new_loop, 1)

# Remove the older redaction marker if present; the final block above owns this now.
text = text.replace('# public_readme_model_redaction_v1\n', '# public_readme_model_redaction_final_v1\n')

path.write_text(text, encoding='utf-8')

check = path.read_text(encoding='utf-8')
try:
    compile(check, str(path), 'exec')
except (IndentationError, SyntaxError) as exc:
    lines = check.splitlines()
    lineno = int(getattr(exc, 'lineno', 0) or 0)
    lo = max(1, lineno - 12)
    hi = min(len(lines), lineno + 12)
    print("----- generated experimental_children_v1.py context -----", flush=True)
    for number in range(lo, hi + 1):
        print(f"{number:04d}: {lines[number - 1]!r}", flush=True)
    print("----- end generated source context -----", flush=True)
    raise
assert 'public_readme_inventory_final_v1' in check
assert 'LITELABS G400 GENRE CANDIDATES' in check
assert 'ESSENTIA GENRE CANDIDATES' not in check
print('LiteLABS final research README composition verified')
