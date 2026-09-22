from pathlib import Path

path = Path('/app/experimental_children_v1.py')
text = path.read_text(encoding='utf-8')

# Flatten experimental outputs into the ZIP root. Parent audio is removed below,
# so an /experimental/ subfolder only adds friction for users.
text = text.replace(
    '        experimental = final / "experimental"\n',
    '        experimental = final\n',
    1,
)

# Defensive compatibility with older addons that passed the internal
# <track>-<16hex job key> as payload.filename.
old_track = '        track = _safe_name(Path(str(payload.get("filename") or raw_name)).stem)\n'
new_track = '''        track = _safe_name(Path(str(payload.get("filename") or raw_name)).stem)
        if len(track) > 17 and track[-17] == "-" and all(
            ch in "0123456789abcdefABCDEF" for ch in track[-16:]
        ):
            track = track[:-17]
'''
if old_track in text:
    text = text.replace(old_track, new_track, 1)

# Research download packs are for experimental outputs only. RoFormer parent
# stems remain available internally for routing, QA and child reconstruction,
# but they are not duplicated into the downloadable ZIP.
parent_copy = '''        for stem in SW_STEMS:
            _copy_as_flac(stems[stem], final / f"{track}_{stem}.flac")
'''
if parent_copy in text:
    text = text.replace(
        parent_copy,
        '''        # experimental_pack_only_v1
        # Parent stems stay internal; do not copy them into the research pack.
''',
        1,
    )

instrumental_export = '        _write_flac(final / f"{track}_instrumental.flac", mixture[:n] - vocals[:n], mix_sr)\n'
if instrumental_export in text:
    text = text.replace(
        instrumental_export,
        '''        # Keep the derived instrumental internal as detector input only.
''',
        1,
    )

text = text.replace(
    'emit("Packaging Parent and Experimental Stems", 92)',
    'emit("Packaging Experimental Stems", 92)',
)
text = text.replace(
    'archive = root / f"{track}_parent_plus_experimental.zip"',
    'archive = root.parent / f"{root.name}_{track}_experimental_stems.zip"',
)
text = text.replace(
    '"quality_baseline": "BS-RoFormer-SW parent stems at ZIP root",',
    '"quality_baseline": "RoFormer parents used internally for routing/QA only",',
)
text = text.replace(
    '"experimental_policy": "Children are comparison-only and never replace parent stems in this mode",',
    '"experimental_policy": "Download pack contains experimental outputs only; parent stems are internal references",',
)

# Public README should describe the actual research pack, not the old parent+
# children comparison layout.
old_readme = '''        "QUALITY BASELINE",
        "----------------",
        "The FLAC files in the root of this archive are the untouched BS-RoFormer-SW parent stems.",
        "Experimental child stems are stored only inside /experimental/ and do not replace their parent stems.",
        "",
'''
new_readme = '''        "RESEARCH PACK",
        "-------------",
        "This archive contains experimental LiteLABS stems only.",
        "Parent stems are used internally for routing, reconstruction and QA and are not included in this download.",
        "",
'''
if old_readme in text:
    text = text.replace(old_readme, new_readme, 1)

# The result field name was historical and misleading once parent audio is no
# longer exported. Keep a compact list of root metadata files instead.
text = text.replace(
    '"root_parent_files": sorted(p.name for p in final.iterdir() if p.is_file()),',
    '"root_metadata_files": sorted(p.name for p in final.iterdir() if p.is_file()),',
)


path.write_text(text, encoding='utf-8')

check = path.read_text(encoding='utf-8')
if 'experimental_pack_only_v1' not in check:
    # Marker can disappear when an earlier patch has already removed the exact
    # parent-copy loop. Stamp the policy marker independently of that anchor.
    check = '# experimental_pack_only_v1\n' + check
    path.write_text(check, encoding='utf-8')
compile(check, str(path), 'exec')
assert 'experimental_pack_only_v1' in check
assert '_parent_plus_experimental.zip' not in check
assert '_experimental_stems.zip' in check
assert 'archive = root.parent / f"{root.name}_{track}_experimental_stems.zip"' in check
assert 'Packaging Experimental Stems' in check
assert 'root_parent_files' not in check
assert 'experimental = final / "experimental"' not in check
assert 'track = track[:-17]' in check
print('LiteLABS research experimental-only pack policy applied')
