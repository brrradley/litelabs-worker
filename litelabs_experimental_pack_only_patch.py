from pathlib import Path

path = Path('/app/experimental_children_v1.py')
text = path.read_text(encoding='utf-8')

# Parents are internal routing/QA material only. Remove the historical root
# exports when present; the final archive is also curated explicitly below.
parent_copy = '''        for stem in SW_STEMS:
            _copy_as_flac(stems[stem], final / f"{track}_{stem}.flac")
'''
if parent_copy in text:
    text = text.replace(
        parent_copy,
        '''        # experimental_pack_only_v3
        # Parent stems stay internal; do not copy them into the customer pack.
''',
        1,
    )

instrumental_export = '        _write_flac(final / f"{track}_instrumental.flac", mixture[:n] - vocals[:n], mix_sr)\n'
if instrumental_export in text:
    text = text.replace(
        instrumental_export,
        '''        # Derived instrumental remains internal detector input only.
''',
        1,
    )

text = text.replace(
    '"quality_baseline": "BS-RoFormer-SW parent stems at ZIP root",',
    '"quality_baseline": "RoFormer parents used internally for routing/QA only",',
)
text = text.replace(
    '"experimental_policy": "Children are comparison-only and never replace parent stems in this mode",',
    '"experimental_policy": "Download pack contains curated Experimental outputs only",',
)

old_readme = '''        "QUALITY BASELINE",
        "----------------",
        "The FLAC files in the root of this archive are the untouched BS-RoFormer-SW parent stems.",
        "Experimental child stems are stored only inside /experimental/ and do not replace their parent stems.",
        "",
'''
new_readme = '''        "EXPERIMENTAL PACK",
        "-----------------",
        "This archive contains curated LiteLABS Experimental stems only.",
        "Parent stems are used internally for routing, reconstruction and QA and are not included.",
        "",
'''
if old_readme in text:
    text = text.replace(old_readme, new_readme, 1)

# Replace the historical open-ended ZIP walk with an explicit curated staging
# folder. Only Experimental audio that passes the dead-stem guard is copied.
start_markers = (
    '        emit("Packaging Parent and Experimental Stems", 92)\n',
    '        emit("Packaging Experimental Stems", 92)\n',
)
start = -1
for marker in start_markers:
    start = text.find(marker)
    if start >= 0:
        break
end = text.find('        uploaded = False\n', start)
if start < 0 or end < 0:
    raise RuntimeError('Could not locate Experimental packaging block')

packaging = '''        # experimental_pack_only_v3
        emit("Packaging Experimental Stems", 92)
        import re as _pack_re
        import shutil as _pack_shutil

        worker_match = _pack_re.search(r"-(?P<key>[0-9a-fA-F]{16})$", track)
        if worker_match:
            worker_key = worker_match.group("key")
            public_track = track[:worker_match.start()]
        else:
            worker_key = str(
                payload.get("worker_key")
                or payload.get("job_key")
                or payload.get("progress_job_id")
                or ""
            ).strip()
            public_track = track

        # Staging remains internal only. The ZIP itself is deliberately flat.
        pack_root = root / "curated_experimental_pack"
        pack_root.mkdir(parents=True, exist_ok=True)

        def _pack_audio_metrics(audio_path):
            try:
                audio, _sr = _read(audio_path)
                arr = np.asarray(audio, dtype=np.float32)
                if arr.size == 0:
                    return {
                        "dead": True,
                        "rms_dbfs": -120.0,
                        "peak_dbfs": -120.0,
                        "active_ratio": 0.0,
                    }
                mono = np.mean(arr, axis=1) if arr.ndim > 1 else arr
                rms = float(np.sqrt(np.mean(np.asarray(mono, dtype=np.float64) ** 2) + 1e-12))
                peak = float(np.max(np.abs(mono)) + 1e-12)
                rms_dbfs = _db(rms)
                peak_dbfs = _db(peak)
                activity_floor = max(10.0 ** (-60.0 / 20.0), peak * 0.02)
                active_ratio = float(np.mean(np.abs(mono) >= activity_floor))
                duration_seconds = float(len(mono)) / max(float(_sr), 1.0)
                bytes_per_second = float(audio_path.stat().st_size) / max(duration_seconds, 1.0)

                # Be conservative: remove only stems that are clearly empty or
                # near-empty. Quiet/sparse musical material should survive.
                dead = bool(
                    peak_dbfs <= -55.0
                    or (rms_dbfs <= -55.0 and active_ratio < 0.005)
                    or (
                        duration_seconds >= 60.0
                        and bytes_per_second < 1800.0
                        and rms_dbfs <= -48.0
                        and active_ratio < 0.01
                    )
                )
                return {
                    "dead": dead,
                    "rms_dbfs": round(rms_dbfs, 3),
                    "peak_dbfs": round(peak_dbfs, 3),
                    "active_ratio": round(active_ratio, 6),
                    "duration_seconds": round(duration_seconds, 3),
                    "bytes_per_second": round(bytes_per_second, 3),
                }
            except Exception as exc:
                return {
                    "dead": False,
                    "analysis_error": str(exc),
                }

        exported_files = []
        omitted_files = []
        stem_name_map = {
            "drums_5stem_kick": "kick",
            "drums_5stem_snare": "snare",
            "drums_5stem_toms": "toms",
            "drums_5stem_hats": "hats",
            "wind_brass_family": "wind",
            "sax_specialist_saxophone": "saxophone",
        }

        for src in sorted(experimental.glob("*.flac")):
            stem_id = src.stem
            if stem_id.startswith(track):
                stem_id = stem_id[len(track):].lstrip("_-")
            stem_id = stem_name_map.get(stem_id, stem_id)

            # Complements/residuals are technical evidence, not customer stems.
            if stem_id.endswith("_residual") or "residual" in stem_id:
                omitted_files.append({
                    "source": src.name,
                    "reason": "technical_residual",
                })
                continue

            metrics = _pack_audio_metrics(src)
            if metrics.get("dead"):
                omitted_files.append({
                    "source": src.name,
                    "reason": "dead_or_near_silent",
                    "metrics": metrics,
                })
                print(
                    f"LiteLABS pack omitted dead stem {src.name}: {metrics}",
                    flush=True,
                )
                continue

            dest = pack_root / f"{public_track}-{stem_id}.flac"
            _pack_shutil.copy2(src, dest)
            exported_files.append(dest.name)

        # Rebuild public detector evidence at the final packaging boundary so
        # earlier README composition cannot leave stale/empty instrumentation.
        public_detected_by_family = {}
        for family, members in family_map.items():
            if family == "Vocals":
                continue
            found = sorted({
                name for name in detected_instruments
                if name in members
            })
            if found:
                public_detected_by_family[family] = found

        g400_items = []
        broad_families = []
        public_genre = "unverified"
        public_genre_reason = "LiteLABS G400 evidence unavailable; heuristic genre kept internal only"
        if essentia_report.get("ok"):
            g400_items = list(essentia_report.get("genre_top10") or [])
            broad_families = list(essentia_report.get("genre_broad_families") or [])
            if g400_items:
                top_genre = g400_items[0]
                public_genre = str(top_genre.get("label") or "unverified").replace("---", " / ").replace("_", " ")
                public_genre_reason = (
                    "LiteLABS G400 top classification "
                    f"(mean {float(top_genre.get('mean', 0.0)):.3f}, "
                    f"p90 {float(top_genre.get('p90', 0.0)):.3f})"
                )

        readme_source = final / "README.txt"
        if readme_source.is_file():
            readme_lines = readme_source.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        else:
            readme_lines = [
                "LiteLABS Stem Extraction Tools",
                "==============================",
                "",
                "TRACK INFORMATION",
                "-----------------",
                f"Track: {payload.get('filename') or public_track}",
                "Pack: Experimental",
                "Output format: FLAC",
                "",
                "ABOUT THIS PACK",
                "---------------",
                "This stem pack was created using LiteLABS Stem Extraction Tools.",
            ]

        # Remove any older detector/G400 block so there is only one source of
        # truth in the final README.
        if "DETECTED INSTRUMENTS" in readme_lines:
            detector_start = readme_lines.index("DETECTED INSTRUMENTS")
            included_candidates = [
                idx for idx, line in enumerate(readme_lines)
                if line == "INCLUDED STEMS" and idx > detector_start
            ]
            detector_end = included_candidates[0] if included_candidates else len(readme_lines)
            del readme_lines[detector_start:detector_end]

        # Replace stale heuristic genre text with G400 evidence (or explicitly
        # unverified metadata when G400 did not return evidence).
        genre_seen = False
        reason_seen = False
        for idx, line in enumerate(readme_lines):
            if line.startswith("Detected genre:"):
                readme_lines[idx] = f"Detected genre: {public_genre}"
                genre_seen = True
            elif line.startswith("Genre reason:"):
                readme_lines[idx] = f"Genre reason: {public_genre_reason}"
                reason_seen = True

        output_format_idx = next(
            (idx for idx, line in enumerate(readme_lines) if line.startswith("Output format:")),
            None,
        )
        if not genre_seen:
            insert_idx = (output_format_idx + 1) if output_format_idx is not None else 0
            readme_lines.insert(insert_idx, f"Detected genre: {public_genre}")
            genre_seen = True
            if not reason_seen:
                readme_lines.insert(insert_idx + 1, f"Genre reason: {public_genre_reason}")
                reason_seen = True
        elif not reason_seen:
            genre_idx = next(
                idx for idx, line in enumerate(readme_lines)
                if line.startswith("Detected genre:")
            )
            readme_lines.insert(genre_idx + 1, f"Genre reason: {public_genre_reason}")

        # Replace Included Stems with exactly what survived final curation.
        if "INCLUDED STEMS" in readme_lines:
            included_idx = readme_lines.index("INCLUDED STEMS")
            about_idx = next(
                (
                    idx for idx, line in enumerate(readme_lines)
                    if line == "ABOUT THIS PACK" and idx > included_idx
                ),
                len(readme_lines),
            )
            included_block = [
                "INCLUDED STEMS",
                "--------------",
                *[f"- {name}" for name in sorted(exported_files)],
                "",
            ]
            readme_lines[included_idx:about_idx] = included_block
        else:
            about_idx = next(
                (
                    idx for idx, line in enumerate(readme_lines)
                    if line == "ABOUT THIS PACK"
                ),
                len(readme_lines),
            )
            readme_lines[about_idx:about_idx] = [
                "INCLUDED STEMS",
                "--------------",
                *[f"- {name}" for name in sorted(exported_files)],
                "",
            ]

        # Insert final detector and G400 evidence immediately before the Included
        # Stems section. This section must never silently disappear.
        included_idx = readme_lines.index("INCLUDED STEMS")
        detector_lines = [
            "DETECTED INSTRUMENTS",
            "--------------------",
        ]
        if public_detected_by_family:
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
            for family, names in public_detected_by_family.items():
                pretty = [
                    display_names.get(name, name.replace("-", " ").title())
                    for name in names
                ]
                detector_lines.append(f"{family}: {', '.join(pretty)}")
        else:
            detector_lines.append("No instruments reached the current detector confidence threshold.")

        detector_lines.extend([
            "",
            "LITELABS G400 GENRE CANDIDATES",
            "-----------------------------",
        ])
        if g400_items:
            for item in g400_items[:5]:
                detector_lines.append(
                    f"{item.get('label')}: mean {float(item.get('mean', 0.0)):.3f}"
                )
            if broad_families:
                detector_lines.append("")
                detector_lines.append(
                    "Broad families: " + ", ".join(
                        f"{item.get('family')} {float(item.get('score', 0.0)):.3f}"
                        for item in broad_families[:4]
                    )
                )
        else:
            detector_lines.append("No G400 genre evidence was available for this run.")
        detector_lines.append("")
        readme_lines[included_idx:included_idx] = detector_lines

        (pack_root / "README.txt").write_text(
            "\\n".join(readme_lines).rstrip() + "\\n",
            encoding="utf-8",
        )

        report["packaging"] = {
            "policy": "curated_experimental_only_v3",
            "archive_layout": "flat",
            "public_track": public_track,
            "worker_key": worker_key or None,
            "exported_files": sorted(exported_files),
            "omitted_files": omitted_files,
            "public_detected_by_family": public_detected_by_family,
            "public_genre": public_genre,
            "g400_available": bool(g400_items),
        }
        (pack_root / f"{public_track}-EXPERIMENTAL_REPORT.json").write_text(
            json.dumps(_json_safe(report), indent=2),
            encoding="utf-8",
        )

        archive = root / f"{track}_experimental_stems.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
            for p in sorted(pack_root.iterdir()):
                if p.is_file():
                    bundle.write(p, arcname=p.name)

'''
text = text[:start] + packaging + text[end:]

# Historical result field names are misleading after curation.
text = text.replace(
    '"root_parent_files": sorted(p.name for p in final.iterdir() if p.is_file()),',
    '"archive_layout": "flat",\n            "exported_files": sorted(exported_files),\n            "omitted_files": omitted_files,',
)
text = text.replace(
    '"root_metadata_files": sorted(p.name for p in final.iterdir() if p.is_file()),',
    '"archive_layout": "flat",\n            "exported_files": sorted(exported_files),\n            "omitted_files": omitted_files,',
)
text = text.replace(
    '"experimental_files": sorted(p.name for p in experimental.iterdir() if p.is_file()),',
    '"experimental_files": sorted(exported_files),',
)

path.write_text(text, encoding='utf-8')

check = path.read_text(encoding='utf-8')
compile(check, str(path), 'exec')
assert 'experimental_pack_only_v3' in check
assert 'archive_layout": "flat' in check
assert 'curated_experimental_only_v3' in check
assert 'drums_5stem_kick": "kick"' in check
assert 'wind_brass_family": "wind"' in check
assert 'technical_residual' in check
assert 'dead_or_near_silent' in check
assert '_parent_plus_experimental.zip' not in check
assert '_experimental_stems.zip' in check
print('LiteLABS curated Experimental-only flat pack policy v3 applied')
