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
        '''        # experimental_pack_only_v2
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

packaging = '''        # experimental_pack_only_v2
        emit("Packaging Experimental Stems", 92)
        import re as _pack_re
        import shutil as _pack_shutil

        # The worker key stays in the pack folder for traceability, but public
        # audio filenames use the clean source track name.
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

        pack_folder_name = (
            f"{public_track}-{worker_key}-litelabs-experimental"
            if worker_key
            else f"{public_track}-litelabs-experimental"
        )
        pack_root = root / pack_folder_name
        pack_root.mkdir(parents=True, exist_ok=True)

        def _pack_audio_metrics(audio_path):
            try:
                audio, _sr = _read(audio_path)
                arr = np.asarray(audio, dtype=np.float32)
                if arr.size == 0:
                    return {"dead": True, "rms_dbfs": -120.0, "peak_dbfs": -120.0, "active_ratio": 0.0}
                mono = np.mean(arr, axis=1) if arr.ndim > 1 else arr
                rms = float(np.sqrt(np.mean(np.asarray(mono, dtype=np.float64) ** 2) + 1e-12))
                peak = float(np.max(np.abs(mono)) + 1e-12)
                rms_dbfs = _db(rms)
                peak_dbfs = _db(peak)
                activity_floor = max(10.0 ** (-55.0 / 20.0), peak * 0.03)
                active_ratio = float(np.mean(np.abs(mono) >= activity_floor))
                duration_seconds = float(len(mono)) / max(float(_sr), 1.0)
                bytes_per_second = float(audio_path.stat().st_size) / max(duration_seconds, 1.0)
                dead = bool(
                    peak_dbfs <= -42.0
                    or (rms_dbfs <= -52.0 and active_ratio < 0.01)
                    or active_ratio < 0.001
                    or (duration_seconds >= 60.0 and bytes_per_second < 3000.0)
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

            # Residual complements are routing evidence, not customer stems.
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

        # Rewrite the public README's Included Stems block to match the curated
        # names, while preserving detector/genre sections composed earlier.
        readme_source = final / "README.txt"
        if readme_source.is_file():
            readme_text = readme_source.read_text(encoding="utf-8", errors="replace")
            heading = "INCLUDED STEMS\\n--------------"
            if heading in readme_text:
                before, after = readme_text.split(heading, 1)
                tail = after
                split_at = tail.find("\\n\\n")
                if split_at >= 0:
                    tail = tail[split_at + 2:]
                included = "\\n".join(f"- {name}" for name in sorted(exported_files))
                readme_text = before + heading + "\\n" + included + "\\n\\n" + tail
            else:
                included = "\\n".join(f"- {name}" for name in sorted(exported_files))
                readme_text += "\\n\\n" + heading + "\\n" + included + "\\n"
            (pack_root / "README.txt").write_text(readme_text, encoding="utf-8")

        report["packaging"] = {
            "policy": "curated_experimental_only_v2",
            "pack_folder": pack_folder_name,
            "public_track": public_track,
            "worker_key": worker_key or None,
            "exported_files": sorted(exported_files),
            "omitted_files": omitted_files,
        }
        (pack_root / f"{public_track}-EXPERIMENTAL_REPORT.json").write_text(
            json.dumps(_json_safe(report), indent=2),
            encoding="utf-8",
        )

        archive = root / f"{track}_experimental_stems.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
            for p in sorted(pack_root.rglob("*")):
                if p.is_file():
                    bundle.write(p, arcname=str(p.relative_to(root)))

'''
text = text[:start] + packaging + text[end:]

# Historical result field names are misleading after curation.
text = text.replace(
    '"root_parent_files": sorted(p.name for p in final.iterdir() if p.is_file()),',
    '"pack_folder": pack_folder_name,\n            "exported_files": sorted(exported_files),\n            "omitted_files": omitted_files,',
)
text = text.replace(
    '"root_metadata_files": sorted(p.name for p in final.iterdir() if p.is_file()),',
    '"pack_folder": pack_folder_name,\n            "exported_files": sorted(exported_files),\n            "omitted_files": omitted_files,',
)
text = text.replace(
    '"experimental_files": sorted(p.name for p in experimental.iterdir() if p.is_file()),',
    '"experimental_files": sorted(exported_files),',
)

path.write_text(text, encoding='utf-8')

check = path.read_text(encoding='utf-8')
compile(check, str(path), 'exec')
assert 'experimental_pack_only_v2' in check
assert 'pack_folder_name' in check
assert 'curated_experimental_only_v2' in check
assert 'drums_5stem_kick": "kick"' in check
assert 'wind_brass_family": "wind"' in check
assert 'technical_residual' in check
assert 'dead_or_near_silent' in check
assert '_parent_plus_experimental.zip' not in check
assert '_experimental_stems.zip' in check
print('LiteLABS curated Experimental-only pack policy v2 applied')
