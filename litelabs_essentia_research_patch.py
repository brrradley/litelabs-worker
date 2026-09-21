from pathlib import Path

path = Path('/app/experimental_children_v1.py')
text = path.read_text(encoding='utf-8')

if 'essentia_research_second_opinion_v1' not in text:
    anchor = '        global_inventory_report = {\n'
    block = '''        # essentia_research_second_opinion_v1
        # Use Essentia as an independent second opinion over the same sampled
        # instrumental audio plus a representative source sample. Mega53 remains
        # the primary inventory router; Essentia can boost family confidence and
        # provides richer genre evidence.
        essentia_report = {
            "ok": False,
            "research_only": True,
            "error": "not_run",
        }
        try:
            import os as _ess_os
            import subprocess as _ess_subprocess
            import sys as _ess_sys

            genre_sample_path = root / "essentia_genre_sample.flac"
            genre_audio, genre_sr = _read(source)
            genre_segment_len = max(1, int(genre_sr * 4.0))
            genre_total = len(genre_audio)
            genre_parts = []
            for frac in (0.08, 0.24, 0.40, 0.56, 0.72, 0.88):
                centre = int(genre_total * frac)
                start = max(0, min(max(genre_total - genre_segment_len, 0), centre - genre_segment_len // 2))
                piece = genre_audio[start:start + genre_segment_len]
                if len(piece):
                    genre_parts.append(piece)
            genre_sample = np.concatenate(genre_parts, axis=0) if genre_parts else genre_audio[:genre_segment_len]
            _write_flac(genre_sample_path, genre_sample, genre_sr)

            # Isolate Essentia's legacy TensorFlow runtime from the main worker.
            # The package targets CUDA 11 while production workers use CUDA 13;
            # running it CPU-only in a child process prevents native TF failures
            # from killing the extraction worker or polluting later GPU stages.
            if progress:
                progress("LiteLABS Inst-MTG / G400", 41)
            essentia_json_path = root / "essentia_research_result.json"
            essentia_env = _ess_os.environ.copy()
            essentia_env["CUDA_VISIBLE_DEVICES"] = "-1"
            essentia_env["TF_CPP_MIN_LOG_LEVEL"] = "2"
            essentia_proc = _ess_subprocess.run(
                [
                    _ess_sys.executable,
                    "/app/essentia_research.py",
                    str(global_inventory_in / "instrumental.flac"),
                    str(genre_sample_path),
                    str(essentia_json_path),
                ],
                env=essentia_env,
                stdout=_ess_subprocess.PIPE,
                stderr=_ess_subprocess.PIPE,
                text=True,
                timeout=300,
            )
            if essentia_proc.returncode != 0:
                err_tail = (essentia_proc.stderr or essentia_proc.stdout or "")[-5000:]
                raise RuntimeError(
                    f"Essentia subprocess failed ({essentia_proc.returncode}): {err_tail}"
                )
            essentia_report = json.loads(
                essentia_json_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            essentia_report = {
                "ok": False,
                "research_only": True,
                "error": str(exc),
                "error_type": exc.__class__.__name__,
            }
            print(f"LiteLABS Essentia research analysis skipped: {exc}", flush=True)

        essentia_detected = set()
        if essentia_report.get("ok"):
            essentia_detected = {
                str(name).lower().replace("_", "-")
                for name, item in (essentia_report.get("detected_instruments") or {}).items()
                if str((item or {}).get("confidence") or "") in {"strong", "likely"}
            }

        essentia_to_mega = {
            "acousticbassguitar": "bass",
            "acousticguitar": "acoustic-guitar",
            "bell": "bells",
            "classicalguitar": "guitar",
            "doublebass": "double-bass",
            "drummachine": "drums",
            "electricguitar": "electric-guitar",
            "electricpiano": "digital-piano",
            "horn": "french-horn",
            "keyboard": "keys",
            "pipeorgan": "organ",
            "synthesizer": "synth",
        }
        essentia_normalized = {
            essentia_to_mega.get(name, name)
            for name in essentia_detected
        }

        for name in sorted(essentia_normalized):
            if name not in detected_instruments:
                detected_instruments.append(name)

        # Rebuild the public family inventory after second-opinion evidence is
        # merged. The earlier family snapshot only contains the primary detector
        # and can otherwise leave README/report instrumentation stale or empty.
        detected_by_family = {}
        for family, members in family_map.items():
            if family == "Vocals":
                continue
            found = sorted({name for name in detected_instruments if name in members})
            if found:
                detected_by_family[family] = found

        # Genre remains secondary routing evidence, but G400 is the canonical
        # public/QA genre when available. The legacy parent heuristic is retained
        # only as an internal fallback when G400 genuinely has no result.
        research_genre = str(genre or "mixed_or_unknown")
        research_genre_reason = str(genre_reason or "")
        if essentia_report.get("ok") and (essentia_report.get("genre_top10") or []):
            research_genre_item = (essentia_report.get("genre_top10") or [])[0]
            research_genre = str(
                research_genre_item.get("label") or "mixed_or_unknown"
            ).replace("---", " / ").replace("_", " ")
            research_genre_reason = (
                "LiteLABS G400 top classification "
                f"(mean {float(research_genre_item.get('mean', 0.0)):.3f}, "
                f"p90 {float(research_genre_item.get('p90', 0.0)):.3f})"
            )
        # Feed the same canonical evidence into silent QA. This is deliberately
        # a source-text patch because the QA call lives in the inherited
        # production implementation.
        # Family routing can be boosted by Essentia, but expensive individual
        # specialists still require the existing Mega53/specialist evidence.
        essentia_brass = bool(
            {"brass", "trumpet", "trombone", "french-horn"}.intersection(essentia_normalized)
        )
        essentia_woodwind = bool(
            {"saxophone", "flute", "clarinet", "oboe"}.intersection(essentia_normalized)
        )
        essentia_strings = bool(
            {"strings", "violin", "viola", "cello", "double-bass"}.intersection(essentia_normalized)
        )

'''
    if anchor not in text:
        raise RuntimeError('Could not locate global inventory report anchor for Essentia insertion')
    text = text.replace(anchor, block + anchor, 1)

# Promote merged detector and G400 evidence into the canonical inventory report
# at construction time. The Essentia block is inserted before the report dict,
# so mutating global_inventory_report there would reference it before assignment.
if '"essentia": essentia_report' not in text:
    anchor = '            "evidence": global_inventory,\n'
    if anchor not in text:
        raise RuntimeError('Could not locate inventory evidence field')
    fields = (
        '            "detected": sorted(detected_instruments),\n'
        '            "detected_by_family": detected_by_family,\n'
        '            "evidence": global_inventory,\n'
        '            "essentia": essentia_report,\n'
        '            "detected_genre": research_genre,\n'
        '            "genre_reason": research_genre_reason,\n'
        '            "genre_top10": list(essentia_report.get("genre_top10") or [])[:10],\n'
    )
    text = text.replace(
        '            "detected": sorted(detected_instruments),\n'
        '            "detected_by_family": detected_by_family,\n'
        '            "evidence": global_inventory,\n',
        fields,
        1,
    )

# Boost wind-family routing using Essentia's independent detection.
old = '''        brass_detected = any(inv_score(x) for x in ("trumpet", "brass", "trombone", "french-horn", "tuba"))
        woodwind_detected = any(inv_score(x) for x in ("saxophone", "wind", "woodwind", "clarinet", "flute", "oboe", "bassoon"))
'''
new = '''        brass_detected = any(inv_score(x) for x in ("trumpet", "brass", "trombone", "french-horn", "tuba")) or essentia_brass
        woodwind_detected = any(inv_score(x) for x in ("saxophone", "wind", "woodwind", "clarinet", "flute", "oboe", "bassoon")) or essentia_woodwind
'''
if old in text:
    text = text.replace(old, new, 1)

# Add Essentia genre candidates to the README without overriding the existing
# genre label until we have benchmarked the classifier on our recent tracks.
if 'ESSENTIA GENRE CANDIDATES' not in text:
    marker = '            lines = ["DETECTED INSTRUMENTS", "--------------------"]\n'
    addition = '''            genre_lines = []
            if essentia_report.get("ok"):
                genre_lines = ["ESSENTIA GENRE CANDIDATES", "-------------------------"]
                for item in (essentia_report.get("genre_top10") or [])[:5]:
                    genre_lines.append(
                        f"{item.get('label')}: mean {float(item.get('mean', 0.0)):.3f}"
                    )
                broad = essentia_report.get("genre_broad_families") or []
                if broad:
                    genre_lines.append("")
                    genre_lines.append(
                        "Broad families: " + ", ".join(
                            f"{item.get('family')} {float(item.get('score', 0.0)):.3f}"
                            for item in broad[:4]
                        )
                    )
                genre_lines.append("")

'''
    if marker not in text:
        raise RuntimeError('Could not locate README inventory lines anchor')
    text = text.replace(marker, addition + marker, 1)

    end_marker = '            section = "\\n".join(lines)\n'
    replacement = '''            section = "\\n".join(lines)
            if genre_lines:
                section += "\\n" + "\\n".join(genre_lines)
'''
    if end_marker not in text:
        raise RuntimeError('Could not locate README inventory section join')
    text = text.replace(end_marker, replacement, 1)

text = text.replace(
    '            genre=genre,\n',
    '            genre=research_genre,\n',
)
text = text.replace(
    '            genre_reason=genre_reason,\n',
    '            genre_reason=research_genre_reason,\n',
)

# Add merged detector/G400 evidence to the QA extra payload so downstream
# LiteRECORDS packaging has access to the same canonical metadata.
qa_call = text.find('build_research_qa(')
if qa_call >= 0:
    extra_pos = text.find('extra={', qa_call)
    if extra_pos >= 0:
        insert_pos = extra_pos + len('extra={')
        qa_extra = (
            '\n                "detected_instruments": sorted(detected_instruments),'
            '\n                "detected_by_family": detected_by_family,'
            '\n                "genre_top10": list(essentia_report.get("genre_top10") or [])[:10],'
            '\n                "genre_broad_families": list(essentia_report.get("genre_broad_families") or [])[:8],'
        )
        if '"detected_instruments": sorted(detected_instruments)' not in text[qa_call:]:
            text = text[:insert_pos] + qa_extra + text[insert_pos:]

path.write_text(text, encoding='utf-8')

check = path.read_text(encoding='utf-8')
assert 'essentia_research_second_opinion_v1' in check
assert '"essentia": essentia_report' in check
assert 'ESSENTIA GENRE CANDIDATES' in check
print('LiteLABS Essentia second-opinion inventory + genre research patch applied')
