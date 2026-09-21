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
            from essentia_research import run_essentia_research

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

            essentia_report = run_essentia_research(
                global_inventory_in / "instrumental.flac",
                genre_sample_path,
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

# Include Essentia evidence in the instrument inventory report.
if '"essentia": essentia_report' not in text:
    anchor = '            "evidence": global_inventory,\n'
    if anchor not in text:
        raise RuntimeError('Could not locate inventory evidence field')
    text = text.replace(
        anchor,
        anchor + '            "essentia": essentia_report,\n',
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
    replacement = '''            section = "\n".join(lines)
            if genre_lines:
                section += "\n" + "\n".join(genre_lines)
'''
    if end_marker not in text:
        raise RuntimeError('Could not locate README inventory section join')
    text = text.replace(end_marker, replacement, 1)

path.write_text(text, encoding='utf-8')

check = path.read_text(encoding='utf-8')
assert 'essentia_research_second_opinion_v1' in check
assert '"essentia": essentia_report' in check
assert 'ESSENTIA GENRE CANDIDATES' in check
print('LiteLABS Essentia second-opinion inventory + genre research patch applied')
