from pathlib import Path

path = Path('/app/master_pack.py')
text = path.read_text(encoding='utf-8')

old_function = '''def detect_genre_from_audio(decisions: list[StemDecision], core_stats: dict[str, dict], original_stats: dict) -> tuple[str, str]:
    """Return a practical audio-derived genre/track-type label plus a short explanation.

    This deliberately does not use file metadata. It uses the separated stem activity as the first
    reliable signal, because that is what we can validate and tune from real LiteRECORDS uploads.
    """
    included = {d.label for d in decisions if d.include}
    optional_scores = {d.label: d.score for d in decisions}

    vocals = score_of(core_stats, "Vocals")
    drums = score_of(core_stats, "Drums")
    bass = score_of(core_stats, "Bass")
    guitar = optional_scores.get("Guitar", 0.0)
    piano = optional_scores.get("Piano / Keys", 0.0)
    synth_other = optional_scores.get("Synths / Strings / Other", 0.0)
    original_active = float(original_stats.get("active_ratio", 0.0))

    strong_rhythm = drums >= 0.46 and bass >= 0.34
    strong_vocal = vocals >= 0.45
    strong_guitar = "Guitar" in included and guitar >= 0.42
    strong_piano = "Piano / Keys" in included and piano >= 0.42
    strong_synth = "Synths / Strings / Other" in included and synth_other >= 0.42

    if strong_rhythm and strong_synth and not strong_guitar:
        return "electronic_dance", "strong drums/bass with active synth/other and no confident guitar"
    if strong_rhythm and strong_guitar:
        return "rock_band", "strong drums with confident guitar activity"
    if strong_piano and strong_vocal and drums < 0.42:
        return "piano_vocal_or_pop_ballad", "confident piano/keys with strong vocal and lighter drums"
    if strong_vocal and drums >= 0.35 and bass >= 0.25 and not strong_guitar:
        return "vocal_pop", "strong vocal with moderate rhythm section and no confident guitar"
    if strong_vocal and original_active > 0.35 and drums < 0.30 and bass < 0.30:
        return "acoustic_or_sparse", "strong vocal with low drum/bass activity"
    if strong_rhythm and not strong_vocal:
        return "instrumental_or_dance", "strong drums/bass with weaker vocal presence"
    return "mixed_or_unknown", "audio features did not strongly match a known route"
'''

new_function = '''def detect_genre_from_audio(decisions: list[StemDecision], core_stats: dict[str, dict], original_stats: dict) -> tuple[str, str]:
    """Return a practical audio-derived genre/track-type label plus a short explanation.

    This deliberately does not use file metadata. It uses separated stem activity. The dance rule is
    intentionally allowed to beat guitar, because house/trance/pop-dance often contains sampled or
    reconstructed guitar parts that should not make the whole track rock.
    """
    included = {d.label for d in decisions if d.include}
    optional_scores = {d.label: d.score for d in decisions}

    vocals = score_of(core_stats, "Vocals")
    drums = score_of(core_stats, "Drums")
    bass = score_of(core_stats, "Bass")
    guitar = optional_scores.get("Guitar", 0.0)
    piano = optional_scores.get("Piano / Keys", 0.0)
    synth_other = optional_scores.get("Synths / Strings / Other", 0.0)
    original_active = float(original_stats.get("active_ratio", 0.0))

    strong_rhythm = drums >= 0.44 and bass >= 0.30
    strong_vocal = vocals >= 0.45
    strong_guitar = "Guitar" in included and guitar >= 0.42
    dominant_guitar = strong_guitar and guitar > max(synth_other + 0.18, 0.66)
    strong_piano = "Piano / Keys" in included and piano >= 0.42
    strong_synth = "Synths / Strings / Other" in included and synth_other >= 0.38
    dance_like = strong_rhythm and (strong_synth or not dominant_guitar or bass >= 0.42)

    if dance_like:
        details = ["strong drums/bass"]
        if strong_synth:
            details.append("active synth/other")
        if strong_guitar and not dominant_guitar:
            details.append("guitar appears secondary/sample-like")
        return "electronic_dance", ", ".join(details)
    if strong_rhythm and dominant_guitar:
        return "rock_band", "strong drums with dominant confident guitar activity"
    if strong_piano and strong_vocal and drums < 0.42:
        return "piano_vocal_or_pop_ballad", "confident piano/keys with strong vocal and lighter drums"
    if strong_vocal and drums >= 0.35 and bass >= 0.25 and not dominant_guitar:
        return "vocal_pop", "strong vocal with moderate rhythm section and no dominant guitar"
    if strong_vocal and original_active > 0.35 and drums < 0.30 and bass < 0.30:
        return "acoustic_or_sparse", "strong vocal with low drum/bass activity"
    if strong_rhythm and not strong_vocal:
        return "instrumental_or_dance", "strong drums/bass with weaker vocal presence"
    return "mixed_or_unknown", "audio features did not strongly match a known route"


def tune_optional_stems_for_genre(decisions: list[StemDecision], detected_genre: str) -> list[StemDecision]:
    if detected_genre != "electronic_dance":
        return decisions

    tuned: list[StemDecision] = []
    for decision in decisions:
        if decision.label == "Guitar" and decision.include and decision.score < 0.68:
            tuned.append(StemDecision(
                decision.label,
                decision.source,
                False,
                "omitted for electronic/dance route: low-confidence guitar/sample bleed",
                decision.score,
                decision.active_ratio,
                decision.mean_db,
                decision.max_db,
            ))
            continue
        if decision.label == "Piano / Keys" and decision.include and decision.score < 0.60:
            tuned.append(StemDecision(
                decision.label,
                decision.source,
                False,
                "omitted for electronic/dance route: low-confidence keys/bleed",
                decision.score,
                decision.active_ratio,
                decision.mean_db,
                decision.max_db,
            ))
            continue
        tuned.append(decision)
    return tuned
'''

if old_function not in text:
    raise SystemExit('Expected detect_genre_from_audio function was not found')
text = text.replace(old_function, new_function)

old_call = '''    detected_genre, genre_reason = detect_genre_from_audio(optional_decisions, core_stats, original_stats)
    print(f"LiteLABS detected genre: {detected_genre} ({genre_reason})", flush=True)
'''
new_call = '''    detected_genre, genre_reason = detect_genre_from_audio(optional_decisions, core_stats, original_stats)
    optional_decisions = tune_optional_stems_for_genre(optional_decisions, detected_genre)
    print(f"LiteLABS detected genre: {detected_genre} ({genre_reason})", flush=True)
'''
if old_call not in text:
    raise SystemExit('Expected genre detection call was not found')
text = text.replace(old_call, new_call)

path.write_text(text, encoding='utf-8')
