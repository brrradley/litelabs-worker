from pathlib import Path

path = Path('/app/research_benchmark_v2.py')
text = path.read_text(encoding='utf-8')
marker = '# litelabs_research_full_matrix_v1\n'
if marker not in text:
    constant_anchor = 'VOCAL_CHALLENGER = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"\n'
    if constant_anchor not in text:
        raise RuntimeError('Could not locate vocal challenger constant')
    text = text.replace(
        constant_anchor,
        constant_anchor + 'KIMBERLEY_CHALLENGER = "vocals_mel_band_roformer.ckpt"\n',
        1,
    )

    published_anchor = '''    "karaoke_challenger": {\n        "label": "Anvuew BS-RoFormer Karaoke",\n        "published_lead_sdr": 10.22,\n    },\n'''
    if published_anchor not in text:
        raise RuntimeError('Could not locate published karaoke metadata')
    text = text.replace(
        published_anchor,
        '''    "kimberley_vocal_challenger": {\n        "label": "Kimberley Jensen MelBand RoFormer Vocals",\n        "note": "Independent public vocal-parent challenger tested locally in the same job.",\n    },\n''' + published_anchor,
        1,
    )

    run_anchor = '''        emit("Running current lead/back baseline", 36)\n        baseline_dir = root / "karaoke_baseline"\n'''
    if run_anchor not in text:
        raise RuntimeError('Could not locate current lead/back benchmark anchor')
    kimberley_block = '''        # litelabs_research_full_matrix_v1\n        emit("Testing Kimberley vocal-parent challenger", 31)\n        kimberley_dir = root / "vocal_challenger_kimberley"\n        kimberley = _run_separator(source, kimberley_dir, KIMBERLEY_CHALLENGER, [], timeout)\n        report["tests"]["vocal_parent_kimberley"] = kimberley\n        if kimberley.get("returncode") == 0:\n            kimberley_vocals = Path(kimberley["primary"])\n            kimberley_inst = Path(kimberley["secondary"])\n            kimberley["pair_metrics"] = _pair_metrics(source, kimberley_vocals, kimberley_inst)\n            kimberley["vs_current_sw_vocals"] = _similarity(sw_vocals, kimberley_vocals)\n            if challenger.get("returncode") == 0:\n                kimberley["vs_viperx_129755"] = _similarity(Path(challenger["primary"]), kimberley_vocals)\n            _copy_named(kimberley_vocals, outputs / "02b_kimberley_vocals.flac")\n            _copy_named(kimberley_inst, outputs / "02b_kimberley_instrumental.flac")\n\n'''
    text = text.replace(run_anchor, kimberley_block + run_anchor, 1)
    path.write_text(text, encoding='utf-8')

check = path.read_text(encoding='utf-8')
compile(check, str(path), 'exec')
assert '# litelabs_research_full_matrix_v1' in check
assert 'KIMBERLEY_CHALLENGER = "vocals_mel_band_roformer.ckpt"' in check
assert 'vocal_parent_kimberley' in check
assert 'vs_viperx_129755' in check
print('LiteLABS consolidated research matrix patch applied')
