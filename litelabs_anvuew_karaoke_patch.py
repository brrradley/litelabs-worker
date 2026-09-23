from pathlib import Path

path = Path('/app/experimental_children_v1.py')
text = path.read_text(encoding='utf-8')

marker = '# litelabs_anvuew_karaoke_v1'
if marker not in text:
    old_model = '"mel_band_roformer_karaoke_becruily.ckpt"'
    if old_model not in text:
        raise RuntimeError('Could not locate current Becruily karaoke model in production vocal route')
    text = text.replace(old_model, '"bs_roformer_karaoke_anvuew.ckpt"', 1)

    # Production candidate selected from the blind shootout: Anvuew with
    # autocast + torch.compile on the already-clean SW vocal parent.
    autocast_anchor = '            "--use_autocast",\n'
    if autocast_anchor not in text:
        raise RuntimeError('Could not locate karaoke autocast flag')
    text = text.replace(
        autocast_anchor,
        autocast_anchor + '            "--use_torch_compile",\n',
        1,
    )

    replacements = {
        'Becruily Karaoke vocals output from SW vocal parent':
            'Anvuew Karaoke autocast+compile vocals output from SW vocal parent',
        'Becruily Karaoke instrumental output from SW vocal parent':
            'Anvuew Karaoke autocast+compile instrumental output from SW vocal parent',
        'LiteLABS-VX v2 single-pass vocal parent route':
            'LiteLABS-VX v3 Anvuew autocast+compile vocal parent route',
        '"benchmark_id": "vocal_route_v2_single_pass"':
            '"benchmark_id": "vocal_route_v3_anvuew_ac_compile"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Keep model metadata/report labels honest if the inherited image exposes
    # any of the historical Becruily names.
    text = text.replace(
        '"lead_backing_vocals": "BS-RoFormer Karaoke (frazer/becruily)"',
        '"lead_backing_vocals": "BS-RoFormer Karaoke (Anvuew, autocast + torch.compile)"',
    )
    text = text.replace(
        '"lead_backing_vocals": "Becruily Karaoke"',
        '"lead_backing_vocals": "Anvuew Karaoke (autocast + torch.compile)"',
    )

    text = marker + '\n' + text
    path.write_text(text, encoding='utf-8')

check = path.read_text(encoding='utf-8')
compile(check, str(path), 'exec')
assert marker in check
assert 'bs_roformer_karaoke_anvuew.ckpt' in check
assert 'mel_band_roformer_karaoke_becruily.ckpt' not in check
assert '"--use_autocast"' in check
assert '"--use_torch_compile"' in check
assert 'vocal_route_v3_anvuew_ac_compile' in check
print('LiteLABS-VX v3 Anvuew autocast+compile karaoke route applied')
