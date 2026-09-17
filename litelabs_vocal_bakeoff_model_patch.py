from pathlib import Path

path = Path('/app/experimental_vocal_cascade_bakeoff.py')
text = path.read_text(encoding='utf-8')

const = 'SW_MODEL = "BS-Roformer-SW.ckpt"\n'
insert = const + 'CURRENT_KARAOKE_CONFIG = Path("/models/karaoke_bs_roformer/config.yaml")\nCURRENT_KARAOKE_CHECKPOINT = Path("/models/karaoke_bs_roformer/model.ckpt")\nMSS_REPO = Path("/opt/music-source-separation-training")\n'
if 'CURRENT_KARAOKE_CONFIG' not in text:
    if const not in text:
        raise RuntimeError('Could not locate vocal bakeoff constants')
    text = text.replace(const, insert, 1)

anchor = '\ndef _pair_metrics(\n'
helper = r'''

def _run_current_set_karaoke(source: Path, output_dir: Path, timeout: int) -> tuple[np.ndarray, np.ndarray | None, float, str]:
    """Run the exact current SET karaoke checkpoint through MSS directly.

    The installed audio-separator build can run the MelBand Becruily candidate,
    but its model registry does not expose the frazer/becruily BS-RoFormer used
    by SET production.  Production already runs that checkpoint through MSS,
    so the benchmark must do the same rather than relying on the registry.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = output_dir / 'input'
    stems_dir = output_dir / 'stems'
    input_dir.mkdir(parents=True, exist_ok=True)
    stems_dir.mkdir(parents=True, exist_ok=True)
    local_source = input_dir / 'vocals.flac'
    import shutil
    shutil.copy2(source, local_source)

    missing = [str(p) for p in (CURRENT_KARAOKE_CONFIG, CURRENT_KARAOKE_CHECKPOINT, MSS_REPO / 'inference.py') if not p.is_file()]
    if missing:
        raise RuntimeError(f'Current SET karaoke assets missing: {missing}')

    cmd = [
        'python', str(MSS_REPO / 'inference.py'),
        '--model_type', 'bs_roformer',
        '--config_path', str(CURRENT_KARAOKE_CONFIG),
        '--start_check_point', str(CURRENT_KARAOKE_CHECKPOINT),
        '--input_folder', str(input_dir),
        '--store_dir', str(stems_dir),
        '--device_ids', '0',
        '--disable_detailed_pbar',
        '--filename_template', '{file_name}/{instr}',
    ]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=MSS_REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    elapsed = time.monotonic() - t0
    tail = '\n'.join((proc.stdout or '').splitlines()[-40:])
    if proc.returncode != 0:
        raise RuntimeError(f'Current SET BS-RoFormer Karaoke failed: {tail}')

    lead_path = _find_output(stems_dir, 'vocals')
    native_backing_path = _find_output(stems_dir, 'instrumental')
    if lead_path is None:
        raise RuntimeError(f'Current SET karaoke produced no vocals output. Files: {[p.name for p in stems_dir.rglob("*") if p.is_file()]}')
    lead, _sr = mt._load(lead_path)
    native_backing = mt._load(native_backing_path)[0] if native_backing_path else None
    return lead, native_backing, elapsed, tail
'''
if '_run_current_set_karaoke' not in text:
    if anchor not in text:
        raise RuntimeError('Could not locate pair metrics anchor')
    text = text.replace(anchor, helper + anchor, 1)

old = '''        set_lead, set_native_backing, set_elapsed, set_tail = _run_separator(\n            CURRENT_KARAOKE, sw_vocals_path, root / "set_karaoke", model_dir, timeout\n        )'''
new = '''        set_lead, set_native_backing, set_elapsed, set_tail = _run_current_set_karaoke(\n            sw_vocals_path, root / "set_karaoke", timeout\n        )'''
if old in text:
    text = text.replace(old, new, 1)
elif '_run_current_set_karaoke(' not in text[text.find('progress("Testing current SET'):]:]:
    raise RuntimeError('Could not locate current SET karaoke invocation')

path.write_text(text, encoding='utf-8')
print('vocal bakeoff now uses exact current SET karaoke checkpoint through MSS')
