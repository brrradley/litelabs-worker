from pathlib import Path

path = Path('/app/mss_candidate_lab.py')
text = path.read_text(encoding='utf-8')

builder_marker = '\n\ndef build_mss_candidate_lab(payload: dict, progress=None) -> dict:\n'
helper = r'''


def _mega53_output_name(path: Path) -> str:
    name = path.stem.strip().lower().replace('_', '-').replace(' ', '-')
    while '--' in name:
        name = name.replace('--', '-')
    return name.strip('-') or 'unknown'


def _run_mega53_discovery(payload: dict, progress=None) -> dict:
    model_id = 'mvsep-mega53-piano-keys'
    audio_url = str(payload.get('audio_url') or payload.get('source_url') or '').strip()
    if not audio_url:
        return {'ok': False, 'mode': 'mss_candidate_lab', 'action': 'mega53_discovery', 'error': 'audio_url is required'}

    timeout_seconds = int(payload.get('timeout_seconds') or 1800)
    audible_rms_floor = float(payload.get('audible_rms_floor_dbfs') or -55.0)
    active_floor = float(payload.get('active_ratio_floor') or 0.01)

    inventory = _inventory()
    models = {item['id']: item for item in inventory['registered_models']}
    model = models.get(model_id)
    if not model:
        return {'ok': False, 'mode': 'mss_candidate_lab', 'action': 'mega53_discovery', 'error': f'Missing registry model: {model_id}'}

    auto_installed = False
    if not model.get('research_ready'):
        missing_files_only = bool(model.get('validation_errors')) and all(
            str(error).startswith('missing config:') or str(error).startswith('missing checkpoint:')
            for error in model.get('validation_errors', [])
        )
        if missing_files_only:
            if progress:
                progress('Installing Mega53 discovery model', 5)
            installed = _install_candidate({'model_id': model_id}, progress=None)
            if not installed.get('ok'):
                return {'ok': False, 'mode': 'mss_candidate_lab', 'action': 'mega53_discovery', 'failed_stage': 'install', 'result': installed}
            auto_installed = True
            inventory = _inventory()
            models = {item['id']: item for item in inventory['registered_models']}
            model = models.get(model_id)
    if not model or not model.get('research_ready'):
        return {'ok': False, 'mode': 'mss_candidate_lab', 'action': 'mega53_discovery', 'error': 'Mega53 is not research-ready', 'model': model}

    repo_dir = Path(inventory['framework']['repo_dir'])
    with tempfile.TemporaryDirectory(prefix='litelabs_mega53_discovery_') as temp:
        root = Path(temp)
        input_dir = root / 'input'
        output_dir = root / 'output'
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = unquote(Path(urlparse(audio_url).path).name) or 'track.flac'
        downloaded = root / filename
        if progress:
            progress('Downloading source for Mega53 discovery', 10)
        _download(audio_url, downloaded)

        source = input_dir / f'{Path(filename).stem}.wav'
        converted = subprocess.run(
            ['ffmpeg', '-y', '-i', str(downloaded), '-ar', '44100', '-ac', '2', str(source)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=300,
        )
        if converted.returncode != 0:
            return {
                'ok': False,
                'mode': 'mss_candidate_lab',
                'action': 'mega53_discovery',
                'failed_stage': 'convert_source',
                'runtime_log': '\n'.join((converted.stdout or '').splitlines()[-80:]),
            }

        command = [
            'python', str(repo_dir / 'inference.py'),
            '--model_type', model['model_type'],
            '--config_path', model['config_path'],
            '--start_check_point', model['checkpoint_path'],
            '--input_folder', str(input_dir),
            '--store_dir', str(output_dir),
            '--device_ids', '0',
            '--disable_detailed_pbar',
            '--filename_template', '{file_name}/{instr}',
        ]
        if progress:
            progress('Running Mega53 instrument discovery', 25)
        completed = subprocess.run(
            command,
            cwd=repo_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
        )
        if completed.returncode != 0:
            return {
                'ok': False,
                'mode': 'mss_candidate_lab',
                'action': 'mega53_discovery',
                'failed_stage': 'inference',
                'return_code': completed.returncode,
                'runtime_log': '\n'.join((completed.stdout or '').splitlines()[-100:]),
            }

        output_paths = sorted(
            item for item in output_dir.rglob('*')
            if item.is_file() and item.suffix.lower() in {'.wav', '.flac'}
        )
        if not output_paths:
            return {
                'ok': False,
                'mode': 'mss_candidate_lab',
                'action': 'mega53_discovery',
                'failed_stage': 'collect_outputs',
                'error': 'Mega53 produced no readable audio outputs',
                'runtime_log': '\n'.join((completed.stdout or '').splitlines()[-100:]),
            }

        evidence = []
        for index, output_path in enumerate(output_paths, start=1):
            try:
                metrics = _candidate_audio_metrics(output_path, source)
            except Exception as exc:
                evidence.append({
                    'instrument': _mega53_output_name(output_path),
                    'output_file': str(output_path.relative_to(output_dir)),
                    'metrics_error': str(exc),
                    'status': 'unscored',
                })
                continue
            audible = metrics.get('rms_dbfs', -240.0) >= audible_rms_floor and metrics.get('active_ratio', 0.0) >= active_floor
            score = (
                float(metrics.get('rms_dbfs', -240.0))
                + min(float(metrics.get('active_ratio', 0.0)), 1.0) * 8.0
                - abs(float(metrics.get('mixture_cosine', 0.0))) * 2.0
            )
            evidence.append({
                'instrument': _mega53_output_name(output_path),
                'output_file': str(output_path.relative_to(output_dir)),
                'metrics': metrics,
                'status': 'audible_candidate' if audible else 'weak_or_absent',
                'discovery_score': round(score, 3),
            })
            if progress and index % 5 == 0:
                progress('Scoring Mega53 instrument outputs', 70 + int(20 * index / max(1, len(output_paths))))

        evidence.sort(key=lambda item: item.get('discovery_score', -999.0), reverse=True)
        audible = [item for item in evidence if item.get('status') == 'audible_candidate']
        weak = [item for item in evidence if item.get('status') == 'weak_or_absent']

        mtg_interest = [
            str(item).strip().lower().replace('_', '-').replace(' ', '-')
            for item in (payload.get('mtg_interest') or [])
            if str(item).strip()
        ]
        mtg_crosscheck = []
        if mtg_interest:
            by_name = {item['instrument']: item for item in evidence}
            for name in mtg_interest:
                exact = by_name.get(name)
                if exact:
                    mtg_crosscheck.append({'instrument': name, 'mega53': exact})
                else:
                    related = [item for item in evidence if name in item['instrument'] or item['instrument'] in name]
                    mtg_crosscheck.append({'instrument': name, 'mega53': related[:5]})

        if progress:
            progress('Mega53 discovery complete', 100)
        return {
            'ok': True,
            'mode': 'mss_candidate_lab',
            'schema_version': 1,
            'action': 'mega53_discovery',
            'research_only': True,
            'model_id': model_id,
            'auto_installed_on_worker': auto_installed,
            'audio_url': audio_url,
            'output_count': len(output_paths),
            'audible_rms_floor_dbfs': audible_rms_floor,
            'active_ratio_floor': active_floor,
            'audible_candidates': audible,
            'weak_or_absent': weak,
            'all_outputs_ranked': evidence,
            'mtg_crosscheck': mtg_crosscheck,
            'warning': 'Mega53 stems can overlap and are discovery evidence, not ground-truth proof that an instrument exists.',
            'next_action': 'Merge Mega53 evidence with MTG detections and listening truth before routing specialist extraction.',
        }
'''

if 'def _run_mega53_discovery(' not in text:
    if builder_marker not in text:
        raise RuntimeError('Could not locate MSS candidate lab builder for Mega53 discovery')
    text = text.replace(builder_marker, helper + builder_marker, 1)

route_anchor = '    if action == "inventory":\n        return _inventory()\n'
route = route_anchor + '    if action == "mega53_discovery":\n        return _run_mega53_discovery(payload, progress=progress)\n'
if route_anchor in text and 'if action == "mega53_discovery"' not in text:
    text = text.replace(route_anchor, route, 1)
elif 'if action == "mega53_discovery"' not in text:
    raise RuntimeError('Could not add Mega53 discovery route')

path.write_text(text, encoding='utf-8')
print('Mega53 individual instrument discovery patch applied')
