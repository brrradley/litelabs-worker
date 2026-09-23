from pathlib import Path

path = Path('/app/research_benchmark_v2.py')
text = path.read_text(encoding='utf-8')
marker = '# litelabs_research_focused_v2\n'
if marker not in text:
    text += r'''

# litelabs_research_focused_v2

def _focused_pcm16_copy(source: Path, destination: Path) -> None:
    """Normalise FLOAT parents before audio-separator FLAC export.

    audio-separator 0.47 inherits FLOAT from the source WAV and soundfile cannot
    write FLOAT subtype into FLAC. 16-bit PCM is the same input format used by
    the successful 60-second precision matrix and lets the full-track A/B finish.
    """
    rc, _, log = _run([
        "ffmpeg", "-y", "-i", str(source), "-ar", "44100", "-ac", "2",
        "-c:a", "pcm_s16le", str(destination),
    ], timeout=300)
    if rc != 0:
        raise RuntimeError("Failed to create PCM16 research parent: " + log[-2000:])


def _focused_drumsep(parent: Path, root: Path, timeout: int) -> dict:
    """Measure the exact production DrumSep route without changing its maths."""
    from routed_extraction_v1 import _collect_named_outputs, _read
    from experimental_children_v1 import DRUM5, DRUM5_CONFIG, DRUM5_CHECKPOINT

    repo_dir = Path('/opt/music-source-separation-training')
    input_dir = root / 'drum_input'
    output_dir = root / 'drum_output'
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    drums, sr = _read(parent)
    sf.write(input_dir / 'drums.wav', drums.astype(np.float32), sr, subtype='FLOAT')
    cmd = [
        'python', str(repo_dir / 'inference.py'),
        '--model_type', 'mdx23c',
        '--config_path', str(DRUM5_CONFIG),
        '--start_check_point', str(DRUM5_CHECKPOINT),
        '--input_folder', str(input_dir),
        '--store_dir', str(output_dir),
        '--device_ids', '0',
        '--disable_detailed_pbar',
        '--filename_template', '{file_name}/{instr}',
    ]
    rc, elapsed, log = _run(cmd, cwd=repo_dir, timeout=timeout)
    result = {
        'returncode': rc,
        'runtime_seconds': round(elapsed, 3),
        'log_tail': '\n'.join(log.splitlines()[-80:]),
        'backend': 'production_mdx23c_drumsep_5stem',
        'precision_matrix': 'not_exposed_by_current_inference_cli',
    }
    if rc != 0:
        return result

    paths = _collect_named_outputs(output_dir, DRUM5)
    loaded = {name: _read(p)[0] for name, p in paths.items()}
    missing = [name for name in DRUM5 if name not in loaded]
    if missing:
        result['missing'] = missing
        return result

    n = min([len(drums)] + [len(loaded[name]) for name in DRUM5])
    parent_audio = drums[:n]
    child_sum = np.sum(np.stack([loaded[name][:n] for name in DRUM5], axis=0), axis=0)
    residual = parent_audio - child_sum
    result['quality'] = {
        'parent_vs_children_sum_cosine': round(float(_cos(parent_audio.reshape(-1), child_sum.reshape(-1))), 8),
        'residual_relative_to_parent_db': round(_db_ratio(residual, parent_audio), 3),
    }
    result['children'] = {name: _stats(paths[name]) for name in DRUM5}
    result['paths'] = {name: str(paths[name]) for name in DRUM5}
    return result


def _focused_corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n <= 0:
        return 0.0
    aa = a[:n].astype(np.float64, copy=False)
    bb = b[:n].astype(np.float64, copy=False)
    aa = aa - np.mean(aa)
    bb = bb - np.mean(bb)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / denom) if denom > 1e-12 else 0.0


def _focused_unmixx_chunked(input_path: Path, root: Path, timeout: int, *, chunk_seconds: float = 20.0, overlap_seconds: float = 2.0) -> dict:
    """Run UNMIXX in bounded-memory chunks and preserve speaker continuity."""
    audio, sr = sf.read(str(input_path), always_2d=True, dtype='float32')
    mono = np.mean(audio, axis=1, dtype=np.float32)
    chunk = max(1, int(sr * chunk_seconds))
    overlap = max(0, min(chunk // 2, int(sr * overlap_seconds)))
    step = max(1, chunk - overlap)
    starts = list(range(0, len(mono), step))
    if starts and starts[-1] >= len(mono):
        starts.pop()

    assembled = [np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)]
    chunk_reports = []
    started = time.monotonic()

    for index, start in enumerate(starts):
        segment = mono[start:min(len(mono), start + chunk)]
        if len(segment) < max(1, sr // 2):
            continue
        chunk_dir = root / f'unmixx_chunk_{index:03d}'
        out_dir = chunk_dir / 'out'
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_path = chunk_dir / 'input.wav'
        sf.write(chunk_path, segment, sr, subtype='PCM_16')

        rc, elapsed, log = _run([
            'python', str(UNMIXX_DIR / 'inference.py'),
            '--conf_path', str(UNMIXX_CONF),
            '--ckpt_path', str(UNMIXX_CKPT),
            '--audio_path', str(chunk_path),
            '--output_dir', str(out_dir),
            '--device', 'cuda',
        ], cwd=UNMIXX_DIR, timeout=timeout)
        info = {
            'index': index,
            'start_seconds': round(start / sr, 3),
            'duration_seconds': round(len(segment) / sr, 3),
            'runtime_seconds': round(elapsed, 3),
            'returncode': rc,
        }
        if rc != 0:
            info['log_tail'] = '\n'.join(log.splitlines()[-50:])
            chunk_reports.append(info)
            return {
                'returncode': rc,
                'runtime_seconds': round(time.monotonic() - started, 3),
                'chunk_seconds': chunk_seconds,
                'overlap_seconds': overlap_seconds,
                'chunks': chunk_reports,
                'failed_chunk': index,
            }

        speakers = sorted(out_dir.rglob('spk*.wav'))
        if len(speakers) < 2:
            info['error'] = f'Expected >=2 speaker files, found {len(speakers)}'
            chunk_reports.append(info)
            return {
                'returncode': 2,
                'runtime_seconds': round(time.monotonic() - started, 3),
                'chunks': chunk_reports,
                'failed_chunk': index,
            }

        current = []
        for speaker in speakers[:2]:
            data, out_sr = sf.read(str(speaker), always_2d=True, dtype='float32')
            if int(out_sr) != int(sr):
                raise RuntimeError(f'UNMIXX chunk sample rate changed: {out_sr} != {sr}')
            current.append(np.mean(data, axis=1, dtype=np.float32))

        if not len(assembled[0]):
            assembled = [current[0], current[1]]
            info['speaker_swap'] = False
        else:
            ov = min(overlap, len(assembled[0]), len(current[0]), len(current[1]))
            swap = False
            if ov > 0:
                tail0 = assembled[0][-ov:]
                tail1 = assembled[1][-ov:]
                same = abs(_focused_corr(tail0, current[0][:ov])) + abs(_focused_corr(tail1, current[1][:ov]))
                crossed = abs(_focused_corr(tail0, current[1][:ov])) + abs(_focused_corr(tail1, current[0][:ov]))
                swap = crossed > same
            if swap:
                current = [current[1], current[0]]
            info['speaker_swap'] = bool(swap)

            for speaker_index in (0, 1):
                ov = min(overlap, len(assembled[speaker_index]), len(current[speaker_index]))
                if ov > 0:
                    fade_in = np.linspace(0.0, 1.0, ov, endpoint=False, dtype=np.float32)
                    fade_out = 1.0 - fade_in
                    joined = assembled[speaker_index][-ov:] * fade_out + current[speaker_index][:ov] * fade_in
                    assembled[speaker_index] = np.concatenate([
                        assembled[speaker_index][:-ov], joined, current[speaker_index][ov:]
                    ])
                else:
                    assembled[speaker_index] = np.concatenate([assembled[speaker_index], current[speaker_index]])
        chunk_reports.append(info)

    target = len(mono)
    stitched_dir = root / 'unmixx_stitched'
    stitched_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in (0, 1):
        output = assembled[index][:target]
        if len(output) < target:
            output = np.pad(output, (0, target - len(output)))
        dest = stitched_dir / f'spk{index + 1}.wav'
        sf.write(dest, output, sr, subtype='FLOAT')
        paths.append(dest)

    return {
        'returncode': 0,
        'runtime_seconds': round(time.monotonic() - started, 3),
        'chunk_seconds': chunk_seconds,
        'overlap_seconds': overlap_seconds,
        'chunk_count': len(chunk_reports),
        'speaker_swaps': sum(1 for item in chunk_reports if item.get('speaker_swap')),
        'chunks': chunk_reports,
        'files': [p.name for p in paths],
        'paths': [str(p) for p in paths],
        'pair_metrics': _pair_metrics(input_path, paths[0], paths[1]),
    }



def _focused_upload_archive(archive: Path, put_url: str, payload: dict) -> dict:
    """Upload research artifacts through the LiteRECORDS chunk receiver."""
    import math
    import time as _upload_time
    import requests

    size = archive.stat().st_size
    chunk_mb = int(payload.get("result_chunk_size_mb") or 16)
    chunk_mb = max(4, min(32, chunk_mb))
    chunk_bytes = chunk_mb * 1024 * 1024
    total = max(1, int(math.ceil(size / chunk_bytes)))

    print(
        f"LiteLABS research chunked result upload: {total} part(s) at up to {chunk_mb} MiB each",
        flush=True,
    )

    with archive.open("rb") as handle:
        for part in range(total):
            data = handle.read(chunk_bytes)
            if not data:
                raise RuntimeError(
                    f"Research archive ended before chunk {part + 1}/{total}"
                )

            for attempt in range(1, 4):
                try:
                    response = requests.post(
                        put_url,
                        params={"part": part, "total": total, "size": size},
                        data=data,
                        headers={"Content-Type": "application/octet-stream"},
                        timeout=(30, 600),
                    )
                    if response.status_code == 413:
                        return {
                            "uploaded": False,
                            "status_code": 413,
                            "part": part,
                        }
                    response.raise_for_status()
                    try:
                        body = response.json()
                    except ValueError:
                        body = {}

                    # XenForo responses can be returned directly or wrapped by
                    # the JSON reply layer. Normalize the common wrappers before
                    # interpreting the receiver acknowledgement.
                    ack = body
                    for key in ("json", "data", "response"):
                        if isinstance(ack, dict) and isinstance(ack.get(key), dict):
                            ack = ack[key]
                            break

                    if isinstance(ack, dict) and ack and ack.get("ok") is False:
                        raise RuntimeError(
                            str(ack.get("error") or "Chunk receiver rejected research upload")
                        )

                    if part == total - 1:
                        # Do not require a particular XenForo JSON envelope here.
                        # The chunk receiver writes the request body before returning
                        # and assembles synchronously when all parts are present.
                        # Therefore a successful 2xx response on our sequential final
                        # POST is sufficient; explicit receiver errors are handled above.
                        print(
                            "LiteLABS research final chunk accepted by receiver",
                            flush=True,
                        )
                    print(
                        f"LiteLABS research upload chunk {part + 1}/{total} complete",
                        flush=True,
                    )
                    break
                except Exception as exc:
                    if attempt >= 3:
                        raise
                    wait = 2 ** (attempt - 1)
                    print(
                        f"LiteLABS research upload chunk {part + 1}/{total} attempt "
                        f"{attempt} failed: {exc}; retrying in {wait}s",
                        flush=True,
                    )
                    _upload_time.sleep(wait)

    return {"uploaded": True, "mode": "chunked", "chunks": total}


def _run_research_benchmark_focused(payload: dict, progress=None) -> dict:
    """Focused SET research: viable lead/back, drums and bounded-memory multi-vocal."""
    audio_url = str(payload.get('audio_url') or payload.get('source_url') or '').strip()
    if not audio_url:
        return {'ok': False, 'mode': MODE, 'error': 'audio_url is required'}

    timeout = max(300, int(payload.get('timeout_seconds') or 2400))
    heartbeat = max(5, int(payload.get('heartbeat_seconds') or 15))
    filename = str(payload.get('filename') or unquote(Path(urlparse(audio_url).path).name) or 'track.wav')
    track = _safe_name(Path(filename).stem)
    model_dir = Path(str(payload.get('model_dir') or '/models/bs_roformer_sw'))

    def emit(message: str, percent: int) -> None:
        print(f'[{MODE}] {message} ({percent}%)', flush=True)
        if progress:
            progress(message, percent)

    build_sha = os.getenv('LITELABS_BUILD_SHA', 'unknown')
    listening_only = bool(payload.get('listening_only'))
    report = {
        'schema_version': 3,
        'mode': 'research_benchmark_focused_v2',
        'build_sha': build_sha,
        'track': track,
        'policy': {
            'quality_first': True,
            'precision_target_difference_db': -69.0,
            'dropped_candidates': ['Viperx BS-RoFormer ep317', 'Kimberley Jensen MelBand RoFormer Vocals'],
        },
        'listening_only': listening_only,
        'tests': {},
    }

    started = time.monotonic()
    archive = Path('/tmp') / f'litelabs-focused-research-{uuid.uuid4().hex[:10]}-{track}.zip'
    with tempfile.TemporaryDirectory(prefix='litelabs_focused_research_') as temp:
        root = Path(temp)
        source_dir = root / 'source'
        sw_dir = root / 'sw'
        outputs = root / 'outputs'
        source_dir.mkdir()
        sw_dir.mkdir()
        outputs.mkdir()

        raw_name = unquote(Path(urlparse(audio_url).path).name) or 'input.audio'
        downloaded = root / raw_name
        source = source_dir / f'{track}.wav'

        emit('Downloading benchmark source', 2)
        _download(audio_url, downloaded)
        rc, elapsed, log = _run(['ffmpeg', '-y', '-i', str(downloaded), '-ar', '44100', '-ac', '2', str(source)], timeout=300)
        if rc != 0:
            raise RuntimeError('Source conversion failed: ' + log[-2000:])
        report['tests']['source_conversion'] = {'runtime_seconds': round(elapsed, 3)}

        emit('Running current SW vocal parent baseline', 8)
        sw_config, sw_checkpoint, _ = _resolve_model_files(model_dir, progress=None)
        rc, sw_elapsed = _run_polled([
            'bs-roformer-infer', '--config_path', str(sw_config), '--model_path', str(sw_checkpoint),
            '--input_folder', str(source_dir), '--store_dir', str(sw_dir),
        ], cwd=None, timeout=timeout, log_path=root / 'sw.log', progress=None,
           stage_name='Research SW baseline', start_percent=8, end_percent=22, heartbeat_seconds=heartbeat)
        if rc != 0:
            raise RuntimeError('Current SW parent baseline failed')
        sw_stems = _collect_sw_stems(sw_dir)
        sw_vocals = sw_stems.get('vocals')
        sw_drums = sw_stems.get('drums')
        if not sw_vocals:
            raise RuntimeError('SW baseline did not produce vocals')
        if not listening_only and not sw_drums:
            raise RuntimeError('SW baseline did not produce drums')
        report['tests']['sw_parent_baseline'] = {
            'runtime_seconds': round(sw_elapsed, 3),
            'vocals': _stats(sw_vocals),
        }
        if sw_drums:
            report['tests']['sw_parent_baseline']['drums'] = _stats(sw_drums)
        _copy_named(sw_vocals, outputs / '01_current_sw_vocals.flac')
        if not listening_only and sw_drums:
            _copy_named(sw_drums, outputs / '01_current_sw_drums.flac')

            emit('Benchmarking production DrumSep route', 25)
            drum_result = _focused_drumsep(sw_drums, root, timeout)
            report['tests']['drumsep_current'] = drum_result
            if drum_result.get('returncode') == 0:
                for name, source_path in (drum_result.get('paths') or {}).items():
                    _copy_named(Path(source_path), outputs / f'02_drumsep_{name}.wav')

        emit('Preparing full-track lead/back compatibility parent', 47)
        vocal_parent = root / 'sw_vocals_pcm16.wav'
        _focused_pcm16_copy(sw_vocals, vocal_parent)

        emit('Running current Becruily lead/back baseline', 50)
        current_dir = root / 'leadback_current'
        current = _run_separator(vocal_parent, current_dir, KARAOKE_BASELINE, [], timeout)
        report['tests']['lead_back_current_fp32'] = current
        current_lead = current_back = None
        if current.get('returncode') == 0:
            current_lead = Path(current['primary'])
            current_back = Path(current['secondary'])
            current['pair_metrics'] = _pair_metrics(vocal_parent, current_lead, current_back)
            _copy_named(current_lead, outputs / '03_current_lead.flac')
            _copy_named(current_back, outputs / '03_current_backing.flac')

        emit('Running Anvuew lead/back FP32', 60)
        anvuew_fp32_dir = root / 'leadback_anvuew_fp32'
        anvuew_fp32 = _run_separator(vocal_parent, anvuew_fp32_dir, KARAOKE_CHALLENGER, [], timeout)
        report['tests']['lead_back_anvuew_fp32'] = anvuew_fp32
        anvuew_ref = None
        if anvuew_fp32.get('returncode') == 0:
            lead = Path(anvuew_fp32['primary'])
            back = Path(anvuew_fp32['secondary'])
            anvuew_ref = (lead, back)
            anvuew_fp32['pair_metrics'] = _pair_metrics(vocal_parent, lead, back)
            if current_lead and current_back:
                anvuew_fp32['vs_current'] = {
                    'lead': _similarity(current_lead, lead),
                    'backing': _similarity(current_back, back),
                }
            _copy_named(lead, outputs / '04_anvuew_fp32_lead.flac')
            _copy_named(back, outputs / '04_anvuew_fp32_backing.flac')

        emit('Running Anvuew lead/back autocast + compile', 70)
        anvuew_ac_dir = root / 'leadback_anvuew_autocast_compile'
        anvuew_ac = _run_separator(vocal_parent, anvuew_ac_dir, KARAOKE_CHALLENGER, ['--use_autocast', '--use_torch_compile'], timeout)
        report['tests']['lead_back_anvuew_autocast_compile'] = anvuew_ac
        if anvuew_ac.get('returncode') == 0:
            lead = Path(anvuew_ac['primary'])
            back = Path(anvuew_ac['secondary'])
            anvuew_ac['pair_metrics'] = _pair_metrics(vocal_parent, lead, back)
            if anvuew_ref:
                anvuew_ac['vs_anvuew_fp32'] = {
                    'lead': _similarity(anvuew_ref[0], lead),
                    'backing': _similarity(anvuew_ref[1], back),
                }
            _copy_named(lead, outputs / '05_anvuew_autocast_compile_lead.flac')
            _copy_named(back, outputs / '05_anvuew_autocast_compile_backing.flac')

        # Build a genuinely blind A/B/C listening pack. The mapping is kept only
        # in a deliberately named answer-key file so it can remain unopened
        # until after the listener has scored lead and backing quality.
        if current_lead and current_back and anvuew_ref and anvuew_ac.get('returncode') == 0:
            ac_lead = Path(anvuew_ac['primary'])
            ac_back = Path(anvuew_ac['secondary'])
            candidates = [
                ('Current Becruily FP32', current_lead, current_back),
                ('Anvuew FP32', anvuew_ref[0], anvuew_ref[1]),
                ('Anvuew autocast + compile', ac_lead, ac_back),
            ]
            blinded = sorted(
                candidates,
                key=lambda item: uuid.uuid5(uuid.NAMESPACE_DNS, f'{build_sha}:{track}:{item[0]}').int,
            )
            blind_root = root / 'blind_listening_pack'
            blind_root.mkdir(parents=True, exist_ok=True)
            answer_lines = ['LiteLABS Blind Listening Answer Key', '===============================', '']
            for letter, (label, lead_path, back_path) in zip(('A', 'B', 'C'), blinded):
                _crop_center(lead_path, blind_root / f'{letter}_lead.flac', 60.0)
                _crop_center(back_path, blind_root / f'{letter}_backing.flac', 60.0)
                answer_lines.append(f'{letter} = {label}')
            (blind_root / 'LISTEN_FIRST.txt').write_text(
                'LiteLABS Lead / Backing Blind Listening Test\n'
                '===========================================\n\n'
                'Do not open ANSWER_KEY_AFTER_LISTENING.txt until scoring is complete.\n\n'
                'Listen to A/B/C lead stems first, then A/B/C backing stems.\n'
                'Use the same headphones/speakers and level for every file.\n\n'
                'Score each candidate 1-10 for:\n'
                '- vocal isolation / bleed\n'
                '- retained harmonies and ad-libs\n'
                '- artefacts / warble\n'
                '- naturalness\n'
                '- usefulness of the backing-vocal stem\n\n'
                'Then give one overall preference for LEAD and one for BACKING.\n',
                encoding='utf-8',
            )
            (blind_root / 'SCORECARD.txt').write_text(
                'A lead: isolation __  retention __  artefacts __  naturalness __\n'
                'B lead: isolation __  retention __  artefacts __  naturalness __\n'
                'C lead: isolation __  retention __  artefacts __  naturalness __\n\n'
                'A backing: isolation __  retention __  artefacts __  naturalness __  usefulness __\n'
                'B backing: isolation __  retention __  artefacts __  naturalness __  usefulness __\n'
                'C backing: isolation __  retention __  artefacts __  naturalness __  usefulness __\n\n'
                'Preferred lead: __\nPreferred backing: __\nOverall preference: __\n',
                encoding='utf-8',
            )
            (blind_root / 'ANSWER_KEY_AFTER_LISTENING.txt').write_text(
                '\n'.join(answer_lines) + '\n', encoding='utf-8'
            )
            blind_zip = outputs / 'BLIND_LEAD_BACK_LISTENING_TEST.zip'
            with zipfile.ZipFile(blind_zip, 'w', compression=zipfile.ZIP_STORED) as blind_bundle:
                for blind_file in sorted(blind_root.iterdir()):
                    if blind_file.is_file():
                        blind_bundle.write(blind_file, arcname=blind_file.name)
            report['blind_listening_pack'] = {
                'created': True,
                'excerpt_seconds': 60,
                'filename': blind_zip.name,
                'candidates': 3,
                'mapping_hidden_in_answer_key': True,
            }

        if not listening_only:
            emit('Testing bounded-memory UNMIXX multi-vocal separation', 80)
            unmixx_input = root / 'unmixx_vocals.wav'
            rc, _, log = _run(['ffmpeg', '-y', '-i', str(sw_vocals), '-ar', '24000', '-ac', '1', '-c:a', 'pcm_s16le', str(unmixx_input)], timeout=300)
            if rc != 0:
                raise RuntimeError('UNMIXX input conversion failed: ' + log[-2000:])
            unmixx = _focused_unmixx_chunked(
                unmixx_input, root, timeout,
                chunk_seconds=float(payload.get('unmixx_chunk_seconds') or 20.0),
                overlap_seconds=float(payload.get('unmixx_overlap_seconds') or 2.0),
            )
            report['tests']['unmixx_chunked'] = unmixx
            if unmixx.get('returncode') == 0:
                for index, source_path in enumerate(unmixx.get('paths') or [], start=1):
                    _copy_named(Path(source_path), outputs / f'06_unmixx_voice_{index}.wav')

        report['total_runtime_seconds'] = round(time.monotonic() - started, 3)
        (outputs / 'research_benchmark_report.json').write_text(json.dumps(_json_safe(report), indent=2), encoding='utf-8')
        (outputs / 'README.txt').write_text(
            'LiteLABS Focused Research Benchmark\n'
            '==================================\n\n'
            'Quality-first SET research pack.\n'
            'Dropped: Viperx and Kimberley vocal-parent candidates.\n'
            'Retained: current Becruily vs Anvuew lead/back, production DrumSep baseline,\n'
            'and chunked UNMIXX multi-vocal research.\n'
            'The target for precision-only changes is <= -69 dB difference from FP32.\n',
            encoding='utf-8',
        )

        emit('Packaging focused research comparison', 94)
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as bundle:
            for output in sorted(outputs.rglob('*')):
                if output.is_file():
                    bundle.write(output, arcname=output.name)

    archive_size = archive.stat().st_size
    uploaded = False
    result_url = payload.get('result_public_url')
    put_url = str(payload.get('result_put_url') or '').strip()
    upload_details = None
    if put_url:
        emit('Uploading focused research comparison', 96)
        upload_details = _focused_upload_archive(archive, put_url, payload)
        if int(upload_details.get('status_code') or 0) == 413:
            return _json_safe({
                'ok': False,
                'mode': 'research_benchmark_focused_v2',
                'track': track,
                'build_sha': build_sha,
                'failed_stage': 'result_upload',
                'error_code': 'result_too_large',
                'http_status': 413,
                'archive_name': archive.name,
                'archive_size_bytes': archive_size,
                'uploaded': False,
                'result_url': None,
                'report': report,
            })
        uploaded = bool(upload_details.get('uploaded'))

    result = {
        'ok': True,
        'mode': 'research_benchmark_focused_v2',
        'track': track,
        'build_sha': build_sha,
        'archive_name': archive.name,
        'archive_size_bytes': archive_size,
        'uploaded': uploaded,
        'result_url': result_url if uploaded else None,
        'upload': upload_details,
        'report': report,
    }
    if uploaded:
        archive.unlink(missing_ok=True)
    emit('Focused research benchmark complete', 100)
    return _json_safe(result)


# The suite wrapper resolves this global when each child track starts, so this
# safely replaces the old kitchen-sink single-track benchmark without changing
# the established audio_urls request shape.
_run_research_benchmark_single = _run_research_benchmark_focused
'''
    path.write_text(text, encoding='utf-8')

check = path.read_text(encoding='utf-8')
compile(check, str(path), 'exec')
assert '# litelabs_research_focused_v2' in check
assert '_run_research_benchmark_single = _run_research_benchmark_focused' in check
assert 'unmixx_chunked' in check
assert 'lead_back_anvuew_autocast_compile' in check
assert 'drumsep_current' in check
print('LiteLABS focused quality-first research benchmark applied')
