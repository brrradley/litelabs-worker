from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import soundfile as sf

from sw_residual_allocator import _download, _resolve_model_files

MEGA53_CONFIG = Path('/models/mss_training/mvsep-mega53/config.yaml')
MEGA53_CHECKPOINT = Path('/models/mss_training/mvsep-mega53/model.ckpt')
TARGETS = ('saxophone', 'trumpet')


def _read(path: Path):
    audio, sr = sf.read(path, always_2d=True, dtype='float32')
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    return audio[:, :2].astype(np.float64), int(sr)


def _db(x: float) -> float:
    return round(20.0 * np.log10(max(float(x), 1e-12)), 6)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    x, y = a[:n].reshape(-1), b[:n].reshape(-1)
    d = float(np.linalg.norm(x) * np.linalg.norm(y))
    return 0.0 if d <= 1e-12 else float(np.dot(x, y) / d)


def _metrics(audio: np.ndarray, parent: np.ndarray) -> dict:
    n = min(len(audio), len(parent))
    audio, parent = audio[:n], parent[:n]
    rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))
    peak = float(np.max(np.abs(audio)) if audio.size else 0.0)
    active = float(np.mean(np.max(np.abs(audio), axis=1) > 1e-4)) if len(audio) else 0.0
    return {
        'rms_dbfs': _db(rms),
        'peak_dbfs': _db(peak),
        'active_ratio': round(active, 6),
        'parent_cosine': round(_cos(audio, parent), 6),
    }


def _tail(path: Path, lines: int = 120) -> str:
    try:
        return '\n'.join(path.read_text(encoding='utf-8', errors='replace').splitlines()[-lines:])
    except Exception:
        return ''


def _emit(progress, message: str, percent: int) -> None:
    print(f'[wind_brass_v2] {message} ({percent}%)', flush=True)
    if progress:
        try:
            progress(message, percent)
        except Exception as exc:
            print(f'[wind_brass_v2] progress callback failed: {exc!r}', flush=True)


def _run_polled(
    cmd: list[str],
    *,
    cwd: Path | None,
    timeout: int,
    log_path: Path,
    progress,
    stage_name: str,
    start_percent: int,
    end_percent: int,
    heartbeat_seconds: int = 15,
) -> tuple[int, float]:
    started = time.monotonic()
    deadline = started + timeout
    next_heartbeat = started
    env = os.environ.copy()
    env.setdefault('PYTHONUNBUFFERED', '1')

    with log_path.open('w', encoding='utf-8') as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
        while True:
            rc = proc.poll()
            now = time.monotonic()
            elapsed = now - started
            if rc is not None:
                _emit(progress, f'{stage_name} finished in {elapsed:.1f}s', end_percent)
                return int(rc), elapsed

            if now >= deadline:
                try:
                    os.killpg(proc.pid, 15)
                    proc.wait(timeout=10)
                except Exception:
                    try:
                        os.killpg(proc.pid, 9)
                    except Exception:
                        proc.kill()
                raise subprocess.TimeoutExpired(cmd, timeout)

            if now >= next_heartbeat:
                span = max(end_percent - start_percent, 1)
                soft_fraction = min(elapsed / max(float(timeout), 1.0), 0.90)
                pct = start_percent + int(span * soft_fraction)
                _emit(progress, f'{stage_name} running — {elapsed:.0f}s elapsed', min(pct, end_percent - 1))
                next_heartbeat = now + heartbeat_seconds

            time.sleep(1.0)


def build_wind_brass_decomposition_v2(payload: dict, progress=None) -> dict:
    job_started = time.monotonic()
    stage_timings: dict[str, float] = {}
    last_stage = 'initialise'

    def mark(stage: str, message: str, percent: int) -> None:
        nonlocal last_stage
        last_stage = stage
        _emit(progress, message, percent)

    audio_url = str(payload.get('audio_url') or payload.get('source_url') or '').strip()
    if not audio_url:
        return {'ok': False, 'mode': 'wind_brass_decomposition_v2', 'error': 'audio_url is required'}
    if not MEGA53_CONFIG.is_file() or not MEGA53_CHECKPOINT.is_file():
        return {'ok': False, 'mode': 'wind_brass_decomposition_v2', 'failed_stage': 'model_setup', 'error': 'Baked Mega53 model files are missing'}

    timeout = int(payload.get('timeout_seconds') or 1800)
    heartbeat_seconds = max(5, int(payload.get('heartbeat_seconds') or 15))
    rms_gate = float(payload.get('child_rms_gate_dbfs') or -40.0)
    parent_cos_gate = float(payload.get('child_parent_cosine_gate') or 0.55)
    overlap_gate = float(payload.get('child_pairwise_cosine_gate') or 0.20)
    residual_keep_gate = float(payload.get('residual_keep_gate_db') or -35.0)

    repo_dir = Path(str(payload.get('mss_repo_dir') or '/opt/music-source-separation-training'))
    mark('model_setup', 'Resolving baked models', 2)
    t0 = time.monotonic()
    sw_config, sw_checkpoint, sw_installed = _resolve_model_files(Path(str(payload.get('model_dir') or '/models/bs_roformer_sw')), progress=progress)
    stage_timings['model_setup'] = round(time.monotonic() - t0, 3)

    with tempfile.TemporaryDirectory(prefix='litelabs_wind_brass_v2_') as temp:
        root = Path(temp)
        srcdir, swout, parentdir, outdir, logdir = [root / n for n in ('src','sw','parent','mega53','logs')]
        for d in (srcdir, swout, parentdir, outdir, logdir):
            d.mkdir(parents=True, exist_ok=True)

        try:
            mark('download', 'Downloading source audio', 5)
            t0 = time.monotonic()
            name = unquote(Path(urlparse(audio_url).path).name) or 'track.flac'
            downloaded = root / name
            _download(audio_url, downloaded)
            stage_timings['download'] = round(time.monotonic() - t0, 3)

            mark('convert', 'Converting source to stereo 44.1 kHz WAV', 8)
            t0 = time.monotonic()
            source = srcdir / f'{Path(name).stem}.wav'
            conv_log = logdir / 'ffmpeg.log'
            with conv_log.open('w', encoding='utf-8') as log:
                conv = subprocess.run(
                    ['ffmpeg','-y','-i',str(downloaded),'-ar','44100','-ac','2',str(source)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=300,
                )
            stage_timings['convert'] = round(time.monotonic() - t0, 3)
            if conv.returncode != 0:
                return {'ok': False, 'mode': 'wind_brass_decomposition_v2', 'failed_stage': 'convert', 'runtime_log': _tail(conv_log), 'stage_timings': stage_timings}

            mark('parent_separation', 'Starting BS-RoFormer Other separation', 12)
            sw_log = logdir / 'bs_roformer.log'
            rc, elapsed = _run_polled(
                ['bs-roformer-infer','--config_path',str(sw_config),'--model_path',str(sw_checkpoint),'--input_folder',str(srcdir),'--store_dir',str(swout)],
                cwd=None,
                timeout=timeout,
                log_path=sw_log,
                progress=progress,
                stage_name='BS-RoFormer Other separation',
                start_percent=12,
                end_percent=42,
                heartbeat_seconds=heartbeat_seconds,
            )
            stage_timings['parent_separation'] = round(elapsed, 3)
            if rc != 0:
                return {'ok': False, 'mode': 'wind_brass_decomposition_v2', 'failed_stage': 'parent_separation', 'runtime_log': _tail(sw_log), 'stage_timings': stage_timings}

            mark('collect_parent', 'Collecting Other parent', 44)
            matches = [p for p in swout.rglob('*.wav') if p.name.lower().endswith('_other.wav')]
            if not matches:
                matches = [p for p in swout.rglob('*.wav') if 'other' in p.name.lower()]
            if not matches:
                return {'ok': False, 'mode': 'wind_brass_decomposition_v2', 'failed_stage': 'collect_parent', 'error': 'SW Other stem not found', 'stage_timings': stage_timings}

            t0 = time.monotonic()
            parent_audio, sr = _read(matches[0])
            parent_path = parentdir / 'other.wav'
            sf.write(parent_path, parent_audio.astype(np.float32), sr, subtype='FLOAT')
            stage_timings['collect_parent'] = round(time.monotonic() - t0, 3)

            mark('mega53', 'Starting Mega53 saxophone/trumpet separation', 48)
            mega_log = logdir / 'mega53.log'
            rc, elapsed = _run_polled(
                [
                    'python', str(repo_dir/'inference.py'), '--model_type','bs_roformer',
                    '--config_path',str(MEGA53_CONFIG), '--start_check_point',str(MEGA53_CHECKPOINT),
                    '--input_folder',str(parentdir), '--store_dir',str(outdir), '--device_ids','0',
                    '--disable_detailed_pbar','--filename_template','{file_name}/{instr}'
                ],
                cwd=repo_dir,
                timeout=timeout,
                log_path=mega_log,
                progress=progress,
                stage_name='Mega53 separation',
                start_percent=48,
                end_percent=88,
                heartbeat_seconds=heartbeat_seconds,
            )
            stage_timings['mega53'] = round(elapsed, 3)
            if rc != 0:
                return {'ok': False, 'mode': 'wind_brass_decomposition_v2', 'failed_stage': 'mega53', 'runtime_log': _tail(mega_log), 'stage_timings': stage_timings}

            mark('analysis', 'Analysing saxophone/trumpet ownership and residual', 90)
            t0 = time.monotonic()
            files = [p for p in outdir.rglob('*') if p.is_file() and p.suffix.lower() in {'.wav','.flac'}]
            by_name = {p.stem.lower().replace('_','-'): p for p in files}
            loaded = {}
            evidence = []
            for target in TARGETS:
                path = by_name.get(target)
                if not path:
                    evidence.append({'instrument': target, 'approved': False, 'reason': 'missing_output'})
                    continue
                audio, _ = _read(path)
                loaded[target] = audio
                metrics = _metrics(audio, parent_audio)
                evidence.append({'instrument': target, **metrics})

            overlap = None
            if all(t in loaded for t in TARGETS):
                overlap = round(_cos(loaded['saxophone'], loaded['trumpet']), 6)

            approved = []
            for item in evidence:
                if 'rms_dbfs' not in item:
                    continue
                reasons = []
                if item['rms_dbfs'] < rms_gate:
                    reasons.append('too_quiet')
                if item['parent_cosine'] < parent_cos_gate:
                    reasons.append('weak_parent_relation')
                if overlap is not None and abs(overlap) > overlap_gate:
                    reasons.append('pairwise_overlap_too_high')
                item['approved'] = not reasons
                item['rejection_reasons'] = reasons
                if item['approved']:
                    approved.append(item['instrument'])

            if approved:
                n = min([len(parent_audio)] + [len(loaded[x]) for x in approved])
                parent = parent_audio[:n]
                child_sum = np.sum(np.stack([loaded[x][:n] for x in approved], axis=0), axis=0)
                residual = parent - child_sum
                parent_rms = float(np.sqrt(np.mean(parent * parent) + 1e-12))
                residual_rms = float(np.sqrt(np.mean(residual * residual) + 1e-12))
                residual_relative_db = _db(residual_rms / max(parent_rms, 1e-12))
                residual_metrics = _metrics(residual, parent)
                child_sum_cos = round(_cos(parent, child_sum), 6)
                residual_is_meaningful = residual_relative_db >= residual_keep_gate
            else:
                residual_relative_db = 0.0
                residual_metrics = None
                child_sum_cos = 0.0
                residual_is_meaningful = True

            partial_decomposition = bool(approved) and residual_is_meaningful
            export = list(approved)
            if residual_is_meaningful:
                export.append('other_residual')
            stage_timings['analysis'] = round(time.monotonic() - t0, 3)
            stage_timings['total'] = round(time.monotonic() - job_started, 3)

            mark('complete', 'Wind/brass partial decomposition decision complete', 100)
            return {
                'ok': True,
                'mode': 'wind_brass_decomposition_v2',
                'schema_version': 3,
                'research_only': True,
                'audio_url': audio_url,
                'parent': {**_metrics(parent_audio, parent_audio), 'source': 'BS-RoFormer-SW Other'},
                'children': evidence,
                'approved_children': approved,
                'pairwise_sax_trumpet_cosine': overlap,
                'residual': {
                    'relative_to_parent_db': residual_relative_db,
                    'metrics': residual_metrics,
                    'meaningful': residual_is_meaningful,
                },
                'diagnostics': {
                    'approved_children_sum_vs_parent_cosine': child_sum_cos,
                    'child_rms_gate_dbfs': rms_gate,
                    'child_parent_cosine_gate': parent_cos_gate,
                    'child_pairwise_cosine_gate': overlap_gate,
                    'residual_keep_gate_db': residual_keep_gate,
                    'heartbeat_seconds': heartbeat_seconds,
                    'stage_timings': stage_timings,
                },
                'export_decision': {
                    'partial_decomposition_passed': partial_decomposition,
                    'drop_original_other_parent': partial_decomposition,
                    'export': export if partial_decomposition else ['other'],
                    'policy': 'approved children plus meaningful residual; otherwise preserve original parent',
                },
                'sw_model_auto_installed': sw_installed,
                'mega53_baked': True,
                'warning': 'Mega53 remains research evidence. Promotion still requires listening and broader multi-track validation.'
            }

        except subprocess.TimeoutExpired as exc:
            stage_timings['total'] = round(time.monotonic() - job_started, 3)
            return {
                'ok': False,
                'mode': 'wind_brass_decomposition_v2',
                'failed_stage': last_stage,
                'error': f'internal subprocess timeout after {exc.timeout}s',
                'stage_timings': stage_timings,
            }
        except Exception as exc:
            stage_timings['total'] = round(time.monotonic() - job_started, 3)
            return {
                'ok': False,
                'mode': 'wind_brass_decomposition_v2',
                'failed_stage': last_stage,
                'error': repr(exc),
                'stage_timings': stage_timings,
            }
