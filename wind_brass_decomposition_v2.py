from __future__ import annotations

import subprocess
import tempfile
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


def build_wind_brass_decomposition_v2(payload: dict, progress=None) -> dict:
    audio_url = str(payload.get('audio_url') or payload.get('source_url') or '').strip()
    if not audio_url:
        return {'ok': False, 'mode': 'wind_brass_decomposition_v2', 'error': 'audio_url is required'}
    if not MEGA53_CONFIG.is_file() or not MEGA53_CHECKPOINT.is_file():
        return {'ok': False, 'mode': 'wind_brass_decomposition_v2', 'failed_stage': 'model_setup', 'error': 'Baked Mega53 model files are missing'}

    timeout = int(payload.get('timeout_seconds') or 1800)
    rms_gate = float(payload.get('child_rms_gate_dbfs') or -40.0)
    parent_cos_gate = float(payload.get('child_parent_cosine_gate') or 0.55)
    overlap_gate = float(payload.get('child_pairwise_cosine_gate') or 0.20)
    residual_keep_gate = float(payload.get('residual_keep_gate_db') or -35.0)

    repo_dir = Path(str(payload.get('mss_repo_dir') or '/opt/music-source-separation-training'))
    sw_config, sw_checkpoint, sw_installed = _resolve_model_files(Path(str(payload.get('model_dir') or '/models/bs_roformer_sw')), progress=progress)

    with tempfile.TemporaryDirectory(prefix='litelabs_wind_brass_v2_') as temp:
        root = Path(temp)
        srcdir, swout, parentdir, outdir = [root / n for n in ('src','sw','parent','mega53')]
        for d in (srcdir, swout, parentdir, outdir):
            d.mkdir(parents=True, exist_ok=True)

        name = unquote(Path(urlparse(audio_url).path).name) or 'track.flac'
        downloaded = root / name
        _download(audio_url, downloaded)
        source = srcdir / f'{Path(name).stem}.wav'
        conv = subprocess.run(['ffmpeg','-y','-i',str(downloaded),'-ar','44100','-ac','2',str(source)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300)
        if conv.returncode != 0:
            return {'ok': False, 'mode': 'wind_brass_decomposition_v2', 'failed_stage': 'convert'}

        if progress:
            progress('Isolating parent Other stem', 20)
        sw = subprocess.run(['bs-roformer-infer','--config_path',str(sw_config),'--model_path',str(sw_checkpoint),'--input_folder',str(srcdir),'--store_dir',str(swout)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        if sw.returncode != 0:
            return {'ok': False, 'mode': 'wind_brass_decomposition_v2', 'failed_stage': 'parent_separation', 'runtime_log': '\n'.join((sw.stdout or '').splitlines()[-100:])}
        matches = [p for p in swout.rglob('*.wav') if p.name.lower().endswith('_other.wav')]
        if not matches:
            matches = [p for p in swout.rglob('*.wav') if 'other' in p.name.lower()]
        if not matches:
            return {'ok': False, 'mode': 'wind_brass_decomposition_v2', 'failed_stage': 'collect_parent', 'error': 'SW Other stem not found'}

        parent_audio, sr = _read(matches[0])
        parent_path = parentdir / 'other.wav'
        sf.write(parent_path, parent_audio.astype(np.float32), sr, subtype='FLOAT')

        if progress:
            progress('Separating saxophone and trumpet candidates', 48)
        run = subprocess.run([
            'python', str(repo_dir/'inference.py'), '--model_type','bs_roformer',
            '--config_path',str(MEGA53_CONFIG), '--start_check_point',str(MEGA53_CHECKPOINT),
            '--input_folder',str(parentdir), '--store_dir',str(outdir), '--device_ids','0',
            '--disable_detailed_pbar','--filename_template','{file_name}/{instr}'
        ], cwd=repo_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        if run.returncode != 0:
            return {'ok': False, 'mode': 'wind_brass_decomposition_v2', 'failed_stage': 'mega53', 'runtime_log': '\n'.join((run.stdout or '').splitlines()[-120:])}

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

        if progress:
            progress('Wind/brass partial decomposition decision complete', 100)

        return {
            'ok': True,
            'mode': 'wind_brass_decomposition_v2',
            'schema_version': 2,
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
