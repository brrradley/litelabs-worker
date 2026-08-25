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
SPECIFIC = ('saxophone','trumpet','trombone','french-horn','tuba','clarinet','flute','oboe','bassoon')
GENERIC = ('brass','wind','woodwind')


def _read(path: Path):
    audio, sr = sf.read(path, always_2d=True, dtype='float32')
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    return audio[:, :2].astype(np.float64), int(sr)


def _db(x: float) -> float:
    return round(20.0 * np.log10(max(float(x), 1e-12)), 6)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a.reshape(-1), b.reshape(-1)
    d = float(np.linalg.norm(x) * np.linalg.norm(y))
    return 0.0 if d <= 1e-12 else float(np.dot(x, y) / d)


def _metrics(audio: np.ndarray, parent: np.ndarray) -> dict:
    n = min(len(audio), len(parent))
    audio, parent = audio[:n], parent[:n]
    rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))
    peak = float(np.max(np.abs(audio)) if audio.size else 0.0)
    active = float(np.mean(np.max(np.abs(audio), axis=1) > 1e-4)) if len(audio) else 0.0
    return {'rms_dbfs': _db(rms), 'peak_dbfs': _db(peak), 'active_ratio': round(active, 6), 'parent_cosine': round(_cos(audio, parent), 6)}


def build_wind_brass_decomposition_v1(payload: dict, progress=None) -> dict:
    audio_url = str(payload.get('audio_url') or payload.get('source_url') or '').strip()
    if not audio_url:
        return {'ok': False, 'mode': 'wind_brass_decomposition_v1', 'error': 'audio_url is required'}
    if not MEGA53_CONFIG.is_file() or not MEGA53_CHECKPOINT.is_file():
        return {'ok': False, 'mode': 'wind_brass_decomposition_v1', 'failed_stage': 'model_setup', 'error': 'Baked Mega53 model files are missing'}

    timeout = int(payload.get('timeout_seconds') or 1800)
    repo_dir = Path(str(payload.get('mss_repo_dir') or '/opt/music-source-separation-training'))
    sw_config, sw_checkpoint, sw_installed = _resolve_model_files(Path(str(payload.get('model_dir') or '/models/bs_roformer_sw')), progress=progress)

    with tempfile.TemporaryDirectory(prefix='litelabs_wind_brass_') as temp:
        root = Path(temp)
        srcdir, swout, parentdir, outdir = [root / n for n in ('src','sw','parent','mega53')]
        for d in (srcdir, swout, parentdir, outdir): d.mkdir(parents=True, exist_ok=True)
        name = unquote(Path(urlparse(audio_url).path).name) or 'track.flac'
        downloaded = root / name
        _download(audio_url, downloaded)
        source = srcdir / f'{Path(name).stem}.wav'
        conv = subprocess.run(['ffmpeg','-y','-i',str(downloaded),'-ar','44100','-ac','2',str(source)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=300)
        if conv.returncode != 0:
            return {'ok': False, 'mode': 'wind_brass_decomposition_v1', 'failed_stage': 'convert'}

        if progress: progress('Isolating parent Other stem', 20)
        sw = subprocess.run(['bs-roformer-infer','--config_path',str(sw_config),'--model_path',str(sw_checkpoint),'--input_folder',str(srcdir),'--store_dir',str(swout)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        if sw.returncode != 0:
            return {'ok': False, 'mode': 'wind_brass_decomposition_v1', 'failed_stage': 'parent_separation', 'runtime_log': '\n'.join((sw.stdout or '').splitlines()[-100:])}
        matches = [p for p in swout.rglob('*.wav') if p.name.lower().endswith('_other.wav')]
        if not matches: matches = [p for p in swout.rglob('*.wav') if 'other' in p.name.lower()]
        if not matches:
            return {'ok': False, 'mode': 'wind_brass_decomposition_v1', 'failed_stage': 'collect_parent', 'error': 'SW Other stem not found'}
        parent_audio, sr = _read(matches[0])
        parent_path = parentdir / 'other.wav'
        sf.write(parent_path, parent_audio.astype(np.float32), sr, subtype='FLOAT')

        if progress: progress('Running Mega53 on Other parent', 48)
        run = subprocess.run(['python',str(repo_dir/'inference.py'),'--model_type','bs_roformer','--config_path',str(MEGA53_CONFIG),'--start_check_point',str(MEGA53_CHECKPOINT),'--input_folder',str(parentdir),'--store_dir',str(outdir),'--device_ids','0','--disable_detailed_pbar','--filename_template','{file_name}/{instr}'], cwd=repo_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        if run.returncode != 0:
            return {'ok': False, 'mode': 'wind_brass_decomposition_v1', 'failed_stage': 'mega53', 'runtime_log': '\n'.join((run.stdout or '').splitlines()[-120:])}

        files = [p for p in outdir.rglob('*') if p.is_file() and p.suffix.lower() in {'.wav','.flac'}]
        by_name = {p.stem.lower().replace('_','-'): p for p in files}
        evidence, loaded = [], {}
        for name in SPECIFIC + GENERIC:
            p = by_name.get(name)
            if not p: continue
            audio, _ = _read(p)
            loaded[name] = audio
            m = _metrics(audio, parent_audio)
            m.update({'instrument': name, 'status': 'audible_candidate' if m['rms_dbfs'] >= -55 and m['active_ratio'] >= 0.01 else 'weak_or_absent'})
            evidence.append(m)
        evidence.sort(key=lambda x: x['rms_dbfs'], reverse=True)

        specific_audible = [x['instrument'] for x in evidence if x['instrument'] in SPECIFIC and x['status']=='audible_candidate']
        pairwise = []
        for i, a in enumerate(specific_audible):
            for b in specific_audible[i+1:]:
                n = min(len(loaded[a]), len(loaded[b]))
                pairwise.append({'a':a,'b':b,'cosine':round(_cos(loaded[a][:n], loaded[b][:n]),6)})
        max_overlap = max([abs(x['cosine']) for x in pairwise], default=0.0)
        if specific_audible:
            n = min([len(parent_audio)] + [len(loaded[x]) for x in specific_audible])
            child_sum = np.sum(np.stack([loaded[x][:n] for x in specific_audible], axis=0), axis=0)
            parent = parent_audio[:n]
            residual = parent - child_sum
            parent_rms = float(np.sqrt(np.mean(parent*parent)+1e-12))
            residual_rms = float(np.sqrt(np.mean(residual*residual)+1e-12))
            sum_cos = round(_cos(parent, child_sum),6)
            residual_db = _db(residual_rms/max(parent_rms,1e-12))
        else:
            sum_cos, residual_db = 0.0, 0.0

        if progress: progress('Wind/brass decomposition diagnostics complete', 100)
        return {
            'ok': True,
            'mode': 'wind_brass_decomposition_v1',
            'schema_version': 1,
            'research_only': True,
            'audio_url': audio_url,
            'parent': {**_metrics(parent_audio, parent_audio), 'source': 'BS-RoFormer-SW Other'},
            'evidence': evidence,
            'specific_audible': specific_audible,
            'pairwise_overlap': pairwise,
            'diagnostics': {'max_specific_pairwise_cosine': round(max_overlap,6), 'specific_children_sum_vs_parent_cosine': sum_cos, 'residual_relative_to_parent_db': residual_db},
            'export_decision': {'replace_parent_other': False, 'reason': 'v1 diagnostic only: Mega53 stems may overlap; require listening/specialist verification before retiring Other'},
            'sw_model_auto_installed': sw_installed,
            'mega53_baked': True,
            'warning': 'Mega53 is discovery/decomposition evidence and its outputs may overlap; this does not yet prove stem ownership.'
        }
