FROM ghcr.io/brrradley/litelabs-worker:f6d1cbbff61f1c6ce9cf8d2001c2b9f2819db2c4

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /app

# Increment from the last verified production image. It already contains the
# current v3 preset/routing/QA/upload stack; this layer only promotes the locked
# vocal benchmark and the in-memory hh+cymbals -> hats output policy.
COPY litelabs_drum_hats_compat_patch.py /app/litelabs_drum_hats_compat_patch.py
COPY litelabs_locked_vocal_hats_patch.py /app/litelabs_locked_vocal_hats_patch.py
COPY litelabs_qa_learning_hotfix.py /app/litelabs_qa_learning_hotfix.py
COPY benchmarks/vocal_benchmark_v1.json /app/benchmarks/vocal_benchmark_v1.json

# Bake the two Becruily MelBand models used by the locked benchmark so customer
# jobs never download multi-GB checkpoints at runtime. URLs are pinned to the
# verified upstream commits and checkpoints are SHA256 checked.
RUN mkdir -p /models/audio_separator /app/benchmarks \
    && python - <<'PY'
from pathlib import Path
import hashlib
import time
import requests

ASSETS = [
    (
        'https://huggingface.co/becruily/mel-band-roformer-vocals/resolve/af457f56e56eb23fa8322929eef6a63b455e5858/config_vocals_becruily.yaml?download=true',
        Path('/models/audio_separator/config_vocals_becruily.yaml'),
        None,
    ),
    (
        'https://huggingface.co/becruily/mel-band-roformer-vocals/resolve/af457f56e56eb23fa8322929eef6a63b455e5858/mel_band_roformer_vocals_becruily.ckpt?download=true',
        Path('/models/audio_separator/mel_band_roformer_vocals_becruily.ckpt'),
        'a05961310cc55fbb901290c2e8be02682942f73522b6ac76bf2ec11e347ed95a',
    ),
    (
        'https://huggingface.co/becruily/mel-band-roformer-karaoke/resolve/0c149975cfaa261c7d87baf54330a9da85bcf888/config_karaoke_becruily.yaml?download=true',
        Path('/models/audio_separator/config_karaoke_becruily.yaml'),
        None,
    ),
    (
        'https://huggingface.co/becruily/mel-band-roformer-karaoke/resolve/0c149975cfaa261c7d87baf54330a9da85bcf888/mel_band_roformer_karaoke_becruily.ckpt?download=true',
        Path('/models/audio_separator/mel_band_roformer_karaoke_becruily.ckpt'),
        'd3aa262ac01df870b9fc033e9c7b6cad33fe04fc9c148b6c40841326a515a0e0',
    ),
]


def download(url: str, path: Path, attempts: int = 5) -> None:
    if path.is_file():
        return
    tmp = path.with_suffix(path.suffix + '.part')
    for attempt in range(1, attempts + 1):
        try:
            tmp.unlink(missing_ok=True)
            with requests.get(url, stream=True, timeout=(30, 1800)) as response:
                response.raise_for_status()
                with tmp.open('wb') as handle:
                    for chunk in response.iter_content(4 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            tmp.replace(path)
            return
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            if attempt >= attempts:
                raise
            wait = min(30, 2 ** attempt)
            print(f'download attempt {attempt}/{attempts} failed for {path.name}: {exc}; retrying in {wait}s', flush=True)
            time.sleep(wait)


for url, path, expected in ASSETS:
    download(url, path)
    if expected:
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected:
            raise RuntimeError(f'SHA256 mismatch for {path.name}: {actual}')
print('Locked Becruily vocal benchmark models baked and verified')
PY

# Normalise the production image, then repair the silent QA telemetry inherited
# from f6d1. QA is post-delivery research evidence and must never fail a finished
# extraction because learning_observation was not initialised.
RUN python /app/litelabs_drum_hats_compat_patch.py \
    && python /app/litelabs_locked_vocal_hats_patch.py \
    && python /app/litelabs_qa_learning_hotfix.py \
    && python -m py_compile /app/handler.py /app/experimental_children_v1.py /app/preset_pack.py /app/qa_research.py /app/litelabs_drum_hats_compat_patch.py /app/litelabs_locked_vocal_hats_patch.py /app/litelabs_qa_learning_hotfix.py \
    && python - <<'PY'
from pathlib import Path
import json
import sys
sys.path.insert(0, '/app')
from preset_pack import PRESETS, STEM_LABELS, preset_capabilities

source = Path('/app/experimental_children_v1.py').read_text(encoding='utf-8')
qa_source = Path('/app/qa_research.py').read_text(encoding='utf-8')
benchmark = json.loads(Path('/app/benchmarks/vocal_benchmark_v1.json').read_text(encoding='utf-8'))

assert benchmark['benchmark_id'] == 'vocal_benchmark_v1'
assert benchmark['status'] == 'locked'
assert benchmark['locked_recipes']['lead_vocals']['quality_score'] == 65.48
assert benchmark['locked_recipes']['backing_vocals']['strict_quality_score'] == 19.77

# Locked vocal production recipes.
assert '"benchmark_id": "vocal_benchmark_v1"' in source
assert '0.25 * np.asarray(sw_vocals_audio' in source
assert '0.75 * np.asarray(alt_vocals_audio' in source
assert 'mel_band_roformer_vocals_becruily.ckpt' in source
assert source.count('mel_band_roformer_karaoke_becruily.ckpt') >= 2
assert 'best_backing = fast_parent - fast_lead' in source
assert 'best_stems_share_single_parent_pair' in source

# HH and cymbals remain one DrumSep inference but are merged before encoding.
assert 'in_memory_hh_plus_cymbals_before_final_encode' in source
assert '"hats":' in source and '"hh"' in source and '"cymbals"' in source
assert 'for name, audio in final_children.items()' in source
assert 'hats' in PRESETS['experimental']
assert 'hi_hats' not in PRESETS['experimental']
assert 'cymbals' not in PRESETS['experimental']
assert STEM_LABELS['hats'] == 'Hats'
capabilities = preset_capabilities()
experimental = next(item for item in capabilities['presets'] if item['id'] == 'experimental')
assert 'Hats' in experimental['stems']
assert 'Hi-Hats' not in experimental['stems']
assert 'Cymbals' not in experimental['stems']

# Regression guard for the production failure e9532db0...-e2.
assert 'learning_observation = {' in qa_source
assert '"learning_observation": learning_observation' in qa_source
assert qa_source.index('learning_observation = {') < qa_source.index('"learning_observation": learning_observation')
assert '"drum_children": ("kick", "snare", "toms", "hats")' in qa_source
assert '"drums_5stem_hats" in lower' in source

# Required frozen assets.
for path in (
    '/models/audio_separator/config_vocals_becruily.yaml',
    '/models/audio_separator/mel_band_roformer_vocals_becruily.ckpt',
    '/models/audio_separator/config_karaoke_becruily.yaml',
    '/models/audio_separator/mel_band_roformer_karaoke_becruily.ckpt',
):
    assert Path(path).is_file(), path

print('LiteLABS locked vocal benchmark + hats + QA learning hotfix verified')
PY

# Smoke-test the real final handler startup path without entering RunPod's
# blocking serverless loop.
RUN python - <<'PY'
import runpy
import runpod.serverless
captured = {}
original_start = runpod.serverless.start
def fake_start(config):
    captured['config'] = config
runpod.serverless.start = fake_start
try:
    runpy.run_path('/app/handler.py', run_name='__main__')
finally:
    runpod.serverless.start = original_start
assert callable((captured.get('config') or {}).get('handler'))
print('LiteLABS serverless boot smoke test passed')
PY

CMD ["python", "-u", "/app/handler.py"]
