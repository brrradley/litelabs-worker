FROM ghcr.io/brrradley/litelabs-research-worker:94dbad201e293bb1fe1cba5c5d0e98e2784375ed

ENV PYTHONUNBUFFERED=1 \
    STEMFORGE_MODEL_DIR=/models/bs_roformer_sw \
    LITELABS_AUDIO_SEPARATOR_MODEL_DIR=/models/audio_separator \
    LITELABS_ENGINE=LiteLABS-Experimental-Children \
    LITELABS_RELEASE=3.0.0-beta

WORKDIR /app

# Verified routed runtime provides MSS, Mega53, DrumSep and baked SW.
# Main is our real-world v3 beta test harness: clean BS-RoFormer parents stay
# at ZIP root while candidate child stems are isolated in /experimental/.
COPY master_pack.py /app/legacy_master_pack.py
COPY master_pack_v2.py /app/master_pack.py
COPY handler.py /app/handler.py
COPY litelabs_live_patch.py /app/litelabs_live_patch.py
COPY litelabs_quality_status_patch.py /app/litelabs_quality_status_patch.py
COPY experimental_children_v1.py /app/experimental_children_v1.py
COPY litelabs_experimental_main_patch.py /app/litelabs_experimental_main_patch.py
COPY litelabs_family_first_patch.py /app/litelabs_family_first_patch.py
COPY litelabs_quality_router_patch.py /app/litelabs_quality_router_patch.py
COPY litelabs_progress_percent_patch.py /app/litelabs_progress_percent_patch.py
COPY litelabs_v3_fast_router_vocals_patch.py /app/litelabs_v3_fast_router_vocals_patch.py

# Candidate child-separation assets.
RUN mkdir -p /models/drumsep_5stem /models/sax_demucs /models/audio_separator /models/karaoke_bs_roformer \
    && python - <<'PY'
from pathlib import Path
import hashlib
import requests

assets = [
    (
        'https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/bbd47058b34a68c370e460bb9b3fe426222f5a30/mdx23c/mdx23c_drumsep_5stem_aufr33_jarredou.ckpt?download=true',
        Path('/models/drumsep_5stem/mdx23c_drumsep_5stem_aufr33_jarredou.ckpt'),
        '1f8e636fb674b88a52c8399fde9a4ebe2b72b065ca07eed4e03ab1c9f0bfb2e0',
    ),
    (
        'https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/eef67d3ac3fd64c144473d72887eb47a55aef7a6/mdx23c/mdx23c_drumsep_5stem_aufr33_jarredou_config.yaml?download=true',
        Path('/models/drumsep_5stem/mdx23c_drumsep_5stem_aufr33_jarredou_config.yaml'),
        None,
    ),
    (
        'https://huggingface.co/xavriley/demucs_v3_saxophone_separation/resolve/b9aab0e3eb2c1df749989ca321a46e7a2b54e214/filosax_demucs_v3_14.22_SDR.th?download=true',
        Path('/models/sax_demucs/filosax_demucs_v3_14.22_SDR.th'),
        '4a801bc00f0e21d476edef1d9fa25dbcb088e3bae0a4092e590167ae26dbe360',
    ),
    (
        'https://huggingface.co/Blane187/all_public_uvr_models/resolve/main/17_HP-Wind_Inst-UVR.pth?download=true',
        Path('/models/audio_separator/17_HP-Wind_Inst-UVR.pth'),
        'acc6d472b4b478da9c9ab5af45b167749e05a7f65b30c7d5988b3700a513aeee',
    ),
    (
        'https://huggingface.co/becruily/bs-roformer-karaoke/resolve/main/bs_roformer_karaoke_frazer_becruily.ckpt?download=true',
        Path('/models/karaoke_bs_roformer/model.ckpt'),
        'eb90ee24c1154d83fbcfd27e96182f19e061557cc6e4746953125e08c29389f9',
    ),
    (
        'https://huggingface.co/becruily/bs-roformer-karaoke/resolve/main/config_karaoke_frazer_becruily.yaml?download=true',
        Path('/models/karaoke_bs_roformer/config.yaml'),
        None,
    ),
]
for url, path, expected in assets:
    if not path.exists():
        with requests.get(url, stream=True, timeout=(30, 1800)) as r:
            r.raise_for_status()
            with path.open('wb') as f:
                for chunk in r.iter_content(4 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
    if expected:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
                h.update(chunk)
        actual = h.hexdigest()
        if actual != expected:
            raise RuntimeError(f'candidate model hash mismatch for {path.name}: {actual}')
print('main experimental child candidate models baked')
PY

# The published saxophone checkpoint was exported for Demucs 3.0.6. Keep an
# isolated launcher, reuse the image's CUDA-enabled torch/torchaudio, and add
# only Demucs' minimal separation-time dependencies so the main stack is not
# downgraded or replaced.
RUN python -m venv --system-site-packages /opt/demucs3 \
    && /opt/demucs3/bin/pip install --no-deps 'demucs==3.0.6' \
    && /opt/demucs3/bin/pip install 'dora-search' 'einops' 'julius>=0.2.3' 'lameenc>=1.2' 'openunmix' 'pyyaml' 'tqdm' \
    && /opt/demucs3/bin/python - <<'PY'
import demucs
from demucs.pretrained import get_model
import dora
print('Demucs v3 sax runtime ready:', demucs.__file__)
print('Dora runtime ready:', dora.__file__)
PY

RUN python /app/litelabs_live_patch.py \
    && python /app/litelabs_quality_status_patch.py \
    && python /app/litelabs_experimental_main_patch.py \
    && python /app/litelabs_family_first_patch.py \
    && python /app/litelabs_progress_percent_patch.py \
    && python /app/litelabs_quality_router_patch.py \
    && python /app/litelabs_v3_fast_router_vocals_patch.py \
    && python -m py_compile /app/handler.py /app/routed_extraction_v1.py /app/experimental_children_v1.py /app/drum_decomposition_v1.py /app/wind_brass_decomposition_v2.py /app/sw_residual_allocator.py \
    && python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, '/app')
import master_pack
import experimental_children_v1
assert master_pack.ENGINE_NAME == 'BS-RoFormer-SW'
assert hasattr(experimental_children_v1, 'build_experimental_children_v1')
source = Path('/app/experimental_children_v1.py').read_text(encoding='utf-8')
assert 'common_export_gain' in source
assert 'centers = (0.18, 0.50, 0.82)' in source
assert 'Running Mega53 Instrument Inventory' in source
assert 'family_route = ' in source
assert 'DEMUCS3_PYTHON' in source
assert 'KARAOKE_CHECKPOINT' in source
assert '_lead_vocals.flac' in source
assert '_backing_vocals.flac' in source
assert '"drums": "percussion"' in source
assert '"guitar": "strings"' in source
assert '"piano": "keys"' in source
assert '_write_readme(experimental' in source
assert '(experimental / f"{track}_EXPERIMENTAL_REPORT.json")' in source
progress_source = Path('/app/wind_brass_decomposition_v2.py').read_text(encoding='utf-8')
assert "f'{stage_name} — {pct}%'" in progress_source
assert 'elapsed:.0f}s elapsed' not in progress_source
handler_source = Path('/app/handler.py').read_text(encoding='utf-8')
assert 'experimental_children_v1' in handler_source
assert Path('/opt/demucs3/bin/python').is_file()
assert Path('/models/bs_roformer_sw/BS-Rofo-SW-Fixed.ckpt').is_file()
assert Path('/models/drumsep_5stem/mdx23c_drumsep_5stem_aufr33_jarredou.ckpt').is_file()
assert Path('/models/mss_training/mvsep-mega53/model.ckpt').is_file()
assert Path('/models/sax_demucs/filosax_demucs_v3_14.22_SDR.th').is_file()
assert Path('/models/audio_separator/17_HP-Wind_Inst-UVR.pth').is_file()
assert Path('/models/karaoke_bs_roformer/model.ckpt').is_file()
assert Path('/models/karaoke_bs_roformer/config.yaml').is_file()
print('LiteLABS v3 beta fast adaptive child image ready')
PY

# Execute the real final handler startup path during the image build, but
# replace RunPod's blocking serverless start call with a capture function.
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

config = captured.get('config') or {}
assert callable(config.get('handler')), 'handler.py did not register a callable RunPod handler'
print('LiteLABS serverless boot smoke test passed')
PY

CMD ["python", "-u", "/app/handler.py"]
