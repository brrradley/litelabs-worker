FROM ghcr.io/brrradley/litelabs-research-worker:94dbad201e293bb1fe1cba5c5d0e98e2784375ed

ENV PYTHONUNBUFFERED=1 \
    STEMFORGE_MODEL_DIR=/models/bs_roformer_sw \
    LITELABS_AUDIO_SEPARATOR_MODEL_DIR=/models/audio_separator \
    LITELABS_ENGINE=LiteLABS-Experimental-Children \
    LITELABS_RELEASE=3.0.0-beta

WORKDIR /app

# Verified routed runtime provides MSS, Mega53, DrumSep 6-stem and baked SW.
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

# Candidate child-separation assets.
RUN mkdir -p /models/drumsep_5stem /models/sax_demucs /models/audio_separator \
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

# The published saxophone checkpoint was exported for Demucs 3.0.6. Keep a
# tiny isolated launcher that reuses the image's existing torch/audio deps.
RUN python -m venv --system-site-packages /opt/demucs3 \
    && /opt/demucs3/bin/pip install --no-deps 'demucs==3.0.6' \
    && /opt/demucs3/bin/python - <<'PY'
import demucs
from demucs.pretrained import get_model
print('Demucs v3 sax runtime ready:', demucs.__file__)
PY

RUN python /app/litelabs_live_patch.py \
    && python /app/litelabs_quality_status_patch.py \
    && python /app/litelabs_experimental_main_patch.py \
    && python /app/litelabs_family_first_patch.py \
    && python /app/litelabs_quality_router_patch.py \
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
assert 'Running Mega53 Instrument Inventory' in source
assert 'family_route = ' in source
assert 'DEMUCS3_PYTHON' in source
assert '_write_readme(experimental' in source
assert '(experimental / f"{track}_EXPERIMENTAL_REPORT.json")' in source
handler_source = Path('/app/handler.py').read_text(encoding='utf-8')
assert 'experimental_children_v1' in handler_source
assert Path('/opt/demucs3/bin/python').is_file()
assert Path('/models/bs_roformer_sw/BS-Rofo-SW-Fixed.ckpt').is_file()
assert Path('/models/drumsep_5stem/mdx23c_drumsep_5stem_aufr33_jarredou.ckpt').is_file()
assert Path('/models/mss_training/mvsep-mega53/model.ckpt').is_file()
assert Path('/models/sax_demucs/filosax_demucs_v3_14.22_SDR.th').is_file()
assert Path('/models/audio_separator/17_HP-Wind_Inst-UVR.pth').is_file()
print('LiteLABS v3 beta quality router image ready')
PY

CMD ["python", "-u", "/app/handler.py"]
