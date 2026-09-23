FROM ghcr.io/brrradley/litelabs-worker:dc2ef73ab4f28452337100fc79250f27d3ec5272

ARG LITELABS_BUILD_SHA=unknown
ENV LITELABS_BUILD_SHA=${LITELABS_BUILD_SHA}

WORKDIR /app

# Production is layered from the exact green dc2ef73 image. Experimental stays
# on that validated foundation. Basic/Core use the repository's deterministic
# preset source so this overlay is not coupled to historical base-image anchors.
COPY preset_pack.py /app/preset_pack.py
COPY litelabs_parent_preset_analysis_patch.py /app/litelabs_parent_preset_analysis_patch.py
COPY litelabs_chunked_result_upload_patch.py /app/litelabs_chunked_result_upload_patch.py
COPY multilead_research.py /app/multilead_research.py
COPY litelabs_unmixx_multilead_patch.py /app/litelabs_unmixx_multilead_patch.py

RUN rm -rf /opt/unmixx /opt/medleyvox /models/medleyvox \
    && git clone https://github.com/jihoojung0106/unmixx.git /opt/unmixx \
    && cd /opt/unmixx \
    && git checkout 8e750521b5942f4717656cac86a23cf0bd90dea5 \
    && test -s /opt/unmixx/ckpt/best.ckpt \
    && test -s /opt/unmixx/ckpt/conf.yml \
    && cd /app \
    && python /app/litelabs_parent_preset_analysis_patch.py \
    && python /app/litelabs_chunked_result_upload_patch.py \
    && python /app/litelabs_unmixx_multilead_patch.py \
    && python -m py_compile /app/preset_pack.py /app/experimental_children_v1.py /app/multilead_research.py /app/litelabs_parent_preset_analysis_patch.py /app/litelabs_chunked_result_upload_patch.py /app/litelabs_unmixx_multilead_patch.py /app/litelabs_anvuew_karaoke_patch.py \
    && python - <<'PY'
from pathlib import Path

preset = Path('/app/preset_pack.py').read_text(encoding='utf-8')
experimental = Path('/app/experimental_children_v1.py').read_text(encoding='utf-8')
for marker in (
    'parent_analysis_v3',
    '/app/genre_probe.py',
    '/app/instrument_probe.py',
    'Detected instruments:',
    '"instrument_inventory": instrument_inventory',
    '_litelabs_mmss(execution_seconds)',
):
    assert marker in preset, marker

# Chunked POST upload is the production default for both parent and Experimental
# packs. A single PUT is retained only as an explicit diagnostic override.
for source in (preset, experimental):
    assert 'def _litelabs_upload_archive(' in source
    assert 'requested_mode = str(payload.get("result_upload_mode")' in source
    assert 'mode = "single" if requested_mode == "single" else "chunked"' in source
    assert 'LiteLABS chunked result upload' in source
    assert 'requests.post(' in source

# The green base already contains the validated probes and models. Fail the
# image build if that contract ever changes.
for required in (
    '/app/genre_probe.py',
    '/app/instrument_probe.py',
    '/models/essentia/discogs-effnet-bs64-1.pb',
    '/models/essentia/genre_discogs400-discogs-effnet-1.pb',
    '/models/essentia/mtg_jamendo_instrument-discogs-effnet-1.pb',
):
    path = Path(required)
    assert path.is_file() and path.stat().st_size > 0, required

assert Path('/opt/unmixx/ckpt/best.ckpt').is_file()
assert Path('/opt/unmixx/ckpt/conf.yml').is_file()
assert Path('/models/audio_separator/bs_roformer_karaoke_anvuew.ckpt').is_file()
assert Path('/models/audio_separator/bs_roformer_karaoke_anvuew.ckpt').stat().st_size > 0
assert 'bs_roformer_karaoke_anvuew.ckpt' in experimental
assert 'mel_band_roformer_karaoke_becruily.ckpt' not in experimental
assert '"--use_autocast"' in experimental
assert '"--use_torch_compile"' in experimental
assert 'vocal_route_v3_anvuew_ac_compile' in experimental
multi = Path('/app/multilead_research.py').read_text(encoding='utf-8')
assert 'UNMIXX chunked' in multi
assert 'MedleyVox' not in multi
assert 'multi_lead_unmixx' in experimental

print('LiteLABS production overlay verified: analysis + chunked upload + UNMIXX + Anvuew AC+C vocals enabled')
PY
