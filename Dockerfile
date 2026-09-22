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

RUN python /app/litelabs_parent_preset_analysis_patch.py \
    && python /app/litelabs_chunked_result_upload_patch.py \
    && python -m py_compile /app/preset_pack.py /app/experimental_children_v1.py /app/litelabs_parent_preset_analysis_patch.py /app/litelabs_chunked_result_upload_patch.py \
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

print('LiteLABS production overlay verified: validated analysis + automatic chunked result upload enabled')
PY
