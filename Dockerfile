FROM ghcr.io/brrradley/litelabs-research-worker:94dbad201e293bb1fe1cba5c5d0e98e2784375ed

ENV PYTHONUNBUFFERED=1 \
    STEMFORGE_MODEL_DIR=/models/bs_roformer_sw \
    LITELABS_AUDIO_SEPARATOR_MODEL_DIR=/models/audio_separator \
    LITELABS_ENGINE=LiteLABS-Routed \
    LITELABS_RELEASE=2.1

WORKDIR /app

# The verified research image already contains the routed extraction runtime,
# MSS training framework, DrumSep, Mega53 and the baked BS-RoFormer-SW model.
# Overlay the production pack builder and production handler so storage,
# callbacks and legacy rollback behaviour remain production-controlled.
COPY master_pack.py /app/legacy_master_pack.py
COPY master_pack_v2.py /app/master_pack.py
COPY handler.py /app/handler.py
COPY litelabs_live_patch.py /app/litelabs_live_patch.py

RUN python /app/litelabs_live_patch.py \
    && python -m py_compile /app/handler.py /app/routed_extraction_v1.py /app/drum_decomposition_v1.py /app/wind_brass_decomposition_v2.py /app/sw_residual_allocator.py \
    && python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, '/app')
import master_pack
import routed_extraction_v1
assert master_pack.ENGINE_NAME == 'BS-RoFormer-SW'
assert hasattr(routed_extraction_v1, 'build_routed_extraction_v1')
assert Path('/models/bs_roformer_sw/BS-Rofo-SW-Fixed.ckpt').is_file()
assert Path('/models/drumsep_mdx23c/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt').is_file()
assert Path('/models/mss_training/mvsep-mega53/model.ckpt').is_file()
print('LiteLABS routed production image ready')
PY

CMD ["python", "-u", "/app/handler.py"]
