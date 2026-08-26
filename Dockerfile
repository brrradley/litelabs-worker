FROM ghcr.io/brrradley/litelabs-worker:58aea79a4ad81ea592cb18bce8a92bd18f26462c

WORKDIR /app

# Incremental v3 beta build: all current models/runtimes are already baked into
# the verified 58aea79 image. Apply only lightweight routing/tuning patches so
# routine changes never redownload large checkpoints from external hosts.
COPY litelabs_wind_vr_fast_patch.py /app/litelabs_wind_vr_fast_patch.py
COPY litelabs_v3_inventory_recall_patch.py /app/litelabs_v3_inventory_recall_patch.py

RUN python /app/litelabs_wind_vr_fast_patch.py \
    && python /app/litelabs_v3_inventory_recall_patch.py \
    && python -m py_compile /app/handler.py /app/experimental_children_v1.py \
    && python - <<'PY'
from pathlib import Path
source = Path('/app/experimental_children_v1.py').read_text(encoding='utf-8')
assert '"--vr_batch_size", "32"' in source
assert '"--vr_window_size", "1024"' in source
assert '"--use_autocast"' in source
assert 'centers = (0.08, 0.25, 0.42, 0.58, 0.75, 0.92)' in source
assert 'segment_len = max(1, int(other_sr * 3.0))' in source
assert '>= -45.0' in source
assert '>= 0.12' in source
assert 'run_sax_specialist' in source
assert Path('/models/drumsep_5stem/mdx23c_drumsep_5stem_aufr33_jarredou.ckpt').is_file()
assert Path('/models/mss_training/mvsep-mega53/model.ckpt').is_file()
assert Path('/models/sax_demucs/filosax_demucs_v3_14.22_SDR.th').is_file()
assert Path('/models/audio_separator/17_HP-Wind_Inst-UVR.pth').is_file()
assert Path('/models/karaoke_bs_roformer/model.ckpt').is_file()
print('LiteLABS v3 beta incremental fast-wind + inventory-recall image ready')
PY

# Execute the real final handler startup path, but intercept RunPod's blocking
# serverless start call. A green image must still boot the actual handler.
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
