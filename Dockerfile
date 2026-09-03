FROM ghcr.io/brrradley/litelabs-worker:58aea79a4ad81ea592cb18bce8a92bd18f26462c

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /app

# Incremental v3 beta build: all current models/runtimes are already baked into
# the verified 58aea79 image. Apply only lightweight routing/tuning patches so
# routine changes never redownload large checkpoints from external hosts.
COPY litelabs_wind_vr_fast_patch.py /app/litelabs_wind_vr_fast_patch.py
COPY litelabs_v3_inventory_recall_patch.py /app/litelabs_v3_inventory_recall_patch.py
COPY litelabs_v3_child_export_fix_patch.py /app/litelabs_v3_child_export_fix_patch.py
COPY demucs3_legacy_loader.py /app/demucs3_legacy_loader.py
COPY preset_pack.py /app/preset_pack.py
COPY litelabs_v3_presets_patch.py /app/litelabs_v3_presets_patch.py

RUN python /app/litelabs_wind_vr_fast_patch.py \
    && python /app/litelabs_v3_inventory_recall_patch.py \
    && python /app/litelabs_v3_child_export_fix_patch.py \
    && python /app/litelabs_v3_presets_patch.py \
    && python -m py_compile /app/handler.py /app/experimental_children_v1.py /app/demucs3_legacy_loader.py /app/preset_pack.py \
    && python - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, '/app')
from preset_pack import PRESETS, PARENT_LABELS, preset_capabilities

source = Path('/app/experimental_children_v1.py').read_text(encoding='utf-8')
handler = Path('/app/handler.py').read_text(encoding='utf-8')
assert '"--vr_batch_size", "8"' in source
assert '"--vr_window_size", "1024"' in source
assert '"--use_autocast"' in source
assert 'centers = (0.08, 0.25, 0.42, 0.58, 0.75, 0.92)' in source
assert 'segment_len = max(1, int(other_sr * 3.0))' in source
assert '>= -45.0' in source
assert '>= 0.12' in source
assert 'run_sax_specialist' in source
assert 'Path(SAX_MODEL).stem' in source
assert '/app/demucs3_legacy_loader.py' in source
assert 'wind_output_ok = False' in source
assert '"output_validated": bool(wind_output_ok)' in source
loader = Path('/app/demucs3_legacy_loader.py').read_text(encoding='utf-8')
assert 'weights_only' in loader and 'False' in loader
assert PRESETS['basic'] == ('instrumental', 'vocals')
assert PRESETS['core'] == ('vocals', 'percussion', 'bass', 'strings', 'keys', 'other')
assert 'lead_vocals' in PRESETS['experimental'] and 'saxophone' in PRESETS['experimental']
assert PARENT_LABELS['drums'] == 'percussion'
assert PARENT_LABELS['guitar'] == 'strings'
assert PARENT_LABELS['piano'] == 'keys'
capabilities = preset_capabilities()
assert capabilities['schema_version'] == 1
assert [item['id'] for item in capabilities['presets']] == ['basic', 'core', 'experimental']
assert capabilities['presets'][0]['stems'] == ['Instrumental', 'Vocals']
assert capabilities['presets'][2]['stems'][-2:] == ['Wind / Brass', 'Saxophone']
assert 'payload.get("capabilities") is True' in handler
assert 'preset_capabilities' in handler
assert 'preset in {"basic", "core"}' in handler
assert 'preset not in {"basic", "core", "experimental"}' in handler
assert '"presets": ["basic", "core", "experimental"]' in handler
assert Path('/models/drumsep_5stem/mdx23c_drumsep_5stem_aufr33_jarredou.ckpt').is_file()
assert Path('/models/mss_training/mvsep-mega53/model.ckpt').is_file()
assert Path('/models/sax_demucs/filosax_demucs_v3_14.22_SDR.th').is_file()
assert Path('/models/audio_separator/17_HP-Wind_Inst-UVR.pth').is_file()
assert Path('/models/karaoke_bs_roformer/model.ckpt').is_file()
print('LiteLABS v3 BASIC/CORE/EXPERIMENTAL preset image ready')
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
