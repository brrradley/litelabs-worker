# Reuse the already-verified Essentia model bundle from the last deployed image.
# This removes essentia.upf.edu from normal production builds entirely.
FROM ghcr.io/brrradley/litelabs-worker:79b4c30943c0d799527b0f16b136a17f40c75d13 AS essentia_model_assets

FROM ghcr.io/brrradley/litelabs-worker:f6d1cbbff61f1c6ce9cf8d2001c2b9f2819db2c4

ARG LITELABS_BUILD_SHA=unknown
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    LITELABS_BUILD_SHA=${LITELABS_BUILD_SHA}

WORKDIR /app

# Start QA from the repository's known v2 source instead of the already-patched
# copy inherited from the base image. This makes the learning hotfix deterministic.
COPY qa_research.py /app/qa_research.py
COPY litelabs_drum_hats_compat_patch.py /app/litelabs_drum_hats_compat_patch.py
COPY litelabs_locked_vocal_hats_patch.py /app/litelabs_locked_vocal_hats_patch.py
COPY litelabs_vocal_duplicate_guard_patch.py /app/litelabs_vocal_duplicate_guard_patch.py
COPY multilead_research.py /app/multilead_research.py
COPY essentia_research.py /app/essentia_research.py
COPY litelabs_multilead_research_patch.py /app/litelabs_multilead_research_patch.py
COPY litelabs_instrument_inventory_research_patch.py /app/litelabs_instrument_inventory_research_patch.py
COPY litelabs_essentia_research_patch.py /app/litelabs_essentia_research_patch.py
COPY litelabs_public_readme_model_redaction_patch.py /app/litelabs_public_readme_model_redaction_patch.py
COPY litelabs_research_readme_finalizer_patch.py /app/litelabs_research_readme_finalizer_patch.py
COPY litelabs_experimental_pack_only_patch.py /app/litelabs_experimental_pack_only_patch.py
COPY litelabs_qa_learning_hotfix.py /app/litelabs_qa_learning_hotfix.py
COPY litelabs_build_identity_patch.py /app/litelabs_build_identity_patch.py
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

# Experimental Essentia second-opinion detector.
# Model files are copied from a pinned GHCR image that already passed production
# build verification. Runtime jobs never download these assets.
COPY --from=essentia_model_assets /models/essentia /models/essentia

RUN python -m pip install --no-cache-dir "essentia-tensorflow==2.1b6.dev1389" \
    && python - <<'PY'
from pathlib import Path
import hashlib
import json

root = Path("/models/essentia")
expected = {
    "discogs-effnet-bs64-1.pb": 18366619,
    "mtg_jamendo_instrument-discogs-effnet-1.pb": 2706836,
    "mtg_jamendo_instrument-discogs-effnet-1.json": 3382,
    "genre_discogs400-discogs-effnet-1.pb": 2057977,
    "genre_discogs400-discogs-effnet-1.json": 14951,
}
for name, size in expected.items():
    path = root / name
    if not path.is_file():
        raise RuntimeError(f"Missing baked Essentia asset: {name}")
    actual = path.stat().st_size
    if actual != size:
        raise RuntimeError(f"Unexpected Essentia asset size for {name}: {actual} != {size}")

known_hashes = {
    "discogs-effnet-bs64-1.pb": "3ed9af50d5367c0b9c795b294b00e7599e4943244f4cbd376869f3bfc87721b1",
    "mtg_jamendo_instrument-discogs-effnet-1.pb": "2e8c3003c722e098da371b6a1f7ad0ce62fac0dcfc09c7c7997d430941196c2a",
}
for name, expected_hash in known_hashes.items():
    digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if digest != expected_hash:
        raise RuntimeError(f"Essentia SHA256 mismatch for {name}: {digest}")

instrument_meta = json.loads((root / "mtg_jamendo_instrument-discogs-effnet-1.json").read_text())
genre_meta = json.loads((root / "genre_discogs400-discogs-effnet-1.json").read_text())
assert len(instrument_meta["classes"]) >= 40
assert len(genre_meta["classes"]) == 400
print("Pinned local Essentia model bundle verified")
PY

RUN CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2 python - <<'PY'
from pathlib import Path
import json
import numpy as np
from essentia.standard import TensorflowPredictEffnetDiscogs
from essentia_research import _make_classifier

root = Path("/models/essentia")
embedder = TensorflowPredictEffnetDiscogs(
    graphFilename=str(root / "discogs-effnet-bs64-1.pb"),
    output="PartitionedCall:1",
)
instrument_model = _make_classifier(
    root / "mtg_jamendo_instrument-discogs-effnet-1.pb"
)
genre_model = _make_classifier(
    root / "genre_discogs400-discogs-effnet-1.pb"
)

# Constructor-only tests missed the production failure. Run real inference over
# a short deterministic signal so graph creation and prediction are both tested.
signal = np.zeros(16000 * 4, dtype=np.float32)
embeddings = np.asarray(embedder(signal), dtype=np.float32)
if embeddings.ndim != 2 or embeddings.shape[0] == 0:
    raise RuntimeError(f"Invalid EffNet embeddings shape: {embeddings.shape}")

instrument_predictions = np.asarray(instrument_model(embeddings))
genre_predictions = np.asarray(genre_model(embeddings))
instrument_classes = json.loads(
    (root / "mtg_jamendo_instrument-discogs-effnet-1.json").read_text()
)["classes"]
genre_classes = json.loads(
    (root / "genre_discogs400-discogs-effnet-1.json").read_text()
)["classes"]
if instrument_predictions.shape[-1] != len(instrument_classes):
    raise RuntimeError(
        f"Instrument prediction mismatch: {instrument_predictions.shape} vs {len(instrument_classes)}"
    )
if genre_predictions.shape[-1] != len(genre_classes):
    raise RuntimeError(
        f"Genre prediction mismatch: {genre_predictions.shape} vs {len(genre_classes)}"
    )
print(
    "Essentia inference smoke test passed:",
    embeddings.shape,
    instrument_predictions.shape,
    genre_predictions.shape,
)
PY

# Experimental MedleyVox duet/co-lead separator. The model runs at 24 kHz and
# is enabled for the Experimental preset, which is restricted by site usergroups.
RUN python -m pip install --no-cache-dir "asteroid==0.7.0" "pyloudnorm>=0.1.1" "praat-parselmouth>=0.4.5" \
    && rm -rf /opt/medleyvox \
    && git clone https://github.com/SUC-DriverOld/MedleyVox-Inference-WebUI.git /opt/medleyvox \
    && cd /opt/medleyvox \
    && git checkout 16a2f1d484226b6774439e13bc354f8ef1e7b240 \
    && mkdir -p "/models/medleyvox/vocals 238" \
    && python - <<'PY'
from pathlib import Path
import hashlib
import requests

assets = [
    (
        "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocals%20238/vocals.json?download=true",
        Path("/models/medleyvox/vocals 238/vocals.json"),
        None,
    ),
    (
        "https://huggingface.co/Cyru5/MedleyVox/resolve/main/vocals%20238/vocals.pth?download=true",
        Path("/models/medleyvox/vocals 238/vocals.pth"),
        "793a3cbdadfdee2f833301d925938cd184f08b79d53a7e65c7ff01d5f2c6b4d5",
    ),
]
for url, path, expected in assets:
    with requests.get(url, stream=True, timeout=(30, 1800)) as response:
        response.raise_for_status()
        with path.open("wb") as handle:
            for chunk in response.iter_content(4 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
    if expected:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(f"MedleyVox SHA256 mismatch for {path.name}: {digest}")
print("Research MedleyVox model downloaded and verified")
PY

# Apply production changes to known source files. QA is deliberately reset above
# before its hotfix so an inherited partial patch cannot leave a local variable
# defined only on some code paths.
RUN python /app/litelabs_drum_hats_compat_patch.py \
    && python /app/litelabs_locked_vocal_hats_patch.py \
    && python /app/litelabs_vocal_duplicate_guard_patch.py \
    && python /app/litelabs_qa_learning_hotfix.py \
    && python /app/litelabs_multilead_research_patch.py \
    && python /app/litelabs_instrument_inventory_research_patch.py \
    && python /app/litelabs_essentia_research_patch.py \
    && python /app/litelabs_public_readme_model_redaction_patch.py \
    && python /app/litelabs_research_readme_finalizer_patch.py \
    && python /app/litelabs_experimental_pack_only_patch.py \
    && python /app/litelabs_build_identity_patch.py \
    && python -m py_compile /app/handler.py /app/experimental_children_v1.py /app/preset_pack.py /app/qa_research.py /app/litelabs_drum_hats_compat_patch.py /app/litelabs_locked_vocal_hats_patch.py /app/litelabs_vocal_duplicate_guard_patch.py /app/multilead_research.py /app/essentia_research.py /app/litelabs_multilead_research_patch.py /app/litelabs_instrument_inventory_research_patch.py /app/litelabs_essentia_research_patch.py /app/litelabs_public_readme_model_redaction_patch.py /app/litelabs_research_readme_finalizer_patch.py /app/litelabs_experimental_pack_only_patch.py /app/litelabs_qa_learning_hotfix.py /app/litelabs_build_identity_patch.py \
    && python - <<'PY'
from pathlib import Path
import json
import sys
sys.path.insert(0, '/app')
from preset_pack import PRESETS, STEM_LABELS, preset_capabilities

source = Path('/app/experimental_children_v1.py').read_text(encoding='utf-8')
qa_source = Path('/app/qa_research.py').read_text(encoding='utf-8')
handler_source = Path('/app/handler.py').read_text(encoding='utf-8')
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
# The karaoke secondary output is discovered by role, not trusted by its
# human-readable filename. Guard the actual lookup path rather than stale prose.
assert source.count('"secondary"') >= 2
assert 'def _vx_find_output(directory, role)' in source
assert 'best_stems_share_single_parent_pair' in source
assert 'suppressed_as_gain_scaled_duplicate' in source
assert 'gain_scaled_duplicate_of_lead' in source
assert 'public_role_correction' in source
assert 'live_validated_vocal_roles_v3' in source
assert 'lead_plus_backing_equals_sw_vocal_parent' in source
assert '_vx_role_evidence' in source
assert 'direct_backing_path = _vx_find_output(fast_out, "vocals")' in source
assert 'public_lead = np.asarray(parent_vocals - public_backing' in source
assert 'backing_subset_valid' in source
assert '"backing_detected": bool(backing_exported)' in source

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

# Static regression guards for the production UnboundLocalError.
assert 'learning_observation = {' in qa_source
assert '"learning_observation": learning_observation' in qa_source
assert qa_source.index('learning_observation = {') < qa_source.index('"learning_observation": learning_observation')
assert '"drum_children": ("kick", "snare", "toms", "hats")' in qa_source
assert 'heuristic_stem_confidence_v3' in qa_source
assert 'confidence_not_fidelity' in qa_source
assert 'complement_residual' in qa_source
assert 'multi_lead_medleyvox' in source
assert 'no_multi_lead_evidence' in source
assert '"multi_lead_children": ("lead_vocals_a", "lead_vocals_b")' in qa_source
assert 'global_instrument_inventory_v1' in source
assert 'global_inventory_reused_for_family_router' in source
assert 'DETECTED INSTRUMENTS' in source
assert 'essentia_research_second_opinion_v1' in source
essentia_source = Path('/app/essentia_research.py').read_text(encoding='utf-8')
assert '_make_classifier' in essentia_source
assert '_graph_contains' in essentia_source
assert 'read_bytes()' in essentia_source
assert 'import tensorflow as tf' not in essentia_source
assert 'candidates = (' not in essentia_source
assert 'serving_default_model_Placeholder' in essentia_source
assert 'PartitionedCall' in essentia_source
assert 'model/Placeholder' in essentia_source
assert 'model/Sigmoid' in essentia_source
assert 'graph_filename.name.startswith("mtg_jamendo_instrument-")' in essentia_source
assert 'output="PartitionedCall:0"' in essentia_source
assert 'def _cli() -> int:' in essentia_source
assert 'CUDA_VISIBLE_DEVICES' in source
assert 'Essentia subprocess failed' in source
multilead_source = Path('/app/multilead_research.py').read_text(encoding='utf-8')
assert 'model.float()' in multilead_source
assert 'torch.autocast' not in multilead_source
assert 'LITELABS G400 GENRE CANDIDATES' in source
assert 'ESSENTIA GENRE CANDIDATES' not in source
assert 'public_readme_inventory_final_v1' in source
assert 'experimental_pack_only_v3' in source
assert 'curated_experimental_only_v3' in source
assert '"archive_layout": "flat"' in source
assert 'bundle.write(p, arcname=p.name)' in source
assert 'public_detected_by_family' in source
assert 'research_genre' in source
assert '"genre": research_genre' in source
assert '"detected_genre": research_genre' in source
assert '"detected_instruments": sorted(detected_instruments)' in source
assert '"detected_by_family": detected_by_family' in source
assert 'genre=research_genre' in source
assert 'genre_reason=research_genre_reason' in source
assert 'research_genre = "unverified"' in source
assert 'research_genre = str(genre or' not in source
assert 'research_genre_reason = str(genre_reason or' not in source
assert source.index('research_genre = "unverified"') < source.index('global_inventory_report = {')
assert source.index('essentia_report = {') < source.index('global_inventory_report = {')
assert 'dead_or_near_silent' in source
assert '"drums_5stem_kick": "kick"' in source
assert '"wind_brass_family": "wind"' in source
assert '_experimental_stems.zip' in source
assert '_parent_plus_experimental.zip' not in source
assert 'root_parent_files' not in source
assert '"drums_5stem_hats" in lower' in source
assert '_BUILD_SHA = os.getenv("LITELABS_BUILD_SHA"' in handler_source
assert 'result.setdefault("build_sha", _BUILD_SHA)' in handler_source
assert 'metadata_key in ("detected_instruments", "detected_by_family", "genre_top10", "genre_broad_families")' in qa_source

# Required frozen assets.
for path in (
    '/models/audio_separator/config_vocals_becruily.yaml',
    '/models/audio_separator/mel_band_roformer_vocals_becruily.ckpt',
    '/models/audio_separator/config_karaoke_becruily.yaml',
    '/models/audio_separator/mel_band_roformer_karaoke_becruily.ckpt',
    '/models/essentia/discogs-effnet-bs64-1.pb',
    '/models/essentia/mtg_jamendo_instrument-discogs-effnet-1.pb',
    '/models/essentia/genre_discogs400-discogs-effnet-1.pb',
):
    assert Path(path).is_file(), path

print('LiteLABS locked vocal benchmark + hats + deterministic QA + build identity verified')
PY

# Execute the QA function on real temporary audio during the image build. Static
# string checks were not enough to catch the previous local-variable failure.
RUN python - <<'PY'
import sys
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
sys.path.insert(0, '/app')
from qa_research import build_research_qa

with tempfile.TemporaryDirectory(prefix='qa-smoke-') as td:
    root = Path(td)
    sr = 8000
    t = np.arange(sr, dtype=np.float32) / sr
    vocals = (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    other = (0.03 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    source = vocals + other
    source_path = root / 'source.wav'
    vocals_path = root / 'vocals.wav'
    other_path = root / 'other.wav'
    sf.write(source_path, np.column_stack([source, source]), sr, subtype='FLOAT')
    sf.write(vocals_path, np.column_stack([vocals, vocals]), sr, subtype='FLOAT')
    sf.write(other_path, np.column_stack([other, other]), sr, subtype='FLOAT')
    qa = build_research_qa(
        source=source_path,
        stems={'vocals': vocals_path, 'other': other_path},
        model_by_stem={'vocals': 'smoke', 'other': 'smoke'},
        filename='smoke.wav',
        input_size_bytes=source_path.stat().st_size,
        input_format='wav',
        genre='test',
        preset='experimental',
        pipeline_revision='build-smoke',
        extra={'timings_seconds': {'smoke': 0.1}},
        genre_reason='build test',
    )
    assert qa['learning_observation']['phase'] == 'post_delivery'
    assert qa['learning_observation']['recipe']['preset'] == 'experimental'
    assert qa['source_metrics']['active_ratio'] >= 0
print('QA runtime smoke test passed')
PY

# Smoke-test the real final handler startup path without entering RunPod's
# blocking serverless loop.
RUN python - <<'PY'
import os
import runpy
import runpod.serverless
captured = {}
original_start = runpod.serverless.start
def fake_start(config):
    captured['config'] = config
runpod.serverless.start = fake_start
try:
    ns = runpy.run_path('/app/handler.py', run_name='__main__')
finally:
    runpod.serverless.start = original_start
handler = (captured.get('config') or {}).get('handler')
assert callable(handler)
health = handler({'input': {'healthcheck': True}})
assert health.get('build_sha') == os.environ.get('LITELABS_BUILD_SHA', 'unknown')
print('LiteLABS serverless boot + build identity smoke test passed')
PY

CMD ["python", "-u", "/app/handler.py"]
