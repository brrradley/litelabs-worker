FROM ghcr.io/brrradley/litelabs-worker:79b4c30943c0d799527b0f16b136a17f40c75d13 AS genre_model_assets

FROM ghcr.io/brrradley/litelabs-worker:f6d1cbbff61f1c6ce9cf8d2001c2b9f2819db2c4

ARG LITELABS_BUILD_SHA=unknown
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    LITELABS_BUILD_SHA=${LITELABS_BUILD_SHA}

WORKDIR /app

# Preserve the known-good production handler inherited from the base image.
# Only inject the isolated genre_probe request route below.
# Start QA from the repository's known v2 source instead of the already-patched
# copy inherited from the base image. This makes the learning hotfix deterministic.
COPY qa_research.py /app/qa_research.py
COPY litelabs_drum_hats_compat_patch.py /app/litelabs_drum_hats_compat_patch.py
COPY litelabs_locked_vocal_hats_patch.py /app/litelabs_locked_vocal_hats_patch.py
COPY litelabs_vocal_duplicate_guard_patch.py /app/litelabs_vocal_duplicate_guard_patch.py
COPY multilead_research.py /app/multilead_research.py
COPY essentia_research.py /app/essentia_research.py
COPY genre_probe.py /app/genre_probe.py
COPY instrument_probe.py /app/instrument_probe.py
COPY litelabs_genre_probe_handler_patch.py /app/litelabs_genre_probe_handler_patch.py
COPY litelabs_instrument_probe_handler_patch.py /app/litelabs_instrument_probe_handler_patch.py
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

# Isolated G400 genre probe.
# Reuse the already-built model bundle from GHCR so production builds never
# depend on essentia.upf.edu availability.
COPY --from=genre_model_assets /models/essentia /models/essentia
RUN python -m pip install --no-cache-dir --pre essentia-tensorflow \
    && python -m py_compile /app/genre_probe.py /app/instrument_probe.py \
    && test -s /models/essentia/discogs-effnet-bs64-1.pb \
    && test -s /models/essentia/genre_discogs400-discogs-effnet-1.pb \
    && test -s /models/essentia/genre_discogs400-discogs-effnet-1.json \
    && test -s /models/essentia/mtg_jamendo_instrument-discogs-effnet-1.pb \
    && test -s /models/essentia/mtg_jamendo_instrument-discogs-effnet-1.json

# Exercise the exact RunPod genre path during the image build. This is real
# inference, not an import/constructor-only smoke test.
RUN CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2 python - <<'PY'
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

audio = Path("/tmp/g400-smoke.wav")
sr = 16000
t = np.arange(sr * 4, dtype=np.float32) / sr
signal = (
    0.08 * np.sin(2 * np.pi * 110 * t)
    + 0.04 * np.sin(2 * np.pi * 440 * t)
).astype(np.float32)
sf.write(audio, signal, sr, subtype="FLOAT")

completed = subprocess.run(
    ["python", "-u", "/app/genre_probe.py", str(audio)],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    env={
        **__import__("os").environ,
        "CUDA_VISIBLE_DEVICES": "-1",
        "TF_CPP_MIN_LOG_LEVEL": "2",
    },
    timeout=120,
    check=False,
)
if completed.returncode != 0:
    raise RuntimeError(
        "G400 smoke inference failed:\n"
        + (completed.stderr or completed.stdout or "")[-5000:]
    )
lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
result = json.loads(lines[-1])
assert result["ok"] is True
assert result["mode"] == "genre_probe"
assert len(result["top10"]) == 10
assert result["embedding_frames"] > 0
print("G400 genre probe inference smoke test passed:", result["genre"])
PY

# Exercise the exact isolated instrument detector too. The classifier is
# intentionally constructed without hard-coded TensorFlow endpoints.
RUN CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2 python - <<'PY'
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

audio = Path("/tmp/instrument-smoke.wav")
sr = 16000
t = np.arange(sr * 4, dtype=np.float32) / sr
signal = (
    0.08 * np.sin(2 * np.pi * 110 * t)
    + 0.04 * np.sin(2 * np.pi * 440 * t)
).astype(np.float32)
sf.write(audio, signal, sr, subtype="FLOAT")

completed = subprocess.run(
    ["python", "-u", "/app/instrument_probe.py", str(audio)],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    env={
        **__import__("os").environ,
        "CUDA_VISIBLE_DEVICES": "-1",
        "TF_CPP_MIN_LOG_LEVEL": "2",
    },
    timeout=120,
    check=False,
)
if completed.returncode != 0:
    raise RuntimeError(
        "Inst-MTG smoke inference failed:\n"
        + (completed.stderr or completed.stdout or "")[-5000:]
    )
lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
result = json.loads(lines[-1])
assert result["ok"] is True
assert result["mode"] == "instrument_probe"
assert result["class_count"] >= 20
assert len(result["top10"]) == 10
assert result["embedding_frames"] > 0
print(
    "Inst-MTG instrument probe inference smoke test passed:",
    [item["label"] for item in result["top10"][:3]],
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
    && python /app/litelabs_genre_probe_handler_patch.py \
    && python /app/litelabs_instrument_probe_handler_patch.py \
    && python /app/litelabs_build_identity_patch.py \
    && python -m py_compile /app/handler.py /app/experimental_children_v1.py /app/preset_pack.py /app/qa_research.py /app/litelabs_drum_hats_compat_patch.py /app/litelabs_locked_vocal_hats_patch.py /app/litelabs_vocal_duplicate_guard_patch.py /app/multilead_research.py /app/essentia_research.py /app/litelabs_multilead_research_patch.py /app/litelabs_instrument_inventory_research_patch.py /app/litelabs_essentia_research_patch.py /app/litelabs_public_readme_model_redaction_patch.py /app/litelabs_research_readme_finalizer_patch.py /app/litelabs_experimental_pack_only_patch.py /app/litelabs_genre_probe_handler_patch.py /app/litelabs_instrument_probe_handler_patch.py /app/litelabs_qa_learning_hotfix.py /app/litelabs_build_identity_patch.py \
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

# Fast production vocal route: one separator pass on the clean vocal parent.
assert '"benchmark_id": "vocal_route_v2_single_pass"' in source
assert 'mel_band_roformer_karaoke_becruily.ckpt' in source
assert 'mel_band_roformer_vocals_becruily.ckpt' not in source
assert 'vocal_alt_parent' not in source
assert 'vocal_fast_karaoke' not in source
assert 'best_backing = fast_parent - fast_lead' not in source
assert 'Becruily Karaoke vocals output from SW vocal parent' in source
assert 'Becruily Karaoke instrumental output from SW vocal parent' in source
assert '"best_stems_share_single_parent_pair": True' in source
assert 'suppressed_as_gain_scaled_duplicate' in source
assert 'gain_scaled_duplicate_of_lead' in source
assert '"backing_detected": bool(duplicate_analysis["backing_detected"])' in source

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
assert 'disabled_by_default_fast_path' in source
# Multi-lead is research-only and disabled by default; its QA labels must never
# block the production image from building.
assert 'validated_instrument_router_v2' in source
assert 'validated_instrument_router_reused_v2' in source
assert 'Analysing Instrument Inventory' not in source
assert 'residual_inventory_fast_skip_v3' in source
assert 'ANALYSIS' in source
assert 'validated_analysis_pre_stem_v2' in source
assert 'validated_analysis_complete_before_stems_v2' in source
assert '["python", "-u", "/app/genre_probe.py", str(downloaded)]' in source
assert '["python", "-u", "/app/instrument_probe.py", str(downloaded)]' in source
assert source.index('validated_analysis_complete_before_stems_v2') < source.index('rc, elapsed = _run_polled(')
assert '"genre_analysis": genre_report' in source
assert '"instrument_analysis": instrument_report' in source
multilead_source = Path('/app/multilead_research.py').read_text(encoding='utf-8')
assert 'model.float()' in multilead_source
assert 'torch.autocast' not in multilead_source
assert 'Detected genre: {detected_genre}' in source
assert 'genre_report.get("genre")' in source
assert 'instrument_report.get("detected_instruments")' in source
assert '"electricguitar": "Electric Guitar"' in source
assert '"doublebass": "Double Bass"' in source
assert 'LITELABS G400 GENRE CANDIDATES' not in source
assert 'ESSENTIA GENRE CANDIDATES' not in source
assert 'public_readme_inventory_final_v2' in source
assert 'experimental_pack_only_v1' in source
assert '_experimental_stems.zip' in source
assert '_parent_plus_experimental.zip' not in source
assert 'root_parent_files' not in source
assert 'experimental = final / "experimental"' not in source
assert 'track = track[:-17]' in source
assert '"drums_5stem_hats" in lower' in source
assert 'experimental_children_v1' in handler_source
assert 'genre_probe_handler_v1' in handler_source
assert 'instrument_probe_handler_v1' in handler_source
assert '_BUILD_SHA = os.getenv("LITELABS_BUILD_SHA"' in handler_source
assert 'result.setdefault("build_sha", _BUILD_SHA)' in handler_source

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

print('LiteLABS fast routing + single-pass vocals + flat pack + deterministic QA verified')
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
# blocking serverless loop. Also invoke the isolated genre_probe route itself so
# a missing/stale inherited handler can never pass the build again.
RUN python - <<'PY'
import os
import runpy
from pathlib import Path

import numpy as np
import runpod.serverless
import soundfile as sf

captured = {}
original_start = runpod.serverless.start

def fake_start(config):
    captured['config'] = config

runpod.serverless.start = fake_start
try:
    runpy.run_path('/app/handler.py', run_name='__main__')
finally:
    runpod.serverless.start = original_start

handler = (captured.get('config') or {}).get('handler')
assert callable(handler)

health = handler({'input': {'healthcheck': True}})
assert health.get('build_sha') == os.environ.get('LITELABS_BUILD_SHA', 'unknown')

audio = Path('/tmp/handler-g400-smoke.wav')
sr = 16000
t = np.arange(sr * 4, dtype=np.float32) / sr
signal = (
    0.08 * np.sin(2 * np.pi * 110 * t)
    + 0.04 * np.sin(2 * np.pi * 440 * t)
).astype(np.float32)
sf.write(audio, signal, sr, subtype='FLOAT')

probe = handler({
    'input': {
        'mode': 'genre_probe',
        'audio_path': str(audio),
        'filename': audio.name,
        'genre_timeout_seconds': 120,
    }
})
assert probe.get('ok') is True, probe
assert probe.get('mode') == 'genre_probe', probe
assert probe.get('model') == 'G400', probe
assert len(probe.get('top10') or []) == 10, probe

instrument_probe = handler({
    'input': {
        'mode': 'instrument_probe',
        'audio_path': str(audio),
        'filename': audio.name,
        'instrument_timeout_seconds': 120,
    }
})
assert instrument_probe.get('ok') is True, instrument_probe
assert instrument_probe.get('mode') == 'instrument_probe', instrument_probe
assert instrument_probe.get('model') == 'Inst-MTG', instrument_probe
assert len(instrument_probe.get('top10') or []) == 10, instrument_probe
assert instrument_probe.get('class_count', 0) >= 20, instrument_probe

print(
    'LiteLABS serverless boot + isolated probe routes passed:',
    probe.get('genre'),
    [item.get('label') for item in instrument_probe.get('top10', [])[:3]],
)
PY

CMD ["python", "-u", "/app/handler.py"]
