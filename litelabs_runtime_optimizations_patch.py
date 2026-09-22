from pathlib import Path

qa_path = Path('/app/qa_research.py')
exp_path = Path('/app/experimental_children_v1.py')

qa = qa_path.read_text(encoding='utf-8')

# Runtime optimisation v1:
# QA is observational only and must not dominate customer delivery time. Sample
# up to 60 seconds spread across the full track rather than decoding every frame
# of every stem. This changes QA telemetry cost, not audio separation or exports.
old_read = '''def _read(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    return np.asarray(audio, dtype=np.float32), int(sr)
'''
new_read = '''QA_SAMPLE_SECONDS = 60.0
QA_SAMPLE_WINDOWS = 12


def _duration_seconds(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.frames) / max(float(info.samplerate), 1.0)


def _read(path: Path) -> tuple[np.ndarray, int]:
    info = sf.info(str(path))
    sr = int(info.samplerate)
    frames = int(info.frames)
    max_frames = int(QA_SAMPLE_SECONDS * sr)

    if frames <= max_frames:
        audio, read_sr = sf.read(
            str(path),
            always_2d=True,
            dtype="float32",
        )
        mono = np.mean(
            np.asarray(audio, dtype=np.float32),
            axis=1,
            dtype=np.float32,
        )
        return mono, int(read_sr)

    window_frames = max(1, max_frames // QA_SAMPLE_WINDOWS)
    last_start = max(0, frames - window_frames)
    starts = np.linspace(
        0,
        last_start,
        QA_SAMPLE_WINDOWS,
        dtype=np.int64,
    )

    chunks = []
    with sf.SoundFile(str(path), mode="r") as handle:
        for start in starts:
            handle.seek(int(start))
            chunk = handle.read(
                window_frames,
                dtype="float32",
                always_2d=True,
            )
            if len(chunk):
                chunks.append(
                    np.mean(
                        np.asarray(chunk, dtype=np.float32),
                        axis=1,
                        dtype=np.float32,
                    )
                )

    if not chunks:
        return np.zeros(0, dtype=np.float32), sr
    return np.concatenate(chunks).astype(np.float32, copy=False), sr
'''
if old_read not in qa:
    raise RuntimeError('Could not locate QA read helper')
qa = qa.replace(old_read, new_read, 1)

source_anchor = '''    source_audio, source_sr = _read(source)
    source_mono = _mono(source_audio)
'''
if source_anchor not in qa:
    raise RuntimeError('Could not locate QA source anchor')
qa = qa.replace(
    source_anchor,
    source_anchor + '    source_duration_seconds = round(_duration_seconds(source), 3)\n',
    1,
)
qa = qa.replace(
    'round(len(source_mono) / max(source_sr, 1), 3)',
    'source_duration_seconds',
)

qa_path.write_text(qa, encoding='utf-8')

exp = exp_path.read_text(encoding='utf-8')

# The inherited QA collector scans both final/ and experimental/. After the pack
# was flattened those are the same directory, so every public parent duplicate
# was loaded twice under different labels. Build parent QA references directly
# from the internal RoFormer stem paths and scan the public directory only for
# actual child/specialist outputs.
collector_start = '''        qa_stems: dict[str, Path] = {}
        qa_models: dict[str, str] = {}

        def add_qa_stem(label: str, candidate: Path, model: str) -> None:
'''
collector_end = '''        qa_pipeline_metrics = {}
'''
start = exp.find(collector_start)
end = exp.find(collector_end, start)
if start < 0 or end < 0:
    raise RuntimeError('Could not locate experimental QA collector block')

lean_collector = '''        # fast_qa_collector_v1
        qa_stems: dict[str, Path] = {}
        qa_models: dict[str, str] = {}

        def add_qa_stem(label: str, candidate: Path, model: str) -> None:
            if candidate.is_file() and label not in qa_stems:
                qa_stems[label] = candidate
                qa_models[label] = model

        # Parent references stay internal. Do not create/read duplicate public
        # FLAC copies just to score QA.
        parent_label_map = {
            "vocals": "vocals",
            "drums": "percussion",
            "bass": "bass",
            "guitar": "strings",
            "piano": "keys",
            "other": "other",
        }
        for source_label, qa_label in parent_label_map.items():
            parent_path = stems.get(source_label)
            if parent_path:
                add_qa_stem(qa_label, Path(parent_path), "BS-RoFormer-SW")

        for candidate in final.glob("*.flac"):
            lower = candidate.name.lower()
            label = None
            model = "experimental-specialist"

            if lower.endswith("_lead_vocals.flac"):
                label, model = (
                    "lead_vocals",
                    "LiteLABS-VX v2 single-pass vocal parent route",
                )
            elif lower.endswith("_backing_vocals.flac"):
                label, model = (
                    "backing_vocals",
                    "LiteLABS-VX v2 single-pass vocal parent route",
                )
            elif lower.endswith("_kick.flac"):
                label, model = "kick", "MDX23C DrumSep 5-stem"
            elif lower.endswith("_snare.flac"):
                label, model = "snare", "MDX23C DrumSep 5-stem"
            elif lower.endswith("_toms.flac"):
                label, model = "toms", "MDX23C DrumSep 5-stem"
            elif lower.endswith("_hats.flac"):
                label, model = (
                    "hats",
                    "MDX23C DrumSep 5-stem (hh+cymbals merged)",
                )
            elif "sax_specialist" in lower and (
                "_sax" in lower or lower.endswith("sax.flac")
            ):
                label, model = (
                    "saxophone",
                    "LiteLABS Sax specialist",
                )
            elif "sax_specialist" in lower:
                label, model = "sax_residual", "LiteLABS Sax specialist"
            elif "wind_brass_residual" in lower:
                label, model = (
                    "wind_brass_residual",
                    "LiteLABS Wind specialist",
                )
            elif "woodwind" in lower or "wind_brass" in lower:
                label, model = "wind_brass", "LiteLABS Wind specialist"

            if label:
                add_qa_stem(label, candidate, model)

'''
exp = exp[:start] + lean_collector + exp[end:]

# Remove parent-like public FLACs immediately before final README/packaging.
# Internal parent paths in stems remain untouched for routing and QA.
pack_anchor_candidates = [
    '        # public_readme_inventory_final_v2\n',
    '        emit("Packaging Experimental Stems", 92)\n',
    '        emit("Packaging Parent and Experimental Stems", 92)\n',
]
pack_pos = -1
for anchor in pack_anchor_candidates:
    pack_pos = exp.find(anchor)
    if pack_pos >= 0:
        break
if pack_pos < 0:
    raise RuntimeError('Could not locate pre-packaging cleanup anchor')

cleanup_block = '''        # public_parent_cleanup_v1
        for parent_label in (
            "vocals",
            "percussion",
            "bass",
            "strings",
            "keys",
            "other",
            "instrumental",
        ):
            duplicate = final / f"{track}_{parent_label}.flac"
            if duplicate.is_file():
                duplicate.unlink()

'''
if 'public_parent_cleanup_v1' not in exp:
    exp = exp[:pack_pos] + cleanup_block + exp[pack_pos:]

# Add timings around QA, packaging and upload so the next run accounts for all
# wall time instead of leaving ~minutes unexplained.
qa_call = '        research_qa = build_research_qa(\n'
if qa_call in exp and 'qa_started = time.monotonic()' not in exp:
    exp = exp.replace(
        qa_call,
        '        qa_started = time.monotonic()\n' + qa_call,
        1,
    )
    qa_print = '        print(f"LiteLABS silent research QA complete for {len(qa_stems)} stems", flush=True)\n'
    if qa_print in exp:
        exp = exp.replace(
            qa_print,
            '        timings["research_qa"] = round(time.monotonic() - qa_started, 3)\n'
            + qa_print,
            1,
        )

package_line = '        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:\n'
if package_line in exp and 'package_started = time.monotonic()' not in exp:
    exp = exp.replace(
        package_line,
        '        package_started = time.monotonic()\n' + package_line,
        1,
    )
    upload_anchor = '        uploaded = False\n'
    if upload_anchor in exp:
        exp = exp.replace(
            upload_anchor,
            '        timings["package_zip"] = round(time.monotonic() - package_started, 3)\n\n'
            + upload_anchor,
            1,
        )

put_anchor = '            with archive.open("rb") as handle:\n'
if put_anchor in exp and 'upload_started = time.monotonic()' not in exp:
    exp = exp.replace(
        put_anchor,
        '            upload_started = time.monotonic()\n' + put_anchor,
        1,
    )
    uploaded_anchor = '            uploaded = True\n'
    if uploaded_anchor in exp:
        exp = exp.replace(
            uploaded_anchor,
            '            timings["upload"] = round(time.monotonic() - upload_started, 3)\n'
            + uploaded_anchor,
            1,
        )

exp_path.write_text(exp, encoding='utf-8')

qa_check = qa_path.read_text(encoding='utf-8')
exp_check = exp_path.read_text(encoding='utf-8')
compile(qa_check, str(qa_path), 'exec')
compile(exp_check, str(exp_path), 'exec')
assert 'QA_SAMPLE_SECONDS = 60.0' in qa_check
assert 'source_duration_seconds' in qa_check
assert 'fast_qa_collector_v1' in exp_check
assert 'public_parent_cleanup_v1' in exp_check
assert 'timings["research_qa"]' in exp_check
assert 'timings["package_zip"]' in exp_check
assert 'timings["upload"]' in exp_check
print('LiteLABS runtime optimisation v1 applied')
