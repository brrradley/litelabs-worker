from pathlib import Path

path = Path("/app/research_benchmark_v2.py")
text = path.read_text(encoding="utf-8")
marker = "# litelabs_set_targeted_research_v1\n"

if marker not in text:
    text += r'''

# litelabs_set_targeted_research_v1
SET_RESEARCH_ROOT = Path("/models/set_research")
SET_RESEARCH_CATALOG = SET_RESEARCH_ROOT / "catalog.json"


def _set_catalog() -> dict:
    if not SET_RESEARCH_CATALOG.is_file():
        return {"known": [], "discovered_124_band": [], "discovered_drum_breakdown": [], "notes": ["catalog missing"]}
    return json.loads(SET_RESEARCH_CATALOG.read_text(encoding="utf-8"))


def _set_direct_model(
    input_path: Path,
    root: Path,
    item: dict,
    timeout: int,
) -> dict:
    repo_dir = Path("/opt/music-source-separation-training")
    input_dir = root / "input"
    output_dir = root / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep a single stable filename so model output names are predictable.
    prepared = input_dir / "input.wav"
    rc, _, log = _run(
        ["ffmpeg", "-y", "-i", str(input_path), "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", str(prepared)],
        timeout=300,
    )
    if rc != 0:
        return {"returncode": rc, "error": "input preparation failed", "log_tail": log[-3000:]}

    model_type = str(item.get("model_type") or "").strip()
    config = Path(str(item.get("config_local") or ""))
    checkpoint = Path(str(item.get("checkpoint_local") or ""))
    if not model_type or not config.is_file() or not checkpoint.is_file():
        return {
            "returncode": 2,
            "error": "candidate files are missing",
            "model_type": model_type,
            "config": str(config),
            "checkpoint": str(checkpoint),
        }

    cmd = [
        "python", str(repo_dir / "inference.py"),
        "--model_type", model_type,
        "--config_path", str(config),
        "--start_check_point", str(checkpoint),
        "--input_folder", str(input_dir),
        "--store_dir", str(output_dir),
        "--device_ids", "0",
        "--disable_detailed_pbar",
        "--filename_template", "{file_name}/{instr}",
    ]
    rc, elapsed, run_log = _run(cmd, cwd=repo_dir, timeout=timeout)
    files = _audio_files(output_dir) if rc == 0 else []
    return {
        "returncode": rc,
        "runtime_seconds": round(elapsed, 3),
        "model_type": model_type,
        "model_id": item.get("id"),
        "checkpoint": str(item.get("checkpoint") or checkpoint.name),
        "config": str(item.get("config") or config.name),
        "checkpoint_bytes": item.get("checkpoint_bytes"),
        "files": [str(p) for p in files],
        "log_tail": "\n".join(run_log.splitlines()[-80:]),
    }


def _set_file_map(result: dict) -> dict[str, Path]:
    mapped: dict[str, Path] = {}
    for raw in result.get("files") or []:
        p = Path(raw)
        name = p.stem.lower().replace("-", "_").replace(" ", "_")
        # More specific names first.
        if "backing" in name or "back_vocal" in name or "backvocal" in name or "_bv" in name:
            mapped.setdefault("backing", p)
        elif "noreverb" in name or "no_reverb" in name or "dry" in name:
            mapped.setdefault("noreverb", p)
        elif "reverb" in name or "echo" in name:
            mapped.setdefault("reverb", p)
        elif "lead" in name or "main_vocal" in name:
            mapped.setdefault("lead", p)
        elif "kick" in name:
            mapped.setdefault("kick", p)
        elif "snare" in name:
            mapped.setdefault("snare", p)
        elif "tom" in name:
            mapped.setdefault("toms", p)
        elif "cymbal" in name or "ride" in name or "crash" in name:
            mapped.setdefault("cymbals", p)
        elif "hihat" in name or "hi_hat" in name or "_hh" in name:
            mapped.setdefault("hh", p)
        elif "instrumental" in name or "accompaniment" in name or name.endswith("_other") or name == "other":
            mapped.setdefault("instrumental", p)
        elif "vocals" in name or "vocal" in name:
            mapped.setdefault("vocals", p)
    return mapped


def _set_choose_lead_back(result: dict) -> tuple[Path | None, Path | None]:
    files = [Path(x) for x in result.get("files") or []]
    mapped = _set_file_map(result)

    # GiantAILAB exposes explicit backing_vocal + vocals + instrumental.
    backing = mapped.get("backing")
    lead = mapped.get("lead") or mapped.get("vocals")
    if lead and backing:
        return lead, backing

    # Two-stem karaoke/BVE models commonly expose vocals + instrumental/other
    # when they are fed an already isolated vocal parent.
    if lead and mapped.get("instrumental"):
        return lead, mapped["instrumental"]

    if len(files) == 2:
        # Last-resort deterministic classification: prefer the file containing
        # "vocal" as lead, otherwise larger output as lead. The report records
        # filenames so a mistaken semantic mapping is obvious.
        vocal = next((p for p in files if "vocal" in p.name.lower() and "back" not in p.name.lower()), None)
        if vocal:
            other = next(p for p in files if p != vocal)
            return vocal, other
        ordered = sorted(files, key=lambda p: p.stat().st_size, reverse=True)
        return ordered[0], ordered[1]
    return None, None


def _set_sdr(reference: Path, estimate: Path) -> dict:
    ref, _ = _mono(reference)
    est, _ = _mono(estimate)
    n = min(len(ref), len(est))
    if n <= 0:
        return {}
    r = ref[:n].astype(np.float64, copy=False)
    e = est[:n].astype(np.float64, copy=False)
    noise = r - e
    sdr = 10.0 * math.log10((float(np.sum(r * r)) + 1e-12) / (float(np.sum(noise * noise)) + 1e-12))

    # SI-SDR: scale estimate onto the reference target.
    r0 = r - np.mean(r)
    e0 = e - np.mean(e)
    scale = float(np.dot(e0, r0)) / (float(np.dot(r0, r0)) + 1e-12)
    target = scale * r0
    residual = e0 - target
    sisdr = 10.0 * math.log10(
        (float(np.sum(target * target)) + 1e-12) /
        (float(np.sum(residual * residual)) + 1e-12)
    )
    return {"sdr_db": round(sdr, 4), "si_sdr_db": round(sisdr, 4)}


def _set_load_references(payload: dict, root: Path) -> dict[str, Path]:
    refs = payload.get("references") or payload.get("reference_urls") or {}
    if not isinstance(refs, dict):
        return {}
    out: dict[str, Path] = {}
    for label, value in refs.items():
        url = str(value or "").strip()
        if not url:
            continue
        raw = root / f"reference_{_safe_name(str(label))}.audio"
        wav = root / f"reference_{_safe_name(str(label))}.wav"
        _download(url, raw)
        rc, _, log = _run(["ffmpeg", "-y", "-i", str(raw), "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", str(wav)], timeout=300)
        if rc != 0:
            raise RuntimeError(f"Reference conversion failed for {label}: {log[-1500:]}")
        out[str(label).lower()] = wav
    return out


def _set_reference_metrics(refs: dict[str, Path], mapping: dict[str, Path]) -> dict:
    aliases = {
        "lead": ("lead", "lead_vocals"),
        "backing": ("backing", "backing_vocals"),
        "vocals": ("vocals", "parent_vocals"),
        "instrumental": ("instrumental", "parent_instrumental"),
        "kick": ("kick",),
        "snare": ("snare",),
        "toms": ("toms", "tom"),
        "cymbals": ("cymbals", "cymbal"),
        "hh": ("hh", "hihat", "hi_hat"),
    }
    metrics: dict = {}
    for logical, candidate_path in mapping.items():
        ref = None
        for key in aliases.get(logical, (logical,)):
            if key in refs:
                ref = refs[key]
                break
        if ref and candidate_path and candidate_path.is_file():
            metrics[logical] = _set_sdr(ref, candidate_path)
    return metrics


def _set_preview(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _crop_center(source, destination, 60.0)


def _set_registered_karaoke(
    vocal_parent: Path,
    root: Path,
    label: str,
    model: str,
    timeout: int,
) -> dict:
    result = _run_separator(vocal_parent, root, model, [], timeout)
    result["label"] = label
    if result.get("returncode") != 0:
        return result
    lead = Path(result["primary"])
    backing = Path(result["secondary"])
    result["lead"] = str(lead)
    result["backing"] = str(backing)
    result["pair_metrics"] = _pair_metrics(vocal_parent, lead, backing)
    return result


def _set_direct_karaoke(
    vocal_parent: Path,
    root: Path,
    item: dict,
    timeout: int,
) -> dict:
    result = _set_direct_model(vocal_parent, root, item, timeout)
    if result.get("returncode") != 0:
        return result
    lead, backing = _set_choose_lead_back(result)

    # Several MSST configs declare both Lead/Back (or vocals/other) but use
    # target_instrument, so inference intentionally writes only the target.
    # On an already-isolated vocal parent the missing complement is exact and
    # should be reconstructed as parent - target instead of treating the model
    # as a failed two-stem separator.
    files = [Path(x) for x in result.get("files") or []]
    if (not lead or not backing) and len(files) == 1:
        only = files[0]
        mapped = _set_file_map(result)
        parent_audio, sr = sf.read(str(vocal_parent), always_2d=True, dtype="float32")
        target_audio, sr2 = sf.read(str(only), always_2d=True, dtype="float32")
        if int(sr) != int(sr2):
            result["returncode"] = 3
            result["error"] = f"single-target sample-rate mismatch: {sr2} != {sr}"
            return result
        n = min(len(parent_audio), len(target_audio))
        complement = (parent_audio[:n] - target_audio[:n]).astype(np.float32)
        complement_path = root / "derived_complement.wav"
        sf.write(str(complement_path), complement, sr, subtype="FLOAT")

        if mapped.get("backing") or "back" in only.stem.lower():
            backing = only
            lead = complement_path
            result["derived_stem"] = "lead"
        else:
            lead = only
            backing = complement_path
            result["derived_stem"] = "backing"
        result["complement_method"] = "vocal_parent_minus_target"
        result["derived_complement"] = str(complement_path)

    if not lead or not backing:
        result["returncode"] = 3
        result["error"] = "could not map lead/backing outputs"
        return result
    result["lead"] = str(lead)
    result["backing"] = str(backing)
    result["pair_metrics"] = _pair_metrics(vocal_parent, lead, backing)
    return result


def _set_run_dereverb(
    source: Path,
    root: Path,
    item: dict,
    timeout: int,
) -> dict:
    result = _set_direct_model(source, root, item, timeout)
    if result.get("returncode") != 0:
        return result
    mapped = _set_file_map(result)
    dry = mapped.get("noreverb") or mapped.get("vocals")
    reverb = mapped.get("reverb") or mapped.get("instrumental")
    if not dry:
        result["returncode"] = 3
        result["error"] = "could not map noreverb output"
        return result
    result["dry"] = str(dry)
    if not reverb:
        source_audio, sr = sf.read(str(source), always_2d=True, dtype="float32")
        dry_audio, sr2 = sf.read(str(dry), always_2d=True, dtype="float32")
        if int(sr) == int(sr2):
            n = min(len(source_audio), len(dry_audio))
            derived = root / "derived_reverb.wav"
            sf.write(str(derived), (source_audio[:n] - dry_audio[:n]).astype(np.float32), sr, subtype="FLOAT")
            reverb = derived
            result["derived_reverb"] = True
    if reverb:
        result["reverb"] = str(reverb)
        result["pair_metrics"] = _pair_metrics(source, dry, reverb)
    result["change_vs_input"] = _similarity(source, dry)
    return result


def _set_targeted_single(payload: dict, progress=None) -> dict:
    audio_url = str(payload.get("audio_url") or payload.get("source_url") or "").strip()
    if not audio_url:
        return {"ok": False, "mode": MODE, "error": "audio_url is required"}

    timeout = max(300, int(payload.get("timeout_seconds") or 3600))
    heartbeat = max(5, int(payload.get("heartbeat_seconds") or 15))
    filename = str(payload.get("filename") or unquote(Path(urlparse(audio_url).path).name) or "track.wav")
    track = _safe_name(Path(filename).stem)
    model_dir = Path(str(payload.get("model_dir") or "/models/bs_roformer_sw"))

    def emit(message: str, percent: int) -> None:
        print(f"[SET targeted research] {message} ({percent}%)", flush=True)
        if progress:
            progress(message, percent)

    catalog = _set_catalog()
    known = {str(x.get("id")): x for x in catalog.get("known") or []}
    dereverb_item = known.get("anvuew_dereverb_2250")

    report: dict = {
        "schema_version": 4,
        "mode": "set_targeted_research_v1",
        "build_sha": os.getenv("LITELABS_BUILD_SHA", "unknown"),
        "track": track,
        "scope": {
            "parent": "current SW vs discovered 124-band candidate(s) only",
            "instrument_identification": "unchanged / not tested",
            "lead_back": "stats-first candidate matrix",
            "dereverb": "Anvuew 22.50 applied independently to successful lead/back candidates",
            "drums": "current DrumSep5 vs any discoverable MelBand/SCNet 4/5-way breakdown candidate",
        },
        "metric_policy": {
            "with_references": "SDR and SI-SDR are calculated when matching reference URLs are supplied",
            "without_references": "reconstruction/correlation metrics are diagnostics only and are not labelled as SDR",
        },
        "catalog": catalog,
        "tests": {},
    }

    started = time.monotonic()
    archive = Path("/tmp") / f"set-targeted-research-{uuid.uuid4().hex[:10]}-{track}.zip"

    with tempfile.TemporaryDirectory(prefix="set_targeted_research_") as temp:
        root = Path(temp)
        source_dir = root / "source"
        sw_dir = root / "sw"
        outputs = root / "outputs"
        source_dir.mkdir()
        sw_dir.mkdir()
        outputs.mkdir()
        refs = _set_load_references(payload, root)

        raw_name = unquote(Path(urlparse(audio_url).path).name) or "input.audio"
        downloaded = root / raw_name
        source = source_dir / f"{track}.wav"

        emit("Downloading source", 2)
        _download(audio_url, downloaded)
        rc, elapsed, log = _run(
            ["ffmpeg", "-y", "-i", str(downloaded), "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", str(source)],
            timeout=300,
        )
        if rc != 0:
            raise RuntimeError("Source conversion failed: " + log[-2000:])
        report["tests"]["source_conversion"] = {"runtime_seconds": round(elapsed, 3)}

        emit("Running current BS-RoFormer-SW parent", 7)
        sw_config, sw_checkpoint, _ = _resolve_model_files(model_dir, progress=None)
        rc, sw_elapsed = _run_polled(
            [
                "bs-roformer-infer",
                "--config_path", str(sw_config),
                "--model_path", str(sw_checkpoint),
                "--input_folder", str(source_dir),
                "--store_dir", str(sw_dir),
            ],
            cwd=None,
            timeout=timeout,
            log_path=root / "sw.log",
            progress=None,
            stage_name="SET research SW parent",
            start_percent=7,
            end_percent=17,
            heartbeat_seconds=heartbeat,
        )
        if rc != 0:
            raise RuntimeError("Current SW parent failed")
        sw_stems = _collect_sw_stems(sw_dir)
        sw_vocals = sw_stems.get("vocals")
        sw_drums = sw_stems.get("drums")
        sw_inst = sw_stems.get("instrumental") or sw_stems.get("other")
        if not sw_vocals or not sw_drums:
            raise RuntimeError("SW parent did not produce required vocals/drums stems")

        sw_parent = {
            "runtime_seconds": round(sw_elapsed, 3),
            "vocals": _stats(sw_vocals),
            "drums": _stats(sw_drums),
        }
        if sw_inst:
            sw_parent["instrumental"] = _stats(sw_inst)
        sw_map = {"vocals": sw_vocals}
        if sw_inst:
            sw_map["instrumental"] = sw_inst
        sw_parent["reference_metrics"] = _set_reference_metrics(refs, sw_map)
        report["tests"]["parent_current_sw"] = sw_parent
        _set_preview(sw_vocals, outputs / "parent_current_sw_vocals_60s.flac")

        # 124-band: test every actually discovered BS-RoFormer candidate. If the
        # mirror does not contain one, say so explicitly instead of mislabelling
        # an unrelated checkpoint.
        parent_candidates: dict = {}
        discovered_124 = catalog.get("discovered_124_band") or []
        if discovered_124:
            for index, item in enumerate(discovered_124, start=1):
                emit(f"Testing discovered 124-band parent {index}/{len(discovered_124)}", 18 + index * 3)
                result = _set_direct_model(source, root / f"parent124_{index}", item, timeout)
                if result.get("returncode") == 0:
                    mapped = _set_file_map(result)
                    vocals = mapped.get("vocals") or mapped.get("lead")
                    inst = mapped.get("instrumental")
                    if vocals and inst:
                        result["pair_metrics"] = _pair_metrics(source, vocals, inst)
                        result["vs_current_sw_vocals"] = _similarity(sw_vocals, vocals)
                        result["reference_metrics"] = _set_reference_metrics(
                            refs, {"vocals": vocals, "instrumental": inst}
                        )
                        _set_preview(vocals, outputs / f"parent_124_{index}_vocals_60s.flac")
                        _set_preview(inst, outputs / f"parent_124_{index}_instrumental_60s.flac")
                    else:
                        result["semantic_warning"] = "could not confidently map vocals + instrumental outputs"
                parent_candidates[str(item.get("id") or f"candidate_{index}")] = result
        else:
            parent_candidates["status"] = {
                "available": False,
                "reason": "No actual 124-band vocal/instrumental BS-RoFormer config was discoverable in mvsepless_resources at image-build time.",
            }
        report["tests"]["parent_124_band_candidates"] = parent_candidates

        emit("Running current DrumSep5 baseline", 31)
        current_drums = _focused_drumsep(sw_drums, root / "drum_current", timeout)
        if current_drums.get("returncode") == 0:
            current_paths = {k: Path(v) for k, v in (current_drums.get("paths") or {}).items()}
            current_drums["reference_metrics"] = _set_reference_metrics(refs, current_paths)
            for label, p in current_paths.items():
                _set_preview(p, outputs / f"drums_current_{label}_60s.flac")
        report["tests"]["drums_current_drumsep5"] = current_drums

        drum_candidates: dict = {}
        discovered_drums = catalog.get("discovered_drum_breakdown") or []
        if discovered_drums:
            for index, item in enumerate(discovered_drums, start=1):
                emit(f"Testing discovered drum breakdown {index}/{len(discovered_drums)}", 35 + index * 4)
                result = _set_direct_model(sw_drums, root / f"drum_candidate_{index}", item, timeout)
                if result.get("returncode") == 0:
                    mapped = _set_file_map(result)
                    # Merge hh into cymbals only for comparison/output if a 5-way
                    # candidate is discovered; the desired customer route is 4-way.
                    if mapped.get("hh") and mapped.get("cymbals"):
                        hh, sr = sf.read(str(mapped["hh"]), always_2d=True, dtype="float32")
                        cym, sr2 = sf.read(str(mapped["cymbals"]), always_2d=True, dtype="float32")
                        if sr == sr2:
                            n = min(len(hh), len(cym))
                            merged = root / f"drum_candidate_{index}" / "cymbals_merged.wav"
                            sf.write(str(merged), (hh[:n] + cym[:n]).astype(np.float32), sr, subtype="FLOAT")
                            mapped["cymbals"] = merged
                    wanted = {k: v for k, v in mapped.items() if k in {"kick", "snare", "toms", "cymbals"}}
                    result["mapped_outputs"] = {k: str(v) for k, v in wanted.items()}
                    result["reference_metrics"] = _set_reference_metrics(refs, wanted)
                    if len(wanted) == 4:
                        arrays = []
                        parent_audio, _ = sf.read(str(sw_drums), always_2d=True, dtype="float32")
                        for key in ("kick", "snare", "toms", "cymbals"):
                            data, _ = sf.read(str(wanted[key]), always_2d=True, dtype="float32")
                            arrays.append(data)
                        n = min([len(parent_audio)] + [len(x) for x in arrays])
                        summed = np.sum(np.stack([x[:n] for x in arrays], axis=0), axis=0)
                        residual = parent_audio[:n] - summed
                        result["reconstruction"] = {
                            "parent_vs_children_sum_cosine": round(
                                _cos(parent_audio[:n].reshape(-1), summed.reshape(-1)), 8
                            ),
                            "residual_relative_to_parent_db": round(
                                _db_ratio(residual, parent_audio[:n]), 3
                            ),
                        }
                        for key, p in wanted.items():
                            _set_preview(p, outputs / f"drums_candidate_{index}_{key}_60s.flac")
                    else:
                        result["semantic_warning"] = f"mapped only {sorted(wanted)}"
                drum_candidates[str(item.get("id") or f"candidate_{index}")] = result
        else:
            drum_candidates["status"] = {
                "available": False,
                "reason": "Mirror contains no discoverable MelBand/SCNet 4/5-way drum-breakdown checkpoint. Current DrumSep5 still runs as runtime/quality baseline.",
                "wanted": ["kick", "snare", "toms", "cymbals"],
            }
        report["tests"]["drums_new_candidates"] = drum_candidates

        emit("Preparing vocal parent for lead/back matrix", 53)
        vocal_parent = root / "sw_vocals_pcm16.wav"
        _focused_pcm16_copy(sw_vocals, vocal_parent)

        karaoke_results: dict[str, dict] = {}
        registered = [
            ("becruily_melband", "Becruily MelBand Karaoke", KARAOKE_BASELINE),
            ("anvuew_bs", "Anvuew BS-RoFormer Karaoke", KARAOKE_CHALLENGER),
        ]
        for index, (key, label, model) in enumerate(registered, start=1):
            emit(f"Lead/back stats: {label}", 55 + index * 4)
            result = _set_registered_karaoke(vocal_parent, root / f"karaoke_{key}", label, model, timeout)
            if result.get("returncode") == 0:
                lead = Path(result["lead"])
                backing = Path(result["backing"])
                result["reference_metrics"] = _set_reference_metrics(refs, {"lead": lead, "backing": backing})
                _set_preview(lead, outputs / f"leadback_{key}_lead_60s.flac")
                _set_preview(backing, outputs / f"leadback_{key}_backing_60s.flac")
            karaoke_results[key] = result

        direct_items = [
            item for item in (catalog.get("known") or [])
            if item.get("category") == "lead_back"
        ]
        for index, item in enumerate(direct_items, start=1):
            key = str(item.get("id") or f"direct_{index}")
            emit(f"Lead/back stats: {key}", min(80, 63 + index * 4))
            result = _set_direct_karaoke(vocal_parent, root / f"karaoke_{key}", item, timeout)
            if result.get("returncode") == 0:
                lead = Path(result["lead"])
                backing = Path(result["backing"])
                result["reference_metrics"] = _set_reference_metrics(refs, {"lead": lead, "backing": backing})
                _set_preview(lead, outputs / f"leadback_{key}_lead_60s.flac")
                _set_preview(backing, outputs / f"leadback_{key}_backing_60s.flac")
            karaoke_results[key] = result
        report["tests"]["lead_back_candidates"] = karaoke_results

        # Dereverb is intentionally evaluated on every successful lead/back pair.
        # This avoids prematurely picking a winner from reconstruction proxies.
        dereverb_results: dict = {}
        if dereverb_item:
            successful = [
                (key, value) for key, value in karaoke_results.items()
                if value.get("returncode") == 0 and value.get("lead") and value.get("backing")
            ]
            total = max(1, len(successful))
            for index, (key, value) in enumerate(successful, start=1):
                emit(f"Dereverb {index}/{total}: {key}", min(93, 81 + int(index / total * 11)))
                per_candidate = {}
                for stem_name in ("lead", "backing"):
                    source_stem = Path(value[stem_name])
                    result = _set_run_dereverb(
                        source_stem,
                        root / f"dereverb_{key}_{stem_name}",
                        dereverb_item,
                        timeout,
                    )
                    if result.get("returncode") == 0 and result.get("dry"):
                        dry = Path(result["dry"])
                        ref_key = "lead" if stem_name == "lead" else "backing"
                        result["reference_metrics"] = _set_reference_metrics(refs, {ref_key: dry})
                        _set_preview(dry, outputs / f"dereverb_{key}_{stem_name}_dry_60s.flac")
                    per_candidate[stem_name] = result
                dereverb_results[key] = per_candidate
        else:
            dereverb_results["status"] = {
                "available": False,
                "reason": "Anvuew 22.50 dereverb assets were not available in the model catalog.",
            }
        report["tests"]["dereverb_anvuew_2250"] = dereverb_results

        # Produce concise stats-first summary. Reference rankings only exist when
        # genuine references were supplied; we never rank separation quality from
        # reconstruction proxies alone.
        ranking: dict = {"lead": [], "backing": [], "drums": {}, "parent": []}
        for key, result in karaoke_results.items():
            rm = result.get("reference_metrics") or {}
            for stem in ("lead", "backing"):
                metric = rm.get(stem)
                if metric:
                    ranking[stem].append({
                        "candidate": key,
                        "sdr_db": metric.get("sdr_db"),
                        "si_sdr_db": metric.get("si_sdr_db"),
                        "runtime_seconds": result.get("runtime_seconds"),
                    })
        for stem in ("lead", "backing"):
            ranking[stem].sort(key=lambda x: (x.get("si_sdr_db") is not None, x.get("si_sdr_db") or -999), reverse=True)

        for key, result in drum_candidates.items():
            if not isinstance(result, dict):
                continue
            ranking["drums"][key] = {
                "runtime_seconds": result.get("runtime_seconds"),
                "reference_metrics": result.get("reference_metrics") or {},
                "reconstruction": result.get("reconstruction") or {},
            }
        ranking["drums"]["current_drumsep5"] = {
            "runtime_seconds": current_drums.get("runtime_seconds"),
            "reference_metrics": current_drums.get("reference_metrics") or {},
            "quality": current_drums.get("quality") or {},
        }

        for key, result in parent_candidates.items():
            if not isinstance(result, dict):
                continue
            rm = result.get("reference_metrics") or {}
            if rm:
                ranking["parent"].append({
                    "candidate": key,
                    "runtime_seconds": result.get("runtime_seconds"),
                    "reference_metrics": rm,
                })
        report["stats_first_summary"] = ranking
        report["total_runtime_seconds"] = round(time.monotonic() - started, 3)

        (outputs / "SET_RESEARCH_REPORT.json").write_text(
            json.dumps(_json_safe(report), indent=2), encoding="utf-8"
        )
        (outputs / "README.txt").write_text(
            "SET targeted research pack\n"
            "==========================\n\n"
            "One request tested only the current SET priorities:\n"
            "- current SW parent vs actual discovered 124-band candidate(s)\n"
            "- current DrumSep5 vs discoverable 4/5-way modern drum candidates\n"
            "- lead/backing candidate matrix\n"
            "- Anvuew 22.50 dereverb on successful lead/backing outputs\n\n"
            "Files in this ZIP are 60-second centre previews. Full-track statistics\n"
            "are in SET_RESEARCH_REPORT.json.\n\n"
            "Important: reconstruction/correlation values are diagnostics, NOT SDR.\n"
            "True SDR/SI-SDR is only reported when reference stem URLs were supplied.\n",
            encoding="utf-8",
        )

        emit("Packaging research report and previews", 95)
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
            for output in sorted(outputs.rglob("*")):
                if output.is_file():
                    bundle.write(output, arcname=output.name)

    archive_size = archive.stat().st_size
    uploaded = False
    result_url = payload.get("result_public_url")
    upload_details = None
    put_url = str(payload.get("result_put_url") or "").strip()
    if put_url:
        emit("Uploading targeted research pack", 97)
        upload_details = _focused_upload_archive(archive, put_url, payload)
        uploaded = bool(upload_details.get("uploaded"))
        if int(upload_details.get("status_code") or 0) == 413:
            return _json_safe({
                "ok": False,
                "mode": "set_targeted_research_v1",
                "error_code": "result_too_large",
                "archive_size_bytes": archive_size,
                "report": report,
            })

    result = {
        "ok": True,
        "mode": "set_targeted_research_v1",
        "track": track,
        "build_sha": os.getenv("LITELABS_BUILD_SHA", "unknown"),
        "archive_name": archive.name,
        "archive_size_bytes": archive_size,
        "uploaded": uploaded,
        "result_url": result_url if uploaded else None,
        "upload": upload_details,
        "report": report,
    }
    if uploaded:
        archive.unlink(missing_ok=True)
    emit("SET targeted research complete", 100)
    return _json_safe(result)


# Keep the existing one-request multi-track suite wrapper, but swap each child
# to this deliberately narrow SET research pass.
_run_research_benchmark_single = _set_targeted_single
'''
    path.write_text(text, encoding="utf-8")

check = path.read_text(encoding="utf-8")
compile(check, str(path), "exec")
assert marker in check
assert "_run_research_benchmark_single = _set_targeted_single" in check
assert "dereverb_anvuew_2250" in check
assert "drums_new_candidates" in check
assert "parent_124_band_candidates" in check
print("LiteLABS SET targeted research pass applied")
