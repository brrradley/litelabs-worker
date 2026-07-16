from pathlib import Path

path = Path('/app/mss_candidate_lab.py')
text = path.read_text(encoding='utf-8')

# Preserve release metadata and optional aggregate stem definitions in validated models.
old_meta = '''        "notes": entry.get("notes", ""),
    }
'''
new_meta = '''        "notes": entry.get("notes", ""),
        "release_repo": entry.get("release_repo"),
        "release_tag": entry.get("release_tag"),
        "asset_name_contains": entry.get("asset_name_contains") or [],
        "aggregate_output_names": entry.get("aggregate_output_names") or [],
    }
'''
if old_meta in text:
    text = text.replace(old_meta, new_meta, 1)
elif '"aggregate_output_names": entry.get("aggregate_output_names")' not in text:
    raise RuntimeError('Could not extend validated MSS model metadata')

registry_marker = '\n\ndef _install_candidate(payload: dict, progress=None) -> dict:\n'
helpers = r'''

def _resolve_release_downloads(entry: dict) -> tuple[str, str, dict]:
    repo = str(entry.get("release_repo") or "").strip()
    tag = str(entry.get("release_tag") or "").strip()
    if not repo or not tag:
        return "", "", {}
    response = requests.get(
        f"https://api.github.com/repos/{repo}/releases/tags/{tag}",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "LiteLABS-Research"},
        timeout=(30, 120),
    )
    response.raise_for_status()
    release = response.json()
    assets = [asset for asset in release.get("assets", []) if isinstance(asset, dict)]
    tokens = [str(token).lower() for token in entry.get("asset_name_contains", []) if str(token).strip()]

    def relevant(asset: dict) -> bool:
        name = str(asset.get("name") or "").lower()
        return not tokens or all(token in name for token in tokens)

    filtered = [asset for asset in assets if relevant(asset)]
    pool = filtered or assets
    configs = [asset for asset in pool if str(asset.get("name") or "").lower().endswith((".yaml", ".yml"))]
    checkpoints = [asset for asset in pool if str(asset.get("name") or "").lower().endswith((".ckpt", ".th", ".pt", ".pth"))]
    if not configs or not checkpoints:
        raise RuntimeError(
            f"Release {repo}@{tag} did not expose a usable config/checkpoint pair. "
            f"Assets: {[asset.get('name') for asset in assets]}"
        )
    config = configs[0]
    checkpoint = max(checkpoints, key=lambda asset: int(asset.get("size") or 0))
    return (
        str(config.get("browser_download_url") or ""),
        str(checkpoint.get("browser_download_url") or ""),
        {
            "release_repo": repo,
            "release_tag": tag,
            "config_asset": config.get("name"),
            "checkpoint_asset": checkpoint.get("name"),
        },
    )


def _aggregate_candidate_outputs(paths: list[Path], destination: Path) -> Path:
    if not paths:
        raise RuntimeError("No candidate outputs supplied for aggregation")
    arrays = []
    sample_rate = None
    minimum_length = None
    for source in paths:
        audio, sr = sf.read(source, always_2d=True, dtype="float32")
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise RuntimeError(f"Aggregate sample-rate mismatch: {sr} vs {sample_rate}")
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        audio = audio[:, :2]
        minimum_length = len(audio) if minimum_length is None else min(minimum_length, len(audio))
        arrays.append(audio)
    combined = np.sum([audio[:minimum_length] for audio in arrays], axis=0)
    peak = float(np.max(np.abs(combined)))
    if peak > 0.999:
        combined = combined * (0.999 / peak)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, combined, int(sample_rate), subtype="FLOAT")
    return destination
'''
if 'def _resolve_release_downloads(' not in text:
    if registry_marker not in text:
        raise RuntimeError('Could not locate MSS install function marker')
    text = text.replace(registry_marker, helpers + registry_marker, 1)

old_urls = '''    config_url = str(entry.get("config_url") or "").strip()
    checkpoint_url = str(entry.get("checkpoint_url") or "").strip()
    if not config_url or not checkpoint_url:
        return {"ok": False, "mode": "mss_candidate_lab", "action": "install", "error": "Candidate has no install URLs"}
'''
new_urls = '''    config_url = str(entry.get("config_url") or "").strip()
    checkpoint_url = str(entry.get("checkpoint_url") or "").strip()
    resolved_release = {}
    if not config_url or not checkpoint_url:
        try:
            config_url, checkpoint_url, resolved_release = _resolve_release_downloads(entry)
        except Exception as exc:
            return {
                "ok": False,
                "mode": "mss_candidate_lab",
                "action": "install",
                "error": f"Could not resolve release assets: {exc}",
                "error_type": exc.__class__.__name__,
            }
    if not config_url or not checkpoint_url:
        return {"ok": False, "mode": "mss_candidate_lab", "action": "install", "error": "Candidate has no install URLs"}
'''
if old_urls in text:
    text = text.replace(old_urls, new_urls, 1)
elif 'resolved_release = {}' not in text:
    raise RuntimeError('Could not add dynamic GitHub release resolution')

old_install_tail = '''        "checkpoint_size_bytes": checkpoint_path.stat().st_size if checkpoint_path.exists() else 0,
        "production_status": "blocked_pending_licence_and_benchmark" if not model["production_ready"] else "approved",
'''
new_install_tail = '''        "checkpoint_size_bytes": checkpoint_path.stat().st_size if checkpoint_path.exists() else 0,
        "resolved_release": resolved_release or None,
        "production_status": "blocked_pending_licence_and_benchmark" if not model["production_ready"] else "approved",
'''
if old_install_tail in text:
    text = text.replace(old_install_tail, new_install_tail, 1)
elif '"resolved_release": resolved_release or None' not in text:
    raise RuntimeError('Could not expose resolved release metadata')

old_target = '''        target_matches = [path for path in output_paths if model["target_stem"] in path.name.lower()]
        target_metrics = _candidate_audio_metrics(target_matches[0], source) if succeeded and target_matches else None
'''
new_target = '''        aggregate_names = [str(name).lower() for name in model.get("aggregate_output_names", [])]
        if aggregate_names:
            target_matches = [
                path for path in output_paths
                if path.stem.lower() in aggregate_names or any(path.stem.lower().endswith(name) for name in aggregate_names)
            ]
        else:
            target_matches = [path for path in output_paths if model["target_stem"] in path.name.lower()]
        target_audio = None
        if succeeded and target_matches:
            if len(target_matches) == 1:
                target_audio = target_matches[0]
            else:
                target_audio = _aggregate_candidate_outputs(target_matches, root / f"{model_id}-aggregate-{model['target_stem']}.wav")
        target_metrics = _candidate_audio_metrics(target_audio, source) if target_audio else None
'''
if old_target in text:
    text = text.replace(old_target, new_target, 1)
elif 'aggregate_names = [str(name).lower()' not in text:
    raise RuntimeError('Could not add candidate stem aggregation')

old_result = '''            "target_metrics": target_metrics,
            "archive_name": archive.name if archive else None,
'''
new_result = '''            "target_metrics": target_metrics,
            "target_output_files": [str(item.relative_to(output_dir)) for item in target_matches if item.is_relative_to(output_dir)],
            "aggregate_output_names": model.get("aggregate_output_names", []),
            "archive_name": archive.name if archive else None,
'''
if old_result in text:
    text = text.replace(old_result, new_result, 1)
elif '"aggregate_output_names": model.get("aggregate_output_names", [])' not in text:
    raise RuntimeError('Could not expose aggregate candidate outputs')

path.write_text(text, encoding='utf-8')
print('MSS batch candidate release resolution and aggregate-output patch applied')
