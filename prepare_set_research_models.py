from __future__ import annotations

import json
import re
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "noblebarkrr/mvsepless_resources"
ROOT = Path("/models/set_research")
ROOT.mkdir(parents=True, exist_ok=True)

KNOWN = [
    {
        "id": "gonzaluigi_bve",
        "category": "lead_back",
        "model_type": "mel_band_roformer",
        "checkpoint": "mel_band_roformer/mbr_bve_gonzaluigi.ckpt",
        "config": "mel_band_roformer/mbr_bve_gonzaluigi_config.yaml",
    },
    {
        "id": "giantailab_karaoke_3stem",
        "category": "lead_back",
        "model_type": "bs_roformer",
        "checkpoint": "bs_roformer/bs_karaoke_3stem_giantailab.ckpt",
        "config": "bs_roformer/bs_karaoke_3stem_giantailab_config.yaml",
    },
    {
        "id": "gabox_karaoke",
        "category": "lead_back",
        "model_type": "bs_roformer",
        "checkpoint": "bs_roformer/bs_karaoke_gabox.ckpt",
        "config": "bs_roformer/bs_karaoke_gabox_config.yaml",
    },
    {
        "id": "anvuew_dereverb_2250",
        "category": "dereverb",
        "model_type": "bs_roformer",
        "checkpoint": "bs_roformer/bs_dereverb_2250_anvuew.ckpt",
        "config": "bs_roformer/bs_dereverb_2250_anvuew_config.yaml",
    },
]


def _download(repo_path: str) -> Path:
    local = Path(hf_hub_download(REPO_ID, repo_path))
    dest = ROOT / repo_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.write_bytes(local.read_bytes())
    return dest


def _read_config(repo_path: str) -> str:
    return _download(repo_path).read_text(encoding="utf-8", errors="replace")


def _list_items(text: str, key: str) -> list[str]:
    lines = text.splitlines()
    out: list[str] = []
    for i, line in enumerate(lines):
        m = re.match(rf"^(\s*){re.escape(key)}\s*:\s*(.*)$", line)
        if not m:
            continue
        indent = len(m.group(1))
        tail = m.group(2).strip()
        if tail.startswith("[") and tail.endswith("]"):
            return [x.strip().strip("'\"") for x in tail[1:-1].split(",") if x.strip()]
        for following in lines[i + 1 :]:
            if not following.strip():
                continue
            leading = len(following) - len(following.lstrip())
            if leading <= indent:
                break
            item = re.match(r"^\s*-\s*(.+?)\s*$", following)
            if item:
                out.append(item.group(1).strip().strip("'\""))
        if out:
            return out
    return out


def _band_count(text: str) -> int | None:
    values = _list_items(text, "freqs_per_bands")
    if values:
        return len(values)
    match = re.search(r"\bnum_bands\s*:\s*(\d+)", text)
    return int(match.group(1)) if match else None


def _instrument_names(text: str) -> list[str]:
    for key in ("instruments", "instrument_names", "sources"):
        values = _list_items(text, key)
        if values:
            return [v.lower().replace("-", "_").replace(" ", "_") for v in values]
    # Some MSST configs use training.instruments / model.instruments but the
    # line-oriented parser above still catches the nested key itself.
    return []


def _target_name(text: str) -> str:
    for key in ("target_instrument", "target"):
        match = re.search(rf"^\s*{key}\s*:\s*['\"]?([^'\"#\n]+)", text, re.M)
        if match:
            return match.group(1).strip().lower()
    return ""


def _paired_checkpoint(config_path: str, file_set: set[str]) -> str | None:
    candidates = [
        config_path.replace("_config.yaml", ".ckpt"),
        config_path.replace("_config.yml", ".ckpt"),
        config_path.replace("_config.yaml", ".chpt"),
        config_path.replace("_config.yml", ".chpt"),
    ]
    for candidate in candidates:
        if candidate in file_set:
            return candidate
    return None


def main() -> None:
    api = HfApi()
    files = api.list_repo_files(REPO_ID)
    file_set = set(files)

    catalog: dict = {
        "repo": REPO_ID,
        "known": [],
        "discovered_124_band": [],
        "discovered_drum_breakdown": [],
        "notes": [],
    }

    # Exact candidates we deliberately want to test.
    for item in KNOWN:
        if item["checkpoint"] not in file_set or item["config"] not in file_set:
            catalog["notes"].append(f"missing known asset: {item['id']}")
            continue
        ckpt = _download(item["checkpoint"])
        cfg = _download(item["config"])
        enriched = dict(item)
        enriched.update({
            "checkpoint_local": str(ckpt),
            "config_local": str(cfg),
            "checkpoint_bytes": ckpt.stat().st_size,
        })
        catalog["known"].append(enriched)

    # Scan configs instead of trusting filenames. This lets the research build
    # catch oddly named/private-ish mirror assets such as a 124-band BS RoFormer
    # candidate if it is actually present in the repository.
    config_paths = [
        p for p in files
        if p.endswith(("_config.yaml", "_config.yml"))
        and p.split("/", 1)[0] in {"bs_roformer", "mel_band_roformer", "scnet"}
    ]

    found_124: list[dict] = []
    found_drums: list[dict] = []
    for cfg_path in config_paths:
        try:
            text = _read_config(cfg_path)
        except Exception as exc:
            catalog["notes"].append(f"config read failed {cfg_path}: {exc}")
            continue

        ckpt_path = _paired_checkpoint(cfg_path, file_set)
        if not ckpt_path:
            continue

        architecture = cfg_path.split("/", 1)[0]
        bands = _band_count(text)
        instruments = _instrument_names(text)
        target = _target_name(text)
        lower = text.lower()

        if architecture == "bs_roformer" and bands == 124:
            # Parent candidate must plausibly separate vocals from accompaniment.
            vocabulary = set(instruments)
            vocalish = ("vocal" in lower) or any("vocal" in x for x in vocabulary) or "vocal" in target
            accompaniment = (
                any(x in lower for x in ("instrumental", "other"))
                or any(x in {"instrumental", "other"} for x in vocabulary)
            )
            if vocalish and accompaniment:
                found_124.append({
                    "id": Path(ckpt_path).stem,
                    "category": "parent_124_band",
                    "model_type": "bs_roformer",
                    "checkpoint": ckpt_path,
                    "config": cfg_path,
                    "bands": bands,
                    "instruments": instruments,
                    "target": target,
                })

        normalized = {x.replace("hi_hat", "hh").replace("hihat", "hh") for x in instruments}
        four = {"kick", "snare", "toms", "cymbals"}
        five = {"kick", "snare", "toms", "hh", "cymbals"}
        drum_match = normalized == four or normalized == five
        if drum_match and architecture in {"mel_band_roformer", "scnet"}:
            found_drums.append({
                "id": Path(ckpt_path).stem,
                "category": "drum_breakdown",
                "model_type": architecture,
                "checkpoint": ckpt_path,
                "config": cfg_path,
                "instruments": instruments,
                "output_count": len(normalized),
            })

    # Avoid exploding the image if the mirror happens to contain lots of
    # fine-tunes. We preserve metadata for every match but download a small,
    # deterministic research set.
    found_124.sort(key=lambda x: (x["checkpoint"], x["config"]))
    found_drums.sort(key=lambda x: (0 if x["output_count"] == 4 else 1, x["checkpoint"]))

    for item in found_124[:3]:
        ckpt = _download(item["checkpoint"])
        cfg = _download(item["config"])
        item.update({
            "checkpoint_local": str(ckpt),
            "config_local": str(cfg),
            "checkpoint_bytes": ckpt.stat().st_size,
        })
        catalog["discovered_124_band"].append(item)
    if len(found_124) > 3:
        catalog["notes"].append(f"{len(found_124) - 3} extra 124-band candidate(s) not baked into image")

    for item in found_drums[:3]:
        ckpt = _download(item["checkpoint"])
        cfg = _download(item["config"])
        item.update({
            "checkpoint_local": str(ckpt),
            "config_local": str(cfg),
            "checkpoint_bytes": ckpt.stat().st_size,
        })
        catalog["discovered_drum_breakdown"].append(item)
    if not found_drums:
        catalog["notes"].append(
            "No MelBand-RoFormer/SCNet 4- or 5-way drum-breakdown checkpoint was discoverable in mvsepless_resources."
        )
    elif len(found_drums) > 3:
        catalog["notes"].append(f"{len(found_drums) - 3} extra drum candidate(s) not baked into image")

    path = ROOT / "catalog.json"
    path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
