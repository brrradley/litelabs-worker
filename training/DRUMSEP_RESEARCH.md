# SET DrumSep MelBand RoFormer research

Goal: replace the current MDX23C DrumSep5 child route with a four-output MelBand RoFormer trained specifically for:

- kick
- snare
- toms
- cymbals (closed/open hi-hat + crash + ride)

The production parent remains SET's current BS-RoFormer SW drums stem.

## Training framework

Use ZFTurbo/Music-Source-Separation-Training (MSST). The repository supports MelBand RoFormer training, partial compatible checkpoint loading, dataset types 1-7, SDR/SI-SDR/bleedless/fullness validation, EMA, augmentation, LoRA and experiment tracking.

## Primary dataset

StemGMD:
https://zenodo.org/records/7860223

StemGMD contains 1,224 hours of isolated nine-piece drum-kit audio at 44.1 kHz. Dataset license: CC BY 4.0.

Mapping used by SET:

- kick = kick
- snare = snare
- toms = high tom + low-mid tom + floor tom
- cymbals = closed hi-hat + open hi-hat + crash + ride
- mixture = sum(kick, snare, toms, cymbals)

Run:

```bash
python training/prepare_stemgmd_4stem.py \
  --source /workspace/StemGMD \
  --output /workspace/set-drumsep-data
```

The generated train/valid folders use MSST aligned dataset layout and include an explicit mixture.wav.

## Starting checkpoint

Preferred first experiment: Aname-Tommy MelBand RoFormer 4-stem Large.

Repository:
https://huggingface.co/Aname-Tommy/melbandroformer4stems

The model repository declares Apache-2.0 and has a four-output MelBand RoFormer checkpoint. Its original semantics are vocals/drums/bass/other; SET uses it only as an architecture-compatible initialization and retrains all four output heads on drum components.

Checkpoint:
https://huggingface.co/Aname-Tommy/melbandroformer4stems/resolve/main/mel_band_roformer_4stems_large_ver1.ckpt

Also run a from-scratch control. Do not promote either route until the resulting checkpoint license/provenance and validation are recorded.

## Training

From scratch:

```bash
DATA_ROOT=/workspace/set-drumsep-data \
RESULTS_ROOT=/workspace/set-drumsep-results/scratch \
bash training/train_drumsep_melband_4stem.sh
```

Warm-start:

```bash
DATA_ROOT=/workspace/set-drumsep-data \
RESULTS_ROOT=/workspace/set-drumsep-results/aname-init \
START_CKPT=/workspace/models/mel_band_roformer_4stems_large_ver1.ckpt \
bash training/train_drumsep_melband_4stem.sh
```

The launcher validates with SDR, SI-SDR, bleedless and fullness and uses SDR as the scheduler metric.

## Research comparator only

LarsNet:
https://github.com/polimi-ispl/larsnet

LarsNet is useful as a benchmark because it separates kick/snare/toms/hi-hat/cymbals faster than real time and was trained on StemGMD. Its released pretrained checkpoints are CC BY-NC 4.0, so they must not be copied into SET production. They may be used for non-commercial research comparison subject to that license.

## Validation target

MVSep's closed DrumSep benchmark is useful as an external target. Their current four-stem MelBand route reports approximately:

- kick SDR 22.22
- snare SDR 17.09
- toms SDR 15.86
- hh+cymbals SDR 11.87

We cannot train against MVSep's hidden references, so these numbers are a target/benchmark rather than a reproducible training objective.

Promotion criteria for SET should therefore be based on:

1. held-out StemGMD validation;
2. a real-acoustic validation set not used for training;
3. direct A/B against current MDX23C DrumSep5 on SW drum parents;
4. runtime and VRAM on the same RunPod GPU;
5. listening for transient smearing, cymbal wash, kick/snare bleed and missing ghost notes.
