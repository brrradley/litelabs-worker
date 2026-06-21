# LiteLABS Worker

GPU-backed stem extraction worker for **LiteLABS by LiteRECORDS**.

This repository contains the RunPod Serverless worker used by LiteRECORDS to process uploaded audio into a downloadable LiteLABS stem pack. It is designed to be called by the LiteLABS XenForo add-on rather than used directly by forum members.

## Overview

The worker accepts a public audio URL, processes the track, creates a ZIP stem pack, and can upload the finished archive back to the LiteRECORDS server.

The public output pack contains clean filenames such as:

```text
01_track_vocals.flac
02_track_drums.flac
03_track_bass.flac
04_track_guitar.flac
05_track_piano_keys.flac
06_track_synth_strings_other.flac
07_track_instrumental_clean.flac
README.txt
```

The ZIP archive is named:

```text
track-litelabs-stem-pack.zip
```

## Architecture

```text
XenForo LiteLABS add-on
  -> temporary public audio upload
  -> RunPod Serverless job
  -> worker downloads the audio
  -> worker creates the LiteLABS ZIP
  -> worker uploads the ZIP back to LiteRECORDS
  -> XenForo provides the download and cleanup
```

The XenForo add-on handles permissions, DBTech Credits, cooldowns, queue control, user interface, download handling, and cleanup. This repository handles the GPU processing step.

## Container image

GitHub Actions builds and publishes the worker image to GitHub Container Registry:

```text
ghcr.io/brrradley/litelabs-worker:latest
ghcr.io/brrradley/litelabs-worker:<commit-sha>
```

For deployment, use a commit SHA tag rather than `latest`.

## Model volume

Large model files are not committed to this repository. They should live on the RunPod network volume.

The worker reads the model directory from:

```text
STEMFORGE_MODEL_DIR
```

The current RunPod volume path is expected to be similar to:

```text
/runpod-volume/models/bs_roformer_sw
```

## RunPod settings

Recommended beta settings:

```text
Type: Queue based
Max workers: 1
Active workers: 1 for warm testing, 0 for cheaper cold-start testing
FlashBoot: enabled
Network volume: attached
Start command: blank
```

Keeping max workers at 1 protects LiteRECORDS from running multiple expensive jobs at once. With active workers set to 0, cold starts can take several minutes.

## Healthcheck

```json
{
  "input": {
    "healthcheck": true
  }
}
```

Expected response:

```json
{
  "ok": true,
  "status": "ready"
}
```

## Processing input

```json
{
  "input": {
    "audio_url": "https://example.com/input/example.mp3",
    "filename": "example.mp3",
    "result_put_url": "https://example.com/result-receiver.php?file=example-litelabs-stem-pack.zip",
    "result_public_url": "https://example.com/results/example-litelabs-stem-pack.zip"
  }
}
```

### Fields

| Field | Required | Description |
| --- | --- | --- |
| `audio_url` | Yes | Public URL the worker can download. |
| `filename` | No | Original filename used to derive a safe track name. |
| `result_put_url` | No | Receiver endpoint for the finished ZIP. |
| `result_public_url` | No | Public URL returned to the caller after upload. |
| `model_dir` | No | Optional model directory override. |

## Successful response

```json
{
  "ok": true,
  "track": "example",
  "archive_size_bytes": 123456789,
  "uploaded": true,
  "result_url": "https://example.com/results/example-litelabs-stem-pack.zip",
  "stems": [
    "01_example_vocals.flac",
    "02_example_drums.flac",
    "03_example_bass.flac",
    "04_example_guitar.flac",
    "05_example_piano_keys.flac",
    "06_example_synth_strings_other.flac",
    "07_example_instrumental_clean.flac",
    "README.txt"
  ]
}
```

## Server limits

The returned ZIP can be much larger than the original audio upload. The receiver server must allow large request bodies.

Recommended tested values:

```ini
upload_max_filesize = 1024M
post_max_size = 1100M
memory_limit = 512M
max_execution_time = 900
max_input_time = 900
```

If nginx is in front of PHP, set:

```nginx
client_max_body_size 1100m;
```

Without this, larger result uploads may fail with `413 Request Entity Too Large` before the receiver script runs.

## Cleanup

Temporary processing files are scoped to the worker job. Long-term storage cleanup is handled by the XenForo add-on:

- original uploads are removed after success, failure, or abandonment
- ZIP files are removed after successful download
- abandoned jobs are cleaned after the configured timeout

## Notes

- The container is CUDA/PyTorch based and expects GPU execution.
- `ffmpeg` is required and installed in the Docker image.
- ZIP archives are stored without extra compression because the stems are already FLAC.
- Model files should stay outside the GitHub repository.
- Do not commit API keys, receiver secrets, or model checkpoints.
