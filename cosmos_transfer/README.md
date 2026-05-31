# cosmos_transfer

Cosmos Transfer 2.5 **distilled edge** sim-to-real for DreamBC robot videos.

- **Default:** upload inputs to GCS → RunPod Serverless (H100 80GB) → download outputs.
- **Local:** `--local` uses a local `cosmos-transfer2.5` venv (see `setup_cosmos.sh`).

Inputs are produced elsewhere (not in this package):

| Source | Path |
|--------|------|
| Isaac minimal rollout | `outputs/rollouts/<run>/<timestamp>/camera_videos/*.mp4` |
| Curobo dataset previews | `.../camera_videos/*.mp4` |
| Manual | `videos/*.mp4` |

## Setup

### GCS (client + worker)

```bash
export GCS_BUCKET=dreambc_videos
export GCS_PREFIX=cosmos-transfer
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/dreambc-e3a16d04725e.json
```

Service account needs **Storage Object Admin** on `dreambc_videos`.

### RunPod

API key is read from `~/.runpod/config.toml` (`[default].api_key`) if `RUNPOD_API_KEY` is unset.

```bash
export RUNPOD_ENDPOINT_ID=your-serverless-endpoint-id
```

Deploy the worker: [runpod/README.md](runpod/README.md).

### Local GPU (optional)

```bash
export HF_TOKEN=hf_...
bash cosmos_transfer/setup_cosmos.sh
```

## Usage

```bash
pip install -r cosmos_transfer/requirements.txt

# Isaac rollout (typical)
python -m cosmos_transfer \
  --rollout-dir outputs/rollouts/smolvla_pick_cube_eval/20260517_035426 \
  --camera exterior_image_1_left \
  --output-dir outputs/cosmos_transfer/20260517_035426

# All cameras in a rollout
python -m cosmos_transfer --rollout-dir outputs/rollouts/.../<timestamp>

# Legacy videos/ directory
python -m cosmos_transfer --videos-dir videos/

# Local GPU
python -m cosmos_transfer --local --videos-dir videos/

# Backward-compatible entrypoint
python sim2real.py --rollout-dir ...
```

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GCS_BUCKET` | RunPod path | `dreambc_videos` |
| `GOOGLE_APPLICATION_CREDENTIALS` | RunPod path | GCP SA JSON path |
| `RUNPOD_ENDPOINT_ID` | RunPod path | Serverless endpoint ID |
| `RUNPOD_API_KEY` | Optional | Overrides `~/.runpod/config.toml` |
| `GCS_PREFIX` | Optional | Default `cosmos-transfer` |
| `HF_TOKEN` | Local / Docker build | Hugging Face token |

## Notes

- Distilled model is trained for **≤93 frames**; longer videos may warn or degrade.
- RunPod `/run` payload limit ~10MB — always use GCS signed URLs, not inline video bytes.
