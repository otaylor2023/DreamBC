# Cosmos Transfer RunPod Serverless Worker

GPU worker for Cosmos Transfer 2.5 **edge/distilled** (4-step) sim-to-real.

## Requirements

- **GPU:** H100 80GB (or any GPU with ≥65GB VRAM)
- **Secrets on endpoint:**
  - `GCS_BUCKET=dreambc_videos`
  - `GCS_PREFIX=cosmos-transfer` (optional)
  - `GOOGLE_APPLICATION_CREDENTIALS` — path to mounted SA JSON, **or** set env vars from the key file
  - `HF_TOKEN` — only if weights were not baked into the image at build time
  - `COSMOS_EXPERIMENTAL_CHECKPOINTS=1` (set in Dockerfile)

Mount the DreamBC service account JSON at e.g. `/secrets/gcp.json` and set:

```text
GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcp.json
```

## Build and push

From the **DreamBC repo root**:

```bash
export HF_TOKEN=hf_...   # bake distilled edge weights into image
docker build -f cosmos_transfer/runpod/Dockerfile \
  --build-arg HF_TOKEN="$HF_TOKEN" \
  -t YOUR_REGISTRY/dreambc-cosmos-transfer:latest .
docker push YOUR_REGISTRY/dreambc-cosmos-transfer:latest
```

## Create endpoint (RunPod console or CLI)

| Setting | Value |
|---------|--------|
| Image | `YOUR_REGISTRY/dreambc-cosmos-transfer:latest` |
| GPU | H100 80GB |
| Container disk | ≥100 GB recommended |
| Execution timeout | 1800 s (30 min) or higher |
| Idle timeout | per your budget |

Copy the endpoint ID into DreamBC `.env`:

```bash
RUNPOD_ENDPOINT_ID=your-endpoint-id
```

## Job input / output

**Input** (`job["input"]`):

```json
{
  "video_url": "https://storage.googleapis.com/...",
  "prompt": "A robot arm ...",
  "scene_url": "https://storage.googleapis.com/...",
  "name": "exterior_image_1_left"
}
```

**Output:**

```json
{
  "name": "exterior_image_1_left",
  "output_url": "https://storage.googleapis.com/...",
  "output_gs_uri": "gs://dreambc_videos/cosmos-transfer/outputs/..."
}
```

## Local handler smoke test

Requires cosmos-transfer2.5 venv on a GPU machine:

```bash
python handler.py --test_input '{"input": {"video_url": "SIGNED_URL", "prompt": "test"}}'
```
