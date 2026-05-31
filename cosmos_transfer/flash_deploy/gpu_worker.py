"""Flash-deployed pipeline test worker (video pass-through via GCS)."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path

from runpod_flash import Endpoint, GpuType


def _ensure_gcp_credentials() -> None:
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return
    raw = os.environ.get("GCP_SA_JSON")
    if not raw:
        return
    path = Path(tempfile.gettempdir()) / "gcp_sa.json"
    path.write_text(raw)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(path)


_ENDPOINT_ENV: dict[str, str] = {
    "GCS_BUCKET": os.environ.get("GCS_BUCKET", "dreambc_videos"),
    "GCS_PREFIX": os.environ.get("GCS_PREFIX", "cosmos-transfer"),
}
if os.environ.get("GCP_SA_JSON"):
    _ENDPOINT_ENV["GCP_SA_JSON"] = os.environ["GCP_SA_JSON"]


@Endpoint(
    name="dreambc_cosmos_transfer",
    gpu=GpuType.NVIDIA_GEFORCE_RTX_4090,
    workers=(0, 1),
    idle_timeout=300,
    execution_timeout_ms=1_800_000,
    env=_ENDPOINT_ENV,
    dependencies=["google-cloud-storage", "runpod"],
)
async def transfer(job_input: dict) -> dict:
    """Download video from signed URL, upload pass-through output to GCS."""
    _ensure_gcp_credentials()

    from google.cloud import storage
    import urllib.request

    video_url = job_input.get("video_url")
    if not video_url:
        return {"error": "Missing video_url"}

    bucket_name = os.environ["GCS_BUCKET"]
    prefix = os.environ.get("GCS_PREFIX", "cosmos-transfer").strip("/")
    name = job_input.get("name") or "output"
    job_id = uuid.uuid4().hex

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        inp = work / "input.mp4"
        out = work / "output.mp4"
        urllib.request.urlretrieve(video_url, inp)
        out.write_bytes(inp.read_bytes())

        client = storage.Client()
        blob_name = f"{prefix}/outputs/{job_id}/{name}.mp4"
        blob = client.bucket(bucket_name).blob(blob_name)
        blob.upload_from_filename(str(out))

        gs_uri = f"gs://{bucket_name}/{blob_name}"
        url = blob.generate_signed_url(
            version="v4",
            expiration=7200,
            method="GET",
        )
        return {
            "name": name,
            "output_gs_uri": gs_uri,
            "output_url": url,
            "stub": True,
        }


if __name__ == "__main__":
    import asyncio

    print(asyncio.run(transfer({"video_url": "file://test", "name": "test"})))
