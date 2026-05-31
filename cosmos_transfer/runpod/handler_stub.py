"""Stub worker for pipeline testing (copies input video; no Cosmos GPU inference)."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import runpod

from gcs import download_url, signed_url, upload_file
from spec import DEFAULT_PROMPT


def handler(job: dict) -> dict:
    job_id = job.get("id") or uuid.uuid4().hex
    job_input = job.get("input") or {}
    video_url = job_input.get("video_url")
    if not video_url:
        return {"error": "Missing video_url"}

    name = job_input.get("name") or "output"
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        out_path = work / "output.mp4"
        download_url(video_url, work / "input.mp4")
        # Pass-through copy for pipeline validation.
        import shutil

        shutil.copy(work / "input.mp4", out_path)

        gcs_job = job_id.replace("/", "_")[:64]
        output_gs = upload_file(out_path, kind="outputs", job_id=gcs_job)
        return {
            "name": name,
            "output_gs_uri": output_gs,
            "output_url": signed_url(output_gs),
            "stub": True,
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
