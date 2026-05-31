"""
RunPod Serverless worker for Cosmos Transfer 2.5 distilled edge.

Loads the model once at worker start; each job downloads input from signed URLs,
runs inference, uploads output to GCS.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path

import runpod

# Shared modules copied into the image beside this file.
from gcs import download_url, signed_url, upload_file
from spec import DEFAULT_PROMPT, make_spec

os.environ.setdefault("COSMOS_EXPERIMENTAL_CHECKPOINTS", "1")

_INFERENCE = None
_WORKER_BASE = Path("/tmp/cosmos_worker_base")


def _get_inference():
    global _INFERENCE
    if _INFERENCE is not None:
        return _INFERENCE

    from cosmos_oss.init import init_environment

    init_environment()

    from cosmos_transfer2.config import InferenceArguments, InferenceOverrides, SetupArguments
    from cosmos_transfer2.inference import Control2WorldInference

    _WORKER_BASE.mkdir(parents=True, exist_ok=True)
    setup = SetupArguments(
        output_dir=_WORKER_BASE,
        model="edge/distilled",
        disable_guardrails=True,
    )
    _INFERENCE = Control2WorldInference(setup, batch_hint_keys=["edge"])
    return _INFERENCE


def handler(job: dict) -> dict:
    job_id = job.get("id") or uuid.uuid4().hex
    job_input = job.get("input") or {}

    video_url = job_input.get("video_url")
    if not video_url:
        return {"error": "Missing required field: video_url"}

    prompt = job_input.get("prompt") or DEFAULT_PROMPT
    name = job_input.get("name") or "output"
    scene_url = job_input.get("scene_url")

    runpod.serverless.progress_update(job, "Downloading inputs")

    with tempfile.TemporaryDirectory(prefix="cosmos_job_") as tmp:
        work = Path(tmp)
        video_path = work / "input.mp4"
        download_url(video_url, video_path)

        scene_path = None
        if scene_url:
            scene_path = work / "scene.jpg"
            download_url(scene_url, scene_path)

        spec = make_spec(video_path, prompt, scene_path=scene_path, name=name)
        spec_path = work / "spec.json"
        spec_path.write_text(json.dumps(spec, indent=2))

        from cosmos_transfer2.config import InferenceArguments, InferenceOverrides

        samples, _batch_keys = InferenceArguments.from_files(
            [spec_path],
            overrides=InferenceOverrides(),
        )

        out_dir = work / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        runpod.serverless.progress_update(job, "Running Cosmos Transfer inference")
        inference = _get_inference()
        output_paths = inference.generate(samples, output_dir=out_dir)

        if not output_paths:
            return {"error": "Inference produced no output"}

        out_mp4 = Path(output_paths[0])
        if not out_mp4.is_file():
            candidate = out_dir / f"{name}.mp4"
            out_mp4 = candidate if candidate.is_file() else Path(output_paths[0])

        runpod.serverless.progress_update(job, "Uploading output")
        gcs_job = job_id.replace("/", "_")[:64]
        output_gs = upload_file(out_mp4, kind="outputs", job_id=gcs_job)
        output_url = signed_url(output_gs)

        return {
            "name": name,
            "output_gs_uri": output_gs,
            "output_url": output_url,
        }


if __name__ == "__main__":
    # Warm-load model before accepting jobs (best-effort).
    try:
        _get_inference()
    except Exception as exc:
        print(f"WARNING: model preload failed: {exc}")
    runpod.serverless.start({"handler": handler})
