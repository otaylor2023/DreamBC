"""Submit and poll RunPod Serverless jobs."""

from __future__ import annotations

import os
from typing import Any

from cosmos_transfer.runpod_config import load_api_key


def endpoint_id() -> str:
    eid = os.environ.get("RUNPOD_ENDPOINT_ID", "").strip()
    if not eid:
        raise RuntimeError("RUNPOD_ENDPOINT_ID is not set")
    return eid


def _endpoint():
    import runpod

    runpod.api_key = load_api_key()
    return runpod.Endpoint(endpoint_id())


def run_and_wait(job_input: dict[str, Any], *, timeout: int = 3600) -> dict[str, Any]:
    """Submit async job and block until output is ready."""
    job = _endpoint().run(job_input)
    output = job.output(timeout=timeout)
    return _normalize_output(output)


def run_sync(job_input: dict[str, Any], *, timeout: int = 3600) -> dict[str, Any]:
    """Run via /runsync."""
    result = _endpoint().run_sync(job_input, timeout=timeout)
    return _normalize_output(result)


def _normalize_output(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    return {"result": result}
