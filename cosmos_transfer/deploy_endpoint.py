#!/usr/bin/env python3
"""Create or update RunPod Serverless endpoint for cosmos_transfer worker."""

from __future__ import annotations

import argparse
import json
import os
import sys

from cosmos_transfer.runpod_config import load_api_key


def _api_key() -> str:
    return os.environ.get("RUNPOD_API_KEY") or load_api_key()


def create_template(
    *,
    name: str,
    image: str,
    container_disk_gb: int = 100,
    env: dict[str, str] | None = None,
) -> str:
    import runpod
    from runpod.api.graphql import run_graphql_query
    from runpod.api.mutations import templates as template_mutations

    runpod.api_key = _api_key()
    env = env or {}
    mutation = template_mutations.generate_pod_template(
        name=name,
        image_name=image,
        container_disk_in_gb=container_disk_gb,
        is_serverless=True,
        env=env,
    )
    resp = run_graphql_query(mutation)
    if "errors" in resp:
        raise RuntimeError(resp["errors"])
    return resp["data"]["saveTemplate"]["id"]


def create_endpoint(
    *,
    name: str,
    template_id: str,
    gpu_id: str = "NVIDIA H100 80GB HBM3",
    workers_max: int = 1,
    workers_min: int = 0,
    idle_timeout: int = 300,
    execution_timeout: int = 1800,
) -> str:
    import runpod
    from runpod.api import ctl_commands

    runpod.api.key = _api_key()  # noqa: SLF001 — runpod SDK uses .api.key in some versions
    runpod.api_key = _api_key()

    endpoint = ctl_commands.create_endpoint(
        name=name,
        template_id=template_id,
        gpu_ids=gpu_id,
        workers_min=workers_min,
        workers_max=workers_max,
        idle_timeout=idle_timeout,
        scaler_type="QUEUE_DELAY",
        scaler_value=4,
        flashboot=True,
        gpu_count=1,
    )
    return endpoint["id"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy cosmos_transfer RunPod endpoint")
    parser.add_argument("--image", required=True, help="Docker image (e.g. user/dreambc-cosmos-transfer:v1)")
    parser.add_argument("--name", default="dreambc-cosmos-transfer")
    parser.add_argument("--gpu-id", default="NVIDIA H100 80GB HBM3")
    parser.add_argument("--container-disk-gb", type=int, default=100)
    parser.add_argument("--template-only", action="store_true")
    args = parser.parse_args()

    env = {}
    for key in ("GCS_BUCKET", "GCS_PREFIX", "COSMOS_EXPERIMENTAL_CHECKPOINTS"):
        if os.environ.get(key):
            env[key] = os.environ[key]
    gcp = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if gcp and os.path.isfile(gcp):
        env["GOOGLE_APPLICATION_CREDENTIALS"] = "/secrets/gcp.json"

    print(f"Creating template with image {args.image}...")
    template_id = create_template(
        name=f"{args.name}-template",
        image=args.image,
        container_disk_gb=args.container_disk_gb,
        env=env,
    )
    print(f"  template_id={template_id}")

    if args.template_only:
        return

    print("Creating endpoint...")
    endpoint_id = create_endpoint(
        name=args.name,
        template_id=template_id,
        gpu_id=args.gpu_id,
    )
    print(f"  endpoint_id={endpoint_id}")
    print("\nAdd to .env:")
    print(f"RUNPOD_ENDPOINT_ID={endpoint_id}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
