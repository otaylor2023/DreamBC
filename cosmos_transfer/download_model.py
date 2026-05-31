#!/usr/bin/env python3
"""
Pre-download Cosmos-Transfer2.5-2B distilled edge model weights.
Run from cosmos-transfer2.5 venv after setup_cosmos.sh:
    source cosmos-transfer2.5/.venv/bin/activate
    python -m cosmos_transfer.download_model
"""

import os
import sys


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set. Export it first:")
        print("  export HF_TOKEN=hf_...")
        sys.exit(1)

    os.environ["COSMOS_EXPERIMENTAL_CHECKPOINTS"] = "1"

    try:
        from cosmos_oss.checkpoints_transfer2 import register_checkpoints
        from cosmos_transfer2._src.imaginaire.utils.checkpoint_db import download_checkpoint
    except ImportError:
        print("ERROR: cosmos packages not found. Run setup_cosmos.sh first:")
        print("  bash cosmos_transfer/setup_cosmos.sh")
        sys.exit(1)

    register_checkpoints()

    print("Downloading distilled edge controlnet...")
    path = download_checkpoint("41f07f13-f2e4-4e34-ba4c-86f595acbc20")
    print(f"  → {path}")

    print("\nBase backbone, VAE, and text encoder will auto-download on first inference run.")
    print("Done.")


if __name__ == "__main__":
    main()
