#!/usr/bin/env python3
"""
Pre-download Cosmos-Transfer2.5-2B distilled edge model weights.
Run this from inside the cosmos-transfer2.5 venv:
    source cosmos-transfer2.5/.venv/bin/activate
    python download_model.py
"""

import os
import sys

token = os.environ.get("HF_TOKEN")
if not token:
    print("ERROR: HF_TOKEN not set. Export it first:")
    print("  export HF_TOKEN=hf_...")
    sys.exit(1)

# Required to make the distilled edge checkpoint visible in the registry
os.environ["COSMOS_EXPERIMENTAL_CHECKPOINTS"] = "1"

try:
    from cosmos_oss.checkpoints_transfer2 import register_checkpoints
    from cosmos_transfer2._src.imaginaire.utils.checkpoint_db import download_checkpoint
except ImportError:
    print("ERROR: cosmos packages not found. Run this from inside the cosmos venv:")
    print("  source cosmos-transfer2.5/.venv/bin/activate")
    sys.exit(1)

register_checkpoints()

print("Downloading distilled edge controlnet...")
# UUID 41f07f13 = nvidia/Cosmos-Transfer2.5-2B distilled/general/edge checkpoint
path = download_checkpoint("41f07f13-f2e4-4e34-ba4c-86f595acbc20")
print(f"  → {path}")

print("\nBase backbone, VAE, and text encoder will auto-download on first inference run.")
print("Done.")
