"""Repository paths shared across cosmos_transfer."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
COSMOS_DIR = REPO_ROOT / "cosmos-transfer2.5"
COSMOS_PYTHON = COSMOS_DIR / ".venv" / "bin" / "python"
DEFAULT_VIDEOS_DIR = REPO_ROOT / "videos"
DEFAULT_OUTPUT_DIR = DEFAULT_VIDEOS_DIR / "output"
