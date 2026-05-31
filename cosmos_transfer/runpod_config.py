"""Load RunPod API key from environment or ~/.runpod/config.toml."""

from __future__ import annotations

import os
from pathlib import Path


def load_api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if key:
        return key
    config_path = Path.home() / ".runpod" / "config.toml"
    if not config_path.is_file():
        raise RuntimeError(
            "RUNPOD_API_KEY is not set and ~/.runpod/config.toml was not found. "
            "Log in with RunPod CLI or set RUNPOD_API_KEY."
        )
    return _parse_toml_api_key(config_path)


def _parse_toml_api_key(path: Path) -> str:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    with path.open("rb") as f:
        data = tomllib.load(f)
    default = data.get("default") or {}
    if isinstance(default, dict):
        key = default.get("api_key") or default.get("apiKey")
        if key:
            return str(key).strip()
    for section in data.values():
        if isinstance(section, dict):
            key = section.get("api_key") or section.get("apiKey")
            if key:
                return str(key).strip()
    raise RuntimeError(f"No api_key found in {path}")
