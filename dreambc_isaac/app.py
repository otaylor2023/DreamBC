"""Isaac Sim application launch helpers."""

from __future__ import annotations


def simulation_app_config(headless: bool) -> dict:
    config = {"headless": headless}
    if not headless:
        config["extra_args"] = [
            "--enable",
            "omni.kit.test",
            "--/exts/omni.kit.test/runTestsAndQuit=false",
            "--/exts/omni.kit.test/printTestsAndQuit=false",
        ]
    return config

