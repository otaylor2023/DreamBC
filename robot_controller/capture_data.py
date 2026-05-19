#!/usr/bin/env python3
"""Read-only RGB-D capture for DreamBC object datasets.

Live RealSense preview; SPACE or ``s`` saves aligned RGB, depth, and joint state.
Does not write to Redis or command the robot — safe alongside teleop/controllers.

Usage::

  conda activate dreambc_robot
  pip install -r DreamBC/robot_controller/requirements.txt   # first time only
  python DreamBC/robot_controller/capture_data.py apple

Keys: SPACE or ``s`` = save | q / ESC = quit (click the preview window first).

Camera preview uses the same RealSense loop as ``vision/test_camera.py``.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import redis

_ROBOT_CONTROLLER_DIR = Path(__file__).resolve().parent
if str(_ROBOT_CONTROLLER_DIR) not in sys.path:
    sys.path.insert(0, str(_ROBOT_CONTROLLER_DIR))

from vision import test_camera as cam  # noqa: E402

_CAPTURE_DIR_RE = re.compile(r"^capture_(\d+)$", re.IGNORECASE)


def resolve_data_root(path: str | Path) -> Path:
    """Resolve dataset root; relative paths are under ``robot_controller/``."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = _ROBOT_CONTROLLER_DIR / p
    return p.resolve()


_DEFAULT_DATA_ROOT = resolve_data_root("data")
_SAVE_KEYS = frozenset({ord(" "), ord("s")})

# Franka driver may use opensai:: or sai:: depending on webui config.
_JOINT_POSITION_KEYS = (
    "opensai::sensors::FrankaRobot::joint_positions",
    "sai::sensors::FrankaRobot::joint_positions",
)


def parse_redis_list(raw: bytes | str | None) -> np.ndarray | None:
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    raw = raw.strip()
    if not raw:
        return None
    if raw.startswith("["):
        try:
            values = ast.literal_eval(raw)
            if isinstance(values, list):
                return np.array(values, dtype=np.float64)
        except (SyntaxError, ValueError):
            pass
    try:
        values = json.loads(raw)
        if isinstance(values, list):
            return np.array(values, dtype=np.float64)
    except json.JSONDecodeError:
        pass
    # SaiCommon custom vector: "j0 j1 j2 ..." or "j0; j1; ..."
    parts = re.split(r"[;\s]+", raw)
    floats: list[float] = []
    for part in parts:
        if not part:
            continue
        try:
            floats.append(float(part))
        except ValueError:
            floats.clear()
            break
    if len(floats) >= 7:
        return np.array(floats[:7], dtype=np.float64)
    return None


def sanitize_object_name(name: str) -> str:
    """Single folder name under data/ (e.g. ``apple`` → ``data/apple/``)."""
    cleaned = name.strip()
    if not cleaned or cleaned in (".", ".."):
        raise ValueError(f"Invalid object name: {name!r}")
    if "/" in cleaned or "\\" in cleaned:
        raise ValueError(
            f"Object name must be one folder (no path separators), got {name!r}. "
            "Example: python capture_data.py apple"
        )
    return cleaned


def ensure_object_root(data_root: Path, object_name: str) -> Path:
    """Create and return ``<data_root>/<object>/`` (always under robot_controller/data/)."""
    root = resolve_data_root(data_root)
    object_root = (root / sanitize_object_name(object_name)).resolve()
    object_root.mkdir(parents=True, exist_ok=True)
    if not object_root.is_dir():
        raise OSError(f"Could not create object folder: {object_root}")
    return object_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DreamBC read-only RGB-D + robot state capture."
    )
    p.add_argument("object", help="Object label (output folder under data-root).")
    p.add_argument(
        "--data-root",
        type=resolve_data_root,
        default=_DEFAULT_DATA_ROOT,
        help=f"Dataset root (default: {_DEFAULT_DATA_ROOT}).",
    )
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--warmup-frames", type=int, default=30)
    p.add_argument("--timeout-ms", type=int, default=10000)
    p.add_argument("--redis-host", default="localhost")
    p.add_argument("--redis-port", type=int, default=6379)
    p.add_argument(
        "--no-redis",
        action="store_true",
        help="Save RGB-D only (robot.json will have joint_positions=null).",
    )
    return p.parse_args()


def connect_redis(host: str, port: int) -> redis.Redis | None:
    try:
        client = redis.Redis(host=host, port=port, decode_responses=True)
        client.ping()
        return client
    except redis.RedisError as e:
        print(f"Warning: Redis unavailable ({e}). Saves will omit joint positions.", file=sys.stderr)
        return None


def read_robot_state(client: redis.Redis | None) -> dict:
    """Read 7-DOF joint positions from Redis (GET only)."""
    ts = time.time()
    if client is None:
        return {"joint_positions": None, "timestamp_unix": ts, "redis_error": "no redis client"}

    last_err = "no key had 7 joint values"
    for key in _JOINT_POSITION_KEYS:
        raw_q = client.get(key)
        joints = parse_redis_list(raw_q)
        if joints is not None and joints.size == 7:
            return {
                "joint_positions": joints.tolist(),
                "timestamp_unix": ts,
                "joint_positions_key": key,
            }
        last_err = (
            f"{key!r}: "
            f"{'missing' if raw_q is None else repr(raw_q[:80]) + ('...' if len(str(raw_q)) > 80 else '')}"
        )

    return {
        "joint_positions": None,
        "timestamp_unix": ts,
        "redis_error": last_err,
    }


def next_capture_dir(object_root: Path) -> Path:
    object_root.mkdir(parents=True, exist_ok=True)
    max_idx = 0
    if object_root.is_dir():
        for child in object_root.iterdir():
            if not child.is_dir():
                continue
            m = _CAPTURE_DIR_RE.match(child.name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
    cap_dir = object_root / f"capture_{max_idx + 1:04d}"
    cap_dir.mkdir(parents=True, exist_ok=True)
    return cap_dir


def save_capture(
    cap_dir: Path,
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    depth_vis: np.ndarray,
    robot_state: dict,
) -> None:
    rgb_path = cap_dir / "rgb.png"
    depth_npy_path = cap_dir / "depth.npy"
    depth_png_path = cap_dir / "depth.png"
    robot_path = cap_dir / "robot.json"

    if not cv2.imwrite(str(rgb_path), color_bgr):
        raise OSError(f"cv2.imwrite failed: {rgb_path}")
    np.save(depth_npy_path, depth_m.astype(np.float32))
    if not cv2.imwrite(str(depth_png_path), depth_vis):
        raise OSError(f"cv2.imwrite failed: {depth_png_path}")
    with open(robot_path, "w", encoding="utf-8") as f:
        json.dump(robot_state, f, indent=2)
        f.write("\n")

    for p in (rgb_path, depth_npy_path, depth_png_path, robot_path):
        if not p.is_file() or p.stat().st_size == 0:
            raise OSError(f"Save incomplete: {p}")


def draw_overlay(
    display: np.ndarray,
    object_name: str,
    object_root: Path,
    capture_count: int,
    last_id: str | None,
    status: str | None,
) -> np.ndarray:
    out = display.copy()
    lines = [
        f"object: {object_name}",
        f"dir: {object_root}",
        f"captures: {capture_count}",
        "SPACE/s=save  q/ESC=quit (focus this window)",
    ]
    if last_id:
        lines.insert(2, f"last: {last_id}")
    if status:
        lines.append(status)
    y = 28
    for line in lines:
        color = (80, 220, 120) if status and status.startswith("Saved") else (240, 240, 240)
        if status and (status.startswith("Skip") or status.startswith("No frame")):
            color = (80, 80, 255)
        cv2.putText(
            out,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 26
    return out


def start_realsense_pipeline(args: argparse.Namespace):
    """Same startup path as ``vision/test_camera.py`` (proven preview loop)."""
    devices = cam.list_connected_devices()
    if not devices:
        raise RuntimeError(
            "No RealSense devices found. Is the camera plugged in?"
        )
    print(f"Found {len(devices)} RealSense device(s):")
    for d in devices:
        print(f"  - {d}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps
    )
    config.enable_stream(
        rs.stream.depth, args.width, args.height, rs.format.z16, args.fps
    )
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print(f"Depth scale: {depth_scale:.6f} m/unit")

    print(f"Warming up ({args.warmup_frames} frames)...")
    for _ in range(args.warmup_frames):
        try:
            pipeline.wait_for_frames(args.timeout_ms)
        except RuntimeError:
            pass

    return pipeline, align, depth_scale


def run_capture(args: argparse.Namespace) -> int:
    object_name = sanitize_object_name(args.object)
    data_root = resolve_data_root(args.data_root)
    object_root = ensure_object_root(data_root, object_name)

    redis_client = None if args.no_redis else connect_redis(args.redis_host, args.redis_port)

    pipeline = None
    win = f"DreamBC capture: {object_name}"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    capture_count = 0
    last_capture_id: str | None = None
    status_msg: str | None = None
    status_until = 0.0
    consecutive_misses = 0
    max_consecutive_misses = 10
    latched_rgbd: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    print(f"Data root:     {data_root}")
    print(f"Object folder: {object_root}")
    print(f"  saves to     {object_root}/capture_XXXX/{{rgb.png, depth.npy, ...}}")
    print("Read-only mode: no Redis writes or robot commands.")
    print("Click the preview window, then press SPACE or s to save.")

    try:
        pipeline, align, depth_scale = start_realsense_pipeline(args)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1

    try:
        while True:
            ok, frames = pipeline.try_wait_for_frames(args.timeout_ms)
            if not ok:
                consecutive_misses += 1
                print(
                    f"Frame didn't arrive within {args.timeout_ms} ms "
                    f"(miss {consecutive_misses}/{max_consecutive_misses})."
                )
                if consecutive_misses >= max_consecutive_misses:
                    print(
                        "Too many consecutive timeouts. Check USB 3 connection, "
                        "try a different port/cable, or lower --fps / --width / --height.",
                        file=sys.stderr,
                    )
                    return 2
                continue
            consecutive_misses = 0

            frames = align.process(frames)
            display = cam.build_display(frames, show_depth=True)
            latched_rgbd = cam.frames_to_rgbd(frames, None, depth_scale)
            if display is None:
                continue

            now = time.time()
            show_status = status_msg if status_msg and now < status_until else None
            cv2.imshow(
                win,
                draw_overlay(
                    display,
                    object_name,
                    object_root,
                    capture_count,
                    last_capture_id,
                    show_status,
                ),
            )
            key = cv2.waitKey(30) & 0xFF

            if key in (ord("q"), 27):
                break
            if key not in _SAVE_KEYS:
                continue

            if latched_rgbd is None:
                status_msg = "No frame — try again"
                status_until = now + 2.0
                print(status_msg, file=sys.stderr)
                continue

            color_bgr, depth_m, depth_vis = latched_rgbd
            robot_state = read_robot_state(redis_client)

            try:
                cap_dir = next_capture_dir(object_root)
                save_capture(cap_dir, color_bgr, depth_m, depth_vis, robot_state)
            except OSError as e:
                status_msg = f"Save failed: {e}"
                status_until = now + 3.0
                print(status_msg, file=sys.stderr)
                continue

            capture_count += 1
            last_capture_id = cap_dir.name
            if robot_state.get("joint_positions") is None:
                warn = robot_state.get("redis_error", "joint_positions missing")
                status_msg = f"Saved {cap_dir.name} (no joints: {warn})"
                print(status_msg, file=sys.stderr)
            else:
                status_msg = f"Saved {cap_dir.name}"
            status_until = now + 2.5
            print(f"Saved {cap_dir.resolve()}")
    finally:
        if pipeline is not None:
            pipeline.stop()
        cv2.destroyAllWindows()

    return 0


def main() -> int:
    args = parse_args()
    try:
        sanitize_object_name(args.object)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    return run_capture(args)


if __name__ == "__main__":
    sys.exit(main())
