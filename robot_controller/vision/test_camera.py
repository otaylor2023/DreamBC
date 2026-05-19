"""Stream color and depth frames from an Intel RealSense camera.

Usage:
    python DreamBC/robot_controller/vision/test_camera.py
    python DreamBC/robot_controller/vision/test_camera.py --width 1280 --height 720 --fps 30

Press 'q' or ESC in the preview window to quit.
"""

import argparse
import sys

import cv2
import numpy as np
import pyrealsense2 as rs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RealSense streaming test.")
    parser.add_argument("--width", type=int, default=640, help="Frame width.")
    parser.add_argument("--height", type=int, default=480, help="Frame height.")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate.")
    parser.add_argument(
        "--no-depth",
        action="store_true",
        help="Disable depth stream (color only).",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=10000,
        help="Per-frame wait timeout in ms (default 10000).",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=30,
        help="Frames to discard before display (lets auto-exposure settle).",
    )
    return parser.parse_args()


def list_connected_devices() -> list[str]:
    ctx = rs.context()
    return [
        f"{d.get_info(rs.camera_info.name)} "
        f"(SN: {d.get_info(rs.camera_info.serial_number)})"
        for d in ctx.query_devices()
    ]


def build_display(frames, show_depth: bool) -> np.ndarray | None:
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None
    color_image = np.asanyarray(color_frame.get_data())

    if not show_depth:
        return color_image

    depth_frame = frames.get_depth_frame()
    if not depth_frame:
        return None
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
    )
    return np.hstack((color_image, depth_colormap))


def frames_to_rgbd(
    frames,
    align: rs.align | None,
    depth_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Aligned BGR, depth in meters, depth colormap — for capture saves."""
    if align is not None:
        frames = align.process(frames)
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if not color_frame or not depth_frame:
        return None
    color_bgr = np.asanyarray(color_frame.get_data())
    depth_u16 = np.asanyarray(depth_frame.get_data())
    depth_m = depth_u16.astype(np.float32) * depth_scale
    depth_vis = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_u16, alpha=0.03), cv2.COLORMAP_JET
    )
    return color_bgr, depth_m, depth_vis


def main() -> int:
    args = parse_args()

    devices = list_connected_devices()
    if not devices:
        print("No RealSense devices found. Is the camera plugged in?", file=sys.stderr)
        return 1
    print(f"Found {len(devices)} RealSense device(s):")
    for d in devices:
        print(f"  - {d}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps
    )
    if not args.no_depth:
        config.enable_stream(
            rs.stream.depth, args.width, args.height, rs.format.z16, args.fps
        )

    profile = pipeline.start(config)

    align = rs.align(rs.stream.color) if not args.no_depth else None

    depth_scale = 1.0
    if not args.no_depth:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(f"Depth scale: {depth_scale:.6f} m/unit")

    window_name = "RealSense"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    show_depth = not args.no_depth

    print(f"Warming up ({args.warmup_frames} frames)...")
    for _ in range(args.warmup_frames):
        try:
            pipeline.wait_for_frames(args.timeout_ms)
        except RuntimeError:
            pass

    consecutive_misses = 0
    max_consecutive_misses = 10
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

            if align is not None:
                frames = align.process(frames)

            display = build_display(frames, show_depth)
            if display is None:
                continue

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
