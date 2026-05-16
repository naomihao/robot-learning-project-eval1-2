#!/usr/bin/env python3
"""
Eval1 deployment CLI — task1 and task2 robot rollouts.

task1: direct rollout, no prompt preprocessing
task2: capture camera frame at the start of each rollout, detect bowl positions,
       normalize the complex prompt to a canonical color-based form, then roll out.

Usage
-----
  # Task 1 — fixed task string, just run the rollout
  python run_eval.py task1 \\
    --task "Pick up the banana and put it into the red bowl."

  # Task 2 — complex prompt resolved via camera + VLM each rollout
  python run_eval.py task2 \\
    --task "Put the banana into the 2nd bowl from the left from the robot perspective"

  # Run 3 consecutive rollouts
  python run_eval.py task2 --n-rollouts 3 \\
    --task "Put the banana into the bowl that is not red and not blue"

  # Override port / camera / duration
  python run_eval.py task1 \\
    --robot-port /dev/ttyACM1 --camera-index 1 --duration 30 \\
    --task "Pick up the banana and put it into the red bowl."
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


# ── Defaults matching the provided lerobot-rollout example ────────────────────

PRETRAINED_PATH = "RobotLearningVLA/test_eval1"
ROBOT_TYPE = "so101_follower"
ROBOT_ID = "my_awesome_follower_arm"
EMPTY_CAMERAS = 2
STRATEGY_TYPE = "base"
RENAME_MAP = {"observation.images.front": "observation.images.camera1"}
INPUT_FEATURES = {
    "observation.state": {"type": "STATE", "shape": [6]},
    "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]},
}
OUTPUT_FEATURES = {"action": {"type": "ACTION", "shape": [6]}}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _cameras_json(index: int, width: int, height: int, fps: int) -> str:
    return json.dumps({
        "front": {
            "type": "opencv",
            "index_or_path": index,
            "width": width,
            "height": height,
            "fps": fps,
        }
    })


def _build_rollout_cmd(args: argparse.Namespace, task: str) -> list[str]:
    return [
        "lerobot-rollout",
        f"--robot.type={args.robot_type}",
        f"--robot.port={args.robot_port}",
        f"--robot.id={args.robot_id}",
        f"--robot.cameras={_cameras_json(args.camera_index, args.cam_width, args.cam_height, args.cam_fps)}",
        "--policy.type=smolvla",
        f"--policy.pretrained_path={args.pretrained_path}",
        f"--policy.input_features={json.dumps(INPUT_FEATURES)}",
        f"--policy.output_features={json.dumps(OUTPUT_FEATURES)}",
        f"--policy.empty_cameras={args.empty_cameras}",
        f"--strategy.type={args.strategy_type}",
        f"--duration={args.duration}",
        f"--task={task}",
        f"--device={args.device}",
        f"--rename_map={json.dumps(RENAME_MAP)}",
    ]


def _capture_frame(camera_index: int) -> "np.ndarray":
    """Capture a single RGB frame from the camera using OpenCV."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {camera_index}")

    # Discard the first few frames so auto-exposure can settle
    for _ in range(10):
        ret, frame = cap.read()

    cap.release()

    if not ret:
        raise RuntimeError("Failed to read a frame from the camera")

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _load_normalizer(device: str):
    """Load PromptNormalizer backed by SmolVLM2-500M (same backbone as SmolVLA)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "task2_prompt"))
    from eval1_prompt_normalizer import PromptNormalizer

    print("[task2] Loading SmolVLM for prompt normalization...")
    normalizer = PromptNormalizer.from_pretrained(device=device)
    print("[task2] SmolVLM loaded.")
    return normalizer


# ── Task runners ───────────────────────────────────────────────────────────────

def run_task1(args: argparse.Namespace) -> None:
    for i in range(args.n_rollouts):
        if args.n_rollouts > 1:
            print(f"\n[task1] Rollout {i + 1}/{args.n_rollouts}")
        print(f"[task1] task: {args.task!r}")
        cmd = _build_rollout_cmd(args, args.task)
        _print_cmd(cmd)
        subprocess.run(cmd, check=True)


def run_task2(args: argparse.Namespace) -> None:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    normalizer = _load_normalizer(args.device)

    for i in range(args.n_rollouts):
        print(f"\n[task2] Rollout {i + 1}/{args.n_rollouts}")

        # 1. Capture frame
        print(f"[task2] Capturing frame from camera index {args.camera_index}...")
        frame = _capture_frame(args.camera_index)
        print(f"[task2] Frame captured: {frame.shape}")

        # 2. Detect bowl layout (position of objects) at the start of each rollout
        print("[task2] Detecting bowl layout...")
        normalizer._robot_order = None          # reset so layout is re-detected
        layout = normalizer.detect_layout(frame)
        if layout is not None:
            print(f"[task2] Bowl layout (robot L→R): {layout}")
        else:
            print("[task2] Layout detection failed — will use VLM fallback")

        # 3. Normalize the prompt
        print(f"[task2] Raw prompt:        {args.task!r}")
        task = normalizer.normalize(frame, args.task)
        print(f"[task2] Normalized prompt: {task!r}")

        # 4. Run rollout with the canonical task string
        cmd = _build_rollout_cmd(args, task)
        _print_cmd(cmd)
        subprocess.run(cmd, check=True)


def _print_cmd(cmd: list[str]) -> None:
    print("[run]  " + " \\\n       ".join(cmd))


# ── Argument parsing ───────────────────────────────────────────────────────────

def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--task",
        required=True,
        help="Task instruction string (task1: used verbatim; task2: normalized first).",
    )
    p.add_argument("--robot-type",      default=ROBOT_TYPE)
    p.add_argument("--robot-port",      default="/dev/ttyACM0")
    p.add_argument("--robot-id",        default=ROBOT_ID)
    p.add_argument("--camera-index",    type=int, default=0)
    p.add_argument("--cam-width",       type=int, default=640)
    p.add_argument("--cam-height",      type=int, default=480)
    p.add_argument("--cam-fps",         type=int, default=30)
    p.add_argument("--pretrained-path", default=PRETRAINED_PATH)
    p.add_argument("--empty-cameras",   type=int, default=EMPTY_CAMERAS)
    p.add_argument("--strategy-type",   default=STRATEGY_TYPE)
    p.add_argument("--duration",        type=int, default=20,
                   help="Rollout duration in seconds.")
    p.add_argument("--device",          default="cuda")
    p.add_argument("--n-rollouts",      type=int, default=1,
                   help="Number of consecutive rollouts to run.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Eval1 task1 / task2 robot rollout CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser(
        "task1",
        help="Direct rollout — task string used verbatim, no prompt preprocessing.",
    )
    _add_common_args(p1)

    p2 = sub.add_parser(
        "task2",
        help=(
            "Rollout with prompt normalization — captures a camera frame at the "
            "start of each rollout, detects bowl positions, resolves the complex "
            "prompt to a canonical color-based form, then runs the rollout."
        ),
    )
    _add_common_args(p2)

    return parser.parse_args()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    if args.command == "task1":
        run_task1(args)
    elif args.command == "task2":
        run_task2(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
