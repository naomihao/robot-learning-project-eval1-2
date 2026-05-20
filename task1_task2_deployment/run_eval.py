#!/usr/bin/env python3
"""
Eval1 deployment CLI — task1 and task2 robot rollouts.

task1: normalize color prompts to the canonical policy wording, then roll out.
task2: normalize direct, negation, analogy, and spatial prompts to the canonical
       policy wording, then roll out. Spatial prompts use camera color layout.

Usage
-----
  # Task 1 — fixed task string, just run the rollout
  python run_eval.py task1 \\
    --task "Pick up the banana and put it into the red bowl."

  # Task 2 — complex prompt resolved via deterministic preprocessing each rollout
  python run_eval.py task2 \\
    --task "Put the banana into the 2nd bowl from the left from the robot perspective"

  # Run 3 consecutive rollouts
  python run_eval.py task2 --n-rollouts 3 \\
    --task "Put the banana into the bowl that is not red and not blue"

  # Interactive TA/operator interface, one typed instruction per round
  python run_eval.py task1 --interactive --duration 20

  # Interactive with the model loaded once for all rounds
  python run_eval.py task2 --interactive --backend persistent --duration 20

  # Interactive with automatic return to the pose captured before each rollout
  python run_eval.py task1 --interactive --duration 20

  # Interactive with a key press before running the reset command
  python run_eval.py task1 --interactive --duration 20 \\
    --reset-trigger enter --reset-command "python scripts/go_home.py"

  # Override port / camera / duration
  python run_eval.py task1 \\
    --robot-port /dev/ttyACM1 --camera-index 1 --duration 30 \\
    --task "Pick up the banana and put it into the red bowl."
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import select
import shutil
import shlex
import signal
import subprocess
import sys
import time


def _bootstrap_lerobot_path() -> None:
    """Make common LeRobot env launchers visible without a manual export."""
    here = os.path.abspath(os.path.dirname(__file__))
    eval_root = os.path.abspath(os.path.join(here, ".."))
    repo_root = os.path.abspath(os.path.join(eval_root, ".."))
    home = os.path.expanduser("~")

    candidates = [
        os.environ.get("LEROBOT_ENV_BIN"),
        os.path.join(home, ".conda", "envs", "lerobot", "bin"),
        os.path.join(home, "miniforge3", "envs", "lerobot", "bin"),
        os.path.join(home, "miniconda3", "envs", "lerobot", "bin"),
        os.path.join(home, "anaconda3", "envs", "lerobot", "bin"),
        os.path.join(eval_root, ".venv", "bin"),
        os.path.join(repo_root, ".venv", "bin"),
        os.path.join(repo_root, "robot-learning-vla", ".venv", "bin"),
    ]

    old_parts = os.environ.get("PATH", "").split(os.pathsep)
    new_parts: list[str] = []
    for candidate in candidates:
        if candidate and os.path.isdir(candidate) and candidate not in old_parts and candidate not in new_parts:
            new_parts.append(candidate)
    if new_parts:
        os.environ["PATH"] = os.pathsep.join([*new_parts, *old_parts])

    src_dir = os.path.join(repo_root, "src")
    if os.path.isdir(src_dir) and src_dir not in sys.path:
        sys.path.insert(0, src_dir)


_bootstrap_lerobot_path()


# ── Defaults matching the provided lerobot-rollout example ────────────────────

IS_MACOS = sys.platform == "darwin"
BACKEND = os.environ.get("LEROBOT_BACKEND", "auto")
ROLLOUT_COMMAND = os.environ.get("LEROBOT_ROLLOUT_CMD", "lerobot-rollout")
RECORD_COMMAND = os.environ.get("LEROBOT_RECORD_CMD", "lerobot-record")
ROLLOUT_TEMPLATE = os.environ.get("LEROBOT_ROLLOUT_TEMPLATE")
PRETRAINED_PATH = "RobotLearningVLA/test_eval1_v2"
MODEL_CONFIG_PATH = os.environ.get(
    "VLA_MODEL_CONFIG",
    os.path.join(os.path.dirname(__file__), "models.json"),
)
DEFAULT_MODELS = {
    "task1": PRETRAINED_PATH,
    "task2": PRETRAINED_PATH,
}
MACOS_RECORD_REPO_ID = os.environ.get("MACOS_RECORD_REPO_ID", "Naomiihao/eval_test_2")
MACOS_RECORD_PROFILE = os.environ.get("MACOS_RECORD_PROFILE", "0").lower() in {"1", "true", "yes", "on"}
RECORD_REPO_ID = os.environ.get(
    "LEROBOT_RECORD_REPO_ID",
    MACOS_RECORD_REPO_ID if IS_MACOS else "local/eval_interface_rollouts",
)
MACOS_RECORD_EPISODE_TIME_S = float(os.environ.get("MACOS_RECORD_EPISODE_TIME_S", "20"))
MACOS_RECORD_NUM_EPISODES = int(os.environ.get("MACOS_RECORD_NUM_EPISODES", "1"))
DEVICE = os.environ.get("LEROBOT_DEVICE", "auto")
DISPLAY_DATA = os.environ.get("LEROBOT_DISPLAY_DATA", "true").lower() in {"1", "true", "yes", "on"}
DISPLAY_IP = os.environ.get("LEROBOT_DISPLAY_IP")
DISPLAY_PORT = os.environ.get("LEROBOT_DISPLAY_PORT")
ROBOT_TYPE = "so101_follower"
MACOS_ROBOT_PORT = os.environ.get("MACOS_ROBOT_PORT", "/dev/cu.usbmodem5B140317761")
ROBOT_PORT = os.environ.get("LEROBOT_ROBOT_PORT", MACOS_ROBOT_PORT if IS_MACOS else "/dev/ttyACM0")
ROBOT_ID = "my_awesome_follower_arm"
EMPTY_CAMERAS = 2
STRATEGY_TYPE = "base"
TASK2_DEFAULT_ROBOT_ORDER = os.environ.get("TASK2_DEFAULT_ROBOT_ORDER", "blue,red,green")
TASK2_LAYOUT_FRAMES = int(os.environ.get("TASK2_LAYOUT_FRAMES", "8"))
TASK2_LAYOUT_SETTLE_FRAMES = int(os.environ.get("TASK2_LAYOUT_SETTLE_FRAMES", "10"))
TASK2_LAYOUT_FRAME_INTERVAL_S = float(os.environ.get("TASK2_LAYOUT_FRAME_INTERVAL_S", "0.03"))
RESET_CLOSE_GRIPPER = os.environ.get("RESET_CLOSE_GRIPPER", "true").lower() in {"1", "true", "yes", "on"}
RESET_GRIPPER_POS = float(os.environ.get("RESET_GRIPPER_POS", "0"))
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


def _load_model_config(path: str) -> dict[str, str]:
    models = dict(DEFAULT_MODELS)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return models
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in model config {path!r}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Model config {path!r} must be a JSON object")

    configured = data.get("models", data)
    if not isinstance(configured, dict):
        raise ValueError(f"Model config {path!r} must contain a JSON object under 'models'")

    for task_name in ("task1", "task2"):
        value = configured.get(task_name)
        if isinstance(value, str) and value.strip():
            models[task_name] = value.strip()
    return models


def _parse_robot_order(value: str | tuple[str, str, str]) -> tuple[str, str, str]:
    if isinstance(value, str):
        colors = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    else:
        colors = tuple(color.strip().lower() for color in value)
    if len(colors) != 3 or set(colors) != {"red", "green", "blue"}:
        raise ValueError(
            "--task2-default-robot-order must contain red, green, blue exactly once, "
            f"for example blue,red,green; got {value!r}"
        )
    return colors


def _resolve_pretrained_path(args: argparse.Namespace) -> str:
    if args.pretrained_path:
        return args.pretrained_path
    return _load_model_config(args.model_config).get(args.command, PRETRAINED_PATH)


def _quoted_template_values(args: argparse.Namespace, task: str) -> dict[str, str]:
    values = {
        "task": task,
        "duration": args.duration,
        "device": _resolve_device(args.device),
        "robot_type": args.robot_type,
        "robot_port": args.robot_port,
        "robot_id": args.robot_id,
        "camera_index": args.camera_index,
        "cam_width": args.cam_width,
        "cam_height": args.cam_height,
        "cam_fps": args.cam_fps,
        "pretrained_path": args.pretrained_path,
        "record_repo_id": args.record_repo_id,
        "display_data": str(args.display_data).lower(),
        "display_ip": args.display_ip or "",
        "display_port": args.display_port or "",
        "empty_cameras": args.empty_cameras,
        "strategy_type": args.strategy_type,
        "cameras_json": _cameras_json(args.camera_index, args.cam_width, args.cam_height, args.cam_fps),
        "input_features_json": json.dumps(INPUT_FEATURES),
        "output_features_json": json.dumps(OUTPUT_FEATURES),
        "rename_map_json": json.dumps(RENAME_MAP),
    }
    return {key: shlex.quote(str(value)) for key, value in values.items()}


def _record_repo_id(args: argparse.Namespace) -> str:
    if "{run_id}" in args.record_repo_id:
        return args.record_repo_id.format(run_id=time.time_ns())

    org, sep, name = args.record_repo_id.partition("/")
    if not sep:
        org, name = "local", org
    return f"{org}/{name}_{time.time_ns()}"


def _command_exists(command: str) -> bool:
    parts = shlex.split(command)
    if not parts:
        return False
    executable = parts[0]
    if os.path.sep in executable:
        return os.path.exists(executable) and os.access(executable, os.X_OK)
    return shutil.which(executable) is not None


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _select_backend(args: argparse.Namespace) -> str:
    if args.backend == "persistent":
        return "persistent"
    if args.rollout_template:
        return "template"
    if args.backend != "auto":
        return args.backend
    if _command_exists(args.rollout_command):
        return "rollout"
    if _command_exists(args.record_command):
        return "record"
    return "rollout"


def _use_macos_record_profile(args: argparse.Namespace) -> bool:
    return (
        IS_MACOS
        and MACOS_RECORD_PROFILE
        and not getattr(args, "no_macos_record_profile", False)
        and _select_backend(args) == "record"
    )


def _record_episode_time(args: argparse.Namespace) -> int | float:
    if args.record_episode_time_s is not None:
        return args.record_episode_time_s
    if _use_macos_record_profile(args):
        return MACOS_RECORD_EPISODE_TIME_S
    return args.duration


def _record_num_episodes(args: argparse.Namespace) -> int:
    if args.record_num_episodes is not None:
        return args.record_num_episodes
    if _use_macos_record_profile(args):
        return MACOS_RECORD_NUM_EPISODES
    return 1


def _format_cli_number(value: int | float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _display_cli_args(args: argparse.Namespace) -> list[str]:
    display_args = [f"--display_data={str(args.display_data).lower()}"]
    if args.display_ip:
        display_args.append(f"--display_ip={args.display_ip}")
    if args.display_port:
        display_args.append(f"--display_port={args.display_port}")
    return display_args


def _build_rollout_cmd(args: argparse.Namespace, task: str) -> list[str]:
    backend = _select_backend(args)
    device = _resolve_device(args.device)

    if backend == "template":
        try:
            command = args.rollout_template.format_map(_quoted_template_values(args, task))
        except KeyError as exc:
            raise ValueError(f"Unknown --rollout-template placeholder: {exc.args[0]!r}") from exc
        return shlex.split(command)

    if backend == "record":
        if _use_macos_record_profile(args):
            return [
                *shlex.split(args.record_command),
                f"--robot.type={args.robot_type}",
                f"--robot.port={args.robot_port}",
                f"--robot.id={args.robot_id}",
                f"--robot.cameras={_cameras_json(args.camera_index, args.cam_width, args.cam_height, args.cam_fps)}",
                f"--dataset.single_task={task}",
                f"--dataset.repo_id={args.record_repo_id}",
                f"--dataset.episode_time_s={_format_cli_number(_record_episode_time(args))}",
                f"--dataset.num_episodes={_record_num_episodes(args)}",
                *_display_cli_args(args),
                "--dataset.streaming_encoding=true",
                "--dataset.encoder_threads=2",
                f"--dataset.rename_map={json.dumps(RENAME_MAP)}",
                f"--policy.path={args.pretrained_path}",
            ]

        record_repo_id = _record_repo_id(args)
        return [
            *shlex.split(args.record_command),
            f"--robot.type={args.robot_type}",
            f"--robot.port={args.robot_port}",
            f"--robot.id={args.robot_id}",
            f"--robot.cameras={_cameras_json(args.camera_index, args.cam_width, args.cam_height, args.cam_fps)}",
            f"--policy.path={args.pretrained_path}",
            f"--policy.device={device}",
            "--policy.compile_model=false",
            *_display_cli_args(args),
            f"--dataset.repo_id={record_repo_id}",
            f"--dataset.num_episodes={_record_num_episodes(args)}",
            f"--dataset.single_task={task}",
            f"--dataset.episode_time_s={_format_cli_number(_record_episode_time(args))}",
            "--dataset.reset_time_s=0",
            "--dataset.push_to_hub=false",
            f"--dataset.rename_map={json.dumps(RENAME_MAP)}",
        ]

    if backend == "rollout":
        return [
            *shlex.split(args.rollout_command),
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
            f"--device={device}",
            *_display_cli_args(args),
            f"--rename_map={json.dumps(RENAME_MAP)}",
        ]

    if backend == "persistent":
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            args.command,
            "--backend=persistent",
            f"--robot-port={args.robot_port}",
            f"--device={_resolve_device(args.device)}",
            f"--duration={args.duration}",
            f"--task={task}",
        ]
        return cmd

    raise ValueError(f"Unknown backend: {backend!r}")


def _allow_enter_to_stop(args: argparse.Namespace) -> bool:
    return bool(args.interactive and not args.no_enter_to_stop and sys.stdin.isatty())


def _stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGINT)
        else:
            proc.send_signal(signal.SIGINT)
        proc.wait(timeout=3)
        return
    except Exception:
        pass

    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=2)
        return
    except Exception:
        pass

    if proc.poll() is None:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
        proc.wait()


def _run_command(args: argparse.Namespace, cmd: list[str]) -> None:
    _print_cmd(cmd)
    if args.dry_run:
        print("[dry-run] Skipping execution.")
        return

    if not _allow_enter_to_stop(args):
        try:
            subprocess.run(cmd, check=True, cwd=args.rollout_cwd)
        except FileNotFoundError as exc:
            if exc.filename == cmd[0]:
                print(
                    f"error: cannot find rollout command {cmd[0]!r}.\n"
                    "Use --rollout-command to point to the command that works on this machine,\n"
                    "or set LEROBOT_ROLLOUT_CMD. PC station can keep the default.",
                    file=sys.stderr,
                )
            raise
        return

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=args.rollout_cwd,
            start_new_session=(os.name == "posix"),
        )
    except FileNotFoundError as exc:
        if exc.filename == cmd[0]:
            print(
                f"error: cannot find rollout command {cmd[0]!r}.\n"
                "Use --rollout-command to point to the command that works on this machine,\n"
                "or set LEROBOT_ROLLOUT_CMD. PC station can keep the default.",
                file=sys.stderr,
            )
        raise

    print("[run] Rollout is running. Press Enter to stop early and return to start pose.")
    stopped_by_user = False
    while proc.poll() is None:
        readable, _, _ = select.select([sys.stdin], [], [], 0.2)
        if readable:
            sys.stdin.readline()
            stopped_by_user = True
            print("[run] Stop requested. Ending current rollout...")
            _stop_process(proc)
            break

    returncode = proc.wait()
    if stopped_by_user:
        print("[run] Rollout stopped early by operator.")
        return
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)


def _run_preflight(args: argparse.Namespace) -> None:
    task = args.task or "Pick up the banana and put it into the red bowl."
    cmd = _build_rollout_cmd(args, task)
    launcher = cmd[0] if cmd else ""

    print("[preflight] Checking interface configuration only. The robot will not move.")
    print(f"[preflight] Python: {sys.executable}")
    print(f"[preflight] Working directory: {os.getcwd()}")
    print(f"[preflight] Backend: {_select_backend(args)} (requested: {args.backend})")
    print(f"[preflight] macOS record profile: {'on' if _use_macos_record_profile(args) else 'off'}")
    print(f"[preflight] Device: {_resolve_device(args.device)} (requested: {args.device})")
    print(f"[preflight] Rerun display_data: {str(args.display_data).lower()}")
    if args.rollout_cwd:
        exists = os.path.isdir(args.rollout_cwd)
        print(f"[preflight] Rollout working directory: {args.rollout_cwd} ({'ok' if exists else 'missing'})")

    if os.path.sep in launcher:
        exists = os.path.exists(launcher)
        executable = os.access(launcher, os.X_OK)
        status = "ok" if exists and executable else "not executable" if exists else "missing"
        print(f"[preflight] Backend launcher: {launcher} ({status})")
    else:
        found = shutil.which(launcher)
        print(f"[preflight] Backend launcher: {launcher} -> {found or 'not found'}")

    if args.robot_port:
        if args.robot_port.startswith("/"):
            print(f"[preflight] Robot port: {args.robot_port} ({'exists' if os.path.exists(args.robot_port) else 'not found'})")
        else:
            print(f"[preflight] Robot port: {args.robot_port}")

    print(f"[preflight] Camera index: {args.camera_index}")
    print(f"[preflight] Task2 default robot L->R: {_parse_robot_order(args.task2_default_robot_order)}")
    print(f"[preflight] Task2 layout frames: {args.task2_layout_frames}")
    print(
        "[preflight] Auto reset gripper: "
        f"{'closed target ' + str(args.reset_gripper_pos) if args.reset_close_gripper else 'captured pose'}"
    )
    if args.preflight_camera:
        try:
            frames = _capture_frames(
                args.camera_index,
                num_frames=args.task2_layout_frames,
                settle_frames=args.task2_layout_settle_frames,
                frame_interval_s=args.task2_layout_frame_interval_s,
            )
            print(f"[preflight] Camera frames captured: {len(frames)}; latest frame: {frames[-1].shape}")
        except Exception as exc:
            print(f"[preflight] Camera check failed: {type(exc).__name__}: {exc}")

    print(f"[preflight] Model path/repo: {args.pretrained_path}")
    if args.pretrained_path.startswith(("/", ".")):
        print(f"[preflight] Model local path: {'exists' if os.path.exists(args.pretrained_path) else 'not found'}")

    if _select_backend(args) == "persistent":
        print("[preflight] Persistent backend: no per-round subprocess; policy loads once at interface startup.")
        print("[preflight] Equivalent interface launcher:")
    else:
        print("[preflight] Rollout command that would run:")
    _print_cmd(cmd)


def _validate_runtime_args(args: argparse.Namespace) -> None:
    if "XXXX" in args.robot_port:
        print(
            "error: --robot-port still contains the placeholder 'XXXX'. "
            "Replace it with a real port, e.g. /dev/tty.usbmodem5B140317761. "
            "Run `lerobot-find-port` if unsure.",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.robot_port.startswith("/") and not os.path.exists(args.robot_port):
        print(
            f"error: robot port {args.robot_port!r} does not exist on this machine. "
            "Run `ls /dev/tty.usbmodem* /dev/cu.usbmodem*` or `lerobot-find-port`.",
            file=sys.stderr,
        )
        sys.exit(2)


def _effective_reset_mode(args: argparse.Namespace) -> str:
    if args.no_reset_wait:
        return "off"
    if args.reset_mode == "auto" and args.reset_command:
        return "command"
    return args.reset_mode


def _make_reset_robot(args: argparse.Namespace):
    if args.robot_type not in {"so100_follower", "so101_follower"}:
        raise ValueError(f"Automatic reset only supports SO follower robots, got {args.robot_type!r}")

    from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

    cfg = SO100FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        cameras={},
        disable_torque_on_disconnect=False,
    )
    return SO100Follower(cfg)


def _read_joint_pose(robot) -> dict[str, float]:
    obs = robot.get_observation()
    pose = {key: float(obs[key]) for key in robot.action_features if key in obs}
    missing = set(robot.action_features) - set(pose)
    if missing:
        raise RuntimeError(f"Robot observation is missing joint position keys: {sorted(missing)}")
    return pose


def _with_reset_gripper_goal(args: argparse.Namespace, pose: dict[str, float]) -> dict[str, float]:
    target = dict(pose)
    if not args.reset_close_gripper:
        return target

    for key in _gripper_pose_keys(target):
        target[key] = float(args.reset_gripper_pos)
    return target


def _gripper_pose_keys(pose: dict[str, float]) -> list[str]:
    return [key for key in pose if key.endswith("gripper.pos") or key == "gripper.pos"]


def _capture_reset_pose(args: argparse.Namespace) -> dict[str, float] | None:
    if _effective_reset_mode(args) != "auto":
        return None
    if args.dry_run:
        print("[dry-run] Would capture the robot start pose before rollout.")
        return None

    print("[reset] Capturing the robot start pose...")
    robot = _make_reset_robot(args)
    try:
        robot.connect(calibrate=False)
        pose = _read_joint_pose(robot)
    finally:
        if robot.is_connected:
            robot.disconnect()
    if args.reset_close_gripper and any(key.endswith("gripper.pos") or key == "gripper.pos" for key in pose):
        print(f"[reset] Start pose captured; reset gripper target will be {args.reset_gripper_pos:g}.")
    else:
        print("[reset] Start pose captured.")
    return pose


def _wait_before_reset(args: argparse.Namespace) -> None:
    if args.reset_trigger == "enter":
        if args.dry_run:
            print("[dry-run] Reset trigger prompt skipped.")
        else:
            input("[reset] Press Enter to return the robot to the start pose...")
    elif args.reset_delay > 0:
        print(f"[reset] Waiting {args.reset_delay:g}s before reset...")
        if args.dry_run:
            print("[dry-run] Reset delay skipped.")
        else:
            time.sleep(args.reset_delay)


def _return_to_pose(args: argparse.Namespace, reset_pose: dict[str, float] | None) -> None:
    if args.dry_run:
        print("[dry-run] Would return the robot to the captured start pose.")
        return
    if reset_pose is None:
        print("[reset] No captured start pose; automatic reset skipped.")
        return

    target_pose = _with_reset_gripper_goal(args, reset_pose)
    print("[reset] Returning robot to the captured start pose...")
    robot = _make_reset_robot(args)
    try:
        robot.connect(calibrate=False)
        current_pose = _read_joint_pose(robot)
        movement_pose = dict(target_pose)
        if args.reset_close_gripper:
            for key in _gripper_pose_keys(movement_pose):
                if key in current_pose:
                    movement_pose[key] = current_pose[key]
        keys = [key for key in target_pose if key in current_pose]
        steps = max(1, int(args.reset_duration_s * args.reset_fps))
        period_s = 1.0 / args.reset_fps
        for step in range(1, steps + 1):
            ratio = step / steps
            action = {
                key: current_pose[key] + (movement_pose[key] - current_pose[key]) * ratio
                for key in keys
            }
            robot.send_action(action)
            time.sleep(period_s)
        robot.send_action({key: movement_pose[key] for key in keys})
        if args.reset_close_gripper and _gripper_pose_keys(target_pose):
            robot.send_action({key: target_pose[key] for key in keys})
    finally:
        if robot.is_connected:
            robot.disconnect()
    print("[reset] Robot is back at the captured start pose.")


def _run_reset_step(args: argparse.Namespace, reset_pose: dict[str, float] | None = None) -> None:
    """Return the robot to the start pose after a rollout has fully ended."""
    mode = _effective_reset_mode(args)
    if mode == "off":
        return

    _wait_before_reset(args)

    if mode == "manual":
        if args.dry_run:
            print("[dry-run] Manual reset confirmation skipped.")
            return
        input("[reset] Move the robot back to the start pose, then press Enter...")
        return

    if mode == "command":
        reset_cmd = shlex.split(args.reset_command or "")
        print("[reset] Returning robot to the start pose...")
        _print_cmd(reset_cmd)
        if args.dry_run:
            print("[dry-run] Skipping reset command.")
        else:
            subprocess.run(reset_cmd, check=True)

        if args.reset_confirm and not args.dry_run:
            input("[reset] Confirm the robot is back at the start pose, then press Enter...")
        return

    if mode == "auto":
        _return_to_pose(args, reset_pose)
        if args.reset_confirm and not args.dry_run:
            input("[reset] Confirm the robot is back at the start pose, then press Enter...")
        return

    raise ValueError(f"Unknown reset mode: {mode!r}")


def _interactive_prompts(args: argparse.Namespace):
    print("\n[interface] Interactive evaluation mode")
    print("[interface] Type one instruction per round. Press Enter on an empty line, or type q, to stop.")

    round_idx = 0
    while True:
        if args.max_rounds is not None and round_idx >= args.max_rounds:
            print(f"[interface] Reached --max-rounds={args.max_rounds}.")
            return

        raw = input(f"\n[interface] Round {round_idx + 1} instruction> ").strip()
        if raw.lower() in {"", "q", "quit", "exit"}:
            print("[interface] Stopping.")
            return

        round_idx += 1
        yield round_idx, raw


def _capture_frames(
    camera_index: int,
    *,
    num_frames: int,
    settle_frames: int,
    frame_interval_s: float,
) -> list["np.ndarray"]:
    """Capture several RGB frames from the camera using OpenCV."""
    import cv2

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {camera_index}")

    frames = []
    for _ in range(settle_frames):
        ret, frame = cap.read()
        if not ret:
            break

    for _ in range(num_frames):
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if frame_interval_s > 0:
            time.sleep(frame_interval_s)

    cap.release()
    if not frames:
        raise RuntimeError("Failed to read a frame from the camera")
    return frames


def _capture_frame(camera_index: int) -> "np.ndarray":
    """Capture a single RGB frame from the camera using OpenCV."""
    return _capture_frames(
        camera_index,
        num_frames=1,
        settle_frames=TASK2_LAYOUT_SETTLE_FRAMES,
        frame_interval_s=0.0,
    )[0]


def _load_prompt_tools():
    prompt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "task2_prompt"))
    if prompt_dir not in sys.path:
        sys.path.insert(0, prompt_dir)
    from eval1_prompt_normalizer import PromptNormalizer, normalize_prompt_best_effort, normalize_prompt_text

    return PromptNormalizer, normalize_prompt_text, normalize_prompt_best_effort


def _load_normalizer(
    device: str,
    *,
    enable_vlm_fallback: bool = False,
    fallback_robot_order: tuple[str, str, str],
):
    """Load the deterministic normalizer used for eval deployment."""
    PromptNormalizer, _, _ = _load_prompt_tools()

    if enable_vlm_fallback:
        raise ValueError(
            "--enable-vlm-fallback is disabled for eval deployment; "
            "prompt preprocessing uses deterministic rules only."
        )

    print("[task2] Using deterministic prompt preprocessing only (no extra prompt model).")
    return PromptNormalizer(
        vlm=None,
        processor=None,
        device=device,
        fallback_passthrough=True,
        fallback_robot_order=fallback_robot_order,
    )


# ── Persistent in-process backend ─────────────────────────────────────────────

def _apply_optional_smolvla_shim() -> None:
    """Apply the Eval3 SmolVLA compatibility shim when that folder is present."""
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    shim_dir = os.path.join(repo_root, "robot-learning-vla", "scripts")
    if not os.path.isdir(shim_dir):
        return
    if shim_dir not in sys.path:
        sys.path.insert(0, shim_dir)
    try:
        from eval3_lerobot_shim import apply as eval3_shim_apply
    except Exception:
        return
    eval3_shim_apply()

    # The Eval3 shim temporarily hides transformers while importing optional
    # GROOT modules. In this LeRobot checkout that can leave SmolVLA cached with
    # transformer classes set to None. Reload only the SmolVLA modules after the
    # flag is restored so persistent loading can instantiate the policy backbone.
    try:
        import importlib
        import lerobot.utils.import_utils as import_utils

        import_utils._transformers_available = bool(import_utils.is_package_available("transformers"))
        import lerobot.policies.smolvla.smolvlm_with_expert as smolvlm_with_expert

        if smolvlm_with_expert.AutoModelForImageTextToText is None:
            importlib.reload(smolvlm_with_expert)
            import lerobot.policies.smolvla.modeling_smolvla as modeling_smolvla

            importlib.reload(modeling_smolvla)
            import eval3_lerobot_shim

            eval3_lerobot_shim._SMOLVLA_LANG_MASK_PATCHED = False
            eval3_lerobot_shim._patch_smolvla_language_masks()
    except Exception as exc:
        print(f"[persistent] Warning: SmolVLA compatibility reload skipped: {exc}")


def _make_policy_robot(args: argparse.Namespace):
    if args.robot_type not in {"so100_follower", "so101_follower"}:
        raise ValueError(f"Persistent backend only supports SO follower robots, got {args.robot_type!r}")

    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

    cameras = {
        "front": OpenCVCameraConfig(
            index_or_path=args.camera_index,
            width=args.cam_width,
            height=args.cam_height,
            fps=args.cam_fps,
        )
    }
    cfg = SO100FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        cameras=cameras,
        disable_torque_on_disconnect=False,
    )
    return SO100Follower(cfg)


def _load_persistent_policy(args: argparse.Namespace):
    _apply_optional_smolvla_shim()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    policy_cfg = PreTrainedConfig.from_pretrained(args.pretrained_path)
    policy_cfg.pretrained_path = args.pretrained_path
    policy_cfg.device = _resolve_device(args.device)
    if hasattr(policy_cfg, "empty_cameras"):
        policy_cfg.empty_cameras = args.empty_cameras
    if hasattr(policy_cfg, "compile_model"):
        policy_cfg.compile_model = False

    policy_class = get_policy_class(policy_cfg.type)
    if getattr(policy_cfg, "use_peft", False):
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(args.pretrained_path)
        policy = policy_class.from_pretrained(
            pretrained_name_or_path=peft_config.base_model_name_or_path,
            config=policy_cfg,
        )
        policy = PeftModel.from_pretrained(policy, args.pretrained_path, config=peft_config)
    else:
        policy = policy_class.from_pretrained(args.pretrained_path, config=policy_cfg)

    policy.to(policy_cfg.device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": policy_cfg.device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    return policy, preprocessor, postprocessor


def _build_persistent_ds_features(robot):
    from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
    from lerobot.processor import make_default_processors
    from lerobot.utils.feature_utils import combine_feature_dicts

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    ds_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )
    return ds_features, robot_action_processor, robot_observation_processor


def _capture_connected_reset_pose(args: argparse.Namespace, robot) -> dict[str, float] | None:
    if _effective_reset_mode(args) != "auto":
        return None
    if args.dry_run:
        print("[dry-run] Would capture the robot start pose before rollout.")
        return None

    print("[reset] Capturing the robot start pose...")
    pose = _read_joint_pose(robot)
    if args.reset_close_gripper and _gripper_pose_keys(pose):
        print(f"[reset] Start pose captured; reset gripper target will be {args.reset_gripper_pos:g}.")
    else:
        print("[reset] Start pose captured.")
    return pose


def _return_connected_robot_to_pose(
    args: argparse.Namespace,
    robot,
    reset_pose: dict[str, float] | None,
) -> None:
    if args.dry_run:
        print("[dry-run] Would return the robot to the captured start pose.")
        return
    if reset_pose is None:
        print("[reset] No captured start pose; automatic reset skipped.")
        return

    target_pose = _with_reset_gripper_goal(args, reset_pose)
    print("[reset] Returning robot to the captured start pose...")
    current_pose = _read_joint_pose(robot)
    movement_pose = dict(target_pose)
    if args.reset_close_gripper:
        for key in _gripper_pose_keys(movement_pose):
            if key in current_pose:
                movement_pose[key] = current_pose[key]
    keys = [key for key in target_pose if key in current_pose]
    steps = max(1, int(args.reset_duration_s * args.reset_fps))
    period_s = 1.0 / args.reset_fps
    for step in range(1, steps + 1):
        ratio = step / steps
        action = {
            key: current_pose[key] + (movement_pose[key] - current_pose[key]) * ratio
            for key in keys
        }
        robot.send_action(action)
        time.sleep(period_s)
    robot.send_action({key: movement_pose[key] for key in keys})
    if args.reset_close_gripper and _gripper_pose_keys(target_pose):
        robot.send_action({key: target_pose[key] for key in keys})
    print("[reset] Robot is back at the captured start pose.")


def _run_connected_reset_step(
    args: argparse.Namespace,
    session: "_PersistentSession",
    reset_pose: dict[str, float] | None = None,
) -> None:
    mode = _effective_reset_mode(args)
    if mode == "off":
        return

    _wait_before_reset(args)

    if mode == "manual":
        if args.dry_run:
            print("[dry-run] Manual reset confirmation skipped.")
            return
        input("[reset] Move the robot back to the start pose, then press Enter...")
        return

    if mode == "command":
        if session.robot is not None and session.robot.is_connected:
            session.robot.disconnect()
        _run_reset_step(args, reset_pose)
        if session.robot is not None:
            session.robot.connect(calibrate=False)
        return

    if mode == "auto":
        _return_connected_robot_to_pose(args, session.robot, reset_pose)
        if args.reset_confirm and not args.dry_run:
            input("[reset] Confirm the robot is back at the start pose, then press Enter...")
        return

    raise ValueError(f"Unknown reset mode: {mode!r}")


def _maybe_stop_from_enter(args: argparse.Namespace) -> bool:
    if not _allow_enter_to_stop(args):
        return False
    readable, _, _ = select.select([sys.stdin], [], [], 0)
    if not readable:
        return False
    sys.stdin.readline()
    print("[run] Stop requested. Ending current rollout...")
    return True


class _PersistentSession:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.robot = None
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.ds_features = None
        self.robot_action_processor = None
        self.robot_observation_processor = None

    def __enter__(self):
        if self.args.dry_run:
            print("[dry-run] Persistent backend would load the policy and connect the robot once.")
            return self

        print(f"[persistent] Loading policy once: {self.args.pretrained_path}")
        self.policy, self.preprocessor, self.postprocessor = _load_persistent_policy(self.args)

        print("[persistent] Connecting robot and camera once...")
        self.robot = _make_policy_robot(self.args)
        self.ds_features, self.robot_action_processor, self.robot_observation_processor = (
            _build_persistent_ds_features(self.robot)
        )
        self.robot.connect(calibrate=False)

        if self.args.display_data:
            from lerobot.utils.visualization_utils import init_rerun

            port = int(self.args.display_port) if self.args.display_port else None
            init_rerun(session_name="eval1_task_interface", ip=self.args.display_ip, port=port)

        print("[persistent] Ready. The model will stay loaded for this interface session.")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.robot is not None and self.robot.is_connected:
            self.robot.disconnect()
        return False

    def run_round(self, task: str) -> None:
        if self.args.dry_run:
            print(f"[dry-run] Persistent backend would run task: {task!r}")
            return
        reset_pose = _capture_connected_reset_pose(self.args, self.robot)
        try:
            self._deploy_loop(task)
        finally:
            _run_connected_reset_step(self.args, self, reset_pose)

    def capture_layout_frames(self) -> list["np.ndarray"]:
        if self.robot is None or not self.robot.is_connected:
            raise RuntimeError("Persistent robot session is not connected")

        frames = []
        for _ in range(self.args.task2_layout_settle_frames):
            self.robot.get_observation()

        for _ in range(self.args.task2_layout_frames):
            obs = self.robot.get_observation()
            frame = obs.get("front")
            if frame is None:
                frame = obs.get("observation.images.front")
            if frame is None:
                raise RuntimeError("Robot observation does not contain the front camera frame")
            frames.append(frame)
            if self.args.task2_layout_frame_interval_s > 0:
                time.sleep(self.args.task2_layout_frame_interval_s)
        return frames

    def _deploy_loop(self, task: str) -> None:
        import torch

        from lerobot.policies.utils import make_robot_action
        from lerobot.utils.constants import OBS_STR
        try:
            from lerobot.common.control_utils import predict_action
        except ModuleNotFoundError:
            from lerobot.utils.control_utils import predict_action
        from lerobot.utils.device_utils import get_safe_torch_device
        from lerobot.utils.feature_utils import build_dataset_frame
        from lerobot.utils.robot_utils import precise_sleep

        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

        device = get_safe_torch_device(self.policy.config.device)
        control_interval = 1.0 / max(1, self.args.cam_fps)
        timestamp = 0.0
        start_episode_t = time.perf_counter()
        print("[run] Persistent rollout is running. Press Enter to stop early and return to start pose.")

        while timestamp < self.args.duration:
            start_loop_t = time.perf_counter()
            if _maybe_stop_from_enter(self.args):
                break

            obs = self.robot.get_observation()
            obs_processed = self.robot_observation_processor(obs)
            observation_frame = build_dataset_frame(self.ds_features, obs_processed, prefix=OBS_STR)
            with torch.inference_mode():
                action_values = predict_action(
                    observation=observation_frame,
                    policy=self.policy,
                    device=device,
                    preprocessor=self.preprocessor,
                    postprocessor=self.postprocessor,
                    use_amp=self.policy.config.use_amp,
                    task=task,
                    robot_type=self.robot.robot_type,
                )
            act_processed_policy = make_robot_action(action_values, self.ds_features)
            robot_action_to_send = self.robot_action_processor((act_processed_policy, obs))
            self.robot.send_action(robot_action_to_send)

            if self.args.display_data:
                from lerobot.utils.visualization_utils import log_rerun_data

                log_rerun_data(observation=obs_processed, action=robot_action_to_send)

            dt_s = time.perf_counter() - start_loop_t
            sleep_time_s = control_interval - dt_s
            if sleep_time_s < 0:
                print(
                    "[run] Warning: control loop is slower than target "
                    f"({1 / dt_s:.1f} Hz vs {self.args.cam_fps} Hz)."
                )
            precise_sleep(max(sleep_time_s, 0.0))
            timestamp = time.perf_counter() - start_episode_t


def _deployment_session(args: argparse.Namespace):
    if _select_backend(args) == "persistent":
        return _PersistentSession(args)
    return contextlib.nullcontext(None)


# ── Task runners ───────────────────────────────────────────────────────────────

def _run_round(args: argparse.Namespace, task: str, session: _PersistentSession | None = None) -> None:
    if session is not None:
        session.run_round(task)
        return

    reset_pose = _capture_reset_pose(args)
    cmd = _build_rollout_cmd(args, task)
    try:
        _run_command(args, cmd)
    finally:
        _run_reset_step(args, reset_pose)


def _prepare_task1_prompt(raw_task: str) -> str:
    _, _, normalize_prompt_best_effort = _load_prompt_tools()
    task, reason = normalize_prompt_best_effort(raw_task)
    print(f"[task1] Raw prompt:        {raw_task!r}")
    if reason != "strict":
        print(f"[task1] Best-effort prompt fallback: {reason}")
    print(f"[task1] Normalized prompt: {task!r}")
    return task


def run_task1(args: argparse.Namespace) -> None:
    with _deployment_session(args) as session:
        if args.interactive:
            for i, raw_task in _interactive_prompts(args):
                print(f"\n[task1] Rollout {i}")
                try:
                    task = _prepare_task1_prompt(raw_task)
                except ValueError as exc:
                    print(f"[task1] {exc}")
                    continue
                _run_round(args, task, session=session)
            return

        assert args.task is not None
        for i in range(args.n_rollouts):
            if args.n_rollouts > 1:
                print(f"\n[task1] Rollout {i + 1}/{args.n_rollouts}")
            task = _prepare_task1_prompt(args.task)
            _run_round(args, task, session=session)


def run_task2(args: argparse.Namespace) -> None:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    fallback_robot_order = _parse_robot_order(args.task2_default_robot_order)
    normalizer = (
        None
        if args.skip_task2_normalization
        else _load_normalizer(
            _resolve_device(args.device),
            enable_vlm_fallback=args.enable_vlm_fallback,
            fallback_robot_order=fallback_robot_order,
        )
    )

    with _deployment_session(args) as session:
        if args.interactive:
            for i, raw_task in _interactive_prompts(args):
                print(f"\n[task2] Rollout {i}")
                try:
                    task = _prepare_task2_prompt(args, normalizer, raw_task, session=session)
                except ValueError as exc:
                    print(f"[task2] {exc}")
                    continue
                _run_round(args, task, session=session)
            return

        assert args.task is not None
        for i in range(args.n_rollouts):
            print(f"\n[task2] Rollout {i + 1}/{args.n_rollouts}")

            task = _prepare_task2_prompt(args, normalizer, args.task, session=session)
            _run_round(args, task, session=session)


def _prepare_task2_prompt(
    args: argparse.Namespace,
    normalizer,
    raw_task: str,
    session: _PersistentSession | None = None,
) -> str:
    if args.skip_task2_normalization:
        print("[task2] Skipping camera capture and prompt normalization for dry-run interface testing.")
        print(f"[task2] Raw prompt:        {raw_task!r}")
        print(f"[task2] Normalized prompt: {raw_task!r}")
        return raw_task

    _, normalize_prompt_text, _ = _load_prompt_tools()
    print(f"[task2] Raw prompt:        {raw_task!r}")
    try:
        task = normalize_prompt_text(raw_task, fallback_passthrough=False)
    except ValueError:
        task = None
    if task is not None:
        print("[task2] Resolved from text only; no camera frame needed.")
        print(f"[task2] Normalized prompt: {task!r}")
        return task

    fallback_robot_order = _parse_robot_order(args.task2_default_robot_order)

    # 1. Capture several frames and detect bowl layout at the start of each rollout.
    frame = None
    normalizer._robot_order = None          # reset so layout is re-detected
    try:
        print(
            f"[task2] Capturing {args.task2_layout_frames} frame(s) "
            f"from camera index {args.camera_index}..."
        )
        if session is not None:
            frames = session.capture_layout_frames()
        else:
            frames = _capture_frames(
                args.camera_index,
                num_frames=args.task2_layout_frames,
                settle_frames=args.task2_layout_settle_frames,
                frame_interval_s=args.task2_layout_frame_interval_s,
            )
        frame = frames[-1]
        print(f"[task2] Captured {len(frames)} frame(s); latest frame: {frame.shape}")
        print("[task2] Detecting bowl layout from HSV colors...")
        layout = normalizer.detect_layout_from_frames(frames)
        if layout is not None:
            print(f"[task2] Bowl layout (robot L->R): {layout}")
        else:
            normalizer._robot_order = fallback_robot_order
            print(f"[task2] Layout detection unavailable; using default robot L->R: {fallback_robot_order}")
    except Exception as exc:
        normalizer._robot_order = fallback_robot_order
        print(
            f"[task2] Camera/layout detection failed ({type(exc).__name__}: {exc}); "
            f"using default robot L->R: {fallback_robot_order}"
        )

    # 3. Normalize the prompt
    task = normalizer.normalize(frame, raw_task)
    print(f"[task2] Normalized prompt: {task!r}")
    return task


def _print_cmd(cmd: list[str]) -> None:
    print("[run]  " + " \\\n       ".join(cmd))


# ── Argument parsing ───────────────────────────────────────────────────────────

def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--task",
        default=None,
        help="Task instruction string. Eval1/Eval2 prompts are normalized before rollout.",
    )
    p.add_argument("--robot-type",      default=ROBOT_TYPE)
    p.add_argument("--robot-port",      default=ROBOT_PORT)
    p.add_argument("--robot-id",        default=ROBOT_ID)
    p.add_argument("--camera-index",    type=int, default=0)
    p.add_argument("--cam-width",       type=int, default=640)
    p.add_argument("--cam-height",      type=int, default=480)
    p.add_argument("--cam-fps",         type=int, default=30)
    p.add_argument("--task2-default-robot-order", default=TASK2_DEFAULT_ROBOT_ORDER,
                   help="Best-effort robot-perspective L->R bowl order if camera detection is incomplete.")
    p.add_argument("--task2-layout-frames", type=int, default=TASK2_LAYOUT_FRAMES,
                   help="Number of camera frames used to infer task2 bowl order.")
    p.add_argument("--task2-layout-settle-frames", type=int, default=TASK2_LAYOUT_SETTLE_FRAMES,
                   help="Camera frames discarded before task2 layout detection.")
    p.add_argument("--task2-layout-frame-interval-s", type=float, default=TASK2_LAYOUT_FRAME_INTERVAL_S,
                   help="Seconds between task2 layout frames.")
    p.add_argument("--pretrained-path", default=None,
                   help="Override model path/repo. If omitted, read from --model-config.")
    p.add_argument("--model-config",    default=MODEL_CONFIG_PATH,
                   help="JSON file with task1/task2 model paths.")
    p.add_argument("--backend",         choices=("auto", "rollout", "record", "template", "persistent"),
                   default=BACKEND,
                   help=(
                       "Backend selector. auto uses lerobot-rollout if present, otherwise lerobot-record. "
                       "persistent loads the policy once and runs all rounds in this process."
                   ))
    p.add_argument("--rollout-command", default=ROLLOUT_COMMAND,
                   help="Rollout launcher command. Default: env LEROBOT_ROLLOUT_CMD or lerobot-rollout.")
    p.add_argument("--record-command",  default=RECORD_COMMAND,
                   help="Record launcher command for backend=record. Default: env LEROBOT_RECORD_CMD or lerobot-record.")
    p.add_argument("--record-repo-id",  default=RECORD_REPO_ID,
                   help="Local dataset repo id used by backend=record.")
    p.add_argument("--record-episode-time-s", type=float, default=None,
                   help="Episode seconds for backend=record. macOS profile default: 20.")
    p.add_argument("--record-num-episodes", type=int, default=None,
                   help="Number of episodes for backend=record. macOS profile default: 1.")
    p.add_argument("--no-macos-record-profile", action="store_true",
                   help="Disable the macOS record command profile and use generic record settings.")
    p.add_argument("--rollout-template", default=ROLLOUT_TEMPLATE,
                   help=(
                       "Full rollout command template for machines whose LeRobot CLI "
                       "uses different arguments. Placeholders include {task}, "
                       "{duration}, {pretrained_path}, {robot_port}, {camera_index}."
                   ))
    p.add_argument("--rollout-cwd",     default=None,
                   help="Optional working directory for the rollout command, e.g. a local LeRobot checkout.")
    p.add_argument("--empty-cameras",   type=int, default=EMPTY_CAMERAS)
    p.add_argument("--strategy-type",   default=STRATEGY_TYPE)
    p.add_argument("--duration",        type=int, default=20,
                   help="Rollout duration in seconds.")
    p.add_argument("--enable-vlm-fallback", action="store_true",
                   help="Disabled for eval deployment; kept only to reject legacy/offline VLM fallback explicitly.")
    p.add_argument("--device",          default=DEVICE,
                   help="Device for policy. Default: env LEROBOT_DEVICE or auto.")
    p.add_argument("--display-data",    dest="display_data", action="store_true", default=DISPLAY_DATA,
                   help="Open/stream the Rerun display during rollout.")
    p.add_argument("--no-display-data", dest="display_data", action="store_false",
                   help="Disable Rerun display streaming.")
    p.add_argument("--display-ip",      default=DISPLAY_IP,
                   help="Optional Rerun display server IP.")
    p.add_argument("--display-port",    default=DISPLAY_PORT,
                   help="Optional Rerun display server port.")
    p.add_argument("--n-rollouts",      type=int, default=1,
                   help="Number of consecutive rollouts to run.")
    p.add_argument("--interactive",     action="store_true",
                   help="Ask the TA/operator for one language instruction per round.")
    p.add_argument("--max-rounds",      type=int, default=None,
                   help="Maximum interactive rounds. Default: keep asking until q/blank.")
    p.add_argument("--dry-run",         action="store_true",
                   help="Print rollout/reset commands without executing them.")
    p.add_argument("--preflight",       action="store_true",
                   help="Check command paths/configuration and print the rollout command, without moving the robot.")
    p.add_argument("--preflight-camera", action="store_true",
                   help="With --preflight, also try to capture one camera frame.")
    p.add_argument("--reset-command",   default=None,
                   help="Optional command that returns the robot to the start pose after each rollout.")
    p.add_argument("--reset-mode",      choices=("auto", "manual", "command", "off"), default="auto",
                   help="How to return after each rollout. auto captures the start pose and drives back after rollout.")
    p.add_argument("--reset-duration-s", type=float, default=3.0,
                   help="Seconds used for the automatic return-to-start interpolation.")
    p.add_argument("--reset-fps",       type=int, default=30,
                   help="Control rate for automatic return-to-start interpolation.")
    p.add_argument("--reset-close-gripper", dest="reset_close_gripper", action="store_true",
                   default=RESET_CLOSE_GRIPPER,
                   help="During automatic reset, force the gripper target to --reset-gripper-pos.")
    p.add_argument("--no-reset-close-gripper", dest="reset_close_gripper", action="store_false",
                   help="Keep the captured gripper position during automatic reset.")
    p.add_argument("--reset-gripper-pos", type=float, default=RESET_GRIPPER_POS,
                   help="Normalized gripper target for closed reset pose. Default: 0.")
    p.add_argument("--reset-delay",     type=float, default=0.0,
                   help="Seconds to wait after a rollout before running --reset-command.")
    p.add_argument("--reset-trigger",   choices=("auto", "enter"), default="auto",
                   help="Run --reset-command automatically, or only after pressing Enter.")
    p.add_argument("--reset-confirm",   action="store_true",
                   help="Ask for Enter confirmation after --reset-command finishes.")
    p.add_argument("--no-enter-to-stop", action="store_true",
                   help="Interactive mode only: do not use Enter to stop a running rollout early.")
    p.add_argument("--no-reset-wait",   action="store_true",
                   help="Do not pause for manual reset confirmation between rounds.")
    p.add_argument("--skip-task2-normalization", action="store_true",
                   help="Dry-run only: skip task2 prompt/layout normalization and pass prompts through.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Eval1 task1 / task2 robot rollout CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser(
        "task1",
        help="Color prompt rollout — prompt is normalized to the canonical policy wording first.",
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

    try:
        args.pretrained_path = _resolve_pretrained_path(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)

    if not args.interactive and not args.preflight and args.task is None:
        print("error: --task is required unless --interactive is set", file=sys.stderr)
        sys.exit(2)
    if args.skip_task2_normalization and not args.dry_run:
        print("error: --skip-task2-normalization is only allowed with --dry-run", file=sys.stderr)
        sys.exit(2)
    if args.enable_vlm_fallback:
        print(
            "error: --enable-vlm-fallback is disabled for eval deployment; "
            "prompt preprocessing uses deterministic rules only.",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.duration <= 0:
        print("error: --duration must be > 0", file=sys.stderr)
        sys.exit(2)
    if args.reset_delay < 0:
        print("error: --reset-delay must be >= 0", file=sys.stderr)
        sys.exit(2)
    if args.reset_duration_s < 0:
        print("error: --reset-duration-s must be >= 0", file=sys.stderr)
        sys.exit(2)
    if args.reset_fps <= 0:
        print("error: --reset-fps must be > 0", file=sys.stderr)
        sys.exit(2)
    if not 0 <= args.reset_gripper_pos <= 100:
        print("error: --reset-gripper-pos must be between 0 and 100", file=sys.stderr)
        sys.exit(2)
    try:
        _parse_robot_order(args.task2_default_robot_order)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    if args.task2_layout_frames <= 0:
        print("error: --task2-layout-frames must be > 0", file=sys.stderr)
        sys.exit(2)
    if args.task2_layout_settle_frames < 0:
        print("error: --task2-layout-settle-frames must be >= 0", file=sys.stderr)
        sys.exit(2)
    if args.task2_layout_frame_interval_s < 0:
        print("error: --task2-layout-frame-interval-s must be >= 0", file=sys.stderr)
        sys.exit(2)
    if args.reset_mode == "command" and not args.reset_command:
        print("error: --reset-mode command requires --reset-command", file=sys.stderr)
        sys.exit(2)
    if args.record_episode_time_s is not None and args.record_episode_time_s <= 0:
        print("error: --record-episode-time-s must be > 0", file=sys.stderr)
        sys.exit(2)
    if args.record_num_episodes is not None and args.record_num_episodes <= 0:
        print("error: --record-num-episodes must be > 0", file=sys.stderr)
        sys.exit(2)
    if args.backend not in {"auto", "rollout", "record", "template", "persistent"}:
        print("error: --backend must be one of auto, rollout, record, template, persistent", file=sys.stderr)
        sys.exit(2)
    if args.backend == "template" and not args.rollout_template:
        print("error: --backend template requires --rollout-template", file=sys.stderr)
        sys.exit(2)
    if _select_backend(args) == "rollout" and not shlex.split(args.rollout_command):
        print("error: rollout backend requires --rollout-command", file=sys.stderr)
        sys.exit(2)
    if _select_backend(args) == "record" and not shlex.split(args.record_command):
        print("error: record backend requires --record-command", file=sys.stderr)
        sys.exit(2)
    if args.preflight:
        _run_preflight(args)
        return
    if not args.dry_run:
        _validate_runtime_args(args)

    if args.command == "task1":
        run_task1(args)
    elif args.command == "task2":
        run_task2(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
