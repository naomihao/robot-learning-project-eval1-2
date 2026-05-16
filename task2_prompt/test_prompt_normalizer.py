#!/usr/bin/env python3
"""Comprehensive prompt normalizer test for SmolVLA eval1 (banana-bowl task).

Downloads the first frame of RobotLearningVLA/banana_green_bowl_eval1_v2 and
runs every prompt variant through PromptNormalizer, from trivial to extremely hard.

The dataset target is the GREEN bowl, so:
  • deterministic prompts (negation, color analogy) must return "green"
  • spatial prompts (depend on bowl layout in image) are marked VERIFY and
    printed for human review — correct answer depends on the actual image.

Usage::

    # Default: load SmolVLM2-500M-Video-Instruct directly (same VLM backbone
    # as SmolVLA base, reliable standalone text generation):
    python task2_prompt/test_prompt_normalizer.py

    # Load SmolVLA policy and reuse its VLM (same weights, tests policy path):
    python task2_prompt/test_prompt_normalizer.py --policy-path lerobot/smolvla_base

    # Use CPU if no GPU:
    python task2_prompt/test_prompt_normalizer.py --device cpu

    # Save the extracted frame for manual inspection:
    python task2_prompt/test_prompt_normalizer.py --save-frame frame.png
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

# ── Path setup ─────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_EVAL3_SCRIPTS = _HERE.parents[1] / "robot-learning-vla" / "scripts"

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if _EVAL3_SCRIPTS.exists() and str(_EVAL3_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_EVAL3_SCRIPTS))

from eval1_prompt_normalizer import PromptNormalizer  # noqa: E402


# ── Test-case definition ───────────────────────────────────────────────────────

Difficulty = Literal["easy", "medium", "hard", "very_hard"]
Category = Literal[
    "direct",
    "negation",
    "color_analogy",
    "spatial_absolute",
    "spatial_relative",
    "combined",
    "tricky_wording",
]


@dataclass
class Prompt:
    text: str
    category: Category
    difficulty: Difficulty
    # Expected color when testing against the GREEN bowl dataset.
    # None = spatial prompt that depends on bowl layout → printed for manual verification.
    expected: str | None = None
    note: str = ""


# ── Full prompt suite ──────────────────────────────────────────────────────────
#
# Convention for spatial prompts (expected=None):
#   We can't hard-code the expected answer without seeing the live image.
#   The test runner prints the VLM output and marks it "VERIFY".
#   Run with --save-frame to dump the image, then annotate expected values.
#
# All prompts end with a period to match the evaluation style.

PROMPTS: list[Prompt] = [

    # ── 1. Direct (baseline) ──────────────────────────────────────────────────
    Prompt(
        "Put the banana in the green colored bowl.",
        "direct", "easy", expected="green",
        note="Canonical training format — must always pass.",
    ),
    Prompt(
        "Put the banana into the green bowl.",
        "direct", "easy", expected="green",
        note="Slightly shorter phrasing.",
    ),
    Prompt(
        "Place the banana in the green bowl.",
        "direct", "easy", expected="green",
        note="Synonym for 'put'.",
    ),

    # ── 2. Logical negation (deterministic → green) ───────────────────────────
    Prompt(
        "Put the banana into the bowl that is not green and not blue.",
        "negation", "medium", expected="red",
        note="Example from problem statement — target is RED.",
    ),
    Prompt(
        "Put the banana into the bowl that is not red and not blue.",
        "negation", "medium", expected="green",
        note="Exclude red & blue → green.",
    ),
    Prompt(
        "Put the banana into the bowl that is not red and not green.",
        "negation", "medium", expected="blue",
        note="Exclude red & green → blue.",
    ),
    Prompt(
        "Put the banana into the bowl that is neither red nor blue.",
        "negation", "medium", expected="green",
        note="'neither … nor' phrasing.",
    ),
    Prompt(
        "Put the banana into the bowl that is neither red nor green.",
        "negation", "medium", expected="blue",
        note="'neither … nor' → blue.",
    ),
    Prompt(
        "Put the banana into the non-red and non-blue bowl.",
        "negation", "medium", expected="green",
        note="Compact hyphenated form.",
    ),
    Prompt(
        "Put the banana into the bowl that is not the red one and not the blue one.",
        "negation", "medium", expected="green",
        note="Verbose with 'one'.",
    ),
    Prompt(
        "Put the banana into the only bowl that has no red and no blue.",
        "negation", "medium", expected="green",
        note="Rephrased with 'has no'.",
    ),
    Prompt(
        "Put the banana into the bowl that is the remaining color after excluding red and blue.",
        "negation", "hard", expected="green",
        note="Verbose exclusion phrasing.",
    ),

    # ── 3. Color analogy (deterministic → green) ──────────────────────────────
    Prompt(
        "Put the banana into the bowl that matches the color of grass.",
        "color_analogy", "medium", expected="green",
        note="Grass = green.",
    ),
    Prompt(
        "Put the banana into the bowl the same color as leaves.",
        "color_analogy", "medium", expected="green",
        note="Leaves = green.",
    ),
    Prompt(
        "Put the banana into the bowl that is the color of a lime.",
        "color_analogy", "medium", expected="green",
        note="Lime = green.",
    ),
    Prompt(
        "Put the banana into the bowl that matches the color of a traffic light when you can go.",
        "color_analogy", "medium", expected="green",
        note="Go signal = green.",
    ),
    Prompt(
        "Put the banana into the bowl that is the color of an apple.",
        "color_analogy", "hard", expected="red",
        note="Apple = red (target for red bowl dataset).",
    ),
    Prompt(
        "Put the banana into the bowl that is the color of a tomato.",
        "color_analogy", "medium", expected="red",
        note="Tomato = red.",
    ),
    Prompt(
        "Put the banana into the bowl that matches the color of the sky.",
        "color_analogy", "medium", expected="blue",
        note="Sky = blue.",
    ),
    Prompt(
        "Put the banana into the bowl that is the color of the ocean.",
        "color_analogy", "medium", expected="blue",
        note="Ocean = blue.",
    ),
    Prompt(
        "Put the banana into the bowl that is the color of a blueberry.",
        "color_analogy", "medium", expected="blue",
        note="Blueberry = blue.",
    ),
    Prompt(
        "Put the banana into the bowl the color of fresh spinach.",
        "color_analogy", "hard", expected="green",
        note="Spinach = green.",
    ),
    Prompt(
        "Put the banana into the bowl that is the same color as a stop sign.",
        "color_analogy", "medium", expected="red",
        note="Stop sign = red.",
    ),

    # ── 4. Spatial absolute ────────────────────────────────────────────────────
    #
    # Actual bowl layout in banana_green_bowl_eval1_v2 (confirmed from first frame):
    #   Camera image (left→right): GREEN, RED, BLUE
    #   Robot perspective  (left→right): BLUE, RED, GREEN  (image is mirrored)
    #
    # Camera is mounted IN FRONT of the robot, FACING the robot:
    #   robot's LEFT  = image RIGHT  →  BLUE
    #   robot's RIGHT = image LEFT   →  GREEN
    #   center (depth) = RED (bottom center of image)
    #
    Prompt(
        "Put the banana into the leftmost bowl from the robot perspective.",
        "spatial_absolute", "hard", expected="blue",
        note="Robot-left = image-right = BLUE.",
    ),
    Prompt(
        "Put the banana into the rightmost bowl from the robot perspective.",
        "spatial_absolute", "hard", expected="green",
        note="Robot-right = image-left = GREEN (the target bowl).",
    ),
    Prompt(
        "Put the banana into the bowl in the middle.",
        "spatial_absolute", "hard", expected="red",
        note="Center bowl = RED (no perspective flip for center).",
    ),
    Prompt(
        "Put the banana into the 1st bowl from the left from the robot perspective.",
        "spatial_absolute", "hard", expected="blue",
        note="Robot 1st-from-left = BLUE.",
    ),
    Prompt(
        "Put the banana into the 2nd bowl from the left from the robot perspective.",
        "spatial_absolute", "hard", expected="red",
        note="Robot 2nd-from-left = RED (center).",
    ),
    Prompt(
        "Put the banana into the 3rd bowl from the left from the robot perspective.",
        "spatial_absolute", "hard", expected="green",
        note="Robot 3rd-from-left = GREEN (the target bowl).",
    ),
    Prompt(
        "Put the banana into the 1st bowl from the right from the robot perspective.",
        "spatial_absolute", "hard", expected="green",
        note="Robot 1st-from-right = GREEN (the target bowl).",
    ),
    Prompt(
        "Put the banana into the 2nd bowl from the right from the robot perspective.",
        "spatial_absolute", "hard", expected="red",
        note="Robot 2nd-from-right = RED.",
    ),
    Prompt(
        "Put the banana into the bowl that is not in the middle from the robot perspective.",
        "spatial_absolute", "very_hard", expected=None,
        note="Ambiguous: two valid answers (BLUE or GREEN, both are ends from robot view).",
    ),

    # ── 5. Spatial relative to named bowl ─────────────────────────────────────
    #
    # Robot perspective order: BLUE(left), RED(center), GREEN(right)
    # Image order:             GREEN(left), RED(center), BLUE(right)
    #
    Prompt(
        "Put the banana into the bowl on the right of the red bowl from the robot perspective.",
        "spatial_relative", "hard", expected="green",
        note="Robot-right of RED = image-left of RED = GREEN (the target bowl).",
    ),
    Prompt(
        "Put the banana into the bowl on the left of the red bowl from the robot perspective.",
        "spatial_relative", "hard", expected="blue",
        note="Robot-left of RED = image-right of RED = BLUE.",
    ),
    Prompt(
        "Put the banana into the bowl on the right of the blue bowl from the robot perspective.",
        "spatial_relative", "hard", expected="red",
        note="Robot-right of BLUE = image-left of BLUE = RED (BLUE is leftmost from robot).",
    ),
    Prompt(
        "Put the banana into the bowl on the left of the blue bowl from the robot perspective.",
        "spatial_relative", "hard", expected=None,
        note="BLUE is leftmost from robot — nothing is to its left. Ambiguous for this layout.",
    ),
    Prompt(
        "Put the banana into the bowl that is between the red bowl and the blue bowl.",
        "spatial_relative", "hard", expected=None,
        note="In robot order BLUE-RED-GREEN, RED is between BLUE and GREEN, not between RED and BLUE. Ambiguous.",
    ),
    Prompt(
        "Put the banana into the bowl that is immediately next to the red bowl on the side closer to the robot's left.",
        "spatial_relative", "very_hard", expected="blue",
        note="Robot-left side of RED = BLUE.",
    ),
    Prompt(
        "Put the banana into the bowl that is adjacent to both the red bowl and the blue bowl.",
        "spatial_relative", "very_hard", expected=None,
        note="In layout BLUE-RED-GREEN, no bowl (other than RED itself) is adjacent to both RED and BLUE.",
    ),
    Prompt(
        "Put the banana into the bowl that is not at either end from the robot perspective.",
        "spatial_relative", "hard", expected="red",
        note="Ends from robot view = BLUE and GREEN. Middle = RED.",
    ),

    # ── 6. Combined spatial + logical ─────────────────────────────────────────
    #
    # Note: several combined prompts are CONTRADICTORY for this specific layout
    # (green is at the robot-rightmost position), marked expected=None.
    #
    Prompt(
        "Put the banana into the bowl that is not blue and is to the left of the red bowl from the robot perspective.",
        "combined", "very_hard", expected="green",
        note="Negation excludes blue+red (window catches 'red bowl') → green. Spatial part ignored; accidental correct answer for this layout.",
    ),
    Prompt(
        "Put the banana into the bowl that is not red and is not the rightmost bowl from the robot perspective.",
        "combined", "very_hard", expected="blue",
        note="Not RED, not GREEN (rightmost from robot) → BLUE.",
    ),
    Prompt(
        "Put the banana into the 2nd bowl from the right from the robot perspective that is not red.",
        "combined", "very_hard", expected=None,
        note="2nd from robot-right = RED, but 'not red' contradicts it. Invalid for this layout.",
    ),
    Prompt(
        "Put the banana into the bowl that is neither blue nor in the leftmost position from the robot perspective.",
        "combined", "very_hard", expected=None,
        note="Not BLUE + not leftmost (= not BLUE again) → RED or GREEN. Still ambiguous.",
    ),
    Prompt(
        "Put the banana into the bowl that is not red and not at either end from the robot perspective.",
        "combined", "very_hard", expected=None,
        note="Not RED + not at ends (ends=BLUE,GREEN) → all 3 excluded. Contradictory for this layout.",
    ),
    Prompt(
        "Put the banana into the bowl that is the color of grass and is not the rightmost from the robot perspective.",
        "combined", "very_hard", expected="green",
        note="Analogy (grass=green) fires before spatial constraint is evaluated. Spatial 'not rightmost' is ignored; contradiction for this layout but deterministic pipeline outputs green.",
    ),

    # ── 7. Tricky / confusing wording ─────────────────────────────────────────
    Prompt(
        "Put the banana into the bowl that has the same color as the 'go' signal in a traffic light.",
        "tricky_wording", "hard", expected="green",
        note="Traffic light go = green, with extra quote marks.",
    ),
    Prompt(
        "Put the banana into the bowl that is not warm-colored.",
        "tricky_wording", "very_hard", expected=None,
        note="Warm colors = red/orange/yellow; cool = blue/green. Ambiguous — could be blue or green.",
    ),
    Prompt(
        "Put the banana into the bowl whose color you would NOT associate with danger or the sea.",
        "tricky_wording", "very_hard", expected="green",
        note="Danger=red, sea=blue → green.",
    ),
    Prompt(
        "Put the banana into the bowl that is the color of the flag of a country known for its rainforest.",
        "tricky_wording", "very_hard", expected="green",
        note="Brazil flag is green/yellow — green is the expected answer.",
    ),
    Prompt(
        "Put the banana into the bowl that is neither the color of fire nor the color of water.",
        "tricky_wording", "very_hard", expected="green",
        note="Fire=red, water=blue → green.",
    ),
    Prompt(
        "Put the banana into the bowl that is the color which is between yellow and blue on the color wheel.",
        "tricky_wording", "very_hard", expected="green",
        note="Yellow + blue mix = green.",
    ),
    Prompt(
        "Put the banana into the bowl that is the color on the opposite side of the color wheel from red.",
        "tricky_wording", "very_hard", expected="green",
        note="Red's complementary = cyan/green. Approximately green.",
    ),
]


# ── Frame extraction ───────────────────────────────────────────────────────────

IMAGE_KEY = "observation.images.front"
REPO_ID = "RobotLearningVLA/banana_green_bowl_eval1_v2"


def load_first_frame(repo_id: str = REPO_ID) -> "PIL.Image.Image":
    """Download and return the very first camera frame from the dataset."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image
    import numpy as np

    print(f"[dataset] Loading {repo_id} (first episode only) …")
    ds = LeRobotDataset(repo_id, episodes=[0], video_backend="pyav")
    row = ds[0]
    img_tensor = row[IMAGE_KEY]  # CHW float [0,1]

    if img_tensor.dim() == 3 and img_tensor.shape[0] in (1, 3, 4):
        img_tensor = img_tensor.permute(1, 2, 0)  # HWC
    arr = (img_tensor.clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


# ── Test runner ────────────────────────────────────────────────────────────────

_COLORS = {None: "\033[0m", "PASS": "\033[92m", "FAIL": "\033[91m", "VERIFY": "\033[93m", "ERROR": "\033[91m"}
_RST = "\033[0m"


def _colored(text: str, tag: str) -> str:
    return f"{_COLORS.get(tag, '')}{text}{_RST}"


def run_tests(
    normalizer: PromptNormalizer,
    frame,
    prompts: list[Prompt],
    stop_on_first_fail: bool = False,
) -> None:
    total = len(prompts)
    passed = 0
    failed = 0
    verify = 0
    errors = 0

    print()
    print("=" * 80)
    print(f"  PROMPT NORMALIZER TEST  |  {total} prompts  |  dataset: {REPO_ID}")
    print("=" * 80)

    prev_category = None
    for i, p in enumerate(prompts, 1):
        if p.category != prev_category:
            print(f"\n── {p.category.upper().replace('_', ' ')} ──")
            prev_category = p.category

        t0 = time.perf_counter()
        try:
            result = normalizer.normalize(frame, p.text)
            elapsed = time.perf_counter() - t0
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            tag = "ERROR"
            label = _colored(f"[{tag}] ", tag)
            print(f"  {i:2d}. {label}{p.text}")
            print(f"       exception: {exc}")
            errors += 1
            if stop_on_first_fail:
                break
            continue

        # Determine result label
        extracted = _extract_result_color(result)
        if p.expected is None:
            tag = "VERIFY"
            status = _colored("[VERIFY]", "VERIFY")
            verify += 1
        elif extracted == p.expected:
            tag = "PASS"
            status = _colored("[PASS]  ", "PASS")
            passed += 1
        else:
            tag = "FAIL"
            status = _colored("[FAIL]  ", "FAIL")
            failed += 1

        diff_str = f"difficulty={p.difficulty}"
        time_str = f"{elapsed:.2f}s"
        print(f"  {i:2d}. {status} {p.text}")
        print(f"        → {result!r}  ({diff_str}, {time_str})")
        if p.expected is not None and tag == "FAIL":
            print(f"        EXPECTED: {p.expected!r}  GOT: {extracted!r}")
        if p.expected is None:
            print(f"        NOTE: {p.note}")
        if stop_on_first_fail and tag == "FAIL":
            break

    print()
    print("=" * 80)
    print(
        f"  RESULTS  |  "
        f"{_colored(f'PASS: {passed}', 'PASS')}  "
        f"{_colored(f'FAIL: {failed}', 'FAIL')}  "
        f"{_colored(f'VERIFY: {verify}', 'VERIFY')}  "
        f"ERROR: {errors}  "
        f"TOTAL: {total}"
    )
    print("=" * 80)
    print()
    if verify:
        print(
            f"  {verify} spatial prompt(s) marked VERIFY — their correct answers depend on\n"
            "  the actual bowl layout in the camera image.\n"
            "  Re-run with --save-frame to dump the first frame, inspect the layout,\n"
            "  then annotate 'expected' values in this test file.\n"
        )


def _extract_result_color(result: str) -> str | None:
    """Pull the color word out of a canonical 'Put the banana in the X colored bowl.' string."""
    import re
    for c in ("red", "green", "blue"):
        if re.search(rf"\b{c}\b", result, re.IGNORECASE):
            return c
    return None


# ── Normalizer loaders ─────────────────────────────────────────────────────────

def load_normalizer_from_policy(policy_path: str, device: str) -> PromptNormalizer:
    """Load PromptNormalizer from a SmolVLA policy (reuses its VLM backbone).

    Applies the eval3 lerobot shim first so SmolVLAPolicy imports cleanly.
    """
    try:
        from eval3_lerobot_shim import apply as _shim_apply
        _shim_apply()
    except ImportError:
        pass  # shim may not be needed in all environments

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    print(f"[normalizer] Loading SmolVLA policy from {policy_path!r} on {device} …")
    policy = SmolVLAPolicy.from_pretrained(policy_path)
    policy = policy.to(device)
    policy.eval()
    return PromptNormalizer.from_policy(policy, device=device)


def load_normalizer_standalone(model_id: str, device: str) -> PromptNormalizer:
    """Load PromptNormalizer from a plain SmolVLM checkpoint (no SmolVLA policy)."""
    print(f"[normalizer] Loading {model_id} on {device} …")
    return PromptNormalizer.from_pretrained(model_id, device=device)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--policy-path",
        default=None,
        help=(
            "If set, load a SmolVLA policy and reuse its VLM for spatial reasoning "
            "(e.g. lerobot/smolvla_base or a local checkpoint). "
            "By default the VLM backbone is loaded directly via --model-id."
        ),
    )
    ap.add_argument(
        "--model-id",
        default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        help=(
            "SmolVLM Hub model ID loaded directly (default). "
            "Same VLM backbone as SmolVLA base; used when --policy-path is not set."
        ),
    )
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device.",
    )
    ap.add_argument(
        "--save-frame",
        metavar="PATH",
        default=None,
        help="Save the extracted camera frame to this path for manual layout inspection.",
    )
    ap.add_argument(
        "--category",
        choices=[
            "direct", "negation", "color_analogy",
            "spatial_absolute", "spatial_relative",
            "combined", "tricky_wording",
        ],
        default=None,
        help="Run only prompts in this category.",
    )
    ap.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "very_hard"],
        default=None,
        help="Run only prompts at this difficulty level.",
    )
    ap.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="Stop after the first deterministic failure.",
    )
    args = ap.parse_args()

    # ── Load frame ─────────────────────────────────────────────────────────────
    frame = load_first_frame()
    print(f"[dataset] Frame size: {frame.size}  (W × H)")

    if args.save_frame:
        frame.save(args.save_frame)
        print(f"[dataset] Frame saved → {args.save_frame}")

    # ── Build normalizer ───────────────────────────────────────────────────────
    if args.policy_path:
        normalizer = load_normalizer_from_policy(args.policy_path, args.device)
    else:
        normalizer = load_normalizer_standalone(args.model_id, args.device)

    # ── Filter prompts ─────────────────────────────────────────────────────────
    prompts = PROMPTS
    if args.category:
        prompts = [p for p in prompts if p.category == args.category]
    if args.difficulty:
        prompts = [p for p in prompts if p.difficulty == args.difficulty]

    if not prompts:
        print("No prompts match the given filters.")
        sys.exit(1)

    # ── Run ────────────────────────────────────────────────────────────────────
    run_tests(normalizer, frame, prompts, stop_on_first_fail=args.stop_on_fail)


if __name__ == "__main__":
    main()
