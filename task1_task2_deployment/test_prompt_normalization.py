#!/usr/bin/env python3
"""Quick prompt-normalization checks for Eval1/Eval2 without moving the robot."""

from __future__ import annotations

import argparse
import os
import sys
from itertools import permutations


PROMPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "task2_prompt"))
if PROMPT_DIR not in sys.path:
    sys.path.insert(0, PROMPT_DIR)

from eval1_prompt_normalizer import normalize_prompt_best_effort, normalize_prompt_text, validate_robot_order


TASK1_CASES = [
    "put the banana in the red bowl",
    "put the banana in the red color bowl",
    "pick the banana in the red color bowl",
    "pick up banana and place it into the red bowl",
    "grab the banana and drop it in red bowl",
    "move banana to the red colored bowl",
    "place banana inside the blue bowl",
    "put banana into green coloured bowl",
    "banana to the green color bowl",
    "the target is the blue one",
    "use the red cup",
    "put it in bowl red",
]


TASK2_CASES = [
    # Middle / center / second
    "put the banana in the middle of the three bowls",
    "put the banana into the center bowl",
    "put the banana into the centre bowl",
    "put the banana in the central bowl",
    "put banana in the bowl in the middle",
    "put banana in the middle one",
    "place the banana in the center one",
    "drop the banana into the bowl that is in between the other two",
    "put banana in the second bowl",
    "put banana in the second bowl from the left",
    "put banana in the 2nd bowl from left",
    "put banana in the 2nd from the robot's left",
    "put banana in the number two bowl from the left",
    "put banana in bowl number 2 from robot left",
    "put banana in the second from the robot perspective",

    # Absolute left / right / ordinal
    "put banana in the first bowl from the right",
    "put banana in the third bowl from the robot's right",
    "put banana in the first bowl from the left",
    "put banana in the third bowl from the left",
    "put banana in the 1st bowl from the robot's right",
    "put banana in the 3rd bowl from the robot's left",
    "put banana in the rightmost bowl",
    "put banana in the leftmost bowl",
    "put banana in the far left bowl",
    "put banana in the far right bowl",
    "put banana in the right-hand bowl",
    "put banana in the left-hand bowl",
    "put banana in the bowl on the left side",
    "put banana in the bowl on the right side",
    "put banana in the left side bowl",
    "put banana in the right side bowl",
    "put banana in the outer left bowl",
    "put banana in the outer right bowl",

    # Relative to positional landmarks
    "put banana in the bowl to the right of the leftmost bowl",
    "put banana in the bowl to the left of the rightmost bowl",
    "put banana in the bowl left of the middle bowl",
    "put banana in the bowl right of the middle bowl",
    "put banana in the bowl immediately to the left of the middle bowl",
    "put banana in the bowl immediately to the right of the middle bowl",
    "put banana in the bowl beside the rightmost bowl",
    "put banana in the bowl beside the leftmost bowl",

    # Relative to explicit color landmarks
    "put banana in the bowl on the right of the red bowl",
    "put banana in the bowl on the left of the red bowl",
    "put banana in the bowl on the right of the green bowl",
    "put banana in the bowl on the left of the green bowl",
    "put banana in the bowl on the right of the blue bowl",
    "put banana in the bowl on the left of the blue bowl",
    "put banana to the right of the red",
    "put banana to the left of the red",
    "put banana to the left of the blue",
    "put banana to the right of the blue",
    "put banana to the left of the green",
    "put banana to the right of the green",
    "put banana to the left of the avocado colored bowl",
    "put banana to the right of the sky colored bowl",
    "put banana to the left of the grass colored bowl",
    "put banana to the right of the tomato colored bowl",
    "put banana just to the left of the avocado bowl",
    "put banana immediately to the right of the sky bowl",
    "put banana directly to the left of the green colored bowl",
    "put banana directly to the right of the blue colored bowl",
    "Put the banana into the bowl on the right of the red bowl. (from the robot perspective)",
    "put banana in the bowl directly right of the red bowl",
    "put banana in the bowl on the right side of the red bowl",
    "put banana in the bowl just left of the red bowl",
    "put banana in the bowl just to the right of the red bowl",
    "put banana in the bowl immediately to the left of the red bowl",
    "put banana in the red bowl's right",
    "put banana in the red bowl's left",
    "put banana in the red bowl's right neighbor",
    "put banana in the red bowl's left neighbor",
    "put banana in the right neighbor of the red bowl",
    "put banana in the left neighbor of the red bowl",
    "put banana in the neighbor to the left of the green bowl",
    "put banana in the neighbor to the right of the blue bowl",
    "put banana in the bowl next to the red bowl on the right",
    "put banana in the bowl adjacent to the red bowl on the left",
    "put banana in the bowl beside the red bowl on the right",
    "put banana in the bowl beside the green bowl on the left",
    "put banana in the bowl near the blue bowl on the right",
    "put banana in the bowl next to the leftmost bowl",
    "put banana in the bowl adjacent to the rightmost bowl",
    "put banana in the bowl next to the middle bowl on the right",
    "put banana in the bowl next to the middle bowl on the left",
    "put it to the 2nd left of the green",
    "put it to the second left of the avocado colored bowl",
    "put it two bowls to the right of the blue",
    "put it to the 2nd right of the sky colored bowl",

    # Between / separator / adjacent to both
    "put banana in the bowl between blue and green",
    "put banana in between the blue bowl and the green bowl",
    "put banana in the one between the blue and green bowls",
    "put banana in the gap between the blue bowl and the green bowl",
    "put banana in the bowl located between the blue bowl and the green bowl",
    "put banana in the bowl sitting between red and blue",
    "put banana in the bowl separating blue and green",
    "put banana in the bowl that separates blue from green",
    "put banana in the separator bowl between blue and green",
    "put banana in the bowl adjacent to both blue and green",
    "put banana in the one touching both the blue and green bowls",

    # Negation / exclusion
    "put banana in the bowl that is not at either end",
    "put banana in the bowl that is neither leftmost nor rightmost",
    "put banana in the bowl that is not the middle one",
    "put banana in the bowl that is not green and not blue",
    "put banana in the bowl that is not red and not blue",
    "put banana in the bowl that is not red and not green",
    "put banana in the bowl that is neither red nor blue",
    "put banana in the bowl excluding green and blue",
    "put banana in the bowl without red or green",
    "put banana in the non-red non-blue bowl",
    "put banana in the bowl that is not the rightmost and not blue",
    "put banana in the bowl that is neither leftmost nor blue",
    "put banana in the bowl that is not blood or grass",
    "put banana in the bowl that is not blood and not grass",
    "put banana in the bowl that is neither blood nor grass",
    "put banana in the bowl that is not tomato colored or avocado colored",
    "put banana in the bowl that is not sea or leftmost",
    "put banana in the bowl that is not ocean colored and not rightmost",
    "put banana in the bowl that is not blood or rightmost",
    "put banana in the bowl that is neither red nor rightmost",
    "put banana in the bowl that is not avocado colored and not leftmost",
    "put banana in the bowl that is neither sky colored nor middle",
    "put banana in the bowl that is not grass colored and not rightmost",
    "put it to the right of not blue not avocado",
    "put it to the left of not sky not avocado",
    "put it to the second left of not red not blue",
    "put it two bowls right of not blood not grass",

    # Color analogies as direct targets
    "put banana in the color of grass bowl",
    "put banana in the bowl whose color is like the sky",
    "put banana in the avocado colored bowl",
    "put banana in the tomato colored bowl",
    "put banana in the ocean colored bowl",
    "put banana in the bowl the color of a stop sign",
    "put banana in the bowl the color of a leaf",
    "put banana in the bowl the color of blueberries",
    "put banana in the traffic light go bowl",

    # Unknown / fallback sanity checks
    "put banana in the bowl on the left of the red bowl",
    "put banana somewhere safe",
    "put banana in the bowl that is not red",
]


ALL_ROBOT_ORDERS = [
    ("blue", "red", "green"),
    *[
        order
        for order in permutations(("red", "green", "blue"))
        if order != ("blue", "red", "green")
    ],
]


def _parse_robot_order(value: str) -> tuple[str, str, str]:
    return validate_robot_order(tuple(part.strip().lower() for part in value.split(",") if part.strip()))


def _color_from_prompt(normalized: str) -> str:
    for color in ("red", "green", "blue"):
        if f" {color} " in normalized:
            return color
    return "?"


def _normalize_for_print(
    prompt: str,
    robot_order: tuple[str, str, str] | None,
) -> tuple[str, str, str]:
    try:
        normalized = normalize_prompt_text(prompt, robot_order=robot_order)
        reason = "strict"
    except Exception as exc:
        normalized, reason = normalize_prompt_best_effort(prompt, robot_order=robot_order)
        reason = f"best-effort: {reason} (strict failed: {type(exc).__name__})"
    return normalized, _color_from_prompt(normalized), reason


def _print_cases(
    name: str,
    prompts: list[str],
    robot_order: tuple[str, str, str] | None,
    *,
    compact: bool,
) -> tuple[int, int]:
    print(f"\n[{name}] robot_order={robot_order}")
    strict_count = 0
    best_effort_count = 0
    for prompt in prompts:
        normalized, color, reason = _normalize_for_print(prompt, robot_order)
        if reason == "strict":
            strict_count += 1
        else:
            best_effort_count += 1
        if compact:
            tag = "S" if reason == "strict" else "B"
            print(f"{tag} {color:5s} | {prompt}")
        else:
            print(f"- {prompt!r}\n  -> {normalized}\n     {reason}")
    print(f"[summary] strict={strict_count}  best_effort={best_effort_count}  total={len(prompts)}")
    return strict_count, best_effort_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("task1", "task2", "all"), default="all")
    parser.add_argument(
        "--robot-order",
        default="blue,red,green",
        help="Robot-perspective L->R order for task2 spatial prompts.",
    )
    parser.add_argument(
        "--all-robot-orders",
        action="store_true",
        help="For task2/custom prompts, run all 6 robot-perspective bowl orders.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print one line per prompt: S/B color | prompt.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Custom prompt to test. Can be passed multiple times.",
    )
    args = parser.parse_args()

    robot_orders = [_parse_robot_order(args.robot_order)]
    if args.all_robot_orders and args.task != "task1":
        robot_orders = ALL_ROBOT_ORDERS

    if args.prompt:
        if args.task == "task1":
            _print_cases("custom", args.prompt, None, compact=args.compact)
        else:
            for robot_order in robot_orders:
                _print_cases("custom", args.prompt, robot_order, compact=args.compact)
        return

    if args.task in {"task1", "all"}:
        _print_cases("task1", TASK1_CASES, None, compact=args.compact)
    if args.task in {"task2", "all"}:
        for robot_order in robot_orders:
            _print_cases("task2", TASK2_CASES, robot_order, compact=args.compact)


if __name__ == "__main__":
    main()
