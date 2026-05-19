#!/usr/bin/env python3
"""Quick prompt-normalization checks for Eval1/Eval2 without moving the robot."""

from __future__ import annotations

import argparse
import os
import sys


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
    "put the banana in the middle of the three bowls",
    "put the banana into the center bowl",
    "put the banana in the central bowl",
    "put banana in the second bowl",
    "put banana in the second bowl from the left",
    "put banana in the 2nd bowl from left",
    "put banana in the 2nd from the robot's left",
    "put banana in the first bowl from the right",
    "put banana in the third bowl from the robot's right",
    "put banana in the rightmost bowl",
    "put banana in the leftmost bowl",
    "put banana in the far left bowl",
    "put banana in the right-hand bowl",
    "put banana in the bowl on the left side",
    "put banana in the bowl to the right of the leftmost bowl",
    "put banana in the bowl left of the middle bowl",
    "put banana in the bowl right of the middle bowl",
    "put banana in the bowl beside the rightmost bowl",
    "put banana in the bowl on the right of the red bowl",
    "Put the banana into the bowl on the right of the red bowl. (from the robot perspective)",
    "put banana in the bowl directly right of the red bowl",
    "put banana in the bowl on the right side of the red bowl",
    "put banana in the bowl on the left of the red bowl",
    "put banana in the bowl just left of the red bowl",
    "put banana in the red bowl's right",
    "put banana in the red bowl's left",
    "put banana in the red bowl's right neighbor",
    "put banana in the right neighbor of the red bowl",
    "put banana in the neighbor to the left of the green bowl",
    "put banana in the bowl next to the red bowl on the right",
    "put banana in the bowl adjacent to the red bowl on the left",
    "put banana in the bowl beside the red bowl on the right",
    "put banana in the bowl next to the leftmost bowl",
    "put banana in the bowl adjacent to the rightmost bowl",
    "put banana in the bowl next to the middle bowl on the right",
    "put banana in the bowl between blue and green",
    "put banana in between the blue bowl and the green bowl",
    "put banana in the one between the blue and green bowls",
    "put banana in the gap between the blue bowl and the green bowl",
    "put banana in the bowl separating blue and green",
    "put banana in the bowl that separates blue from green",
    "put banana in the bowl adjacent to both blue and green",
    "put banana in the bowl that is not at either end",
    "put banana in the bowl that is neither leftmost nor rightmost",
    "put banana in the bowl that is not the middle one",
    "put banana in the bowl that is not green and not blue",
    "put banana in the bowl that is not red and not blue",
    "put banana in the bowl that is not red and not green",
    "put banana in the color of grass bowl",
    "put banana in the bowl whose color is like the sky",
    "put banana in the bowl on the left of the red bowl",
    "put banana somewhere safe",
    "put banana in the bowl that is not red",
]


def _parse_robot_order(value: str) -> tuple[str, str, str]:
    return validate_robot_order(tuple(part.strip().lower() for part in value.split(",") if part.strip()))


def _print_cases(name: str, prompts: list[str], robot_order: tuple[str, str, str] | None) -> None:
    print(f"\n[{name}] robot_order={robot_order}")
    for prompt in prompts:
        try:
            normalized = normalize_prompt_text(prompt, robot_order=robot_order)
            reason = "strict"
        except Exception as exc:
            normalized, reason = normalize_prompt_best_effort(prompt, robot_order=robot_order)
            reason = f"best-effort: {reason} (strict failed: {type(exc).__name__})"
        print(f"- {prompt!r}\n  -> {normalized}\n     {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("task1", "task2", "all"), default="all")
    parser.add_argument(
        "--robot-order",
        default="blue,red,green",
        help="Robot-perspective L->R order for task2 spatial prompts.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Custom prompt to test. Can be passed multiple times.",
    )
    args = parser.parse_args()

    robot_order = _parse_robot_order(args.robot_order)

    if args.prompt:
        _print_cases("custom", args.prompt, robot_order if args.task != "task1" else None)
        return

    if args.task in {"task1", "all"}:
        _print_cases("task1", TASK1_CASES, None)
    if args.task in {"task2", "all"}:
        _print_cases("task2", TASK2_CASES, robot_order)


if __name__ == "__main__":
    main()
