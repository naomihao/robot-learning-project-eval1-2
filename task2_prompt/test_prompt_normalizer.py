#!/usr/bin/env python3
"""Offline prompt-normalizer smoke test for the banana/bowl eval tasks.

This script is intentionally outside the deployment path. By default it uses
only the deterministic prompt normalizer. Pass --model-id only for legacy
offline VLM fallback experiments.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from eval1_prompt_normalizer import DEFAULT_ROBOT_ORDER, PromptNormalizer, validate_robot_order


@dataclass(frozen=True)
class PromptCase:
    text: str
    expected: str | None
    category: str


PROMPTS = [
    PromptCase("Put the banana in the red colored bowl.", "red", "direct"),
    PromptCase("Pick up the banana and place it into the green bowl.", "green", "direct"),
    PromptCase("Place it to blue bowl.", "blue", "direct"),
    PromptCase("Put the banana into the bowl that is not red and not blue.", "green", "negation"),
    PromptCase("Put the banana into the bowl that is neither red nor green.", "blue", "negation"),
    PromptCase("Put the banana into the non-red and non-blue bowl.", "green", "negation"),
    PromptCase("Put the banana into the bowl that matches the color of grass.", "green", "analogy"),
    PromptCase("Put the banana into the bowl that is the color of a tomato.", "red", "analogy"),
    PromptCase("Put the banana into the bowl that matches the color of the sky.", "blue", "analogy"),
    PromptCase("Put the banana into the leftmost bowl from the robot perspective.", None, "spatial"),
    PromptCase("Put the banana into the rightmost bowl from the robot perspective.", None, "spatial"),
    PromptCase("Put the banana into the 2nd bowl from the left from the robot perspective.", None, "spatial"),
    PromptCase("Put the banana to the left of the avocado colored bowl.", None, "spatial"),
    PromptCase("Put the banana to the right of the sky colored bowl.", None, "spatial"),
    PromptCase("Put the banana into the bowl that is not blood or grass.", "blue", "combined"),
    PromptCase("Put the banana into the bowl that is not sea or leftmost.", None, "combined"),
]


def _extract_color(result: str) -> str | None:
    match = re.search(r"\b(red|green|blue)\b", result, flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def _parse_order(text: str) -> tuple[str, str, str]:
    return validate_robot_order(tuple(part.strip().lower() for part in text.split(",")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        default=str(Path(__file__).with_name("first_frame.png")),
        help="Camera frame used for spatial cases.",
    )
    parser.add_argument(
        "--fallback-robot-order",
        default=",".join(DEFAULT_ROBOT_ORDER),
        help="Robot-perspective fallback bowl order, e.g. blue,red,green.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Optional standalone SmolVLM model id for legacy/offline fallback.",
    )
    parser.add_argument("--device", default="cuda", help="Device for optional standalone SmolVLM fallback.")
    parser.add_argument("--category", default=None, help="Run only one category.")
    args = parser.parse_args()

    frame = Image.open(args.image).convert("RGB")
    fallback_order = _parse_order(args.fallback_robot_order)
    if args.model_id:
        normalizer = PromptNormalizer.from_pretrained(
            args.model_id,
            device=args.device,
            fallback_robot_order=fallback_order,
        )
    else:
        normalizer = PromptNormalizer(fallback_robot_order=fallback_order)

    prompts = [case for case in PROMPTS if args.category is None or case.category == args.category]
    if not prompts:
        raise SystemExit(f"No prompts matched category {args.category!r}")

    print(f"[test] image={args.image}")
    print(f"[test] fallback robot L->R={fallback_order}")
    for i, case in enumerate(prompts, start=1):
        normalizer._robot_order = None
        result = normalizer.normalize(frame, case.text)
        got = _extract_color(result)
        if case.expected is None:
            status = "CHECK"
        elif got == case.expected:
            status = "PASS"
        else:
            status = f"FAIL expected={case.expected} got={got}"
        print(f"{i:02d}. [{status}] {case.text}")
        print(f"    -> {result}")


if __name__ == "__main__":
    main()
