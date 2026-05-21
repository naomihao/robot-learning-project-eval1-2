"""
Normalizes complex/indirect task prompts to canonical color-based format for SmolVLA eval1.

Input:  any phrasing of which bowl to target, e.g.
    "Put the banana into the 2nd bowl from the left from the robot perspective"
    "Put the banana into the bowl on the right of the red bowl from the robot perspective"
    "Put the banana into the bowl that is not green and not blue"
Output: "Put the banana in the [color] colored bowl."

Architecture — hybrid deterministic normalizer
----------------------------------------------
Prompts fall into two fundamentally different categories:

  1. Language-only (negation, color analogy) — no image needed, deterministic:
       "not red and not blue"          → green  (pure logic)
       "the color of grass"            → green  (knowledge lookup)
     Handled by ``_try_negation()`` and ``_try_analogy()`` without any extra model.

  2. Visual-spatial — requires the camera image to detect bowl layout:
       "2nd bowl from the left from the robot perspective"
       "bowl on the right of the red bowl"
     Handled by HSV color segmentation, then deterministic spatial rules.

Eval-day prompt preprocessing is deterministic: no extra language or vision
model is loaded to interpret the prompt.

Camera perspective (spatial prompts only)
------------------------------------------
The camera is physically mounted IN FRONT of the robot arm, pointing TOWARD the robot.
This makes the camera image a LEFT-RIGHT MIRROR of the robot's own field of view:

    robot's LEFT  → RIGHT side of the camera image
    robot's RIGHT → LEFT side of the camera image
"""
from __future__ import annotations

import logging
import re
from itertools import permutations
from typing import Union

try:
    import numpy as np
except ImportError:  # Allows text-only prompt normalization outside the robot env.
    np = None  # type: ignore[assignment]

try:
    from PIL import Image
except ImportError:  # Allows text-only prompt normalization outside the robot env.
    Image = None  # type: ignore[assignment]

try:
    import torch
except ImportError:  # Allows text-only prompt normalization outside the robot env.
    torch = None  # type: ignore[assignment]


BOWL_COLORS = ("red", "green", "blue")
CANONICAL_TEMPLATE = "Put the banana in the {color} colored bowl."
DEFAULT_ROBOT_ORDER = ("blue", "red", "green")
_SPATIAL_WORD_RE = re.compile(
    r"\b(?:left|right|middle|center|centre|between|adjacent|next|nearest|farthest|"
    r"leftmost|rightmost|first|second|third|1st|2nd|3rd|from)\b",
    re.IGNORECASE,
)
_POSITION_WORD_RE = re.compile(
    r"\b(?:left|right|middle|center|centre|between|adjacent|next|nearest|farthest|"
    r"leftmost|rightmost|first|second|third|1st|2nd|3rd)\b",
    re.IGNORECASE,
)
_COLOR_ALIASES: dict[str, tuple[str, ...]] = {
    "red": ("red", "reddish", "scarlet", "crimson"),
    "green": ("green", "greenish"),
    "blue": ("blue", "bluish"),
}


def canonical_prompt(color: str) -> str:
    """Return the exact prompt form expected by the deployed policy."""
    if color not in BOWL_COLORS:
        raise ValueError(f"Unknown bowl color {color!r}; expected one of {BOWL_COLORS}")
    return CANONICAL_TEMPLATE.format(color=color)


def validate_robot_order(robot_order: tuple[str, str, str]) -> tuple[str, str, str]:
    """Validate a left-to-right robot-perspective color order."""
    colors = tuple(color.strip().lower() for color in robot_order)
    if len(colors) != 3 or set(colors) != set(BOWL_COLORS):
        raise ValueError(f"Robot order must contain red, green, blue exactly once, got {robot_order!r}")
    return colors


def _find_color_mentions(prompt: str) -> list[str]:
    lower = prompt.lower()
    mentions: list[tuple[int, str]] = []
    for color, aliases in _COLOR_ALIASES.items():
        for alias in aliases:
            for match in re.finditer(rf"\b{re.escape(alias)}\b", lower):
                mentions.append((match.start(), color))
    mentions.sort()
    return [color for _, color in mentions]


def _analogy_color(keyword: str, color: str) -> str:
    return "blue" if color == "lake" else color


def _normalize_spatial_landmarks(prompt: str) -> str:
    """Rewrite color-like landmark words so spatial rules can use them.

    Example: "left of the avocado colored bowl" becomes
    "left of the green bowl". This must only be used in spatial parsing;
    non-spatial prompts can still use analogy rules as direct targets.
    """
    lower = prompt.lower()
    replacements: list[tuple[str, str]] = []
    for color, aliases in _COLOR_ALIASES.items():
        for alias in aliases:
            replacements.append((alias, color))
    for keyword, color in _ANALOGY_MAP.items():
        replacements.append((keyword, _analogy_color(keyword, color)))

    for keyword, color in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        lower = re.sub(
            rf"\b{re.escape(keyword)}\b(?:\s+(?:colou?red|colou?r))?",
            color,
            lower,
        )
    return lower


def _collect_excluded_colors(prompt: str) -> set[str]:
    lower = _normalize_spatial_landmarks(prompt)
    excluded: set[str] = set()
    for m in _NEG_MARKERS_RE.finditer(lower):
        window = lower[m.start() : min(len(lower), m.end() + 60)]
        for color in BOWL_COLORS:
            if re.search(rf"\b{color}\b", window):
                excluded.add(color)
        for concept, color in _CONCEPT_COLOR.items():
            if re.search(rf"\b{re.escape(concept)}\b", window):
                excluded.add(color)
    for color in BOWL_COLORS:
        if re.search(rf"\bnon[-\s]{color}\b", lower):
            excluded.add(color)
    return excluded


def _collect_excluded_position_colors(
    prompt: str,
    robot_order: tuple[str, str, str],
) -> set[str]:
    lower = _normalize_spatial_landmarks(prompt)
    robot = validate_robot_order(robot_order)
    excluded: set[str] = set()
    patterns = [
        (
            r"\bleftmost\b|\bfar\s+left\b|\bleft[-\s]?hand\b|"
            r"\bleft\s+(?:bowl|one|side)\b|\b(?:bowl|one)\s+on\s+the\s+left\b|"
            r"\b(?:1st|first)\s+(?:bowl|one)?\s*from\s+(?:the\s+)?left\b",
            robot[0],
        ),
        (
            r"\brightmost\b|\bfar\s+right\b|\bright[-\s]?hand\b|"
            r"\bright\s+(?:bowl|one|side)\b|\b(?:bowl|one)\s+on\s+the\s+right\b|"
            r"\b(?:1st|first)\s+(?:bowl|one)?\s*from\s+(?:the\s+)?right\b",
            robot[-1],
        ),
        (
            r"\bmiddle\b|\bcent(?:er|re)\b|\bcentral\b|\b(?:2nd|second)\s+(?:bowl|one)?\b",
            robot[1],
        ),
        (r"\b(?:3rd|third)\s+(?:bowl|one)?\s*from\s+(?:the\s+)?left\b", robot[2]),
        (r"\b(?:3rd|third)\s+(?:bowl|one)?\s*from\s+(?:the\s+)?right\b", robot[0]),
    ]

    for marker in _NEG_MARKERS_RE.finditer(lower):
        window = lower[marker.start() : min(len(lower), marker.end() + 80)]
        for pattern, color in patterns:
            if re.search(pattern, window):
                excluded.add(color)
    return excluded


def _choose_from_remaining(
    remaining: list[str],
    robot_order: tuple[str, str, str] | None,
) -> str:
    if robot_order is not None:
        for color in (robot_order[1], robot_order[0], robot_order[2]):
            if color in remaining:
                return color
    return remaining[0]


def _best_effort_color(
    raw_prompt: str,
    robot_order: tuple[str, str, str] | None = None,
    *,
    default_color: str | None = None,
) -> tuple[str, str]:
    """Choose a likely target color when strict parsing cannot resolve a prompt."""
    lower = _normalize_spatial_landmarks(raw_prompt)
    robot = validate_robot_order(robot_order) if robot_order is not None else None
    if default_color is not None and default_color not in BOWL_COLORS:
        raise ValueError(f"default_color must be one of {BOWL_COLORS}, got {default_color!r}")

    excluded = _collect_excluded_colors(raw_prompt)
    if robot is not None:
        excluded |= _collect_excluded_position_colors(raw_prompt, robot)
    if excluded:
        remaining = [color for color in BOWL_COLORS if color not in excluded]
        if remaining:
            return _choose_from_remaining(remaining, robot), f"excluded {sorted(excluded)}"

    mentions = _find_color_mentions(lower)
    unique_mentions = list(dict.fromkeys(mentions))

    if robot is not None and mentions:
        landmark = mentions[-1]
        if re.search(r"\bright\b", lower):
            idx = robot.index(landmark)
            best_idx = min(idx + 1, len(robot) - 1)
            return robot[best_idx], f"clamped right-of-{landmark}"
        if re.search(r"\bleft\b", lower):
            idx = robot.index(landmark)
            best_idx = max(idx - 1, 0)
            return robot[best_idx], f"clamped left-of-{landmark}"
        if re.search(r"\b(?:between|in\s+between|gap|separat(?:e|es|ing)|adjacent\s+to\s+both)\b", lower):
            return robot[1], "between/separator fallback to middle"

    if robot is not None:
        if re.search(r"\bnot\b.*\b(?:middle|cent(?:er|re))\b", lower):
            return robot[0], "not-middle fallback to leftmost"
        if re.search(r"\b(?:middle|cent(?:er|re))\b", lower):
            return robot[1], "middle keyword"
        if re.search(r"\b(?:rightmost|right)\b", lower):
            return robot[-1], "right keyword"
        if re.search(r"\b(?:leftmost|left)\b", lower):
            return robot[0], "left keyword"
        if re.search(r"\b(?:first|1st)\b", lower):
            return robot[0], "first keyword"
        if re.search(r"\b(?:second|2nd)\b", lower):
            return robot[1], "second keyword"
        if re.search(r"\b(?:third|3rd)\b", lower):
            return robot[2], "third keyword"

    if unique_mentions:
        return unique_mentions[-1], "last color mention"

    color = default_color or (robot[1] if robot is not None else "red")
    return color, "default color"


def normalize_prompt_best_effort(
    raw_prompt: str,
    robot_order: tuple[str, str, str] | None = None,
    *,
    default_color: str | None = None,
) -> tuple[str, str]:
    """Normalize with strict rules first, then return a canonical best-effort prompt."""
    try:
        return normalize_prompt_text(raw_prompt, robot_order=robot_order), "strict"
    except ValueError:
        color, reason = _best_effort_color(raw_prompt, robot_order, default_color=default_color)
        return canonical_prompt(color), reason

# ── Deterministic pre-processors ──────────────────────────────────────────────

# ── Negative marker set (used in several patterns below) ──────────────────────
_NEG_MARKERS_RE = re.compile(
    r"\b(?:not|neither|nor|no|without|excluding|except)\b",
    re.IGNORECASE,
)
_NON_COLOR_RE = re.compile(r"\bnon[-\s](?:red|green|blue)\b", re.IGNORECASE)

# Color knowledge: keyword → bowl color.
# Keys are lowercase substrings to search for in the prompt.
_ANALOGY_MAP: dict[str, str] = {
    # → green
    "grass":         "green",
    "leaf":          "green",
    "leaves":        "green",
    "lime":          "green",
    "spinach":       "green",
    "broccoli":      "green",
    "frog":          "green",
    "avocado":       "green",
    "cucumber":      "green",
    "pea":           "green",
    "mint":          "green",
    "basil":         "green",
    "rainforest":    "green",
    "jungle":        "green",
    "forest":        "green",
    # → red
    "tomato":        "red",
    "apple":         "red",
    "cherry":        "red",
    "stop sign":     "red",
    "fire engine":   "red",
    "fire truck":    "red",
    "rose":          "red",
    "strawberry":    "red",
    "blood":         "red",
    "chili":         "red",
    "ketchup":       "red",
    "ladybug":       "red",
    # → blue
    "sky":           "blue",
    "ocean":         "blue",
    "sea":           "blue",
    "blueberry":     "blue",
    "blueberries":   "blue",
    "sapphire":      "blue",
    "cobalt":        "blue",
    "navy":          "blue",
    "denim":         "blue",
    "jeans":         "blue",
    "water":         "blue",
    "river":         "blue",
    "lake":          "lake",   # placeholder — resolved below
}

# Traffic light special case (regex).
_TRAFFIC_GO = re.compile(r"traffic.{0,20}go|go.{0,20}traffic", re.IGNORECASE)

# Color-wheel reasoning patterns (wider window for "opposite side of the color wheel from red").
_BETWEEN_YELLOW_BLUE = re.compile(
    r"between\s+yellow\s+and\s+blue|yellow.{0,15}blue.{0,15}color\s+wheel",
    re.IGNORECASE,
)
_OPPOSITE_RED = re.compile(
    r"opposite.{0,40}red|complement.{0,40}red|red.{0,40}complement",
    re.IGNORECASE,
)

# Concept → bowl color, used for concept-negation prompts like
# "not associated with danger or the sea" → exclude red, exclude blue → green.
_CONCEPT_COLOR: dict[str, str] = {
    "danger":    "red",
    "fire":      "red",
    "hot":       "red",
    "warm":      "red",
    "sea":       "blue",
    "ocean":     "blue",
    "water":     "blue",
    "cold":      "blue",
}

# HSV color ranges for bowl detection (used by detect_layout).
# Banana is yellow (~hue 25-35) — does not overlap with any of these.
_HSV_RANGES: dict[str, list[tuple]] = {
    "red":   [(0, 80, 60, 12, 255, 255), (168, 80, 60, 180, 255, 255)],  # two hue bands
    "green": [(38, 60, 40, 90, 255, 255)],
    "blue":  [(95, 60, 40, 135, 255, 255)],
}


def _try_spatial(prompt: str, robot_order: tuple[str, str, str]) -> str | None:
    """Deterministic spatial resolver given the detected bowl layout.

    Parameters
    ----------
    robot_order:
        Tuple of 3 color strings ordered left→right from the ROBOT's perspective
        (i.e. already mirrored from the image order).
    Returns None for ambiguous prompts or unrecognised patterns.
    """
    lower = _normalize_spatial_landmarks(prompt)
    robot = robot_order

    # Eval2 defines spatial language from the robot perspective.  Keep that
    # default even when the TA omits "from the robot perspective".
    has_robot_ctx = True

    number_words = {"one": 0, "two": 1, "three": 2}
    ordinal_words = {
        "first": 1,
        "1st": 1,
        "one": 1,
        "second": 2,
        "2nd": 2,
        "two": 2,
        "third": 3,
        "3rd": 3,
        "three": 3,
    }

    def _neighbor_of(color: str, side: str) -> str | None:
        idx = robot.index(color)
        step = 1 if side == "right" else -1
        target_idx = idx + step
        return robot[target_idx] if 0 <= target_idx < len(robot) else None

    def _position_idx(label: str) -> int | None:
        label = label.lower()
        if re.search(r"\bleftmost\b|\bfar\s+left\b|\bleft[-\s]?hand\b|\b(?:first|1st)\b", label):
            return 0
        if re.search(r"\bmiddle\b|\bcent(?:er|re)\b|\bcentral\b|\b(?:second|2nd)\b", label):
            return 1
        if re.search(r"\brightmost\b|\bfar\s+right\b|\bright[-\s]?hand\b|\b(?:third|3rd|last)\b", label):
            return 2
        return None

    def _neighbor_of_position(label: str, side: str) -> str | None:
        idx = _position_idx(label)
        if idx is None:
            return None
        target_idx = idx + (1 if side == "right" else -1)
        return robot[target_idx] if 0 <= target_idx < len(robot) else None

    def _distance(label: str | None) -> int:
        if not label:
            return 1
        label = label.lower()
        if label.isdigit():
            return int(label)
        return ordinal_words[label]

    def _clean_landmark_phrase(phrase: str) -> str:
        phrase = re.sub(r"\bfrom\s+(?:the\s+)?(?:robot|arm)(?:'s)?\s+(?:perspective|view|side)\b.*", "", phrase)
        phrase = re.sub(r"\b(?:colored?|coloured?)\b", "", phrase)
        phrase = re.sub(r"\b(?:bowl|bowel|one|cup|container|dish|target)\b", "", phrase)
        phrase = re.sub(r"[^a-z0-9\s-]", " ", phrase)
        return re.sub(r"\s+", " ", phrase).strip()

    def _resolve_landmark_phrase(phrase: str) -> str | None:
        landmark = _clean_landmark_phrase(phrase)
        if not landmark:
            return None

        idx = _position_idx(landmark)
        if idx is not None:
            return robot[idx]

        # Handles landmarks like "not blue not avocado" by first normalizing
        # avocado/sky/tomato/etc. to their bowl colors, then applying negation.
        color = _try_negation(landmark)
        if color:
            return color

        for resolver in (_try_direct, _try_analogy, _try_single_target_color):
            color = resolver(landmark)
            if color:
                return color

        mentions = _find_color_mentions(landmark)
        unique = list(dict.fromkeys(mentions))
        return unique[0] if len(unique) == 1 else None

    def _relative_from_landmark(side: str, landmark_phrase: str, distance: int = 1) -> str | None:
        landmark = _resolve_landmark_phrase(landmark_phrase)
        if landmark is None:
            return None
        idx = robot.index(landmark)
        target_idx = idx + (distance if side == "right" else -distance)
        return robot[target_idx] if 0 <= target_idx < len(robot) else None

    # ── Relative with an optional ordinal distance and a derived landmark.
    # Covers:
    #   "2nd left of green"                  -> two bowls left of green
    #   "second bowl to the right of blue"   -> two bowls right of blue
    #   "right of not blue not avocado"      -> right of the only non-blue/non-green bowl
    ordinal = r"(?P<n>first|1st|one|second|2nd|two|third|3rd|three|\d+)"
    landmark = (
        r"(?P<landmark>.+?)(?=\s*(?:\bfrom\s+(?:the\s+)?(?:robot|arm)(?:'s)?\s+"
        r"(?:perspective|view|side)\b|[,.!?;()]|$))"
    )
    m = re.search(
        rf"\b(?:on|to|at|in)?\s*(?:the\s+)?"
        rf"(?:(?:{ordinal})\s+(?:bowl|one|spot|position|place)?s?\s*(?:to\s+the\s+)?)?"
        rf"(?P<side>right|left)(?:\s+side)?\s+of\s+(?:the\s+)?{landmark}",
        lower,
    )
    if m and has_robot_ctx:
        dist = _distance(m.group("n"))
        return _relative_from_landmark(m.group("side"), m.group("landmark"), dist)

    # ── Absolute ordinal from left ────────────────────────────────────────────
    m = re.search(
        r"\b(?:number\s+)?(one|two|three)\s+(?:bowl|one)?\s*from\s+"
        r"(?:the\s+)?(?:(?:robot|arm)'?s?\s+)?left\b",
        lower,
    )
    if m and has_robot_ctx:
        n = number_words[m.group(1)]
        return robot[n] if 0 <= n < len(robot) else None

    m = re.search(
        r"\b(?:bowl|one)\s+number\s+(one|two|three)\s*from\s+"
        r"(?:the\s+)?(?:(?:robot|arm)'?s?\s+)?left\b",
        lower,
    )
    if m and has_robot_ctx:
        n = number_words[m.group(1)]
        return robot[n] if 0 <= n < len(robot) else None

    m = re.search(
        r"\b(\d+)(?:st|nd|rd|th)?\s+(?:bowl|one)?\s*from\s+"
        r"(?:the\s+)?(?:(?:robot|arm)'?s?\s+)?left\b",
        lower,
    )
    if m and has_robot_ctx:
        n = int(m.group(1)) - 1
        return robot[n] if 0 <= n < len(robot) else None

    m = re.search(
        r"\b(first|second|third)\s+(?:bowl|one)?\s*from\s+"
        r"(?:the\s+)?(?:(?:robot|arm)'?s?\s+)?left\b",
        lower,
    )
    if m and has_robot_ctx:
        n = {"first": 0, "second": 1, "third": 2}[m.group(1)]
        return robot[n] if 0 <= n < len(robot) else None

    m = re.search(
        r"\b(?:second|2nd|two|number\s+two)\s+from\s+"
        r"(?:the\s+)?(?:(?:robot|arm)'?s?\s+)?(?:perspective|view|side)\b",
        lower,
    )
    if m and has_robot_ctx:
        return robot[1]

    m = re.search(r"\b(first|second|third|1st|2nd|3rd)\s+(?:bowl|one)\b(?!\s+from)", lower)
    if m and has_robot_ctx:
        n = {"first": 0, "second": 1, "third": 2, "1st": 0, "2nd": 1, "3rd": 2}[m.group(1)]
        return robot[n] if 0 <= n < len(robot) else None

    # ── Absolute ordinal from right ───────────────────────────────────────────
    m = re.search(
        r"\b(?:number\s+)?(one|two|three)\s+(?:bowl|one)?\s*from\s+"
        r"(?:the\s+)?(?:(?:robot|arm)'?s?\s+)?right\b",
        lower,
    )
    if m and has_robot_ctx:
        n = number_words[m.group(1)]
        return robot[-(n + 1)] if 0 <= n < len(robot) else None

    m = re.search(
        r"\b(?:bowl|one)\s+number\s+(one|two|three)\s*from\s+"
        r"(?:the\s+)?(?:(?:robot|arm)'?s?\s+)?right\b",
        lower,
    )
    if m and has_robot_ctx:
        n = number_words[m.group(1)]
        return robot[-(n + 1)] if 0 <= n < len(robot) else None

    m = re.search(
        r"\b(\d+)(?:st|nd|rd|th)?\s+(?:bowl|one)?\s*from\s+"
        r"(?:the\s+)?(?:(?:robot|arm)'?s?\s+)?right\b",
        lower,
    )
    if m and has_robot_ctx:
        n = int(m.group(1)) - 1
        return robot[-(n + 1)] if 0 <= n < len(robot) else None

    m = re.search(
        r"\b(first|second|third)\s+(?:bowl|one)?\s*from\s+"
        r"(?:the\s+)?(?:(?:robot|arm)'?s?\s+)?right\b",
        lower,
    )
    if m and has_robot_ctx:
        n = {"first": 0, "second": 1, "third": 2}[m.group(1)]
        return robot[-(n + 1)] if 0 <= n < len(robot) else None

    # ── Position landmarks: left/right of the leftmost/middle/rightmost bowl.
    pos_ref = (
        r"(leftmost|rightmost|middle|center|centre|central|"
        r"far\s+left|far\s+right|left[-\s]?hand|right[-\s]?hand|"
        r"first|1st|second|2nd|third|3rd|last)"
    )
    m = re.search(
        rf"\b(?:on|to|at)?\s*(?:the\s+)?(?:immediate(?:ly)?\s+|directly\s+|just\s+)?"
        rf"(right|left)(?:\s+side)?\s+of\s+(?:the\s+)?{pos_ref}(?:\s+(?:bowl|one))?\b",
        lower,
    )
    if m and has_robot_ctx:
        return _neighbor_of_position(m.group(2), m.group(1))

    m = re.search(
        rf"\b(?:next\s+to|adjacent\s+to|beside|near)\s+(?:the\s+)?{pos_ref}"
        rf"(?:\s+(?:bowl|one))?\s+(?:on|to|at)\s+(?:the\s+)?(right|left)(?:\s+side)?\b",
        lower,
    )
    if m and has_robot_ctx:
        return _neighbor_of_position(m.group(1), m.group(2))

    # ── "next to the leftmost/rightmost bowl" has a single unambiguous neighbor.
    m = re.search(
        r"\b(?:next\s+to|adjacent\s+to|beside|near|neighbou?r\s+of)\s+"
        r"(?:the\s+)?leftmost(?:\s+(?:bowl|one))?\b",
        lower,
    )
    if m:
        return robot[1]
    m = re.search(
        r"\b(?:next\s+to|adjacent\s+to|beside|near|neighbou?r\s+of)\s+"
        r"(?:the\s+)?rightmost(?:\s+(?:bowl|one))?\b",
        lower,
    )
    if m:
        return robot[1]

    # ── relative: on the right of [color] from robot ──────────────────────────
    m = re.search(
        r"\b(?:(?:on|to|at)\s+(?:the\s+)?|)(?:immediate(?:ly)?\s+|directly\s+|just\s+)?"
        r"right(?:\s+side)?\s+of\s+(?:the\s+)?(red|green|blue)(?:\s+bowl)?\b",
        lower,
    )
    if m and has_robot_ctx:
        idx = robot.index(m.group(1))
        return robot[idx + 1] if idx + 1 < len(robot) else None

    # ── relative: on the left of [color] from robot ───────────────────────────
    m = re.search(
        r"\b(?:(?:on|to|at)\s+(?:the\s+)?|)(?:immediate(?:ly)?\s+|directly\s+|just\s+)?"
        r"left(?:\s+side)?\s+of\s+(?:the\s+)?(red|green|blue)(?:\s+bowl)?\b",
        lower,
    )
    if m and has_robot_ctx:
        idx = robot.index(m.group(1))
        return robot[idx - 1] if idx - 1 >= 0 else None

    # ── relative: next/beside/neighbor [color] on a named side ────────────────
    m = re.search(
        r"\b(?:next\s+to|adjacent\s+to|beside|near)\s+(?:the\s+)?"
        r"(red|green|blue)(?:\s+bowl)?"
        r".*?\b(?:on|to|at)\s+(?:the\s+)?(right|left)(?:\s+side)?\b",
        lower,
    )
    if m and has_robot_ctx:
        return _neighbor_of(m.group(1), m.group(2))

    m = re.search(
        r"\b(?:the\s+)?(right|left)(?:[-\s]?hand)?\s+"
        r"(?:neighbou?r|adjacent\s+bowl|bowl|one)\s+of\s+(?:the\s+)?"
        r"(red|green|blue)(?:\s+bowl)?\b",
        lower,
    )
    if m and has_robot_ctx:
        return _neighbor_of(m.group(2), m.group(1))

    m = re.search(
        r"\bneighbou?r\s+(?:on|to|at)\s+(?:the\s+)?(right|left)(?:\s+side)?"
        r"\s+of\s+(?:the\s+)?(red|green|blue)(?:\s+bowl)?\b",
        lower,
    )
    if m and has_robot_ctx:
        return _neighbor_of(m.group(2), m.group(1))

    for _pat, _color in (
        (
            r"\bleftmost\b|\bfar\s+left\b|\bleft[-\s]?hand\s+(?:bowl|one|side)\b|"
            r"\bleft\s+(?:bowl|one)\b|\bleft\s+side\s+(?:bowl|one)\b|"
            r"\b(?:bowl|one)\s+on\s+the\s+left(?:\s+side)?\b(?!\s+of)|"
            r"\bon\s+the\s+left\s+side\b(?!\s+of)",
            robot[0],
        ),
        (
            r"\brightmost\b|\bfar\s+right\b|\bright[-\s]?hand\s+(?:bowl|one|side)\b|"
            r"\bright\s+(?:bowl|one)\b|\bright\s+side\s+(?:bowl|one)\b|"
            r"\b(?:bowl|one)\s+on\s+the\s+right(?:\s+side)?\b(?!\s+of)|"
            r"\bon\s+the\s+right\s+side\b(?!\s+of)",
            robot[-1],
        ),
    ):
        m = re.search(_pat, lower)
        if m and has_robot_ctx:
            prefix = lower[max(0, m.start() - 30) : m.start()]
            if not _NEG_MARKERS_RE.search(prefix):
                return _color

    # ── middle / center (no perspective flip needed) ──────────────────────────
    if re.search(r"\b(?:in\s+the\s+)?middle\b|\bcent(?:er|re)\b|\bcentral\b", lower):
        if _NEG_MARKERS_RE.search(lower):
            return None   # "not in the middle" → ambiguous (two valid answers)
        return robot[1]   # RED

    if re.search(r"\b(?:in\s+)?between\s+the\s+other\s+two\b", lower):
        if _NEG_MARKERS_RE.search(lower):
            return None
        return robot[1]

    # ── not at either end → middle ────────────────────────────────────────────
    if re.search(
        r"\b(?:not|neither|no)\b.*?\bends?\b|"
        r"\b(?:not|neither|no)\b.*?\beither\s+(?:end|side)\b|"
        r"\bbetween\s+(?:the\s+)?(?:left\s+and\s+right|two\s+outer)\b",
        lower,
    ):
        return robot[1]   # RED (center is never at an end)

    # ── relative: on the right of [color] from robot ──────────────────────────
    m = re.search(
        r"\b(?:on|to|at)\s+(?:the\s+)?(?:immediate(?:ly)?\s+|directly\s+|just\s+)?"
        r"right(?:\s+side)?\s+of\s+(?:the\s+)?(red|green|blue)(?:\s+bowl)?\b",
        lower,
    )
    if m and has_robot_ctx:
        idx = robot.index(m.group(1))
        return robot[idx + 1] if idx + 1 < len(robot) else None

    m = re.search(r"\b(?:the\s+)?(red|green|blue)(?:\s+bowl)?'?s?\s+right(?:\s+side)?\b", lower)
    if m and has_robot_ctx:
        idx = robot.index(m.group(1))
        return robot[idx + 1] if idx + 1 < len(robot) else None

    m = re.search(
        r"\b(?:next|adjacent)\s+to\s+(?:the\s+)?(red|green|blue)(?:\s+bowl)?"
        r".*?\b(?:on|to)\s+(?:the\s+)?right\b",
        lower,
    )
    if m and has_robot_ctx:
        idx = robot.index(m.group(1))
        return robot[idx + 1] if idx + 1 < len(robot) else None

    # ── relative: on the left of [color] from robot ───────────────────────────
    m = re.search(
        r"\b(?:on|to|at)\s+(?:the\s+)?(?:immediate(?:ly)?\s+|directly\s+|just\s+)?"
        r"left(?:\s+side)?\s+of\s+(?:the\s+)?(red|green|blue)(?:\s+bowl)?\b",
        lower,
    )
    if m and has_robot_ctx:
        idx = robot.index(m.group(1))
        return robot[idx - 1] if idx - 1 >= 0 else None

    m = re.search(r"\b(?:the\s+)?(red|green|blue)(?:\s+bowl)?'?s?\s+left(?:\s+side)?\b", lower)
    if m and has_robot_ctx:
        idx = robot.index(m.group(1))
        return robot[idx - 1] if idx - 1 >= 0 else None

    m = re.search(
        r"\b(?:next|adjacent)\s+to\s+(?:the\s+)?(red|green|blue)(?:\s+bowl)?"
        r".*?\b(?:on|to)\s+(?:the\s+)?left\b",
        lower,
    )
    if m and has_robot_ctx:
        idx = robot.index(m.group(1))
        return robot[idx - 1] if idx - 1 >= 0 else None

    # ── "next to [color]" when only one adjacent bowl exists ──────────────────
    m = re.search(r"\b(?:next|adjacent)\s+to\s+(?:the\s+)?(red|green|blue)(?:\s+bowl)?\b", lower)
    if m:
        suffix = lower[m.end() :]
        if not re.search(r"\b(?:left|right)\b", suffix):
            idx = robot.index(m.group(1))
            neighbors = []
            if idx - 1 >= 0:
                neighbors.append(robot[idx - 1])
            if idx + 1 < len(robot):
                neighbors.append(robot[idx + 1])
            return neighbors[0] if len(neighbors) == 1 else None

    # ── "immediately next to [color] on the side closer to robot's left" ──────
    m = re.search(r"\bnext\s+to\s+(?:the\s+)?(red|green|blue)\b.*?\bleft\b", lower)
    if m:
        idx = robot.index(m.group(1))
        return robot[idx - 1] if idx - 1 >= 0 else None

    # ── between [color1] and [color2] ─────────────────────────────────────────
    m = re.search(
        r"\b(?:between|in\s+between)\s+(?:the\s+)?(red|green|blue)(?:\s+bowl)?"
        r"\s+(?:and|&)\s+(?:the\s+)?(red|green|blue)(?:\s+bowl)?s?\b",
        lower,
    )
    if m:
        i1 = robot.index(m.group(1))
        i2 = robot.index(m.group(2))
        between = [robot[i] for i in range(min(i1, i2) + 1, max(i1, i2))]
        return between[0] if len(between) == 1 else None

    m = re.search(
        r"\bseparat(?:e|es|ing)\s+(?:the\s+)?(red|green|blue)(?:\s+bowl)?"
        r"\s+(?:and|from)\s+(?:the\s+)?(red|green|blue)(?:\s+bowl)?s?\b",
        lower,
    )
    if m:
        i1 = robot.index(m.group(1))
        i2 = robot.index(m.group(2))
        between = [robot[i] for i in range(min(i1, i2) + 1, max(i1, i2))]
        return between[0] if len(between) == 1 else None

    # ── adjacent to both [color1] and [color2] ────────────────────────────────
    m = re.search(
        r"\b(?:adjacent\s+to|next\s+to|touch(?:ing|es)?)\s+both\b.*?"
        r"(red|green|blue).*?(red|green|blue)",
        lower,
    )
    if m:
        c1, c2 = m.group(1), m.group(2)
        i1, i2 = robot.index(c1), robot.index(c2)
        for i, color in enumerate(robot):
            if color in (c1, c2):
                continue
            if abs(i - i1) == 1 and abs(i - i2) == 1:
                return color
        return None   # no bowl adjacent to both

    return None


def _try_negation_with_spatial(prompt: str, robot_order: tuple[str, str, str]) -> str | None:
    """Handle prompts that negate both colors AND positional references.

    Translates positional words (leftmost/rightmost/middle/Nth) into colors using
    the detected layout, then excludes them alongside any explicit color exclusions.
    Returns the sole remaining color, or None.
    """
    lower = _normalize_spatial_landmarks(prompt)
    excluded: set[str] = set()

    # Collect direct color exclusions from negation markers (same as _try_negation).
    for m in _NEG_MARKERS_RE.finditer(lower):
        window = lower[m.start() : min(len(lower), m.end() + 60)]
        for color in BOWL_COLORS:
            if re.search(rf"\b{color}\b", window):
                excluded.add(color)
        for concept, color in _CONCEPT_COLOR.items():
            if re.search(rf"\b{re.escape(concept)}\b", window):
                excluded.add(color)

    # Also translate positional words near negation markers into colors.
    pos_patterns = [
        (
            r"\brightmost\b|\bfar\s+right\b|\bright[-\s]?hand\b|"
            r"\bright\s+(?:bowl|one|side)\b|\b(?:bowl|one)\s+on\s+the\s+right\b|"
            r"\b(?:1st|first)\s+(?:bowl|one)?\s*from\s+(?:the\s+)?right\b|"
            r"\b(?:3rd|third)\s+(?:bowl|one)?\s*from\s+(?:the\s+)?left\b",
            robot_order[-1],
        ),
        (
            r"\bleftmost\b|\bfar\s+left\b|\bleft[-\s]?hand\b|"
            r"\bleft\s+(?:bowl|one|side)\b|\b(?:bowl|one)\s+on\s+the\s+left\b|"
            r"\b(?:1st|first)\s+(?:bowl|one)?\s*from\s+(?:the\s+)?left\b|"
            r"\b(?:3rd|third)\s+(?:bowl|one)?\s*from\s+(?:the\s+)?right\b",
            robot_order[0],
        ),
        (
            r"\bmiddle\b|\bcent(?:er|re)\b|\bcentral\b|"
            r"\b(?:2nd|second)\s+(?:bowl|one)?\b",
            robot_order[1],
        ),
    ]
    for pat, color in pos_patterns:
        # Check if a negation marker appears within 60 chars before the positional word.
        for pm in re.finditer(pat, lower):
            prefix = lower[max(0, pm.start() - 60) : pm.start()]
            if _NEG_MARKERS_RE.search(prefix):
                excluded.add(color)

    remaining = [c for c in BOWL_COLORS if c not in excluded]
    return remaining[0] if len(excluded) >= 2 and len(remaining) == 1 else None


def _try_direct(prompt: str) -> str | None:
    """Return color if the prompt explicitly places the banana 'in/into the [color] bowl'.

    Matches 'into the green bowl', 'in the green colored bowl', 'place … in the red bowl', etc.
    Does NOT match references like 'left of the red bowl' (where red is a landmark, not target).
    """
    if _POSITION_WORD_RE.search(prompt) or _NEG_MARKERS_RE.search(prompt) or _NON_COLOR_RE.search(prompt):
        return None

    # Only match when the target color bowl follows "in/into/to/inside".
    m = re.search(
        r"\b(?:in(?:to)?|inside|to|towards?|onto)\s+(?:the\s+)?"
        r"(red|green|blue)(?:[-\s]+(?:colou?red|colou?r))?\s+bowe?l\b",
        prompt,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).lower()

    # Also accept looser but common forms like "the red one", "red cup",
    # "red container", and "bowl red" when the prompt is not spatial.
    if not _SPATIAL_WORD_RE.search(prompt):
        m = re.search(
            r"\b(red|green|blue)(?:[-\s]+(?:colou?red|colou?r))?\s+"
            r"(?:bowe?l|one|cup|container|dish|target)\b",
            prompt,
            re.IGNORECASE,
        )
        if m:
            return m.group(1).lower()

        m = re.search(
            r"\b(?:bowe?l|one|cup|container|dish|target)\s+"
            r"(?:is\s+)?(?:the\s+)?(red|green|blue)\b",
            prompt,
            re.IGNORECASE,
        )
        if m:
            return m.group(1).lower()

    return None


def _try_single_target_color(prompt: str) -> str | None:
    """Infer the target when exactly one non-spatial color is mentioned."""
    if _SPATIAL_WORD_RE.search(prompt) or _NEG_MARKERS_RE.search(prompt) or _NON_COLOR_RE.search(prompt):
        return None

    mentions = _find_color_mentions(prompt)
    unique = list(dict.fromkeys(mentions))
    if len(unique) == 1:
        return unique[0]

    m = re.search(
        r"\b(?:r|g|b)\b",
        prompt,
        re.IGNORECASE,
    )
    if m:
        return {"r": "red", "g": "green", "b": "blue"}[m.group(0).lower()]
    return None


def _try_negation(prompt: str) -> str | None:
    """Return the only non-excluded bowl color, or None if not determinable.

    Handles:
      "not red and not blue"                         → green  (direct colors)
      "neither red nor blue"                         → green  ('nor' as negative marker)
      "non-red and non-blue"                         → green  (non- prefix)
      "not the red one and not the blue one"         → green  (article before color)
      "excluding red and blue"                       → green  (all colors after marker)
      "neither the color of fire nor of water"       → green  (concept: fire=red, water=blue)
      "not associated with danger or the sea"        → green  (concept: danger=red, sea=blue)
    """
    lower = _normalize_spatial_landmarks(prompt)
    if _POSITION_WORD_RE.search(lower):
        return None

    excluded: set[str] = set()

    # Strategy: for every negative-marker position, collect all bowl colors
    # and concept colors that appear within the next 60 characters.
    for m in _NEG_MARKERS_RE.finditer(lower):
        window = lower[m.start() : min(len(lower), m.end() + 60)]
        # Direct bowl colors in the window.
        for color in BOWL_COLORS:
            if re.search(rf"\b{color}\b", window):
                excluded.add(color)
        # Concept words in the window → map to colors.
        for concept, color in _CONCEPT_COLOR.items():
            if re.search(rf"\b{re.escape(concept)}\b", window):
                excluded.add(color)

    # Non-prefixed forms: "non-red", "non-blue", "non red", "non blue".
    for color in BOWL_COLORS:
        if re.search(rf"\bnon[-\s]{color}\b", lower):
            excluded.add(color)

    remaining = [c for c in BOWL_COLORS if c not in excluded]
    if len(excluded) >= 2 and len(remaining) == 1:
        return remaining[0]
    return None


def _try_analogy(prompt: str) -> str | None:
    """Map color-analogy keywords to a bowl color, or return None.

    Only fires when the keyword appears in a positive (non-negated) context.
    Also handles traffic-light go-signal and color-wheel reasoning.
    """
    lower = prompt.lower()

    # If the prompt says "left/right/next/between ... avocado/sky bowl", the
    # analogy word is a landmark, not the target. Let spatial parsing handle it
    # after the camera/layout step.
    if _POSITION_WORD_RE.search(lower):
        return None

    # Traffic light go-signal: "traffic light when you can go" → green.
    if _TRAFFIC_GO.search(lower):
        return "green"

    # Color wheel: yellow + blue → green; opposite of red ≈ green.
    if _BETWEEN_YELLOW_BLUE.search(lower):
        return "green"
    if _OPPOSITE_RED.search(lower):
        return "green"

    # Keyword lookup — skip if the keyword is within 50 chars of a negative marker.
    for keyword, color in _ANALOGY_MAP.items():
        if keyword not in lower:
            continue
        idx = lower.find(keyword)
        # Widen to 50 chars to catch "not associated with danger or the sea"
        prefix = lower[max(0, idx - 50) : idx]
        if _NEG_MARKERS_RE.search(prefix):
            continue
        if color == "lake":   # resolve placeholder
            color = "blue"
        return color

    return None


def normalize_prompt_text(
    raw_prompt: str,
    robot_order: tuple[str, str, str] | None = None,
    *,
    fallback_passthrough: bool = False,
) -> str:
    """Normalize an Eval1/Eval2 text prompt without loading any extra model.

    Eval1 usually names the target color directly but may vary wording:
    "pick up" vs "put", "in" vs "into", optional articles, etc.  Eval2 may
    use negation, color analogies, or spatial descriptions.  Spatial prompts
    need ``robot_order`` when the target depends on the bowl layout.
    """
    for resolver in (_try_direct, _try_negation, _try_analogy, _try_single_target_color):
        color = resolver(raw_prompt)
        if color:
            return canonical_prompt(color)

    if robot_order is not None:
        for resolver in (_try_spatial, _try_negation_with_spatial):
            color = resolver(raw_prompt, robot_order)
            if color:
                return canonical_prompt(color)

    if fallback_passthrough:
        return raw_prompt
    raise ValueError(
        "Cannot normalize prompt without bowl layout: "
        f"{raw_prompt!r}. Use a color prompt, a two-color negation, or a supported spatial prompt."
    )


# ── HSV layout detection ──────────────────────────────────────────────────────

def _to_pil(image: Union[np.ndarray, "Image.Image", "torch.Tensor"]) -> Image.Image:
    """Convert numpy HWC uint8, float CHW tensor, or PIL Image to RGB PIL."""
    if Image is None:
        raise ImportError("Pillow is required for camera/image-based prompt normalization")
    if Image is not None and isinstance(image, Image.Image):
        return image.convert("RGB")
    if np is not None and isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = (arr.clip(0.0, 1.0) * 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    if torch is not None and isinstance(image, torch.Tensor):
        t = image.detach().cpu().float()
        if t.dim() == 4:
            t = t[0]
        if t.dim() == 3 and t.shape[0] in (1, 3, 4):  # CHW → HWC
            t = t.permute(1, 2, 0)
        if t.max() <= 1.0:
            t = t * 255.0
        return Image.fromarray(t.byte().numpy()).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def _detect_color_centroids(image: Union[np.ndarray, "Image.Image", "torch.Tensor"]) -> dict[str, tuple[float, float]]:
    """Return color -> (x_centroid_px, area_px) for visible bowls in one frame."""
    if np is None:
        raise ImportError("numpy is required for camera/image-based prompt normalization")
    import cv2

    pil_image = _to_pil(image)
    arr = np.array(pil_image)                          # RGB uint8 HWC
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    kernel = np.ones((5, 5), dtype=np.uint8)

    detections: dict[str, tuple[float, float]] = {}
    for color, ranges in _HSV_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo_h, lo_s, lo_v, hi_h, hi_s, hi_v in ranges:
            mask |= cv2.inRange(
                hsv,
                np.array([lo_h, lo_s, lo_v]),
                np.array([hi_h, hi_s, hi_v]),
            )

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if num_labels <= 1:
            continue

        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = float(stats[largest, cv2.CC_STAT_AREA])
        if area >= 200:
            detections[color] = (float(centroids[largest][0]), area)

    return detections


def _aggregate_color_centroids(
    frames: list[Union[np.ndarray, "Image.Image", "torch.Tensor"]],
) -> dict[str, tuple[float, float]]:
    """Aggregate color detections across frames using median x and max area."""
    samples: dict[str, list[tuple[float, float]]] = {color: [] for color in BOWL_COLORS}
    for frame in frames:
        for color, detection in _detect_color_centroids(frame).items():
            samples[color].append(detection)

    centroids: dict[str, tuple[float, float]] = {}
    for color, values in samples.items():
        if not values:
            continue
        xs = [x for x, _ in values]
        areas = [area for _, area in values]
        centroids[color] = (float(np.median(xs)), float(max(areas)))
    return centroids


def _complete_robot_order(
    centroids: dict[str, tuple[float, float]],
    fallback_robot_order: tuple[str, str, str] | None,
) -> tuple[str, str, str] | None:
    """Build robot L-R order from detected camera x positions plus fallback."""
    if not centroids:
        return fallback_robot_order

    detected_image_order = tuple(
        color for color, _ in sorted(centroids.items(), key=lambda item: item[1][0])
    )
    detected_robot_order = tuple(reversed(detected_image_order))
    if len(detected_robot_order) == 3:
        return detected_robot_order
    if fallback_robot_order is None:
        return None

    fallback_robot_order = validate_robot_order(fallback_robot_order)
    candidates: list[tuple[int, tuple[str, str, str]]] = []
    for candidate in permutations(BOWL_COLORS):
        detected_subsequence = tuple(color for color in candidate if color in detected_robot_order)
        if detected_subsequence != detected_robot_order:
            continue
        score = sum(abs(candidate.index(color) - fallback_robot_order.index(color)) for color in BOWL_COLORS)
        candidates.append((score, candidate))

    if not candidates:
        return fallback_robot_order
    return min(candidates, key=lambda item: item[0])[1]


class PromptNormalizer:
    """Translates an Eval1/Eval2 instruction to a canonical color prompt.

    Text is handled by deterministic rules. Task2 spatial prompts use HSV color
    segmentation to infer the red/green/blue bowl order.
    """

    def __init__(
        self,
        *,
        fallback_passthrough: bool = True,
        fallback_robot_order: tuple[str, str, str] | None = DEFAULT_ROBOT_ORDER,
    ) -> None:
        self._fallback_passthrough = fallback_passthrough
        self._fallback_robot_order = (
            validate_robot_order(fallback_robot_order) if fallback_robot_order is not None else None
        )
        # Detected bowl layout: robot perspective L→R, set by detect_layout().
        self._robot_order: tuple[str, str, str] | None = None

    def detect_layout(
        self,
        image: Union[np.ndarray, "Image.Image", "torch.Tensor"],
    ) -> tuple[str, str, str] | None:
        """Detect bowl positions via HSV color segmentation.

        Call once at episode start (before the first normalize() call) to cache
        the layout.  All subsequent spatial prompts are then resolved instantly.

        Finds the centroid of red, green, and blue pixel clusters in the image,
        sorts them left→right, then mirrors to robot perspective.

        Returns the robot-perspective L→R order as a 3-tuple, e.g.
        ``("blue", "red", "green")``. If only one or two colors are found,
        the remaining order is completed from ``fallback_robot_order``.
        """
        return self._finalize_layout(_detect_color_centroids(image), source="single-frame")

    def detect_layout_from_frames(
        self,
        frames: list[Union[np.ndarray, "Image.Image", "torch.Tensor"]],
    ) -> tuple[str, str, str] | None:
        """Detect bowl layout from several frames for eval-day robustness."""
        if not frames:
            return self._finalize_layout({}, source="no-frames")
        return self._finalize_layout(_aggregate_color_centroids(frames), source=f"{len(frames)}-frame")

    def _finalize_layout(
        self,
        centroids: dict[str, tuple[float, float]],
        *,
        source: str,
    ) -> tuple[str, str, str] | None:
        robot_order = _complete_robot_order(centroids, self._fallback_robot_order)
        missing = set(BOWL_COLORS) - set(centroids)

        if robot_order is None:
            logging.warning(
                "[PromptNormalizer] %s layout: no usable bowl layout and no fallback order",
                source,
            )
            return None

        image_order = tuple(reversed(robot_order))
        if missing:
            logging.warning(
                "[PromptNormalizer] %s layout: missing %s, using best-effort robot L→R=%s",
                source,
                sorted(missing),
                robot_order,
            )
        else:
            logging.info(
                "[PromptNormalizer] %s layout: image L→R=%s  robot L→R=%s",
                source,
                image_order,
                robot_order,
            )

        if centroids:
            logging.info(
                "[PromptNormalizer] centroids=%s",
                {c: f"{x:.0f}px/{area:.0f}px" for c, (x, area) in sorted(centroids.items(), key=lambda kv: kv[1][0])},
            )
        self._robot_order = robot_order
        return robot_order

    # ── Core method ────────────────────────────────────────────────────────────

    def normalize(
        self,
        image: Union[np.ndarray, "Image.Image", "torch.Tensor"],
        raw_prompt: str,
    ) -> str:
        """Convert a complex task prompt to the canonical color-based form.

        Routing:
          1. Direct    ("red bowl", "target is blue")      → deterministic
          2. Negation  ("not red and not blue")            → deterministic
          3. Analogy   ("color of grass", "stop sign")     → deterministic
          4. Spatial   ("2nd from left", "right of red")   → HSV layout + rules
          3. Spatial   ("2nd from left", "right of red")   → HSV layout + rules

        Parameters
        ----------
        image:
            Current camera frame.  Accepts numpy HWC uint8 / float,
            CHW float tensor, or PIL Image.  Used only when spatial layout is needed.
        raw_prompt:
            The raw evaluation instruction.

        Returns
        -------
        str
            ``"Put the banana in the red colored bowl."`` (or green / blue).
            Falls back to a deterministic best-effort color when a strict
            match is unavailable.
        """
        # ── Stages 0-2: direct color, negation, and analogy ──────────────────
        try:
            canonical = normalize_prompt_text(raw_prompt)
        except ValueError:
            canonical = None
        if canonical is not None:
            logging.info("[PromptNormalizer] text     %r → %r", raw_prompt, canonical)
            return canonical

        # ── Stage 3: deterministic spatial lookup (image-detected bowl layout) ──
        # Auto-detect layout on first spatial prompt if not already known.
        if self._robot_order is None:
            self.detect_layout(image)
        if self._robot_order is not None:
            color = _try_spatial(raw_prompt, self._robot_order)
            if color:
                canonical = canonical_prompt(color)
                logging.info("[PromptNormalizer] spatial  %r → %r", raw_prompt, canonical)
                return canonical

            # ── Stage 3b: combined negation + spatial ─────────────────────────
            # e.g. "not red and not rightmost" — translate positional negations
            # ("not rightmost" = "not green") then re-run negation.
            color = _try_negation_with_spatial(raw_prompt, self._robot_order)
            if color:
                canonical = canonical_prompt(color)
                logging.info("[PromptNormalizer] neg+spatial %r → %r", raw_prompt, canonical)
                return canonical

        robot_order = self._robot_order or self._fallback_robot_order
        canonical, reason = normalize_prompt_best_effort(raw_prompt, robot_order=robot_order)
        logging.warning(
            "[PromptNormalizer] best-effort fallback (%s) %r → %r",
            reason,
            raw_prompt,
            canonical,
        )
        return canonical

    def __call__(
        self,
        image: Union[np.ndarray, "Image.Image", "torch.Tensor"],
        raw_prompt: str,
    ) -> str:
        return self.normalize(image, raw_prompt)


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    ap = argparse.ArgumentParser(
        description="Test PromptNormalizer with a static image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval1_prompt_normalizer.py frame.png \\
    "Put the banana into the 2nd bowl from the left from the robot perspective"

  python eval1_prompt_normalizer.py frame.png \\
    "Put the banana into the bowl that is not green and not blue"
""",
    )
    ap.add_argument("image", help="Path to a camera frame image (PNG/JPG).")
    ap.add_argument("prompt", help="Complex task prompt to normalize.")
    ap.add_argument(
        "--fallback-robot-order",
        default=",".join(DEFAULT_ROBOT_ORDER),
        help="Robot-perspective fallback bowl order, e.g. blue,red,green.",
    )
    args = ap.parse_args()

    if Image is None:
        raise ImportError("Pillow is required to load the test image")
    fallback_order = tuple(part.strip().lower() for part in args.fallback_robot_order.split(","))
    normalizer = PromptNormalizer(fallback_robot_order=validate_robot_order(fallback_order))
    img = Image.open(args.image)
    result = normalizer.normalize(img, args.prompt)
    print(result)
