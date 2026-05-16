"""
Normalizes complex/indirect task prompts to canonical color-based format for SmolVLA eval1.

Input:  any phrasing of which bowl to target, e.g.
    "Put the banana into the 2nd bowl from the left from the robot perspective"
    "Put the banana into the bowl on the right of the red bowl from the robot perspective"
    "Put the banana into the bowl that is not green and not blue"
Output: "Put the banana in the [color] colored bowl."

Architecture — hybrid two-stage normalizer
------------------------------------------
Prompts fall into two fundamentally different categories:

  1. Language-only (negation, color analogy) — no image needed, deterministic:
       "not red and not blue"          → green  (pure logic)
       "the color of grass"            → green  (knowledge lookup)
     Handled by ``_try_negation()`` and ``_try_analogy()`` before touching the VLM.

  2. Visual-spatial — requires the VLM + camera image:
       "2nd bowl from the left from the robot perspective"
       "bowl on the right of the red bowl"
     Handled by ``PromptNormalizer`` which calls SmolVLM's generate() method.

Routing language-only prompts through a VLM causes the model to fixate on the
visually dominant bowl in the image (typically red) rather than reasoning from text.
Deterministic pre-processing eliminates that error class entirely.

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
from typing import Union

import numpy as np
import torch
from PIL import Image


BOWL_COLORS = ("red", "green", "blue")

# ── Deterministic pre-processors ──────────────────────────────────────────────

# ── Negative marker set (used in several patterns below) ──────────────────────
_NEG_MARKERS_RE = re.compile(
    r"\b(?:not|neither|nor|no|without|excluding|except)\b",
    re.IGNORECASE,
)

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
    Returns None for ambiguous prompts or unrecognised patterns → VLM fallback.
    """
    lower = prompt.lower()
    robot = robot_order

    has_robot_ctx = bool(re.search(r"\brobot\b|\bperspective\b", lower))

    # ── Absolute ordinal from left ────────────────────────────────────────────
    m = re.search(
        r"\b(\d+)(?:st|nd|rd|th)\s+(?:bowl\s+)?from\s+(?:the\s+)?left\b", lower
    )
    if m and has_robot_ctx:
        n = int(m.group(1)) - 1
        return robot[n] if 0 <= n < len(robot) else None

    # ── Absolute ordinal from right ───────────────────────────────────────────
    m = re.search(
        r"\b(\d+)(?:st|nd|rd|th)\s+(?:bowl\s+)?from\s+(?:the\s+)?right\b", lower
    )
    if m and has_robot_ctx:
        n = int(m.group(1)) - 1
        return robot[-(n + 1)] if 0 <= n < len(robot) else None

    # ── leftmost / rightmost (skip if negated — handled by _try_negation_with_spatial)
    for _pat, _color in ((r"\bleftmost\b", robot[0]), (r"\brightmost\b", robot[-1])):
        m = re.search(_pat, lower)
        if m and has_robot_ctx:
            prefix = lower[max(0, m.start() - 30) : m.start()]
            if not _NEG_MARKERS_RE.search(prefix):
                return _color

    # ── middle / center (no perspective flip needed) ──────────────────────────
    if re.search(r"\b(?:in\s+the\s+)?middle\b|\bcenter\b", lower):
        if re.search(r"\bnot\b", lower):
            return None   # "not in the middle" → ambiguous (two valid answers)
        return robot[1]   # RED

    # ── not at either end → middle ────────────────────────────────────────────
    if re.search(r"\bnot\b.*?\bends?\b|\bnot\b.*?\beither\s+end\b", lower):
        return robot[1]   # RED (center is never at an end)

    # ── relative: on the right of [color] from robot ──────────────────────────
    m = re.search(
        r"\b(?:on\s+the|to\s+the)\s+right\s+of\s+(?:the\s+)?(red|green|blue)\b",
        lower,
    )
    if m and has_robot_ctx:
        idx = robot.index(m.group(1))
        return robot[idx + 1] if idx + 1 < len(robot) else None

    # ── relative: on the left of [color] from robot ───────────────────────────
    m = re.search(
        r"\b(?:on\s+the|to\s+the)\s+left\s+of\s+(?:the\s+)?(red|green|blue)\b",
        lower,
    )
    if m and has_robot_ctx:
        idx = robot.index(m.group(1))
        return robot[idx - 1] if idx - 1 >= 0 else None

    # ── "immediately next to [color] on the side closer to robot's left" ──────
    m = re.search(r"\bnext\s+to\s+(?:the\s+)?(red|green|blue)\b.*?\bleft\b", lower)
    if m:
        idx = robot.index(m.group(1))
        return robot[idx - 1] if idx - 1 >= 0 else None

    # ── between [color1] and [color2] ─────────────────────────────────────────
    m = re.search(
        r"\bbetween\s+(?:the\s+)?(red|green|blue)\s+(?:bowl\s+)?and\s+(?:the\s+)?(red|green|blue)\b",
        lower,
    )
    if m:
        i1 = robot.index(m.group(1))
        i2 = robot.index(m.group(2))
        between = [robot[i] for i in range(min(i1, i2) + 1, max(i1, i2))]
        return between[0] if len(between) == 1 else None

    # ── adjacent to both [color1] and [color2] ────────────────────────────────
    m = re.search(
        r"\badjacent\s+to\s+both\b.*?(red|green|blue).*?(red|green|blue)", lower
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
    lower = prompt.lower()
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
        (r"\brightmost\b",  robot_order[-1]),
        (r"\bleftmost\b",   robot_order[0]),
        (r"\bmiddle\b|\bcenter\b", robot_order[1]),
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
    # Only match when the color bowl follows 'in/into [the] [color]'
    m = re.search(
        r"\b(?:in(?:to)?)\s+(?:the\s+)?(red|green|blue)(?:\s+colored)?\s+bowl\b",
        prompt,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).lower()
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
    lower = prompt.lower()
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


# ── VLM prompt (spatial reasoning only) ───────────────────────────────────────

# This prompt is only reached when negation and analogy both fail,
# meaning the prompt requires visual-spatial reasoning from the camera frame.
_SPATIAL_PROMPT = """\
The image shows a yellow banana and three colored bowls: RED, GREEN, BLUE.
Goal: identify which BOWL (red, green, or blue) is at the described position.
The banana is not the target — only bowls matter.

Mirror rule: robot-LEFT = image-RIGHT, robot-RIGHT = image-LEFT.

EXAMPLE 1 — "rightmost bowl from the robot perspective":
  robot-rightmost → flip → image-LEFTMOST bowl → that bowl is green → ANSWER: green

EXAMPLE 2 — "bowl on the right of the red bowl from the robot perspective":
  robot-right-of-red → flip → image-left-of-red bowl → that bowl is blue → ANSWER: blue

Now identify the bowl for the instruction below.
Write ONE reasoning line (robot pos → image pos → bowl color), then ANSWER: red/green/blue."""


def _to_pil(image: Union[np.ndarray, "Image.Image", torch.Tensor]) -> Image.Image:
    """Convert numpy HWC uint8, float CHW tensor, or PIL Image to RGB PIL."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = (arr.clip(0.0, 1.0) * 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    if isinstance(image, torch.Tensor):
        t = image.detach().cpu().float()
        if t.dim() == 4:
            t = t[0]
        if t.dim() == 3 and t.shape[0] in (1, 3, 4):  # CHW → HWC
            t = t.permute(1, 2, 0)
        if t.max() <= 1.0:
            t = t * 255.0
        return Image.fromarray(t.byte().numpy()).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


class PromptNormalizer:
    """Translates a complex task description to a canonical color-based prompt.

    Parameters
    ----------
    vlm:
        ``SmolVLMForConditionalGeneration`` — either the full pretrained model or
        the ``policy.model.vlm_with_expert.vlm`` sub-module from a SmolVLA policy.
    processor:
        Matching ``SmolVLMProcessor``.
    device:
        Where to run VLM generate inference (should match where vlm lives).
    max_new_tokens:
        Generation budget.  150 tokens lets the model reason step-by-step
        before writing "ANSWER: <color>".  10 is far too short.
    fallback_passthrough:
        If True and the VLM output cannot be parsed, return the raw prompt unchanged
        (the robot will try to use it as-is).  If False, raise ValueError.
    """

    def __init__(
        self,
        vlm,
        processor,
        device: str | torch.device = "cuda",
        max_new_tokens: int = 100,
        fallback_passthrough: bool = True,
    ) -> None:
        self._vlm = vlm
        self._processor = processor
        self._device = torch.device(device)
        self._max_new_tokens = max_new_tokens
        self._fallback_passthrough = fallback_passthrough
        # Detected bowl layout: robot perspective L→R, set by detect_layout().
        self._robot_order: tuple[str, str, str] | None = None

    # ── Layout detection ───────────────────────────────────────────────────────

    def detect_layout(
        self,
        image: Union[np.ndarray, "Image.Image", torch.Tensor],
    ) -> tuple[str, str, str] | None:
        """Detect bowl positions via HSV color segmentation — no VLM inference needed.

        Call once at episode start (before the first normalize() call) to cache
        the layout.  All subsequent spatial prompts are then resolved instantly.

        Finds the centroid of red, green, and blue pixel clusters in the image,
        sorts them left→right, then mirrors to robot perspective.

        Returns the robot-perspective L→R order as a 3-tuple, e.g.
        ``("blue", "red", "green")``, or ``None`` if any color is not found
        (spatial prompts fall back to the VLM in that case).
        """
        import cv2

        pil_image = _to_pil(image)
        arr = np.array(pil_image)                          # RGB uint8 HWC
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

        centroids: dict[str, float] = {}
        for color, ranges in _HSV_RANGES.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo_h, lo_s, lo_v, hi_h, hi_s, hi_v in ranges:
                mask |= cv2.inRange(
                    hsv,
                    np.array([lo_h, lo_s, lo_v]),
                    np.array([hi_h, hi_s, hi_v]),
                )
            moments = cv2.moments(mask)
            if moments["m00"] > 200:                       # at least 200 px
                centroids[color] = moments["m10"] / moments["m00"]

        if len(centroids) < 3:
            missing = set(BOWL_COLORS) - set(centroids)
            logging.warning(
                "[PromptNormalizer] detect_layout: could not find bowl(s) %s "
                "— spatial prompts will use VLM fallback",
                missing,
            )
            return None

        # Sort colors by their x-centroid → image left-to-right order
        image_order = tuple(
            color for color, _ in sorted(centroids.items(), key=lambda kv: kv[1])
        )
        # Camera faces the robot → image is left-right mirrored vs robot view
        robot_order = tuple(reversed(image_order))
        self._robot_order = robot_order
        logging.info(
            "[PromptNormalizer] layout: image L→R=%s  robot L→R=%s  centroids=%s",
            image_order,
            robot_order,
            {c: f"{x:.0f}px" for c, x in sorted(centroids.items(), key=lambda kv: kv[1])},
        )
        return robot_order

    # ── Factory methods ────────────────────────────────────────────────────────

    @classmethod
    def from_policy(
        cls,
        policy,
        device: str | torch.device | None = None,
        **kwargs,
    ) -> "PromptNormalizer":
        """Construct from an already-loaded SmolVLAPolicy — zero extra weights.

        The SmolVLA policy contains a full ``SmolVLMForConditionalGeneration``
        (``policy.model.vlm_with_expert.vlm``) that supports ``.generate()``.
        We reuse it here for one-shot color identification before the action loop.

        Usage::

            normalizer = PromptNormalizer.from_policy(policy, device="cuda")
            # Get first camera frame before the action loop
            task = normalizer.normalize(camera_frame, raw_prompt)
            # Then run SmolVLA with the canonical task string
        """
        vwe = policy.model.vlm_with_expert
        dev = device
        if dev is None:
            try:
                dev = next(vwe.vlm.parameters()).device
            except StopIteration:
                dev = "cpu"
        return cls(vlm=vwe.vlm, processor=vwe.processor, device=dev, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        device: str | torch.device = "cuda",
        **kwargs,
    ) -> "PromptNormalizer":
        """Load a standalone SmolVLM (no SmolVLA policy required).

        Useful for testing the normalizer independently or when you want a
        full-capacity VLM instead of the potentially layer-truncated SmolVLA backbone.
        The default model matches the VLM backbone used inside SmolVLA base.
        """
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_id)
        vlm = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(device)
        vlm.eval()
        return cls(vlm=vlm, processor=processor, device=device, **kwargs)

    # ── Core method ────────────────────────────────────────────────────────────

    def normalize(
        self,
        image: Union[np.ndarray, "Image.Image", torch.Tensor],
        raw_prompt: str,
    ) -> str:
        """Convert a complex task prompt to the canonical color-based form.

        Routing:
          1. Negation  ("not red and not blue")            → deterministic, no VLM
          2. Analogy   ("color of grass", "stop sign")     → deterministic, no VLM
          3. Spatial   ("2nd from left", "right of red")   → VLM + camera image

        Parameters
        ----------
        image:
            Current camera frame.  Accepts numpy HWC uint8 / float,
            CHW float tensor, or PIL Image.  Only decoded when VLM is invoked.
        raw_prompt:
            The raw evaluation instruction.

        Returns
        -------
        str
            ``"Put the banana in the red colored bowl."`` (or green / blue).
            Falls back to ``raw_prompt`` if VLM output cannot be parsed and
            ``fallback_passthrough=True``.
        """
        # ── Stage 0: direct color mention ("into the green bowl") ─────────────
        color = _try_direct(raw_prompt)
        if color:
            canonical = f"Put the banana in the {color} colored bowl."
            logging.info("[PromptNormalizer] direct   %r → %r", raw_prompt, canonical)
            return canonical

        # ── Stage 1: deterministic negation ───────────────────────────────────
        color = _try_negation(raw_prompt)
        if color:
            canonical = f"Put the banana in the {color} colored bowl."
            logging.info("[PromptNormalizer] negation %r → %r", raw_prompt, canonical)
            return canonical

        # ── Stage 2: deterministic color-analogy lookup ────────────────────────
        color = _try_analogy(raw_prompt)
        if color:
            canonical = f"Put the banana in the {color} colored bowl."
            logging.info("[PromptNormalizer] analogy  %r → %r", raw_prompt, canonical)
            return canonical

        # ── Stage 3: deterministic spatial lookup (image-detected bowl layout) ──
        # Auto-detect layout on first spatial prompt if not already known.
        if self._robot_order is None:
            self.detect_layout(image)
        if self._robot_order is not None:
            color = _try_spatial(raw_prompt, self._robot_order)
            if color:
                canonical = f"Put the banana in the {color} colored bowl."
                logging.info("[PromptNormalizer] spatial  %r → %r", raw_prompt, canonical)
                return canonical

            # ── Stage 3b: combined negation + spatial ─────────────────────────
            # e.g. "not red and not rightmost" — translate positional negations
            # ("not rightmost" = "not green") then re-run negation.
            color = _try_negation_with_spatial(raw_prompt, self._robot_order)
            if color:
                canonical = f"Put the banana in the {color} colored bowl."
                logging.info("[PromptNormalizer] neg+spatial %r → %r", raw_prompt, canonical)
                return canonical

        # ── Stage 4: VLM fallback (novel/ambiguous spatial prompts) ───────────
        logging.info("[PromptNormalizer] VLM fallback for %r", raw_prompt)
        return self._vlm_normalize(image, raw_prompt)

    def _vlm_normalize(
        self,
        image: Union[np.ndarray, "Image.Image", torch.Tensor],
        raw_prompt: str,
    ) -> str:
        """Inner method: call the VLM for spatial prompts that need image reasoning."""
        pil_image = _to_pil(image)

        user_text = (
            f"{_SPATIAL_PROMPT}\n\n"
            f"Instruction: {raw_prompt}"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text],
            images=[pil_image],
            return_tensors="pt",
        ).to(self._device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            gen_ids = self._vlm.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                repetition_penalty=1.3,   # prevents hallucination loops
            )

        new_tokens = gen_ids[:, input_len:]
        raw_output = self._processor.batch_decode(
            new_tokens, skip_special_tokens=True
        )[0].strip().lower()

        logging.info("[PromptNormalizer] vlm_raw=%r  prompt=%r", raw_output, raw_prompt)

        color = _extract_color_from_reasoning(raw_output)
        if color is None:
            msg = (
                f"[PromptNormalizer] Cannot parse color from VLM output {raw_output!r} "
                f"(prompt={raw_prompt!r})"
            )
            if self._fallback_passthrough:
                logging.warning("%s — passing raw prompt through", msg)
                return raw_prompt
            raise ValueError(msg)

        canonical = f"Put the banana in the {color} colored bowl."
        logging.info("[PromptNormalizer] spatial  %r → %r", raw_prompt, canonical)
        return canonical

    def __call__(
        self,
        image: Union[np.ndarray, "Image.Image", torch.Tensor],
        raw_prompt: str,
    ) -> str:
        return self.normalize(image, raw_prompt)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_color_from_reasoning(text: str) -> str | None:
    """Parse the target color out of a chain-of-thought VLM response.

    Priority order:
      1. Explicit ``ANSWER: <color>`` tag (most reliable).
      2. Last color word in the text — for chain-of-thought, the conclusion
         comes last, so "not red, not blue … green" → "green".
      3. Returns None if no color found at all.
    """
    # 1. Look for explicit ANSWER: tag
    m = re.search(r"\banswer\s*:\s*(red|green|blue)\b", text, re.IGNORECASE)
    if m:
        return m.group(1).lower()

    # 2. Last color word in the full output
    all_matches = re.findall(r"\b(red|green|blue)\b", text, re.IGNORECASE)
    if all_matches:
        return all_matches[-1].lower()

    return None


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
        "--model-id",
        default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        help="SmolVLM Hub model ID for standalone mode.",
    )
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = ap.parse_args()

    normalizer = PromptNormalizer.from_pretrained(args.model_id, device=args.device)
    img = Image.open(args.image)
    result = normalizer.normalize(img, args.prompt)
    print(result)
