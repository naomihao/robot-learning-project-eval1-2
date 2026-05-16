#!/usr/bin/env python3
"""Show the effect of the two training augmentations on Eval 1 frames.

For each slug (red_bowl, blue_bowl, green_bowl) renders a row of panels:

  [original] [bg replaced A] [bg replaced B] [bowls swapped] [both applied]

BackgroundReplaceAugmenter:
  Replaces bg_mask pixels with a random image from the background pool.
  Shown twice with different seeds to illustrate variability.

PrintShuffleAugmenter:
  Swaps the bounding-rect content of other1 and other2 (the two non-target bowls).
  Target bowl is never touched.

Usage:
  python tools/show_eval1_masks.py
  python tools/show_eval1_masks.py --mask-dir /path/to/eval1_masks
  python tools/show_eval1_masks.py --bg-dir /path/to/eval3_backgrounds
  python tools/show_eval1_masks.py --out outputs/my_dir/aug_demo.png
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_REPO_ROOT   = Path(__file__).resolve().parent.parent.parent  # eval1_train/
DEFAULT_MASK_DIR = str(_REPO_ROOT / "outputs" / "eval1_masks")
DEFAULT_BG_DIR   = str(_REPO_ROOT / "outputs" / "eval3_backgrounds")
DEFAULT_OUT      = str(_REPO_ROOT / "outputs" / "eval1_mask_viz" / "augmentation_demo.png")

CELL_W, CELL_H = 320, 240
PAD     = 8
LABEL_H = 24
SLUG_TITLE_H = 32
MAIN_TITLE_H = 48


# ---------------------------------------------------------------------------
# Augmentation helpers (pure numpy/PIL, no torch)
# ---------------------------------------------------------------------------

def _load_mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    arr = np.load(path)
    return arr.astype(bool)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def apply_bg_replace(frame: np.ndarray, bg_mask: np.ndarray, bg_dir: str, seed: int = 0) -> np.ndarray:
    """Replace bg_mask pixels with a randomly chosen background image."""
    rng = random.Random(seed)
    bg_paths = sorted(glob.glob(os.path.join(bg_dir, "*.png")))
    if not bg_paths:
        return frame.copy()
    bg_path = rng.choice(bg_paths)
    h, w = frame.shape[:2]
    bg_arr = np.array(Image.open(bg_path).convert("RGB").resize((w, h), Image.BILINEAR))
    out = frame.copy()
    out[bg_mask] = bg_arr[bg_mask]
    return out


def apply_bowl_shuffle(frame: np.ndarray, other1_mask: np.ndarray, other2_mask: np.ndarray) -> np.ndarray:
    """Swap the bounding-rect content of the two non-target bowls."""
    b1 = _bbox(other1_mask)
    b2 = _bbox(other2_mask)
    if b1 is None or b2 is None:
        return frame.copy()
    x1a, y1a, x2a, y2a = b1
    x1b, y1b, x2b, y2b = b2
    patch_a = frame[y1a:y2a, x1a:x2a].copy()
    patch_b = frame[y1b:y2b, x1b:x2b].copy()
    h_a, w_a = y2a - y1a, x2a - x1a
    h_b, w_b = y2b - y1b, x2b - x1b
    a_into_b = np.array(Image.fromarray(patch_a).resize((w_b, h_b), Image.BILINEAR))
    b_into_a = np.array(Image.fromarray(patch_b).resize((w_a, h_a), Image.BILINEAR))
    out = frame.copy()
    out[y1a:y2a, x1a:x2a] = b_into_a
    out[y1b:y2b, x1b:x2b] = a_into_b
    return out


def apply_both(frame: np.ndarray, bg_mask: np.ndarray, other1_mask: np.ndarray,
               other2_mask: np.ndarray, bg_dir: str, seed: int = 0) -> np.ndarray:
    """bg replace first, then bowl shuffle (matching training order)."""
    out = apply_bg_replace(frame, bg_mask, bg_dir, seed=seed)
    out = apply_bowl_shuffle(out, other1_mask, other2_mask)
    return out


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _cell(arr: np.ndarray, label: str, note: str = "") -> Image.Image:
    """Resize array to cell size, add a dark label bar at the bottom."""
    thumb = Image.fromarray(arr).resize((CELL_W, CELL_H), Image.Resampling.BICUBIC)
    total_h = CELL_H + LABEL_H
    canvas = Image.new("RGB", (CELL_W, total_h), (30, 30, 30))
    canvas.paste(thumb, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, CELL_H, CELL_W, total_h), fill=(25, 25, 25))
    draw.text((6, CELL_H + 3), label, fill=(230, 230, 230), font=_load_font(13))
    if note:
        draw.text((CELL_W - len(note) * 7 - 4, CELL_H + 3), note, fill=(160, 160, 160), font=_load_font(12))
    return canvas


def _draw_bbox_outlines(arr: np.ndarray, bboxes: list[tuple], colors: list[tuple], thickness: int = 2) -> np.ndarray:
    """Draw colored bounding-box outlines on a copy of arr (for the 'bowls swapped' panel)."""
    out = arr.copy()
    for (x1, y1, x2, y2), color in zip(bboxes, colors):
        for t in range(thickness):
            out[y1 + t, x1:x2] = color
            out[y2 - 1 - t, x1:x2] = color
            out[y1:y2, x1 + t] = color
            out[y1:y2, x2 - 1 - t] = color
    return out


def render_slug_row(frame: np.ndarray, bg_mask: np.ndarray | None,
                    other1_mask: np.ndarray | None, other2_mask: np.ndarray | None,
                    bg_dir: str, slug: str, target_color: str) -> Image.Image:
    """Build one horizontal strip of panels for this slug."""

    panels: list[Image.Image] = []

    # 1. Original
    panels.append(_cell(frame, "original"))

    # 2 & 3. Background replaced — two different seeds
    if bg_mask is not None and os.path.isdir(bg_dir):
        for seed, letter in [(7, "A"), (42, "B")]:
            aug = apply_bg_replace(frame, bg_mask, bg_dir, seed=seed)
            panels.append(_cell(aug, f"bg replaced ({letter})", f"seed={seed}"))
    else:
        # Fill placeholders if bg data is missing
        for letter in ("A", "B"):
            panels.append(_cell(frame, f"[no bg pool] ({letter})"))

    # 4. Bowls swapped — annotate the bbox outlines before swap
    if other1_mask is not None and other2_mask is not None:
        b1 = _bbox(other1_mask)
        b2 = _bbox(other2_mask)
        annotated = _draw_bbox_outlines(
            frame,
            [b for b in [b1, b2] if b is not None],
            [(255, 80, 80), (80, 230, 80)],
        )
        swapped = apply_bowl_shuffle(frame, other1_mask, other2_mask)
        # Draw the same outlines on the swapped result
        swapped_annotated = _draw_bbox_outlines(
            swapped,
            [b for b in [b1, b2] if b is not None],
            [(255, 80, 80), (80, 230, 80)],
        )
        panels.append(_cell(annotated, "before swap  [red=other1  green=other2]"))
        panels.append(_cell(swapped_annotated, "bowls swapped"))
    else:
        panels.append(_cell(frame, "[no other masks]"))
        panels.append(_cell(frame, "[no other masks]"))

    # 5. Both augmentations (bg replace seed A + bowl swap)
    if bg_mask is not None and other1_mask is not None and other2_mask is not None and os.path.isdir(bg_dir):
        both = apply_both(frame, bg_mask, other1_mask, other2_mask, bg_dir, seed=7)
        panels.append(_cell(both, "bg replaced + bowls swapped"))
    else:
        panels.append(_cell(frame, "[missing data for combined]"))

    # Assemble panels into one row
    row_w = len(panels) * (CELL_W + PAD) + PAD
    row_h = SLUG_TITLE_H + (CELL_H + LABEL_H) + PAD
    row = Image.new("RGB", (row_w, row_h), (18, 18, 18))
    draw = ImageDraw.Draw(row)

    slug_label = f"  {slug}   target = {target_color} bowl   (target bowl is NEVER touched by either augmentation)"
    draw.rectangle((0, 0, row_w, SLUG_TITLE_H), fill=(45, 45, 55))
    draw.text((8, 7), slug_label, fill=(220, 220, 255), font=_load_font(15))

    for i, panel in enumerate(panels):
        x = PAD + i * (CELL_W + PAD)
        row.paste(panel, (x, SLUG_TITLE_H))

    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mask-dir", default=DEFAULT_MASK_DIR,
                    help=f"Directory of slug subdirs with *.npy masks (default: {DEFAULT_MASK_DIR})")
    ap.add_argument("--bg-dir",   default=DEFAULT_BG_DIR,
                    help=f"Directory of *.png background images (default: {DEFAULT_BG_DIR})")
    ap.add_argument("--out",      default=DEFAULT_OUT,
                    help=f"Output PNG path (default: {DEFAULT_OUT})")
    args = ap.parse_args()

    mask_root = Path(args.mask_dir)
    bg_dir    = args.bg_dir
    out_path  = Path(args.out)

    if not mask_root.exists():
        print(f"ERROR: mask-dir not found: {mask_root}")
        raise SystemExit(1)

    slugs = sorted(p.name for p in mask_root.iterdir() if p.is_dir())
    if not slugs:
        print(f"No subdirectories found in {mask_root}")
        raise SystemExit(1)

    bg_count = len(glob.glob(os.path.join(bg_dir, "*.png")))
    print(f"mask-dir : {mask_root}  ({len(slugs)} slugs)")
    print(f"bg-dir   : {bg_dir}  ({bg_count} images)")
    print(f"out      : {out_path}\n")

    rows: list[Image.Image] = []

    for slug in slugs:
        slug_dir = mask_root / slug
        meta_path = slug_dir / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        target_color = meta.get("target_color", "?")

        frame0 = slug_dir / "frame0.png"
        if not frame0.exists():
            print(f"  [{slug}] SKIP — no frame0.png")
            continue
        frame = np.array(Image.open(frame0).convert("RGB"))

        bg_mask     = _load_mask(slug_dir / "bg_mask.npy")
        other1_mask = _load_mask(slug_dir / "other1_mask.npy")
        other2_mask = _load_mask(slug_dir / "other2_mask.npy")

        row = render_slug_row(frame, bg_mask, other1_mask, other2_mask, bg_dir, slug, target_color)
        rows.append(row)
        print(f"  [{slug}]  target={target_color}  bg_mask={bg_mask.sum() if bg_mask is not None else 'n/a'} px")

    if not rows:
        print("Nothing to render.")
        raise SystemExit(1)

    # Assemble rows into one vertical stack with a global title
    row_w   = max(r.width for r in rows)
    total_h = MAIN_TITLE_H + sum(r.height + PAD for r in rows) + PAD
    canvas  = Image.new("RGB", (row_w, total_h), (10, 10, 10))
    draw    = ImageDraw.Draw(canvas)

    legend = (
        "Eval 1 augmentation demo  —  "
        "cols: original | bg replaced A | bg replaced B | before swap | bowls swapped | both"
    )
    draw.text((PAD, 12), legend, fill=(255, 255, 200), font=_load_font(17))

    y = MAIN_TITLE_H + PAD
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height + PAD

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
