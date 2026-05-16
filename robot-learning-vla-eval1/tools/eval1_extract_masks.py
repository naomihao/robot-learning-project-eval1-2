#!/usr/bin/env python3
"""Manual static masks for Eval 1 banana-in-colored-bowl datasets.

This mirrors the manual mask part of ``tools/eval3_extract_masks.py``: masks
come from manually chosen polygons. Unlike Eval 3, the source frame is still
loaded from the Hugging Face dataset video so this script can refresh
``frame0.png`` from the dataset cache/download.

  bg_mask.npy      bool (H, W); True = background above the table, replaceable
  target_mask.npy  bool (H, W); True = target bowl region
  other1_mask.npy  bool (H, W); True = distractor bowl 1
  other2_mask.npy  bool (H, W); True = distractor bowl 2
  preview.png      overlay sanity check
  meta.json        repo, polygons, pixel counts

Inspect each preview and tune PRESETS if the camera/framing changes.

Usage:
  python tools/eval1_extract_masks.py
  python tools/eval1_extract_masks.py --local-files-only
  python tools/eval1_extract_masks.py --interactive --local-files-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


VIDEO_KEY = "observation.images.front"

# Each dataset has 3 bowl polygons in color positions in the image plane.
# `target` names which bowl the banana was placed into for that dataset.
PRESETS = {
    "red_bowl": {
        "repo_id": "RobotLearningVLA/banana_red_bowl_eval1_v2",
        "table_polygon": [(0, 62), (640, 62), (640, 480), (0, 480)],
        "bowls": {
            "green": [(75, 126), (105, 104), (164, 97), (199, 119), (202, 158), (175, 184), (111, 188), (73, 160)],
            "blue": [(464, 119), (501, 104), (559, 107), (592, 134), (591, 177), (558, 199), (494, 199), (462, 171)],
            "red": [(250, 281), (270, 239), (315, 214), (370, 216), (414, 247), (429, 303), (407, 355), (360, 385), (296, 378), (257, 339)],
        },
        "target": "red",
    },
    "blue_bowl": {
        "repo_id": "RobotLearningVLA/banana_blue_bowl_eval1_v2",
        "table_polygon": [(0, 62), (640, 62), (640, 480), (0, 480)],
        "bowls": {
            "green": [(75, 126), (105, 104), (164, 97), (199, 119), (202, 158), (175, 184), (111, 188), (73, 160)],
            "blue": [(464, 119), (501, 104), (559, 107), (592, 134), (591, 177), (558, 199), (494, 199), (462, 171)],
            "red": [(250, 281), (270, 239), (315, 214), (370, 216), (414, 247), (429, 303), (407, 355), (360, 385), (296, 378), (257, 339)],
        },
        "target": "blue",
    },
    "green_bowl": {
        "repo_id": "RobotLearningVLA/banana_green_bowl_eval1_v2",
        "table_polygon": [(0, 62), (640, 62), (640, 480), (0, 480)],
        "bowls": {
            "green": [(75, 126), (105, 104), (164, 97), (199, 119), (202, 158), (175, 184), (111, 188), (73, 160)],
            "blue": [(464, 119), (501, 104), (559, 107), (592, 134), (591, 177), (558, 199), (494, 199), (462, 171)],
            "red": [(250, 281), (270, 239), (315, 214), (370, 216), (414, 247), (429, 303), (407, 355), (360, 385), (296, 378), (257, 339)],
        },
        "target": "green",
    },
}


def snapshot_root(repo_id: str, local_files_only: bool) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=[
                "meta/info.json",
                f"videos/{VIDEO_KEY}/chunk-000/file-000.mp4",
            ],
            local_files_only=local_files_only,
        )
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def first_video_path(root: Path) -> Path:
    info = read_json(root / "meta" / "info.json")
    template = info.get(
        "video_path",
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    )
    rel = template.format(video_key=VIDEO_KEY, chunk_index=0, file_index=0)
    return root / rel


def load_first_frame(repo_id: str, local_files_only: bool) -> tuple[np.ndarray, Path, Path]:
    root = snapshot_root(repo_id, local_files_only=local_files_only)
    video_path = first_video_path(root)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(str(video_path))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"OpenCV could not read first frame from {video_path}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_rgb, video_path, root


def polygon_to_mask(poly: list[tuple[int, int]], shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).polygon(poly, fill=255)
    return np.array(img) > 0


def draw_polygon_interface(
    frame_rgb: np.ndarray,
    label: str,
    existing: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    points = list(existing or [])
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    window = f"eval1 mask: {label}"

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        canvas = frame_bgr.copy()
        if points:
            pts = np.array(points, dtype=np.int32)
            for idx, (x, y) in enumerate(points, start=1):
                cv2.circle(canvas, (x, y), 4, (0, 255, 255), -1)
                cv2.putText(canvas, str(idx), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            if len(points) > 1:
                cv2.polylines(canvas, [pts], isClosed=False, color=(0, 255, 255), thickness=2)
            if len(points) > 2:
                cv2.polylines(canvas, [pts], isClosed=True, color=(0, 200, 255), thickness=1)

        instructions = [
            f"Draw polygon: {label}",
            "Left click: add point    Right click: undo",
            "Enter/Space: accept      R: reset      Esc/Q: cancel",
        ]
        for row, text in enumerate(instructions):
            y = 24 + row * 22
            cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow(window, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32):
            if len(points) < 3:
                print(f"  {label}: need at least 3 points")
                continue
            cv2.destroyWindow(window)
            return points
        if key in (ord("r"), ord("R")):
            points.clear()
        if key in (27, ord("q"), ord("Q")):
            cv2.destroyWindow(window)
            raise KeyboardInterrupt(f"cancelled while drawing {label}")


def collect_manual_polygons(frame: np.ndarray, slug: str, cfg: dict[str, Any]) -> tuple[list[tuple[int, int]], dict[str, list[tuple[int, int]]]]:
    print(f"  opening manual polygon interface for {slug}")
    print("  draw table first, then green/blue/red bowl polygons")
    table_polygon = draw_polygon_interface(frame, f"{slug}: table", cfg["table_polygon"])
    bowl_polygons = {}
    for color in ("green", "blue", "red"):
        bowl_polygons[color] = draw_polygon_interface(frame, f"{slug}: {color} bowl", cfg["bowls"].get(color))
    return table_polygon, bowl_polygons


def render_preview(frame: np.ndarray, masks: dict[str, np.ndarray], out_path: Path, header: str) -> None:
    """Tint each mask region a distinct color and save a preview PNG."""
    overlay = frame.copy().astype(np.float32)
    tints = {
        "background": (255, 0, 255),   # magenta = will be replaced
        "table": (255, 255, 0),        # yellow  = kept table
        "target": (0, 191, 255),       # cyan    = target bowl
        "other1": (255, 64, 64),       # red     = shuffleable
        "other2": (64, 255, 64),       # green   = shuffleable
    }
    for name, mask in masks.items():
        if mask is None:
            continue
        tint = np.array(tints[name], dtype=np.float32)
        overlay[mask] = overlay[mask] * 0.6 + tint * 0.4

    out = Image.fromarray(overlay.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)
    legend = "magenta=bg(replace)  yellow=table  cyan=TARGET  red/green=shuffleable bowls"
    draw.text((6, 6), header, fill=(0, 0, 0))
    draw.text((6, 22), legend, fill=(0, 0, 0))
    out.save(out_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/eval1_masks")
    ap.add_argument("--local-files-only", action="store_true", help="Use only cached Hugging Face files")
    ap.add_argument("--interactive", action="store_true", help="Click polygons manually on each decoded first frame")
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for slug, cfg in PRESETS.items():
        repo_id = cfg["repo_id"]
        target_color = cfg["target"]
        frame, video_path, root = load_first_frame(repo_id, local_files_only=args.local_files_only)
        h, w = frame.shape[:2]
        print(f"\n[{slug}]  video {video_path}  shape={h}x{w}  target={target_color}")

        # Build masks from manual polygons. With --interactive, polygons are
        # clicked in the UI; otherwise the saved manual PRESETS are used.
        if args.interactive:
            table_polygon, bowl_polygons = collect_manual_polygons(frame, slug, cfg)
        else:
            table_polygon = cfg["table_polygon"]
            bowl_polygons = cfg["bowls"]
        bowl_masks = {color: polygon_to_mask(poly, (h, w)) for color, poly in bowl_polygons.items()}

        # Table mask EXCLUDES the bowl regions; bg = outside table
        table_full = polygon_to_mask(table_polygon, (h, w))
        any_bowl = np.zeros_like(table_full)
        for mask in bowl_masks.values():
            any_bowl |= mask
        table_only = table_full & ~any_bowl
        bg_mask = ~table_full

        # Identify target + the other two
        target_mask = bowl_masks[target_color]
        other_colors = [color for color in ("red", "blue", "green") if color != target_color]
        other1_mask = bowl_masks[other_colors[0]]
        other2_mask = bowl_masks[other_colors[1]]

        ds_dir = out_root / slug
        ds_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(ds_dir / "frame0.png")
        np.save(ds_dir / "bg_mask.npy", bg_mask)
        np.save(ds_dir / "target_mask.npy", target_mask)
        np.save(ds_dir / "other1_mask.npy", other1_mask)
        np.save(ds_dir / "other2_mask.npy", other2_mask)

        render_preview(
            frame,
            {
                "background": bg_mask,
                "table": table_only,
                "target": target_mask,
                "other1": other1_mask,
                "other2": other2_mask,
            },
            ds_dir / "preview.png",
            header=f"{slug}  target={target_color}  {repo_id}",
        )

        meta = {
            "slug": slug,
            "repo_id": repo_id,
            "snapshot_root": str(root),
            "video_path": str(video_path),
            "image_shape_hw": [h, w],
            "table_polygon": table_polygon,
            "bowls": bowl_polygons,
            "target_color": target_color,
            "other_colors": other_colors,
            "bg_mask_pixels": int(bg_mask.sum()),
            "target_mask_pixels": int(target_mask.sum()),
            "other1_mask_pixels": int(other1_mask.sum()),
            "other2_mask_pixels": int(other2_mask.sum()),
        }
        with open(ds_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  bg_mask:     {bg_mask.sum():>7d} px ({bg_mask.mean() * 100:.1f}%)")
        print(f"  target ({target_color:>5}): {target_mask.sum():>7d} px")
        print(f"  other1 ({other_colors[0]:>5}): {other1_mask.sum():>7d} px")
        print(f"  other2 ({other_colors[1]:>5}): {other2_mask.sum():>7d} px")
        print(f"  preview ->  {ds_dir / 'preview.png'}")

    print("\nDone. Inspect each preview.png and adjust PRESETS in this file if a polygon is wrong.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
