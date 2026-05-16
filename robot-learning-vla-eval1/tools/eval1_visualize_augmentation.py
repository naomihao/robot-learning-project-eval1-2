#!/usr/bin/env python3
"""Render torch-free augmentation previews for the Eval 1 banana/red-bowl task.

This avoids importing torch/lerobot, which can bus-error in broken CUDA/PyTorch
environments. By default it downloads/uses the first frame from the Eval 1
dataset video with OpenCV. Pass --input-image to use a local PNG/JPG instead.

Usage:
  python tools/eval1_visualize_augmentation.py
  python tools/eval1_visualize_augmentation.py --repo-id RobotLearningVLA/banana_red_bowl_eval1_v2
  python tools/eval1_visualize_augmentation.py --input-image path/to/frame.png
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


TASK = "Put the banana in the red colored bowel."
DEFAULT_REPO_ID = "RobotLearningVLA/banana_red_bowl_eval1_v2"
DEFAULT_VIDEO_KEY = "observation.images.front"


def make_demo_scene(size: tuple[int, int] = (640, 480)) -> Image.Image:
    """Draw a small synthetic scene so the visualizer works without torch."""
    w, h = size
    img = Image.new("RGB", size, (205, 208, 198))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, int(h * 0.48), w, h), fill=(166, 142, 106))
    draw.ellipse((360, 250, 530, 380), fill=(165, 18, 28), outline=(95, 15, 20), width=8)
    draw.ellipse((390, 275, 500, 340), fill=(80, 18, 22))
    draw.arc((150, 230, 360, 360), 190, 350, fill=(244, 204, 37), width=34)
    draw.arc((150, 230, 360, 360), 190, 350, fill=(92, 68, 20), width=3)
    return img


def load_image(path: str | None, repo_id: str, local_files_only: bool) -> Image.Image:
    if not path:
        try:
            return load_first_dataset_frame(repo_id, local_files_only=local_files_only)
        except Exception as exc:
            print(f"dataset frame load failed ({exc}); using synthetic fallback scene")
            return make_demo_scene()
    return Image.open(path).convert("RGB")


def load_first_dataset_frame(repo_id: str, local_files_only: bool = False) -> Image.Image:
    """Download/read episode 0 frame 0 without importing torch or lerobot."""
    from huggingface_hub import snapshot_download
    import cv2

    root = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=[
                "meta/info.json",
                f"videos/{DEFAULT_VIDEO_KEY}/chunk-000/file-000.mp4",
            ],
            local_files_only=local_files_only,
        )
    )
    info = read_json(root / "meta" / "info.json")
    video_path_template = info.get(
        "video_path",
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    )
    video_rel = video_path_template.format(video_key=DEFAULT_VIDEO_KEY, chunk_index=0, file_index=0)
    video_path = root / video_rel
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(str(video_path))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"OpenCV could not read first frame from {video_path}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    print(f"loaded dataset first frame: {video_path}")
    return Image.fromarray(frame_rgb)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def brightness(img: Image.Image) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(1.35)


def contrast(img: Image.Image) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(1.35)


def saturation(img: Image.Image) -> Image.Image:
    return ImageEnhance.Color(img).enhance(1.45)


def hue(img: Image.Image) -> Image.Image:
    arr = np.array(img.convert("HSV"), dtype=np.int16)
    arr[..., 0] = (arr[..., 0] + 6) % 256
    return Image.fromarray(arr.astype(np.uint8), "HSV").convert("RGB")


def sharpness(img: Image.Image) -> Image.Image:
    return ImageEnhance.Sharpness(img).enhance(1.5)


def affine(img: Image.Image) -> Image.Image:
    w, h = img.size
    angle = math.radians(3.0)
    tx, ty = int(0.03 * w), int(-0.02 * h)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    cx, cy = w / 2.0, h / 2.0
    # PIL affine maps output -> input.
    a = cos_a
    b = sin_a
    c = cx - cos_a * cx - sin_a * cy - tx
    d = -sin_a
    e = cos_a
    f = cy + sin_a * cx - cos_a * cy - ty
    return img.transform(img.size, Image.Transform.AFFINE, (a, b, c, d, e, f), resample=Image.Resampling.BILINEAR)


def perspective(img: Image.Image) -> Image.Image:
    w, h = img.size
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dx, dy = int(0.06 * w), int(0.05 * h)
    dst = [(dx, dy), (w - dx, 0), (w, h - dy), (0, h)]
    coeffs = perspective_coefficients(dst, src)
    return img.transform(img.size, Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)


def resized_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    crop_w, crop_h = int(w * 0.82), int(h * 0.82)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    return img.crop((left, top, left + crop_w, top + crop_h)).resize((w, h), Image.Resampling.BICUBIC)


def gaussian_blur(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=1.4))


def erase(img: Image.Image) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    x1, y1 = int(w * 0.58), int(h * 0.18)
    x2, y2 = int(w * 0.75), int(h * 0.33)
    draw.rectangle((x1, y1, x2, y2), fill=(128, 128, 128))
    return out


def perspective_coefficients(startpoints, endpoints) -> tuple[float, ...]:
    matrix = []
    for p1, p2 in zip(endpoints, startpoints, strict=True):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
    a = np.asarray(matrix, dtype=np.float64)
    b = np.asarray(startpoints, dtype=np.float64).reshape(8)
    return tuple(np.linalg.solve(a, b))


AUGS = {
    "brightness": brightness,
    "contrast": contrast,
    "saturation": saturation,
    "hue": hue,
    "sharpness": sharpness,
    "affine": affine,
    "perspective": perspective,
    "resized crop": resized_crop,
    "gaussian blur": gaussian_blur,
    "erase": erase,
}


def full_stack(img: Image.Image, seed: int = 0) -> Image.Image:
    rng = random.Random(seed)
    names = rng.sample(list(AUGS), k=4)
    out = img
    for name in names:
        out = AUGS[name](out)
    return out


def _load_font(size: int):
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            pass
    return ImageFont.load_default()


def render_grid(img: Image.Image, out_path: Path, label: str) -> None:
    preview_size = (320, 240)
    cells = [("original", img.resize(preview_size, Image.Resampling.BICUBIC))]
    for name, fn in AUGS.items():
        cells.append((name, fn(img).resize(preview_size, Image.Resampling.BICUBIC)))
    cells.append(("full stack", full_stack(img).resize(preview_size, Image.Resampling.BICUBIC)))

    cols = 4
    rows = math.ceil(len(cells) / cols)
    pad = 8
    label_h = 28
    title_h = 46
    cw, ch = preview_size
    canvas = Image.new("RGB", (cols * cw + (cols + 1) * pad, title_h + rows * (ch + label_h + pad) + pad), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 10), label, fill=(0, 0, 0), font=_load_font(18))
    cell_font = _load_font(14)

    for i, (name, cell) in enumerate(cells):
        col = i % cols
        row = i // cols
        x = pad + col * (cw + pad)
        y = title_h + pad + row * (ch + label_h + pad)
        canvas.paste(cell, (x, y))
        draw.text((x + 4, y + ch + 4), name, fill=(35, 35, 35), font=cell_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"wrote {out_path}")


def task_variants() -> list[str]:
    return [
        TASK,
        "Put the banana in the red colored bowl.",
        "Place the banana in the red bowl.",
        "Move the banana into the red colored bowl.",
        "Put the yellow banana in the red bowl.",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-image", default="", help="Optional PNG/JPG frame to augment")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HF dataset repo id for first-frame preview")
    parser.add_argument("--local-files-only", action="store_true", help="Use only cached HF files")
    parser.add_argument("--out-dir", default="outputs/eval1_augmentation_preview")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    img = load_image(args.input_image, args.repo_id, args.local_files_only)
    render_grid(img, out_dir / "banana_red_bowl_aug_preview.png", f"Eval 1 augmentation preview: {TASK}")

    task_json = out_dir / "task_variants.json"
    task_json.parent.mkdir(parents=True, exist_ok=True)
    task_json.write_text(json.dumps(task_variants(), indent=2))
    print(f"wrote {task_json}")


if __name__ == "__main__":
    main()
