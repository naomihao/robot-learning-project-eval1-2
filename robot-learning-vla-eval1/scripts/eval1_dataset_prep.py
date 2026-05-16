"""Dataset preprocessing wrappers for Eval 1 training (banana bowl task).

Adapted from eval3_dataset_prep.py for eval1-specific augmentations.
Applied to each per-colour LeRobotDataset BEFORE concatenation.

1. ``Eval1PrepDataset`` — proxy wrapping a LeRobotDataset with:
   * Episode truncation to first ``max_frames_per_episode`` frames
   * Task string augmentation (varied banana bowl phrasings, same target colour)
   * Background replacement (requires bg_mask.npy + pool of .png images)
   * Bowl shuffle: swap two non-target bowl regions to prevent colour-position
     co-adaptation (analog of eval3 print shuffle)
   * Gamma augmentation applied per sample

2. ``TaskAugmenter`` — varies banana bowl task strings while preserving the
   target bowl colour. Canonical demo wording gets the highest probability since
   that is what the TA will type at evaluation time.

3. ``BackgroundReplaceAugmenter`` — identical logic to eval3 version.
   Needs: ``outputs/eval1_masks/<color>/bg_mask.npy`` and a directory of
   ``.png`` background images (``outputs/eval1_backgrounds``).

4. ``BowlShuffleAugmenter`` — swaps two non-target bowl regions. Needs:
   ``outputs/eval1_masks/<color>/other1_mask.npy`` and ``other2_mask.npy``.
"""
from __future__ import annotations

import glob
import os
import random
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset


BOWL_COLORS = ("green", "blue", "red")


# ── Task augmentation ─────────────────────────────────────────────────────────

class TaskAugmenter:
    """Picklable callable that randomly rewrites Eval 1 banana bowl task strings.

    Detects the target bowl colour from the original task string and emits a
    rephrased variant. The canonical TA wording (highest weight) is:
      "Put the banana in the <color> colored bowl."

    Variant probabilities:
      35 %  "Put the banana in the <color> colored bowl."   (canonical)
      15 %  "Place the banana in the <color> colored bowl."
      15 %  "Put the banana in the <color> bowl."
      15 %  "Place the banana in the <color> bowl."
      10 %  "Move the banana to the <color> bowl."
      10 %  "Pick up the banana and put it in the <color> colored bowl."
    """

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._rng = random.Random(seed)

    def __call__(self, task: str) -> str:
        if not isinstance(task, str):
            return task
        color = next((c for c in BOWL_COLORS if c in task.lower()), None)
        if color is None:
            return task
        roll = self._rng.random()
        if roll < 0.35:
            return f"Put the banana in the {color} colored bowl."
        elif roll < 0.50:
            return f"Place the banana in the {color} colored bowl."
        elif roll < 0.65:
            return f"Put the banana in the {color} bowl."
        elif roll < 0.80:
            return f"Place the banana in the {color} bowl."
        elif roll < 0.90:
            return f"Move the banana to the {color} bowl."
        else:
            return f"Pick up the banana and put it in the {color} colored bowl."

    def __reduce__(self):
        return (self.__class__, (self._seed,))


def make_task_augmenter(seed: int = 42) -> "TaskAugmenter":
    """Factory returning a TaskAugmenter instance."""
    return TaskAugmenter(seed=seed)


# ── Image augmenters ──────────────────────────────────────────────────────────

def _load_mask(mask_path: str) -> np.ndarray:
    arr = np.load(mask_path)
    if arr.dtype != bool:
        arr = arr.astype(bool)
    return arr


def _bbox_of_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


class BackgroundReplaceAugmenter:
    """Replace background pixels with a random image from a pool.

    With probability ``p``, loads a random .png from ``bg_dir``, resizes it to
    match the frame, and pastes it wherever ``bg_mask`` (HxW bool, True =
    background) is True. Foreground pixels (table, bowls, arm, banana) are
    preserved. Identical logic to eval3's BackgroundReplaceAugmenter.
    """

    def __init__(self, mask_path: str, bg_dir: str, p: float = 0.3, seed: int = 0):
        self._mask_path = str(mask_path)
        self._bg_dir = str(bg_dir)
        self._p = float(p)
        self._seed = int(seed)
        self._rng = random.Random(seed)
        self._mask: np.ndarray | None = None
        self._bg_paths: list[str] | None = None

    def _ensure_loaded(self) -> None:
        if self._mask is None:
            self._mask = _load_mask(self._mask_path)
        if self._bg_paths is None:
            self._bg_paths = sorted(glob.glob(os.path.join(self._bg_dir, "*.png")))
            if not self._bg_paths:
                raise FileNotFoundError(f"no .png backgrounds under {self._bg_dir}")

    def __call__(self, img_chw: torch.Tensor) -> torch.Tensor:
        if self._rng.random() >= self._p:
            return img_chw
        self._ensure_loaded()
        c, h, w = img_chw.shape
        bg_path = self._bg_paths[self._rng.randrange(len(self._bg_paths))]
        from PIL import Image
        bg_pil = Image.open(bg_path).convert("RGB").resize((w, h), Image.BILINEAR)
        bg_arr = np.array(bg_pil, dtype=np.float32) / 255.0
        bg_t = torch.from_numpy(bg_arr).permute(2, 0, 1).to(img_chw.dtype).to(img_chw.device)
        m = self._mask
        if m.shape != (h, w):
            m_pil = Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
            m = np.array(m_pil) > 127
        mask_t = torch.from_numpy(m).to(img_chw.device)
        out = img_chw.clone()
        for ch in range(c):
            out[ch][mask_t] = bg_t[ch][mask_t]
        return out

    def __reduce__(self):
        return (self.__class__, (self._mask_path, self._bg_dir, self._p, self._seed))


class BowlShuffleAugmenter:
    """Swap the pixel content of two non-target bowl regions.

    The target bowl is never touched — its position is what the robot aims for,
    and swapping it would break action-image alignment. The two non-target bowls
    are swapped with probability ``p``, preventing the model from encoding bowl
    colour as a fixed spatial position.

    Requires bounding-rectangle mask files for the two non-target bowls
    (``other1_mask.npy``, ``other2_mask.npy``). Masks can differ in size; each
    region is resized to fit the other's bbox before pasting.

    Analog of eval3's PrintShuffleAugmenter.
    """

    def __init__(self, other1_path: str, other2_path: str, p: float = 0.5, seed: int = 0):
        self._other1_path = str(other1_path)
        self._other2_path = str(other2_path)
        self._p = float(p)
        self._seed = int(seed)
        self._rng = random.Random(seed)
        self._bbox1: tuple[int, int, int, int] | None = None
        self._bbox2: tuple[int, int, int, int] | None = None

    def _ensure_loaded(self) -> None:
        if self._bbox1 is None:
            self._bbox1 = _bbox_of_mask(_load_mask(self._other1_path))
        if self._bbox2 is None:
            self._bbox2 = _bbox_of_mask(_load_mask(self._other2_path))

    def __call__(self, img_chw: torch.Tensor) -> torch.Tensor:
        if self._rng.random() >= self._p:
            return img_chw
        self._ensure_loaded()
        x1a, y1a, x2a, y2a = self._bbox1
        x1b, y1b, x2b, y2b = self._bbox2
        if x2a <= x1a or y2a <= y1a or x2b <= x1b or y2b <= y1b:
            return img_chw
        patch_a = img_chw[:, y1a:y2a, x1a:x2a].clone()
        patch_b = img_chw[:, y1b:y2b, x1b:x2b].clone()
        import torchvision.transforms.v2.functional as F  # noqa: N812
        size_b = (y2b - y1b, x2b - x1b)
        size_a = (y2a - y1a, x2a - x1a)
        a_to_b = F.resize(patch_a, size_b, antialias=True)
        b_to_a = F.resize(patch_b, size_a, antialias=True)
        out = img_chw.clone()
        out[:, y1a:y2a, x1a:x2a] = b_to_a
        out[:, y1b:y2b, x1b:x2b] = a_to_b
        return out

    def __reduce__(self):
        return (self.__class__, (self._other1_path, self._other2_path, self._p, self._seed))


# ── Dataset wrapper ───────────────────────────────────────────────────────────

class Eval1PrepDataset(Dataset):
    """Proxy wrapping a LeRobotDataset with episode truncation and augmentation.

    Parameters
    ----------
    dataset : LeRobotDataset
        Underlying dataset (already constructed with delta_timestamps, transforms).
    max_frames_per_episode : int | None
        Keep only the first N frames per episode. None disables truncation.
    task_aug_fn : Callable[[str], str] | None
        Applied to each row's ``task`` string.
    bg_aug_fn : Callable[[Tensor], Tensor] | None
        Background replacement augmenter.
    bowl_shuffle_fn : Callable[[Tensor], Tensor] | None
        Non-target bowl swap augmenter.
    gamma_p : float
        Per-sample probability of applying random gamma correction (0 = off).
    image_key : str
        Observation image key to augment (default: ``observation.images.front``).
    episode_filter : list[int] | None
        If set, only include these episode indices.
    """

    def __init__(
        self,
        dataset,
        max_frames_per_episode: int | None = 200,
        task_aug_fn: Callable[[str], str] | None = None,
        bg_aug_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        bowl_shuffle_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        gamma_p: float = 0.4,
        image_key: str = "observation.images.front",
        episode_filter: list[int] | None = None,
    ):
        self._ds = dataset
        self._task_aug_fn = task_aug_fn
        self._bg_aug_fn = bg_aug_fn
        self._bowl_shuffle_fn = bowl_shuffle_fn
        self._gamma_p = float(gamma_p)
        self._image_key = image_key

        ep_df = dataset.meta.episodes
        from_idxs = list(ep_df["dataset_from_index"])
        to_idxs = list(ep_df["dataset_to_index"])

        valid: list[int] = []
        original_total = 0
        keep_eps = set(int(e) for e in episode_filter) if episode_filter is not None else None
        for ep_idx, (f0, f1) in enumerate(zip(from_idxs, to_idxs)):
            f0i, f1i = int(f0), int(f1)
            original_total += f1i - f0i
            if keep_eps is not None and ep_idx not in keep_eps:
                continue
            cap = (
                min(f0i + int(max_frames_per_episode), f1i)
                if max_frames_per_episode is not None
                else f1i
            )
            valid.extend(range(f0i, cap))

        self._valid_indices = valid
        self._max_frames_per_episode = max_frames_per_episode
        self._original_num_frames = original_total
        self._episode_filter = list(keep_eps) if keep_eps is not None else None

    # ----- Trainer-required attributes ----------------------------------------

    @property
    def meta(self):
        return self._ds.meta

    @property
    def features(self):
        return self._ds.features

    @property
    def repo_id(self) -> str:
        return self._ds.repo_id

    @property
    def num_frames(self) -> int:
        return len(self._valid_indices)

    @property
    def num_episodes(self) -> int:
        return self._ds.num_episodes

    @property
    def episodes(self):
        return self._ds.episodes

    # ----- Dataset protocol ---------------------------------------------------

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx) -> Any:
        original_idx = self._valid_indices[int(idx)]
        row = self._ds[original_idx]
        mutated = False

        if self._task_aug_fn is not None and "task" in row:
            if not mutated:
                row = dict(row)
                mutated = True
            row["task"] = self._task_aug_fn(row["task"])

        # Sample gamma once so we don't enter the image block unnecessarily.
        apply_gamma = self._gamma_p > 0 and torch.rand(1).item() < self._gamma_p
        need_img_aug = (
            self._bg_aug_fn is not None
            or self._bowl_shuffle_fn is not None
            or apply_gamma
        )

        if need_img_aug and self._image_key in row:
            if not mutated:
                row = dict(row)
                mutated = True
            img = row[self._image_key]
            # Order: bg replace → bowl shuffle → gamma.
            # bg replace first so bowl shuffle lands on clean background.
            # Gamma last so it acts on the final pixel values.
            if self._bg_aug_fn is not None:
                img = self._bg_aug_fn(img)
            if self._bowl_shuffle_fn is not None:
                img = self._bowl_shuffle_fn(img)
            if apply_gamma and isinstance(img, torch.Tensor) and img.is_floating_point():
                gamma = 0.8 + 0.4 * torch.rand(1).item()
                img = img.clamp(0.0, 1.0).pow(gamma)
            row[self._image_key] = img

        return row

    # ----- Catch-all proxy ----------------------------------------------------

    def __getattr__(self, name):
        if "_ds" in self.__dict__:
            return getattr(self._ds, name)
        raise AttributeError(name)

    # ----- Debug helpers -------------------------------------------------------

    def truncation_summary(self) -> dict:
        return {
            "repo_id": self.repo_id,
            "max_frames_per_episode": self._max_frames_per_episode,
            "original_num_frames": int(self._original_num_frames),
            "kept_num_frames": int(self.num_frames),
            "dropped_num_frames": int(self._original_num_frames - self.num_frames),
            "kept_fraction": (
                float(self.num_frames) / float(self._original_num_frames)
                if self._original_num_frames
                else 0.0
            ),
        }


# ── Pipeline factory ──────────────────────────────────────────────────────────

def build_eval1_pipeline(
    ds,
    repo_id: str,
    *,
    mask_dir: str = "../outputs/eval1_masks",
    bg_dir: str = "../outputs/eval3_backgrounds",
    max_frames: int = 200,
    gamma_p: float = 0.4,
    bg_p: float = 0.3,
    bs_p: float = 0.5,
    task_aug_fn=None,
    strict: bool = True,
) -> "Eval1PrepDataset":
    """Wrap a LeRobotDataset with the full eval1 augmentation pipeline.

    Parameters
    ----------
    strict : bool
        True (default): raises FileNotFoundError/ValueError when mask or
        background files are missing — used by the training script.
        False: silently skips unavailable augmenters — used by smoke-tests
        where masks may not be present locally.
    """
    if task_aug_fn is None:
        task_aug_fn = make_task_augmenter()

    slug = next(
        (c for c in ("green_bowl", "blue_bowl", "red_bowl") if c in repo_id.lower()), None
    )
    if slug is None:
        if strict:
            raise ValueError(
                f"Cannot determine bowl colour slug from repo_id '{repo_id}'. "
                "Expected one of green_bowl/blue_bowl/red_bowl in the name."
            )

    bg_aug = None
    if slug:
        bg_mask_path = os.path.join(mask_dir, slug, "bg_mask.npy")
        if strict:
            if not os.path.exists(bg_mask_path):
                raise FileNotFoundError(f"[eval1] bg mask not found: {bg_mask_path}")
            if not os.path.isdir(bg_dir):
                raise FileNotFoundError(f"[eval1] background image dir not found: {bg_dir}")
            bg_aug = BackgroundReplaceAugmenter(bg_mask_path, bg_dir, p=bg_p, seed=hash(slug) & 0xFFFF)
        elif os.path.exists(bg_mask_path) and os.path.isdir(bg_dir):
            bg_aug = BackgroundReplaceAugmenter(bg_mask_path, bg_dir, p=bg_p, seed=hash(slug) & 0xFFFF)

    bowl_aug = None
    if slug:
        o1 = os.path.join(mask_dir, slug, "other1_mask.npy")
        o2 = os.path.join(mask_dir, slug, "other2_mask.npy")
        if strict:
            if not os.path.exists(o1):
                raise FileNotFoundError(f"[eval1] bowl-shuffle mask not found: {o1}")
            if not os.path.exists(o2):
                raise FileNotFoundError(f"[eval1] bowl-shuffle mask not found: {o2}")
            bowl_aug = BowlShuffleAugmenter(o1, o2, p=bs_p, seed=(hash(slug) >> 16) & 0xFFFF)
        elif os.path.exists(o1) and os.path.exists(o2):
            bowl_aug = BowlShuffleAugmenter(o1, o2, p=bs_p, seed=(hash(slug) >> 16) & 0xFFFF)

    return Eval1PrepDataset(
        ds,
        max_frames_per_episode=max_frames,
        task_aug_fn=task_aug_fn,
        bg_aug_fn=bg_aug,
        bowl_shuffle_fn=bowl_aug,
        gamma_p=gamma_p,
    )
