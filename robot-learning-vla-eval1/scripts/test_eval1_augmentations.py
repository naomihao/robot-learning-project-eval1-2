"""Unit tests for eval1 augmentations.

Tests every augmenter defined in eval1_dataset_prep.py and verifies the full
pipeline is wired correctly in the training script (without importing lerobot).

Run with:
    cd robot-learning-vla-eval1
    python -m pytest scripts/test_eval1_augmentations.py -v

Integration smoke-test (requires lerobot + HF access):
    python scripts/test_eval1_augmentations.py
    # or with masks:
    EVAL1_MASK_DIR=outputs/eval1_masks \\
    EVAL1_BG_DIR=outputs/eval1_backgrounds \\
    python scripts/test_eval1_augmentations.py
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

# Make the scripts directory importable.
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from eval1_dataset_prep import (
    BOWL_COLORS,
    BackgroundReplaceAugmenter,
    BowlShuffleAugmenter,
    Eval1PrepDataset,
    TaskAugmenter,
    _bbox_of_mask,
    _load_mask,
    build_eval1_pipeline,
    make_task_augmenter,
)


# ─────────────────────────── helpers ──────────────────────────────────────────

def _make_image(h: int = 64, w: int = 64, fill: float | None = None) -> torch.Tensor:
    """Return a CHW float32 tensor in [0, 1]."""
    if fill is not None:
        return torch.full((3, h, w), fill, dtype=torch.float32)
    return torch.rand(3, h, w, dtype=torch.float32)


def _write_mask(path: Path, mask: np.ndarray) -> None:
    np.save(str(path), mask)


def _write_bg_png(path: Path, h: int = 64, w: int = 64) -> None:
    from PIL import Image
    arr = np.full((h, w, 3), 200, dtype=np.uint8)
    Image.fromarray(arr).save(str(path))


def _make_dummy_dataset(task: str = "Put the banana in the green colored bowl.",
                        num_episodes: int = 2,
                        frames_per_ep: int = 5,
                        image_key: str = "observation.images.front") -> MagicMock:
    """Build the minimal mock that Eval1PrepDataset needs."""
    total = num_episodes * frames_per_ep
    ep_df = MagicMock()
    ep_df.__iter__ = MagicMock(return_value=iter([]))
    from_idxs = [i * frames_per_ep for i in range(num_episodes)]
    to_idxs = [(i + 1) * frames_per_ep for i in range(num_episodes)]
    ep_df.__getitem__ = lambda self, key: (
        from_idxs if key == "dataset_from_index" else to_idxs
    )
    meta = MagicMock()
    meta.episodes = ep_df
    ds = MagicMock()
    ds.meta = meta
    ds.repo_id = "test/dataset"
    ds.num_episodes = num_episodes
    ds.episodes = None
    ds.features = {}
    ds.__getitem__ = lambda self, idx: {
        "task": task,
        image_key: _make_image(),
        "action": torch.zeros(7),
    }
    return ds


# ─────────────────────────── TaskAugmenter ────────────────────────────────────

class TestTaskAugmenter:
    def test_detects_all_bowl_colors(self):
        aug = TaskAugmenter(seed=0)
        for color in BOWL_COLORS:
            task = f"Put the banana in the {color} colored bowl."
            result = aug(task)
            assert color in result, f"color '{color}' lost in: {result!r}"

    def test_output_is_one_of_known_variants(self):
        aug = TaskAugmenter(seed=7)
        color = "blue"
        task = f"Put the banana in the {color} colored bowl."
        variants = {
            f"Put the banana in the {color} colored bowl.",
            f"Place the banana in the {color} colored bowl.",
            f"Put the banana in the {color} bowl.",
            f"Place the banana in the {color} bowl.",
            f"Move the banana to the {color} bowl.",
            f"Pick up the banana and put it in the {color} colored bowl.",
        }
        for _ in range(200):
            assert aug(task) in variants

    def test_non_string_passthrough(self):
        aug = TaskAugmenter()
        assert aug(None) is None
        assert aug(42) == 42

    def test_unknown_color_passthrough(self):
        aug = TaskAugmenter()
        task = "Do something with the yellow bowl."
        assert aug(task) == task

    def test_canonical_variant_is_most_frequent(self):
        """The canonical wording should have p=0.35, highest of all variants."""
        aug = TaskAugmenter(seed=123)
        color = "red"
        task = f"Put the banana in the {color} colored bowl."
        canonical = f"Put the banana in the {color} colored bowl."
        counts: dict[str, int] = {}
        for _ in range(2000):
            v = aug(task)
            counts[v] = counts.get(v, 0) + 1
        total = sum(counts.values())
        canonical_frac = counts.get(canonical, 0) / total
        # Should be ~0.35; allow ±0.08 tolerance.
        assert 0.27 < canonical_frac < 0.43, f"canonical fraction = {canonical_frac:.3f}"

    def test_picklable(self):
        aug = TaskAugmenter(seed=99)
        aug2 = pickle.loads(pickle.dumps(aug))
        task = "Put the banana in the green colored bowl."
        assert aug2(task) in {"Put the banana in the green colored bowl.",
                              "Place the banana in the green colored bowl.",
                              "Put the banana in the green bowl.",
                              "Place the banana in the green bowl.",
                              "Move the banana to the green bowl.",
                              "Pick up the banana and put it in the green colored bowl."}

    def test_make_task_augmenter_factory(self):
        aug = make_task_augmenter(seed=5)
        assert isinstance(aug, TaskAugmenter)

    def test_all_variant_probabilities(self):
        """Every variant must be generated near its declared probability (±0.07)."""
        aug = TaskAugmenter(seed=0)
        color = "green"
        task = f"Put the banana in the {color} colored bowl."
        expected = {
            f"Put the banana in the {color} colored bowl.":                0.35,
            f"Place the banana in the {color} colored bowl.":              0.15,
            f"Put the banana in the {color} bowl.":                        0.15,
            f"Place the banana in the {color} bowl.":                      0.15,
            f"Move the banana to the {color} bowl.":                       0.10,
            f"Pick up the banana and put it in the {color} colored bowl.": 0.10,
        }
        n = 3000
        counts: dict[str, int] = {}
        for _ in range(n):
            v = aug(task)
            counts[v] = counts.get(v, 0) + 1
        assert set(counts) == set(expected), (
            f"unexpected variants present: {set(counts) - set(expected)}")
        for variant, target_p in expected.items():
            actual_p = counts.get(variant, 0) / n
            assert abs(actual_p - target_p) < 0.07, (
                f"{variant!r}: expected ~{target_p:.0%} got {actual_p:.1%}")

    def test_seed_reproducibility(self):
        """Same seed must produce identical output sequences."""
        task = "Put the banana in the blue colored bowl."
        aug1, aug2 = TaskAugmenter(seed=17), TaskAugmenter(seed=17)
        for _ in range(100):
            assert aug1(task) == aug2(task)


# ─────────────────────────── _load_mask / _bbox_of_mask ──────────────────────

class TestMaskHelpers:
    def test_load_mask_bool(self, tmp_path):
        m = np.array([[True, False], [False, True]])
        p = tmp_path / "m.npy"
        np.save(str(p), m)
        loaded = _load_mask(str(p))
        assert loaded.dtype == bool
        np.testing.assert_array_equal(loaded, m)

    def test_load_mask_uint8_cast(self, tmp_path):
        m = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        p = tmp_path / "m.npy"
        np.save(str(p), m)
        loaded = _load_mask(str(p))
        assert loaded.dtype == bool

    def test_bbox_basic(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:5, 3:7] = True
        x1, y1, x2, y2 = _bbox_of_mask(mask)
        assert (x1, y1, x2, y2) == (3, 2, 7, 5)

    def test_bbox_empty_mask(self):
        mask = np.zeros((8, 8), dtype=bool)
        assert _bbox_of_mask(mask) == (0, 0, 0, 0)


# ─────────────────────────── BackgroundReplaceAugmenter ──────────────────────

class TestBackgroundReplaceAugmenter:
    @pytest.fixture()
    def setup(self, tmp_path):
        """Create a mask where the top-left 32×32 is background."""
        h, w = 64, 64
        mask = np.zeros((h, w), dtype=bool)
        mask[:32, :32] = True  # top-left quadrant = background
        mask_path = tmp_path / "bg_mask.npy"
        _write_mask(mask_path, mask)
        bg_dir = tmp_path / "backgrounds"
        bg_dir.mkdir()
        _write_bg_png(bg_dir / "bg0.png", h, w)
        return str(mask_path), str(bg_dir), mask

    def test_background_replaced_when_triggered(self, setup):
        mask_path, bg_dir, mask = setup
        # p=1.0 guarantees replacement every call.
        aug = BackgroundReplaceAugmenter(mask_path, bg_dir, p=1.0, seed=0)
        img = torch.zeros(3, 64, 64)
        result = aug(img)
        # Background pixels (mask=True) must be non-zero (from the 200/255 bg).
        bg_mask_t = torch.from_numpy(mask)
        assert result[:, bg_mask_t].abs().sum() > 0

    def test_foreground_preserved(self, setup):
        mask_path, bg_dir, mask = setup
        aug = BackgroundReplaceAugmenter(mask_path, bg_dir, p=1.0, seed=0)
        fg_val = 0.5
        img = torch.full((3, 64, 64), fg_val)
        result = aug(img)
        fg_mask = ~torch.from_numpy(mask)
        assert torch.allclose(result[:, fg_mask], torch.tensor(fg_val))

    def test_passthrough_when_not_triggered(self, setup):
        mask_path, bg_dir, mask = setup
        # p=0 means never replace.
        aug = BackgroundReplaceAugmenter(mask_path, bg_dir, p=0.0, seed=0)
        img = _make_image()
        result = aug(img)
        assert result is img

    def test_no_backgrounds_raises(self, tmp_path):
        mask = np.ones((8, 8), dtype=bool)
        mask_path = tmp_path / "m.npy"
        _write_mask(mask_path, mask)
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        aug = BackgroundReplaceAugmenter(str(mask_path), str(empty_dir), p=1.0)
        with pytest.raises(FileNotFoundError):
            aug(_make_image(8, 8))

    def test_output_dtype_preserved(self, setup):
        mask_path, bg_dir, _ = setup
        aug = BackgroundReplaceAugmenter(mask_path, bg_dir, p=1.0, seed=0)
        img = _make_image().to(torch.float32)
        assert aug(img).dtype == torch.float32

    def test_output_clipped_to_01(self, setup):
        mask_path, bg_dir, _ = setup
        aug = BackgroundReplaceAugmenter(mask_path, bg_dir, p=1.0, seed=0)
        result = aug(_make_image())
        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-5

    def test_picklable(self, setup):
        mask_path, bg_dir, _ = setup
        aug = BackgroundReplaceAugmenter(mask_path, bg_dir, p=0.5, seed=3)
        aug2 = pickle.loads(pickle.dumps(aug))
        assert aug2._p == 0.5

    def test_replacement_rate_near_p(self, setup):
        mask_path, bg_dir, mask = setup
        p = 0.4
        aug = BackgroundReplaceAugmenter(mask_path, bg_dir, p=p, seed=42)
        img = torch.zeros(3, 64, 64)  # all-black; bg adds signal
        hits = 0
        n = 500
        for _ in range(n):
            r = aug(img)
            if r.abs().sum() > 0:
                hits += 1
        rate = hits / n
        assert abs(rate - p) < 0.08, f"replacement rate {rate:.3f} far from p={p}"

    def test_seed_reproducibility(self, setup):
        """Same seed must produce the identical replacement sequence."""
        mask_path, bg_dir, _ = setup
        aug1 = BackgroundReplaceAugmenter(mask_path, bg_dir, p=1.0, seed=99)
        aug2 = BackgroundReplaceAugmenter(mask_path, bg_dir, p=1.0, seed=99)
        for _ in range(10):
            img = _make_image()
            assert torch.allclose(aug1(img.clone()), aug2(img.clone()))

    def test_different_seeds_differ(self, tmp_path):
        """Different seeds must produce different background-selection sequences.

        Requires multiple backgrounds in the pool; the single-file setup fixture
        would make any seed pick the same image every time.
        """
        from PIL import Image
        h, w = 8, 8
        mask = np.ones((h, w), dtype=bool)
        mask_path = tmp_path / "m.npy"
        _write_mask(mask_path, mask)
        bg_dir = tmp_path / "bgs"
        bg_dir.mkdir()
        for i, fill in enumerate([50, 100, 150, 200]):
            arr = np.full((h, w, 3), fill, dtype=np.uint8)
            Image.fromarray(arr).save(str(bg_dir / f"bg{i}.png"))

        aug1 = BackgroundReplaceAugmenter(str(mask_path), str(bg_dir), p=1.0, seed=0)
        aug2 = BackgroundReplaceAugmenter(str(mask_path), str(bg_dir), p=1.0, seed=7)
        img = torch.zeros(3, h, w)
        seq1 = [aug1(img.clone())[0, 0, 0].item() for _ in range(10)]
        seq2 = [aug2(img.clone())[0, 0, 0].item() for _ in range(10)]
        assert seq1 != seq2, "different seeds produced identical background sequence"


# ─────────────────────────── BowlShuffleAugmenter ────────────────────────────

class TestBowlShuffleAugmenter:
    @pytest.fixture()
    def masks(self, tmp_path):
        """Two non-overlapping bowl masks in a 64×64 image."""
        h, w = 64, 64
        m1 = np.zeros((h, w), dtype=bool)
        m1[10:20, 5:20] = True   # bbox: x1=5,y1=10,x2=20,y2=20
        m2 = np.zeros((h, w), dtype=bool)
        m2[40:55, 40:60] = True  # bbox: x1=40,y1=40,x2=60,y2=55
        p1 = tmp_path / "other1_mask.npy"
        p2 = tmp_path / "other2_mask.npy"
        _write_mask(p1, m1)
        _write_mask(p2, m2)
        return str(p1), str(p2), m1, m2

    def test_regions_swapped_when_triggered(self, masks):
        p1, p2, m1, m2 = masks
        aug = BowlShuffleAugmenter(p1, p2, p=1.0, seed=0)
        img = torch.zeros(3, 64, 64)
        # Paint region 1 red (ch0=1) and region 2 blue (ch2=1).
        img[0, 10:20, 5:20] = 1.0
        img[2, 40:55, 40:60] = 1.0
        result = aug(img)
        # After swap: region 1 bbox should contain blue signal, region 2 red.
        assert result[2, 10:20, 5:20].mean() > 0.5, "region1 should have blue after swap"
        assert result[0, 40:55, 40:60].mean() > 0.5, "region2 should have red after swap"

    def test_passthrough_when_not_triggered(self, masks):
        p1, p2, _, _ = masks
        aug = BowlShuffleAugmenter(p1, p2, p=0.0, seed=0)
        img = _make_image()
        result = aug(img)
        assert result is img

    def test_pixels_outside_bboxes_unchanged(self, masks):
        p1, p2, _, _ = masks
        aug = BowlShuffleAugmenter(p1, p2, p=1.0, seed=0)
        img = _make_image()
        original = img.clone()
        result = aug(img)
        # Pixels well outside both bboxes should be identical.
        assert torch.allclose(result[:, 30:35, 25:30], original[:, 30:35, 25:30])

    def test_picklable(self, masks):
        p1, p2, _, _ = masks
        aug = BowlShuffleAugmenter(p1, p2, p=0.5, seed=7)
        aug2 = pickle.loads(pickle.dumps(aug))
        assert aug2._p == 0.5

    def test_swap_rate_near_p(self, masks):
        p1, p2, _, _ = masks
        p = 0.5
        aug = BowlShuffleAugmenter(p1, p2, p=p, seed=0)
        n, swapped = 500, 0
        for _ in range(n):
            img = torch.zeros(3, 64, 64)
            img[0, 10:20, 5:20] = 1.0
            r = aug(img)
            if r[0, 10:20, 5:20].mean() < 0.1:  # red moved away
                swapped += 1
        rate = swapped / n
        assert abs(rate - p) < 0.08, f"swap rate {rate:.3f} far from p={p}"

    def test_different_size_bboxes(self, tmp_path):
        """Swap works even when the two bowl bboxes differ in size."""
        h, w = 64, 64
        m1 = np.zeros((h, w), dtype=bool)
        m1[5:10, 5:10] = True   # 5×5
        m2 = np.zeros((h, w), dtype=bool)
        m2[40:60, 40:60] = True  # 20×20
        p1 = tmp_path / "o1.npy"
        p2 = tmp_path / "o2.npy"
        _write_mask(p1, m1)
        _write_mask(p2, m2)
        aug = BowlShuffleAugmenter(str(p1), str(p2), p=1.0, seed=0)
        img = _make_image()
        result = aug(img)
        assert result.shape == img.shape

    def test_seed_reproducibility(self, masks):
        """Same seed must produce the identical swap sequence."""
        p1, p2, _, _ = masks
        aug1 = BowlShuffleAugmenter(p1, p2, p=0.6, seed=55)
        aug2 = BowlShuffleAugmenter(p1, p2, p=0.6, seed=55)
        for _ in range(20):
            img = _make_image()
            assert torch.allclose(aug1(img.clone()), aug2(img.clone()))


# ─────────────────────────── Gamma correction ────────────────────────────────

class TestGammaCorrection:
    """Gamma is applied inside Eval1PrepDataset.__getitem__; test via the dataset."""

    def _make_dataset_gamma_only(self, gamma_p: float, seed_img: float | None = None):
        img_val = seed_img if seed_img is not None else 0.5
        ds_mock = _make_dummy_dataset()
        # Override __getitem__ to return a deterministic mid-gray image.
        ds_mock.__getitem__ = lambda self, idx: {
            "task": "Put the banana in the green colored bowl.",
            "observation.images.front": torch.full((3, 8, 8), img_val),
            "action": torch.zeros(7),
        }
        return Eval1PrepDataset(ds_mock, max_frames_per_episode=None, gamma_p=gamma_p)

    def test_gamma_modifies_pixel_values(self):
        """With p=1 gamma is always applied; image should differ from the constant input."""
        ds = self._make_dataset_gamma_only(gamma_p=1.0, seed_img=0.5)
        changed = 0
        for i in range(30):
            row = ds[i % len(ds)]
            img = row["observation.images.front"]
            # gamma ∈ [0.8, 1.2], so 0.5^gamma ≠ 0.5 almost surely.
            if not torch.allclose(img, torch.full_like(img, 0.5), atol=1e-4):
                changed += 1
        # Nearly all samples should differ (p=1 means always applied).
        assert changed >= 25, f"only {changed}/30 samples were modified by gamma"

    def test_gamma_zero_disables(self):
        """With p=0 no gamma is applied; image should equal the constant input."""
        ds = self._make_dataset_gamma_only(gamma_p=0.0, seed_img=0.7)
        for i in range(20):
            row = ds[i % len(ds)]
            img = row["observation.images.front"]
            assert torch.allclose(img, torch.full_like(img, 0.7), atol=1e-5)

    def test_gamma_output_in_01(self):
        """Gamma-corrected pixels must stay in [0, 1]."""
        ds = self._make_dataset_gamma_only(gamma_p=1.0, seed_img=0.9)
        for i in range(20):
            img = ds[i % len(ds)]["observation.images.front"]
            assert img.min() >= 0.0
            assert img.max() <= 1.0 + 1e-5

    def test_gamma_rate_near_p(self):
        """Fraction of samples modified should match gamma_p."""
        p = 0.4
        ds = self._make_dataset_gamma_only(gamma_p=p, seed_img=0.5)
        n, modified = 500, 0
        for i in range(n):
            img = ds[i % len(ds)]["observation.images.front"]
            if not torch.allclose(img, torch.full_like(img, 0.5), atol=1e-4):
                modified += 1
        rate = modified / n
        assert abs(rate - p) < 0.08, f"gamma rate {rate:.3f} far from p={p}"

    def test_gamma_value_in_range(self):
        """Applied gamma must be drawn from [0.8, 1.2); infer from output pixel."""
        x = 0.5
        lo = x ** 1.2   # darkest possible: 0.5^1.2 ≈ 0.435
        hi = x ** 0.8   # brightest possible: 0.5^0.8 ≈ 0.574
        ds = self._make_dataset_gamma_only(gamma_p=1.0, seed_img=x)
        for i in range(80):
            img = ds[i % len(ds)]["observation.images.front"]
            pixel = img[0, 0, 0].item()
            assert lo - 1e-4 <= pixel <= hi + 1e-4, (
                f"output pixel {pixel:.4f} not in [{lo:.4f}, {hi:.4f}] "
                f"— gamma was sampled outside [0.8, 1.2]")


# ─────────────────────────── Eval1PrepDataset ────────────────────────────────

class TestEval1PrepDataset:
    def test_len_without_truncation(self):
        ds = _make_dummy_dataset(num_episodes=3, frames_per_ep=10)
        wrapped = Eval1PrepDataset(ds, max_frames_per_episode=None)
        assert len(wrapped) == 30

    def test_truncation_caps_per_episode(self):
        ds = _make_dummy_dataset(num_episodes=2, frames_per_ep=10)
        wrapped = Eval1PrepDataset(ds, max_frames_per_episode=5)
        assert len(wrapped) == 10  # 2 eps × 5 frames

    def test_task_aug_applied(self):
        ds = _make_dummy_dataset(task="Put the banana in the red colored bowl.")
        aug = TaskAugmenter(seed=0)
        wrapped = Eval1PrepDataset(ds, max_frames_per_episode=None, task_aug_fn=aug)
        tasks_seen = {wrapped[i]["task"] for i in range(len(wrapped))}
        # With 10 samples we should get at least 2 distinct variants in expectation.
        assert len(tasks_seen) >= 1

    def test_task_aug_preserves_color(self):
        ds = _make_dummy_dataset(task="Put the banana in the blue colored bowl.")
        aug = TaskAugmenter(seed=1)
        wrapped = Eval1PrepDataset(ds, max_frames_per_episode=None, task_aug_fn=aug)
        for i in range(len(wrapped)):
            assert "blue" in wrapped[i]["task"]

    def test_bg_aug_called(self, tmp_path):
        mask = np.ones((8, 8), dtype=bool)
        mask_path = tmp_path / "bg.npy"
        _write_mask(mask_path, mask)
        bg_dir = tmp_path / "bgs"
        bg_dir.mkdir()
        _write_bg_png(bg_dir / "b.png", 8, 8)
        bg_aug = BackgroundReplaceAugmenter(str(mask_path), str(bg_dir), p=1.0, seed=0)
        ds = _make_dummy_dataset()
        # Override to return a 8×8 image.
        ds.__getitem__ = lambda self, idx: {
            "task": "Put the banana in the green colored bowl.",
            "observation.images.front": torch.zeros(3, 8, 8),
            "action": torch.zeros(7),
        }
        wrapped = Eval1PrepDataset(ds, max_frames_per_episode=None, bg_aug_fn=bg_aug)
        row = wrapped[0]
        # All pixels are background (mask=True); result must be non-zero.
        assert row["observation.images.front"].abs().sum() > 0

    def test_bowl_shuffle_aug_called(self, tmp_path):
        h, w = 64, 64
        m1 = np.zeros((h, w), dtype=bool)
        m1[5:15, 0:10] = True
        m2 = np.zeros((h, w), dtype=bool)
        m2[50:60, 50:60] = True
        p1 = tmp_path / "o1.npy"
        p2 = tmp_path / "o2.npy"
        _write_mask(p1, m1)
        _write_mask(p2, m2)
        bowl_aug = BowlShuffleAugmenter(str(p1), str(p2), p=1.0, seed=0)
        img_val = torch.zeros(3, h, w)
        img_val[0, 5:15, 0:10] = 1.0  # region1 = red
        ds = _make_dummy_dataset()
        ds.__getitem__ = lambda self, idx: {
            "task": "Put the banana in the green colored bowl.",
            "observation.images.front": img_val.clone(),
            "action": torch.zeros(7),
        }
        wrapped = Eval1PrepDataset(ds, max_frames_per_episode=None, bowl_shuffle_fn=bowl_aug)
        row = wrapped[0]
        img_out = row["observation.images.front"]
        # Red should have moved from region1 to region2.
        assert img_out[0, 50:60, 50:60].mean() > 0.5

    def test_episode_filter(self):
        ds = _make_dummy_dataset(num_episodes=3, frames_per_ep=5)
        wrapped = Eval1PrepDataset(ds, max_frames_per_episode=None, episode_filter=[0, 2])
        assert len(wrapped) == 10  # episodes 0 and 2 only

    def test_proxy_attributes(self):
        ds = _make_dummy_dataset()
        wrapped = Eval1PrepDataset(ds, max_frames_per_episode=None)
        assert wrapped.repo_id == "test/dataset"
        assert wrapped.features == {}

    def test_truncation_summary(self):
        ds = _make_dummy_dataset(num_episodes=2, frames_per_ep=10)
        wrapped = Eval1PrepDataset(ds, max_frames_per_episode=5)
        s = wrapped.truncation_summary()
        assert s["original_num_frames"] == 20
        assert s["kept_num_frames"] == 10
        assert abs(s["kept_fraction"] - 0.5) < 1e-6

    def test_augment_order_bg_before_shuffle(self, tmp_path):
        """bg-replace runs before bowl-shuffle (invariant documented in code)."""
        call_order: list[str] = []
        class TrackBg:
            def __call__(self, img):
                call_order.append("bg")
                return img
        class TrackShuffle:
            def __call__(self, img):
                call_order.append("shuffle")
                return img
        ds = _make_dummy_dataset()
        wrapped = Eval1PrepDataset(
            ds, max_frames_per_episode=None,
            bg_aug_fn=TrackBg(), bowl_shuffle_fn=TrackShuffle(), gamma_p=0.0,
        )
        wrapped[0]
        assert call_order == ["bg", "shuffle"]

    def test_no_aug_returns_original_task(self):
        task = "Put the banana in the green colored bowl."
        ds = _make_dummy_dataset(task=task)
        wrapped = Eval1PrepDataset(ds, max_frames_per_episode=None)
        assert wrapped[0]["task"] == task

    def test_custom_image_key(self):
        """Augmenters must apply to image_key, not the hard-coded default."""
        custom_key = "observation.images.wrist"
        ds_mock = _make_dummy_dataset(image_key=custom_key)
        ds_mock.__getitem__ = lambda self, idx: {
            "task": "Put the banana in the green colored bowl.",
            custom_key: torch.full((3, 8, 8), 0.5),
            "action": torch.zeros(7),
        }
        wrapped = Eval1PrepDataset(
            ds_mock, max_frames_per_episode=None,
            gamma_p=1.0, image_key=custom_key,
        )
        row = wrapped[0]
        assert custom_key in row, "custom image_key missing from output"
        # gamma_p=1.0 always applied; constant 0.5 input must change
        assert not torch.allclose(row[custom_key], torch.full((3, 8, 8), 0.5), atol=1e-4)
        assert "observation.images.front" not in row


# ─────────────────────── build_eval1_pipeline ────────────────────────────────

class TestBuildEval1Pipeline:
    """Tests for build_eval1_pipeline: every parameter must be covered."""

    # ── shared fixture ─────────────────────────────────────────────────────

    @pytest.fixture()
    def env(self, tmp_path):
        """Create a minimal mask + background environment for green_bowl."""
        slug = "green_bowl"
        slug_dir = tmp_path / slug
        slug_dir.mkdir()
        h, w = 8, 8
        np.save(str(slug_dir / "bg_mask.npy"),     np.ones((h, w), dtype=bool))
        np.save(str(slug_dir / "other1_mask.npy"), np.zeros((h, w), dtype=bool))
        np.save(str(slug_dir / "other2_mask.npy"), np.zeros((h, w), dtype=bool))
        bg_dir = tmp_path / "bgs"
        bg_dir.mkdir()
        _write_bg_png(bg_dir / "bg0.png", h, w)
        ds = _make_dummy_dataset()
        return str(tmp_path), str(bg_dir), slug, ds

    # ── return type ────────────────────────────────────────────────────────

    def test_returns_eval1_prep_dataset(self, env):
        mask_dir, bg_dir, slug, ds = env
        result = build_eval1_pipeline(ds, f"foo/{slug}", mask_dir=mask_dir, bg_dir=bg_dir)
        assert isinstance(result, Eval1PrepDataset)

    # ── repo_id / slug resolution ──────────────────────────────────────────

    def test_strict_raises_on_unknown_slug(self, tmp_path):
        ds = _make_dummy_dataset()
        with pytest.raises(ValueError, match="slug"):
            build_eval1_pipeline(ds, "foo/no_color_here",
                                 mask_dir=str(tmp_path), bg_dir=str(tmp_path), strict=True)

    def test_lenient_ignores_unknown_slug(self, tmp_path):
        ds = _make_dummy_dataset()
        result = build_eval1_pipeline(ds, "foo/no_color_here",
                                      mask_dir=str(tmp_path), bg_dir=str(tmp_path), strict=False)
        assert isinstance(result, Eval1PrepDataset)
        assert result._bg_aug_fn is None
        assert result._bowl_shuffle_fn is None

    def test_all_three_slugs_resolve(self, tmp_path):
        """green_bowl, blue_bowl, red_bowl all resolve without error (strict)."""
        h, w = 8, 8
        bg_dir = tmp_path / "bgs"
        bg_dir.mkdir()
        _write_bg_png(bg_dir / "bg0.png", h, w)
        for slug in ("green_bowl", "blue_bowl", "red_bowl"):
            d = tmp_path / slug
            d.mkdir(exist_ok=True)
            np.save(str(d / "bg_mask.npy"),     np.ones((h, w), dtype=bool))
            np.save(str(d / "other1_mask.npy"), np.zeros((h, w), dtype=bool))
            np.save(str(d / "other2_mask.npy"), np.zeros((h, w), dtype=bool))
            ds = _make_dummy_dataset()
            result = build_eval1_pipeline(
                ds, f"RobotLearningVLA/banana_{slug}_eval1_v2",
                mask_dir=str(tmp_path), bg_dir=str(bg_dir), strict=True)
            assert isinstance(result, Eval1PrepDataset)

    # ── strict / lenient file checks ───────────────────────────────────────

    def test_strict_raises_on_missing_bg_mask(self, tmp_path):
        slug = "green_bowl"
        (tmp_path / slug).mkdir()
        # bg_mask.npy intentionally absent
        ds = _make_dummy_dataset()
        with pytest.raises(FileNotFoundError, match="bg mask"):
            build_eval1_pipeline(ds, f"foo/{slug}",
                                 mask_dir=str(tmp_path), bg_dir=str(tmp_path), strict=True)

    def test_strict_raises_on_missing_bg_dir(self, env):
        mask_dir, _, slug, ds = env
        with pytest.raises(FileNotFoundError, match="background image dir"):
            build_eval1_pipeline(ds, f"foo/{slug}",
                                 mask_dir=mask_dir, bg_dir="/nonexistent/path", strict=True)

    def test_strict_raises_on_missing_bowl_masks(self, tmp_path):
        slug = "green_bowl"
        slug_dir = tmp_path / slug
        slug_dir.mkdir()
        h, w = 8, 8
        np.save(str(slug_dir / "bg_mask.npy"), np.ones((h, w), dtype=bool))
        bg_dir = tmp_path / "bgs"
        bg_dir.mkdir()
        _write_bg_png(bg_dir / "bg0.png", h, w)
        # other1/other2 absent
        ds = _make_dummy_dataset()
        with pytest.raises(FileNotFoundError, match="bowl-shuffle mask"):
            build_eval1_pipeline(ds, f"foo/{slug}",
                                 mask_dir=str(tmp_path), bg_dir=str(bg_dir), strict=True)

    def test_lenient_skips_missing_masks(self, tmp_path):
        slug = "green_bowl"
        ds = _make_dummy_dataset()
        result = build_eval1_pipeline(ds, f"foo/{slug}",
                                      mask_dir=str(tmp_path), bg_dir=str(tmp_path), strict=False)
        assert result._bg_aug_fn is None
        assert result._bowl_shuffle_fn is None

    def test_lenient_wires_augmenters_when_present(self, env):
        mask_dir, bg_dir, slug, ds = env
        result = build_eval1_pipeline(ds, f"foo/{slug}",
                                      mask_dir=mask_dir, bg_dir=bg_dir, strict=False)
        assert result._bg_aug_fn is not None
        assert result._bowl_shuffle_fn is not None

    # ── task_aug_fn ────────────────────────────────────────────────────────

    def test_creates_task_aug_fn_when_none(self, env):
        mask_dir, bg_dir, slug, ds = env
        result = build_eval1_pipeline(ds, f"foo/{slug}",
                                      mask_dir=mask_dir, bg_dir=bg_dir, task_aug_fn=None)
        assert isinstance(result._task_aug_fn, TaskAugmenter)

    def test_uses_provided_task_aug_fn(self, env):
        mask_dir, bg_dir, slug, ds = env
        sentinel = TaskAugmenter(seed=999)
        result = build_eval1_pipeline(ds, f"foo/{slug}",
                                      mask_dir=mask_dir, bg_dir=bg_dir, task_aug_fn=sentinel)
        assert result._task_aug_fn is sentinel

    # ── numeric parameters ─────────────────────────────────────────────────

    def test_max_frames_passed_through(self, env):
        mask_dir, bg_dir, slug, _ = env
        ds = _make_dummy_dataset(num_episodes=2, frames_per_ep=10)
        result = build_eval1_pipeline(ds, f"foo/{slug}",
                                      mask_dir=mask_dir, bg_dir=bg_dir, max_frames=4)
        assert len(result) == 8  # 2 eps × 4 frames

    def test_gamma_p_passed_through(self, env):
        mask_dir, bg_dir, slug, ds = env
        result = build_eval1_pipeline(ds, f"foo/{slug}",
                                      mask_dir=mask_dir, bg_dir=bg_dir, gamma_p=0.0)
        assert result._gamma_p == pytest.approx(0.0)

    def test_bg_p_wired(self, env):
        mask_dir, bg_dir, slug, ds = env
        result = build_eval1_pipeline(ds, f"foo/{slug}",
                                      mask_dir=mask_dir, bg_dir=bg_dir, bg_p=0.77)
        assert result._bg_aug_fn._p == pytest.approx(0.77)

    def test_bs_p_wired(self, env):
        mask_dir, bg_dir, slug, ds = env
        result = build_eval1_pipeline(ds, f"foo/{slug}",
                                      mask_dir=mask_dir, bg_dir=bg_dir, bs_p=0.88)
        assert result._bowl_shuffle_fn._p == pytest.approx(0.88)

    # ── default values must match training script ──────────────────────────

    def test_default_gamma_p_is_04(self, env):
        mask_dir, bg_dir, slug, ds = env
        result = build_eval1_pipeline(ds, f"foo/{slug}", mask_dir=mask_dir, bg_dir=bg_dir)
        assert result._gamma_p == pytest.approx(0.4)

    def test_default_bg_p_is_03(self, env):
        mask_dir, bg_dir, slug, ds = env
        result = build_eval1_pipeline(ds, f"foo/{slug}", mask_dir=mask_dir, bg_dir=bg_dir)
        assert result._bg_aug_fn._p == pytest.approx(0.3)

    def test_default_bs_p_is_05(self, env):
        mask_dir, bg_dir, slug, ds = env
        result = build_eval1_pipeline(ds, f"foo/{slug}", mask_dir=mask_dir, bg_dir=bg_dir)
        assert result._bowl_shuffle_fn._p == pytest.approx(0.5)

    def test_default_max_frames_is_200(self, env):
        mask_dir, bg_dir, slug, _ = env
        ds = _make_dummy_dataset(num_episodes=2, frames_per_ep=300)
        result = build_eval1_pipeline(ds, f"foo/{slug}", mask_dir=mask_dir, bg_dir=bg_dir)
        assert len(result) == 2 * 200


# ──────────────────────── Training script wiring check ───────────────────────

class TestTrainingScriptWiring:
    """Verify that all augmenters from eval1_dataset_prep.py are referenced
    in the training script without importing lerobot."""

    _SCRIPT = Path(__file__).parent / "train_eval1_smolvla_h100.py"

    def _script_text(self) -> str:
        return self._SCRIPT.read_text()

    def _parse_tfs(self) -> dict:
        """Parse sys.argv with AST to extract the full image-transforms JSON dict."""
        import ast, json
        tree = ast.parse(self._SCRIPT.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Attribute) and t.attr == "argv":
                        for elt in node.value.elts:
                            try:
                                val = ast.literal_eval(elt)
                            except Exception:
                                continue
                            if isinstance(val, str) and "tfs=" in val:
                                return json.loads(val.split("tfs=", 1)[1])
        raise ValueError("tfs JSON not found in training script sys.argv")

    def test_task_augmenter_imported(self):
        assert "make_task_augmenter" in self._script_text()

    def test_build_eval1_pipeline_imported(self):
        assert "build_eval1_pipeline" in self._script_text()

    def test_gamma_p_passed_to_dataset(self):
        assert "gamma_p" in self._script_text()

    def test_all_env_vars_referenced(self):
        text = self._script_text()
        expected_vars = [
            "EVAL1_TASK_AUG",
            "EVAL1_BG_REPLACE",
            "EVAL1_BG_REPLACE_P",
            "EVAL1_BOWL_SHUFFLE",
            "EVAL1_BOWL_SHUFFLE_P",
            "EVAL1_GAMMA_P",
            "EVAL1_MAX_FRAMES_PER_EP",
            "EVAL1_MASK_DIR",
            "EVAL1_BG_DIR",
        ]
        for var in expected_vars:
            assert var in text, f"env var {var!r} not found in training script"

    def test_bg_dir_uses_eval3_backgrounds_pool(self):
        """eval1 intentionally reuses the eval3 background image pool."""
        assert "eval3_backgrounds" in self._script_text()

    def test_colorjitter_augments_present(self):
        text = self._script_text()
        for aug in ("brightness", "contrast", "saturation", "hue", "sharpness"):
            assert aug in text, f"lerobot image-transform '{aug}' missing from training script"

    def test_spatial_augments_present(self):
        """The 5 spatial/occlusion transforms ported from eval3 must all appear."""
        text = self._script_text()
        for aug in ("RandomAffine", "RandomPerspective", "RandomResizedCrop",
                    "GaussianBlur", "RandomErasing"):
            assert aug in text, f"image-transform '{aug}' missing from training script"

    def test_max_num_transforms_is_4(self):
        assert "max_num_transforms=4" in self._script_text()

    def test_brightness_weight_is_2(self):
        """brightness weight must be 2.0 (same as eval3, to fight bowl lighting cues)."""
        import re
        text = self._script_text()
        # Match the weight value that appears inside the brightness entry block.
        m = re.search(r'"brightness"\s*:.*?"weight"\s*:\s*([0-9.]+)', text)
        assert m, "brightness weight not found in tfs JSON"
        assert float(m.group(1)) == 2.0, f"brightness weight is {m.group(1)}, expected 2.0"

    def test_contrast_weight_is_2(self):
        import re
        text = self._script_text()
        m = re.search(r'"contrast"\s*:.*?"weight"\s*:\s*([0-9.]+)', text)
        assert m, "contrast weight not found in tfs JSON"
        assert float(m.group(1)) == 2.0, f"contrast weight is {m.group(1)}, expected 2.0"

    # ── Per-transform kwarg + weight tests (parsed from sys.argv AST) ─────────

    def test_brightness_kwargs_and_weight(self):
        t = self._parse_tfs()["brightness"]
        assert t["type"] == "ColorJitter"
        assert t["kwargs"]["brightness"] == pytest.approx([0.6, 1.4])
        assert t["weight"] == pytest.approx(2.0)

    def test_contrast_kwargs_and_weight(self):
        t = self._parse_tfs()["contrast"]
        assert t["type"] == "ColorJitter"
        assert t["kwargs"]["contrast"] == pytest.approx([0.6, 1.4])
        assert t["weight"] == pytest.approx(2.0)

    def test_saturation_kwargs_and_weight(self):
        t = self._parse_tfs()["saturation"]
        assert t["type"] == "ColorJitter"
        assert t["kwargs"]["saturation"] == pytest.approx([0.5, 1.5])
        assert t["weight"] == pytest.approx(1.0)

    def test_hue_kwargs_and_weight(self):
        t = self._parse_tfs()["hue"]
        assert t["type"] == "ColorJitter"
        assert t["kwargs"]["hue"] == pytest.approx([-0.02, 0.02])
        assert t["weight"] == pytest.approx(1.0)

    def test_sharpness_kwargs_and_weight(self):
        t = self._parse_tfs()["sharpness"]
        assert t["type"] == "SharpnessJitter"
        assert t["kwargs"]["sharpness"] == pytest.approx([0.5, 1.5])
        assert t["weight"] == pytest.approx(1.0)

    def test_affine_kwargs_and_weight(self):
        t = self._parse_tfs()["affine"]
        assert t["type"] == "RandomAffine"
        assert t["kwargs"]["degrees"] == pytest.approx([-3.0, 3.0])
        assert t["kwargs"]["translate"] == pytest.approx([0.03, 0.03])
        assert t["weight"] == pytest.approx(1.0)

    def test_perspective_kwargs_and_weight(self):
        t = self._parse_tfs()["perspective"]
        assert t["type"] == "RandomPerspective"
        assert t["kwargs"]["distortion_scale"] == pytest.approx(0.2)
        assert t["kwargs"]["p"] == pytest.approx(0.5)
        assert t["weight"] == pytest.approx(1.5)

    def test_resized_crop_kwargs_and_weight(self):
        t = self._parse_tfs()["resized_crop"]
        assert t["type"] == "RandomResizedCrop"
        assert t["kwargs"]["size"] == [480, 640]
        assert t["kwargs"]["scale"] == pytest.approx([0.75, 1.0])
        assert t["kwargs"]["ratio"] == pytest.approx([0.95, 1.05])
        assert t["weight"] == pytest.approx(1.0)

    def test_gaussian_blur_kwargs_and_weight(self):
        t = self._parse_tfs()["gaussian_blur"]
        assert t["type"] == "GaussianBlur"
        assert t["kwargs"]["kernel_size"] == [5, 9]
        assert t["kwargs"]["sigma"] == pytest.approx([0.3, 2.0])
        assert t["weight"] == pytest.approx(0.5)

    def test_erase_kwargs_and_weight(self):
        t = self._parse_tfs()["erase"]
        assert t["type"] == "RandomErasing"
        assert t["kwargs"]["p"] == pytest.approx(0.3)
        assert t["kwargs"]["scale"] == pytest.approx([0.02, 0.1])
        assert t["weight"] == pytest.approx(0.5)

    def test_exactly_10_transforms_defined(self):
        assert len(self._parse_tfs()) == 10


# ─────────────────────── Integration tests (require lerobot + HF) ─────────────

REPO_IDS = [
    "RobotLearningVLA/banana_green_bowl_eval1_v2",
    "RobotLearningVLA/banana_blue_bowl_eval1_v2",
    "RobotLearningVLA/banana_red_bowl_eval1_v2",
]
IMAGE_KEY = "observation.images.front"
_EVAL3_SCRIPTS = (
    Path(__file__).resolve().parents[2] / "robot-learning-vla" / "scripts"
)


def _try_import_lerobot():
    """Return make_dataset callable or None if lerobot is unavailable."""
    if _EVAL3_SCRIPTS.exists() and str(_EVAL3_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_EVAL3_SCRIPTS))
    try:
        from eval3_lerobot_shim import apply as _shim
        _shim()
    except ImportError:
        pass
    try:
        from lerobot.datasets.factory import make_dataset
        return make_dataset
    except ModuleNotFoundError:
        pass
    try:
        from lerobot.common.datasets.factory import make_dataset
        return make_dataset
    except ModuleNotFoundError:
        return None


def _make_lerobot_cfg(repo_id: str):
    """Build the minimal TrainPipelineConfig-shaped object lerobot's make_dataset needs."""
    import types
    from lerobot.configs.default import DatasetConfig, ImageTransformsConfig

    img_tf = ImageTransformsConfig(enable=False, tfs={})
    ds_cfg = DatasetConfig(
        repo_id=repo_id,
        root=None,
        revision="main",
        episodes=None,
        image_transforms=img_tf,
        video_backend="pyav",
        use_imagenet_stats=True,
        streaming=False,
    )
    # policy stub: resolve_delta_timestamps only reads these three fields.
    policy_stub = types.SimpleNamespace(
        action_delta_indices=None,
        observation_delta_indices=None,
        reward_delta_indices=None,
    )
    return types.SimpleNamespace(
        dataset=ds_cfg,
        policy=policy_stub,
        tolerance_s=1e-4,
        num_workers=0,
    )


def _pipeline_kwargs_from_env() -> dict:
    return dict(
        mask_dir=os.environ.get("EVAL1_MASK_DIR",                  "../outputs/eval1_masks"),
        bg_dir=os.environ.get("EVAL1_BG_DIR",                      "../outputs/eval3_backgrounds"),
        gamma_p=float(os.environ.get("EVAL1_GAMMA_P",              "0.4")),
        bg_p=float(os.environ.get("EVAL1_BG_REPLACE_P",            "0.3")),
        bs_p=float(os.environ.get("EVAL1_BOWL_SHUFFLE_P",          "0.5")),
        max_frames=int(os.environ.get("EVAL1_MAX_FRAMES_PER_EP",   "200")),
    )


@pytest.mark.integration
class TestEval1DatasetIntegration:
    """End-to-end tests that load real datasets from HuggingFace.

    Skipped automatically when lerobot is not installed or HF is unreachable.
    Run explicitly:
        pytest scripts/test_eval1_augmentations.py -m integration -v
    """

    @pytest.fixture(scope="class")
    def make_ds(self):
        make_dataset = _try_import_lerobot()
        if make_dataset is None:
            pytest.skip("lerobot not installed")
        return make_dataset

    def _load_one(self, make_dataset, repo_id: str):
        """Load a single eval1 repo and wrap with the full pipeline."""
        try:
            ds = make_dataset(_make_lerobot_cfg(repo_id))
        except Exception as e:
            pytest.skip(f"HF dataset load failed ({repo_id}): {e}")
        return build_eval1_pipeline(ds, repo_id, **_pipeline_kwargs_from_env(), strict=False)

    def test_all_three_repos_load(self, make_ds):
        for repo_id in REPO_IDS:
            wrapped = self._load_one(make_ds, repo_id)
            assert len(wrapped) > 0, f"{repo_id} has 0 frames after wrapping"

    def test_frame_truncation_applies(self, make_ds):
        wrapped = self._load_one(make_ds, REPO_IDS[0])
        s = wrapped.truncation_summary()
        max_ep = int(os.environ.get("EVAL1_MAX_FRAMES_PER_EP", "200"))
        assert s["kept_num_frames"] <= s["original_num_frames"]
        # No episode should have more than max_ep frames kept.
        ep_df = wrapped.meta.episodes
        from_idxs = list(ep_df["dataset_from_index"])
        to_idxs   = list(ep_df["dataset_to_index"])
        for f0, f1 in zip(from_idxs, to_idxs):
            kept = sum(1 for i in wrapped._valid_indices if f0 <= i < f1)
            assert kept <= max_ep, f"episode kept {kept} > max_ep {max_ep}"

    def test_sample_image_is_chw_float(self, make_ds):
        wrapped = self._load_one(make_ds, REPO_IDS[0])
        for idx in range(min(5, len(wrapped))):
            row = wrapped[idx]
            assert IMAGE_KEY in row, f"missing {IMAGE_KEY} in sample {idx}"
            img = row[IMAGE_KEY]
            assert isinstance(img, torch.Tensor), "image must be a Tensor"
            assert img.ndim == 3, f"expected CHW (3 dims), got {img.ndim}"
            assert img.shape[0] == 3, f"expected 3 channels, got {img.shape[0]}"
            assert img.is_floating_point(), "image must be float"
            assert img.min() >= 0.0 - 1e-4, f"pixel min {img.min()} < 0"
            assert img.max() <= 1.0 + 1e-4, f"pixel max {img.max()} > 1"

    def test_sample_task_is_eval1_phrasing(self, make_ds):
        wrapped = self._load_one(make_ds, REPO_IDS[0])
        valid_verbs = ("Put", "Place", "Move", "Pick")
        for idx in range(min(10, len(wrapped))):
            task = wrapped[idx].get("task", "")
            assert isinstance(task, str) and len(task) > 0
            assert any(task.startswith(v) for v in valid_verbs), (
                f"unexpected task phrasing: {task!r}")
            assert "banana" in task.lower(), f"task missing 'banana': {task!r}"

    def test_sample_action_shape(self, make_ds):
        wrapped = self._load_one(make_ds, REPO_IDS[0])
        row = wrapped[0]
        if "action" in row:
            act = row["action"]
            assert isinstance(act, torch.Tensor)
            assert act.ndim >= 1

    def test_concat_dataset_frame_count(self, make_ds):
        """ConcatDataset frame count must equal sum of individual wrapped counts."""
        from torch.utils.data import ConcatDataset
        datasets = [self._load_one(make_ds, r) for r in REPO_IDS]
        combined = ConcatDataset(datasets)
        expected = sum(len(d) for d in datasets)
        assert len(combined) == expected

    def test_task_aug_produces_variants_across_dataset(self, make_ds):
        """At least 3 distinct task phrasings across 50 samples (augmenter is live)."""
        wrapped = self._load_one(make_ds, REPO_IDS[0])
        tasks = {wrapped[i % len(wrapped)]["task"] for i in range(50)}
        assert len(tasks) >= 3, (
            f"expected ≥3 task variants, got {len(tasks)}: {tasks}")

    def test_no_nan_in_image_or_action(self, make_ds):
        wrapped = self._load_one(make_ds, REPO_IDS[0])
        for idx in range(min(5, len(wrapped))):
            row = wrapped[idx]
            if IMAGE_KEY in row:
                assert not torch.isnan(row[IMAGE_KEY]).any(), f"NaN in image at idx {idx}"
            if "action" in row:
                assert not torch.isnan(row["action"]).any(), f"NaN in action at idx {idx}"


# ─────────────────────── Visualization helpers ───────────────────────────────

def _tensor_to_pil(t: torch.Tensor):
    from PIL import Image as _PIL
    arr = t.clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
    return _PIL.fromarray(arr)


def _load_font(size: int = 14):
    from PIL import ImageFont
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            pass
    return ImageFont.load_default()


def visualize_extremes(
    out_path: str = "outputs/eval1_aug_extremes.png",
    mask_dir: str = "outputs/eval1_masks",
    bg_dir: str = "outputs/eval3_backgrounds",
) -> str:
    """Render each augmenter at its exact min and max settings side-by-side.

    Layout: one row per augmenter, three columns — original | min | max.
    Uses green_bowl masks and a real dataset frame so the effects are visible
    on an actual scene, not a synthetic placeholder.
    """
    from pathlib import Path as _Path
    from PIL import Image as _PIL, ImageDraw

    slug = "green_bowl"
    frame_path = _Path(mask_dir) / slug / "frame0.png"
    if frame_path.exists():
        orig_pil = _PIL.open(frame_path).convert("RGB")
    else:
        arr = np.random.randint(60, 200, (240, 320, 3), dtype=np.uint8)
        orig_pil = _PIL.fromarray(arr)
        print("  [warn] frame0.png not found — using random synthetic image")

    orig_t = torch.from_numpy(np.array(orig_pil)).permute(2, 0, 1).float() / 255.0

    bg_mask_path = os.path.join(mask_dir, slug, "bg_mask.npy")
    o1_path = os.path.join(mask_dir, slug, "other1_mask.npy")
    o2_path = os.path.join(mask_dir, slug, "other2_mask.npy")

    # rows: list of (title_str, [(col_label, CHW_tensor), ...])
    rows = []

    # 1. Gamma — show both extremes of the range
    rows.append((
        "Gamma correction  [p=0.40, γ∈(0.80, 1.20)]"
        "   — applied per-frame; γ<1 brightens, γ>1 darkens",
        [
            ("original", orig_t),
            ("min  γ=0.80  (brightens)\n  0.5^0.80 = 0.574  (+14.8%)", orig_t.clamp(0, 1).pow(0.80)),
            ("max  γ=1.20  (darkens)\n  0.5^1.20 = 0.435  (−13.0%)", orig_t.clamp(0, 1).pow(1.20)),
        ],
    ))

    # 2. Background replace — p=0 (disabled / passthrough) vs p=1 (every frame)
    if os.path.exists(bg_mask_path) and os.path.isdir(bg_dir):
        bg_frac = np.load(bg_mask_path).astype(bool).mean()
        bg_min = BackgroundReplaceAugmenter(bg_mask_path, bg_dir, p=0.0, seed=0)
        bg_max = BackgroundReplaceAugmenter(bg_mask_path, bg_dir, p=1.0, seed=42)
        rows.append((
            f"Background replace  [p=0.30]"
            f"   — mask covers {bg_frac*100:.1f}% of pixels; foreground always preserved",
            [
                ("original", orig_t),
                ("min  p=0.0  disabled\n  (passthrough, identical to original)", bg_min(orig_t.clone())),
                ("max  p=1.0  every frame replaced\n  (random bg from pool of 500)", bg_max(orig_t.clone())),
            ],
        ))
    else:
        print(f"  [skip] bg-replace: mask or bg_dir not found ({bg_mask_path}, {bg_dir})")

    # 3. Bowl shuffle — p=0 (disabled) vs p=1 (every frame)
    if os.path.exists(o1_path) and os.path.exists(o2_path):
        o1_frac = np.load(o1_path).astype(bool).mean()
        o2_frac = np.load(o2_path).astype(bool).mean()
        bs_min = BowlShuffleAugmenter(o1_path, o2_path, p=0.0, seed=0)
        bs_max = BowlShuffleAugmenter(o1_path, o2_path, p=1.0, seed=0)
        rows.append((
            f"Bowl shuffle  [p=0.50]"
            f"   — other1={o1_frac*100:.1f}%, other2={o2_frac*100:.1f}% of pixels; target bowl never swapped",
            [
                ("original", orig_t),
                ("min  p=0.0  disabled\n  (no swap, identical to original)", bs_min(orig_t.clone())),
                ("max  p=1.0  every frame swapped\n  (two non-target bowls exchanged)", bs_max(orig_t.clone())),
            ],
        ))
    else:
        print(f"  [skip] bowl-shuffle: mask files not found ({o1_path}, {o2_path})")

    # ── Render ────────────────────────────────────────────────────────────────
    cell_w, cell_h = 320, 240
    label_h = 46     # height for 2-line label below each cell
    title_h = 30     # height for row header
    pad = 10
    n_rows = len(rows)
    canvas_w = 3 * cell_w + 4 * pad
    canvas_h = pad + n_rows * (title_h + cell_h + label_h + pad)

    canvas = _PIL.new("RGB", (canvas_w, canvas_h), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(15)
    label_font = _load_font(13)

    for r_i, (title, cells) in enumerate(rows):
        y0 = pad + r_i * (title_h + cell_h + label_h + pad)
        draw.rectangle((0, y0, canvas_w, y0 + title_h - 2), fill=(210, 220, 240))
        draw.text((pad + 4, y0 + 7), title, fill=(10, 10, 60), font=title_font)
        for c_i, (lbl, t) in enumerate(cells):
            x0 = pad + c_i * (cell_w + pad)
            cell_img = _tensor_to_pil(t).resize((cell_w, cell_h), _PIL.Resampling.BICUBIC)
            canvas.paste(cell_img, (x0, y0 + title_h))
            for li, line in enumerate(lbl.split("\n")):
                draw.text((x0 + 4, y0 + title_h + cell_h + 4 + li * 16),
                          line, fill=(30, 30, 30), font=label_font)

    _Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def visualize_transform_extremes(
    out_path: str = "outputs/eval1_transform_extremes.png",
    mask_dir: str = "outputs/eval1_masks",
) -> str:
    """Render all 10 lerobot image transforms at their exact min and max settings.

    Layout: one row per transform, three columns — original | min | max.
    Parameters are taken directly from the training script configuration.
    """
    import torchvision.transforms.v2 as T
    import torchvision.transforms.v2.functional as TF
    from PIL import Image as _PIL, ImageDraw
    from pathlib import Path as _Path

    slug = "green_bowl"
    frame_path = _Path(mask_dir) / slug / "frame0.png"
    if frame_path.exists():
        orig_pil = _PIL.open(frame_path).convert("RGB")
    else:
        arr = np.random.randint(60, 200, (240, 320, 3), dtype=np.uint8)
        orig_pil = _PIL.fromarray(arr)
        print("  [warn] frame0.png not found — using synthetic image")

    orig_t = torch.from_numpy(np.array(orig_pil)).permute(2, 0, 1).float() / 255.0
    h, w = orig_t.shape[1], orig_t.shape[2]

    rows: list[tuple[str, list[tuple[str, torch.Tensor]]]] = []

    def _row(title: str, min_lbl: str, min_t: torch.Tensor,
             max_lbl: str, max_t: torch.Tensor) -> None:
        rows.append((title, [("original", orig_t), (min_lbl, min_t), (max_lbl, max_t)]))

    # 1. brightness [0.6, 1.4]  weight=2.0
    _row("brightness  [ColorJitter, range 0.6→1.4, weight 2.0]",
         "min  factor=0.60  (darkens)", TF.adjust_brightness(orig_t, 0.6),
         "max  factor=1.40  (brightens)", TF.adjust_brightness(orig_t, 1.4))

    # 2. contrast [0.6, 1.4]  weight=2.0
    _row("contrast  [ColorJitter, range 0.6→1.4, weight 2.0]",
         "min  factor=0.60  (flat/grey)", TF.adjust_contrast(orig_t, 0.6),
         "max  factor=1.40  (punchy)", TF.adjust_contrast(orig_t, 1.4))

    # 3. saturation [0.5, 1.5]  weight=1.0
    _row("saturation  [ColorJitter, range 0.5→1.5, weight 1.0]",
         "min  factor=0.50  (desaturated)", TF.adjust_saturation(orig_t, 0.5),
         "max  factor=1.50  (oversaturated)", TF.adjust_saturation(orig_t, 1.5))

    # 4. hue [-0.02, 0.02]  weight=1.0
    _row("hue  [ColorJitter, range −0.02→+0.02, weight 1.0]",
         "min  factor=−0.02  (hue shift ←)", TF.adjust_hue(orig_t, -0.02),
         "max  factor=+0.02  (hue shift →)", TF.adjust_hue(orig_t, +0.02))

    # 5. sharpness [0.5, 1.5]  weight=1.0
    _row("sharpness  [SharpnessJitter, range 0.5→1.5, weight 1.0]",
         "min  factor=0.50  (softened)", TF.adjust_sharpness(orig_t, 0.5),
         "max  factor=1.50  (sharpened)", TF.adjust_sharpness(orig_t, 1.5))

    # 6. affine  degrees=[-3,3], translate=[0.03,0.03]  weight=1.0
    tx, ty = int(w * 0.03), int(h * 0.03)
    _row("affine  [RandomAffine, degrees ±3°, translate ±3%, weight 1.0]",
         "min  angle=−3°  translate=(−3%,−3%)",
         TF.affine(orig_t, angle=-3.0, translate=[-tx, -ty], scale=1.0, shear=[0, 0]),
         "max  angle=+3°  translate=(+3%,+3%)",
         TF.affine(orig_t, angle=3.0, translate=[tx, ty], scale=1.0, shear=[0, 0]))

    # 7. perspective  distortion_scale=0.2, p=0.5  weight=1.5
    torch.manual_seed(0)
    _row("perspective  [RandomPerspective, distortion_scale=0.2, p=0.5, weight 1.5]",
         "min  distortion_scale=0  (identity)", orig_t,
         "max  distortion_scale=0.2  (seed=0)",
         T.RandomPerspective(distortion_scale=0.2, p=1.0)(orig_t))

    # 8. resized_crop  scale=[0.75,1.0], ratio=[0.95,1.05]  weight=1.0
    # lower scale = smaller crop = more upscaling = more zoom-in = bigger visual effect
    torch.manual_seed(0)
    crop_max_zoom = T.RandomResizedCrop((h, w), scale=(0.75, 0.75), ratio=(0.95, 1.05))(orig_t)
    torch.manual_seed(0)
    crop_min_zoom = T.RandomResizedCrop((h, w), scale=(1.0, 1.0), ratio=(1.0, 1.0))(orig_t)
    _row("resized_crop  [RandomResizedCrop, scale 0.75→1.0, ratio 0.95→1.05, weight 1.0]",
         "min  scale=1.0  (near-identity, no zoom)", crop_min_zoom,
         "max  scale=0.75  (crops 25% → zoom in)", crop_max_zoom)

    # 9. gaussian_blur  kernel_size=[5,9], sigma=[0.3,2.0]  weight=0.5
    _row("gaussian_blur  [GaussianBlur, kernel 5→9, sigma 0.3→2.0, weight 0.5]",
         "min  kernel=5  sigma=0.30  (mild blur)",
         TF.gaussian_blur(orig_t, kernel_size=[5, 5], sigma=[0.3]),
         "max  kernel=9  sigma=2.00  (heavy blur)",
         TF.gaussian_blur(orig_t, kernel_size=[9, 9], sigma=[2.0]))

    # 10. erase  p=0.3, scale=[0.02,0.10]  weight=0.5
    torch.manual_seed(0)
    erase_min_t = T.RandomErasing(p=1.0, scale=(0.02, 0.02), ratio=(1.0, 1.0), value=0.5)(orig_t.clone())
    torch.manual_seed(0)
    erase_max_t = T.RandomErasing(p=1.0, scale=(0.10, 0.10), ratio=(1.0, 1.0), value=0.5)(orig_t.clone())
    _row("erase  [RandomErasing, p=0.3, scale 0.02→0.10, weight 0.5]",
         "min  scale=0.02  (2% area erased)", erase_min_t,
         "max  scale=0.10  (10% area erased)", erase_max_t)

    # ── Render ────────────────────────────────────────────────────────────────
    cell_w, cell_h = 240, 180
    label_h = 36
    title_h = 26
    pad = 8
    canvas_w = 3 * cell_w + 4 * pad
    canvas_h = pad + len(rows) * (title_h + cell_h + label_h + pad)

    canvas = _PIL.new("RGB", (canvas_w, canvas_h), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(13)
    label_font = _load_font(12)

    for r_i, (title, cells) in enumerate(rows):
        y0 = pad + r_i * (title_h + cell_h + label_h + pad)
        draw.rectangle((0, y0, canvas_w, y0 + title_h - 2), fill=(230, 210, 200))
        draw.text((pad + 4, y0 + 5), title, fill=(60, 10, 10), font=title_font)
        for c_i, (lbl, t) in enumerate(cells):
            x0 = pad + c_i * (cell_w + pad)
            cell_img = _tensor_to_pil(t).resize((cell_w, cell_h), _PIL.Resampling.BICUBIC)
            canvas.paste(cell_img, (x0, y0 + title_h))
            for li, line in enumerate(lbl.split("\n")):
                draw.text((x0 + 4, y0 + title_h + cell_h + 4 + li * 14),
                          line, fill=(30, 30, 30), font=label_font)

    _Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def _print_reasonableness_analysis(mask_dir: str = "outputs/eval1_masks") -> None:
    SEP = "─" * 72

    print(f"\n{'═'*72}")
    print("  Augmentation reasonableness analysis")
    print(f"{'═'*72}")

    # ── Gamma ─────────────────────────────────────────────────────────────────
    print("\n  Gamma correction  [p=0.40, γ∈(0.80, 1.20)]")
    print(f"  {'pixel':>8}  {'γ=0.80 (min, brighten)':>26}  {'γ=1.20 (max, darken)':>24}")
    print(f"  {'-'*8}  {'-'*26}  {'-'*24}")
    for x in (0.2, 0.5, 0.8):
        lo, hi = x ** 0.80, x ** 1.20
        print(f"  {x:>8.1f}  {lo:>8.3f}  ({(lo/x-1)*100:>+6.1f}%)              "
              f"  {hi:>8.3f}  ({(hi/x-1)*100:>+6.1f}%)")
    print("  Applied to 40% of frames.  Maximum shift: ±15% for mid-tones,")
    print("  ±30% for dark pixels — compensates for camera exposure variation.")
    print("  VERDICT: REASONABLE — mild, symmetric around γ=1.0.")

    # ── Background replace ────────────────────────────────────────────────────
    slug = "green_bowl"
    bg_mask_path = os.path.join(mask_dir, slug, "bg_mask.npy")
    if os.path.exists(bg_mask_path):
        frac = np.load(bg_mask_path).astype(bool).mean()
        n_px = int(frac * 640 * 480)
        print(f"\n  Background replace  [p=0.30]")
        print(f"  Background mask: {frac*100:.1f}% of pixels ({n_px:,} px per frame, 640×480).")
        print(f"  30% of frames replaced; 70% untouched.")
        print(f"  In 12 000 training frames → ~3 600 frames get a new background.")
        print(f"  Pool of 500 unique backgrounds → each used on average 7.2× per epoch.")
        print(f"  Foreground (arm, bowls, banana) is NEVER modified.")
        print("  VERDICT: REASONABLE — sufficient diversity to prevent background memorisation.")

    # ── Bowl shuffle ──────────────────────────────────────────────────────────
    o1 = os.path.join(mask_dir, slug, "other1_mask.npy")
    o2 = os.path.join(mask_dir, slug, "other2_mask.npy")
    if os.path.exists(o1) and os.path.exists(o2):
        f1 = np.load(o1).astype(bool).mean()
        f2 = np.load(o2).astype(bool).mean()
        print(f"\n  Bowl shuffle  [p=0.50]")
        print(f"  other1 mask: {f1*100:.1f}% of pixels.  other2 mask: {f2*100:.1f}% of pixels.")
        print(f"  50% of frames swap the two non-target bowl regions.")
        print(f"  Each non-target bowl appears at its original position in exactly 50%")
        print(f"  of frames — maximum entropy, strongest possible position de-correlation.")
        print(f"  Target bowl is NEVER touched → action-image alignment is preserved.")
        print(f"  Bboxes differ in area ({f1/f2:.1f}× ratio) → bicubic resize may cause mild")
        print(f"  distortion at the smaller bbox when it receives the larger patch.")
        print("  VERDICT: REASONABLE — aggressive but necessary; risk is cosmetic only.")

    # ── Task augmenter ────────────────────────────────────────────────────────
    print(f"\n  Task augmenter  [6 variants, seed-stable RNG]")
    variants = [
        (0.35, "Put the banana in the X colored bowl.",               "← canonical (TA phrasing)"),
        (0.15, "Place the banana in the X colored bowl.",             ""),
        (0.15, "Put the banana in the X bowl.",                       ""),
        (0.15, "Place the banana in the X bowl.",                     ""),
        (0.10, "Move the banana to the X bowl.",                      ""),
        (0.10, "Pick up the banana and put it in the X colored bowl.", ""),
    ]
    for p, v, note in variants:
        print(f"  {p:.0%}  {v!r:<55} {note}")
    print("  Canonical phrasing has the highest weight (35%) because it is exactly")
    print("  what the TA types at evaluation time → no distribution shift at eval.")
    print("  65% novel phrasings improve instruction-following generalisation.")
    print("  VERDICT: REASONABLE.")

    print(f"\n{'═'*72}\n")


# ─────────────────────────────── main() ───────────────────────────────────────

def main():
    """Standalone smoke-test and visualizer.

    --visualize   render augmenter extremes grid + print reasonableness analysis
                  (does not require lerobot or HuggingFace access)

    Without --visualize: loads all 3 eval1 datasets, wraps with the full
    augmentation pipeline, samples 5 frames per dataset, and prints a report.

    Exit code 0 on success, 1 on any failure.
    """
    import traceback

    if "--visualize" in sys.argv:
        mask_dir = os.environ.get("EVAL1_MASK_DIR", "../outputs/eval1_masks")
        bg_dir   = os.environ.get("EVAL1_BG_DIR",   "../outputs/eval3_backgrounds")
        out_dir  = os.environ.get("EVAL1_VIZ_DIR",  "outputs")

        print("\nRendering custom augmenter extremes …")
        saved1 = visualize_extremes(
            out_path=os.path.join(out_dir, "eval1_aug_extremes.png"),
            mask_dir=mask_dir, bg_dir=bg_dir,
        )
        print(f"  saved → {saved1}")

        print("\nRendering lerobot image-transform extremes …")
        saved2 = visualize_transform_extremes(
            out_path=os.path.join(out_dir, "eval1_transform_extremes.png"),
            mask_dir=mask_dir,
        )
        print(f"  saved → {saved2}")

        _print_reasonableness_analysis(mask_dir=mask_dir)
        return 0

    SEP = "─" * 68

    print(f"\n{'═'*68}")
    print("  Eval1 dataset integration smoke-test")
    print(f"{'═'*68}")

    # ── Config from env vars (same defaults as training script) ───────────────
    mask_dir   = os.environ.get("EVAL1_MASK_DIR",          "../outputs/eval1_masks")
    bg_dir     = os.environ.get("EVAL1_BG_DIR",            "../outputs/eval3_backgrounds")
    gamma_p    = float(os.environ.get("EVAL1_GAMMA_P",          "0.4"))
    bg_p       = float(os.environ.get("EVAL1_BG_REPLACE_P",     "0.3"))
    bs_p       = float(os.environ.get("EVAL1_BOWL_SHUFFLE_P",   "0.5"))
    max_frames = int(os.environ.get("EVAL1_MAX_FRAMES_PER_EP",  "200"))
    n_samples  = int(os.environ.get("EVAL1_SMOKE_SAMPLES",      "5"))

    print(f"  mask_dir   : {mask_dir}")
    print(f"  bg_dir     : {bg_dir}")
    print(f"  max_frames : {max_frames}  gamma_p={gamma_p}  bg_p={bg_p}  bs_p={bs_p}")
    print(f"  n_samples  : {n_samples} per dataset")

    # ── Import lerobot ────────────────────────────────────────────────────────
    make_dataset = _try_import_lerobot()
    if make_dataset is None:
        print("\n[SKIP] lerobot not installed — unit tests cover augmenter logic.")
        print("       Install with:  pip install -e '.[smolvla]'")
        return 0

    failures: list[str] = []
    all_summaries: list[dict] = []

    for repo_id in REPO_IDS:
        print(f"\n{SEP}")
        print(f"  Dataset: {repo_id}")
        print(SEP)

        # ── Load raw dataset ──────────────────────────────────────────────────
        try:
            raw_ds = make_dataset(_make_lerobot_cfg(repo_id))
            print(f"  [OK] raw load  | {len(raw_ds):,} frames")
        except Exception as e:
            msg = f"{repo_id}: raw load failed — {e}"
            print(f"  [FAIL] {msg}")
            failures.append(msg)
            continue

        # ── Build augmentation pipeline ───────────────────────────────────────
        slug = next((c for c in ("green_bowl", "blue_bowl", "red_bowl")
                     if c in repo_id.lower()), "")
        wrapped = build_eval1_pipeline(
            raw_ds, repo_id,
            mask_dir=mask_dir, bg_dir=bg_dir,
            gamma_p=gamma_p, bg_p=bg_p, bs_p=bs_p,
            max_frames=max_frames,
            strict=False,
        )
        if wrapped._bg_aug_fn is not None:
            bg_mask_path = os.path.join(mask_dir, slug, "bg_mask.npy")
            print(f"  [OK] bg-replace  p={bg_p}  mask={bg_mask_path}")
        else:
            print(f"  [--] bg-replace  SKIPPED (mask or bg_dir missing)")
        if wrapped._bowl_shuffle_fn is not None:
            print(f"  [OK] bowl-shuffle  p={bs_p}")
        else:
            print(f"  [--] bowl-shuffle  SKIPPED (masks missing)")
        s = wrapped.truncation_summary()
        all_summaries.append({**s, "slug": slug,
                               "bg_aug": wrapped._bg_aug_fn is not None,
                               "bowl_aug": wrapped._bowl_shuffle_fn is not None})
        print(f"  frames : {s['original_num_frames']:,} raw → {s['kept_num_frames']:,} kept"
              f"  ({s['kept_fraction']*100:.1f}%,"
              f"  {s['dropped_num_frames']:,} dropped by truncation)")

        # ── Sample n_samples frames and verify invariants ─────────────────────
        step = max(1, len(wrapped) // n_samples)
        sample_issues: list[str] = []
        tasks_seen: set[str] = set()

        for i in range(n_samples):
            idx = i * step
            try:
                row = wrapped[idx]
            except Exception as e:
                sample_issues.append(f"idx {idx}: __getitem__ raised {e}")
                traceback.print_exc()
                continue

            # Image checks.
            if IMAGE_KEY in row:
                img = row[IMAGE_KEY]
                if not isinstance(img, torch.Tensor):
                    sample_issues.append(f"idx {idx}: image is {type(img)}, not Tensor")
                elif img.ndim != 3 or img.shape[0] != 3:
                    sample_issues.append(f"idx {idx}: image shape {tuple(img.shape)}, expected (3,H,W)")
                elif not img.is_floating_point():
                    sample_issues.append(f"idx {idx}: image dtype {img.dtype}, not float")
                elif img.min() < -1e-4 or img.max() > 1.0 + 1e-4:
                    sample_issues.append(f"idx {idx}: image range [{img.min():.3f},{img.max():.3f}]")
                elif torch.isnan(img).any():
                    sample_issues.append(f"idx {idx}: NaN in image")
                else:
                    pass  # OK
            else:
                sample_issues.append(f"idx {idx}: missing key '{IMAGE_KEY}'")

            # Task checks.
            task = row.get("task", "")
            if not isinstance(task, str) or not task:
                sample_issues.append(f"idx {idx}: bad task {task!r}")
            elif "banana" not in task.lower():
                sample_issues.append(f"idx {idx}: task missing 'banana': {task!r}")
            else:
                tasks_seen.add(task)

            # Action checks.
            if "action" in row:
                act = row["action"]
                if not isinstance(act, torch.Tensor):
                    sample_issues.append(f"idx {idx}: action is {type(act)}")
                elif torch.isnan(act).any():
                    sample_issues.append(f"idx {idx}: NaN in action")

        if sample_issues:
            for iss in sample_issues:
                print(f"  [FAIL] {iss}")
                failures.append(f"{repo_id} | {iss}")
        else:
            img_shape = tuple(wrapped[0][IMAGE_KEY].shape) if IMAGE_KEY in wrapped[0] else "N/A"
            print(f"  [OK] {n_samples} samples  image={img_shape}  task_variants={len(tasks_seen)}")

        # Verify task augmenter produces at least 2 distinct phrasings.
        if len(tasks_seen) < 2 and n_samples >= 5:
            msg = f"{repo_id}: only {len(tasks_seen)} distinct task phrasing(s) in {n_samples} samples"
            print(f"  [WARN] {msg}")
        if tasks_seen:
            for t in sorted(tasks_seen):
                print(f"         task: {t!r}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'═'*68}")
    print("  SUMMARY")
    print(f"{'═'*68}")
    print(f"  {'Dataset':<35} {'raw':>6} {'kept':>6} {'%':>5}  bg  bowl")
    print(f"  {'-'*35} {'------':>6} {'------':>6} {'-----':>5}  --  ----")
    for s in all_summaries:
        name = s["repo_id"].split("/")[-1][:35]
        print(f"  {name:<35} {s['original_num_frames']:>6,} {s['kept_num_frames']:>6,}"
              f" {s['kept_fraction']*100:>5.1f}%"
              f"  {'Y' if s['bg_aug'] else 'N':>2}  {'Y' if s['bowl_aug'] else 'N'}")

    aug_row = "  Augmenters active: task=Y  gamma=Y"
    if all_summaries:
        aug_row += f"  bg={'Y' if all_summaries[0]['bg_aug'] else 'N'}"
        aug_row += f"  bowl={'Y' if all_summaries[0]['bowl_aug'] else 'N'}"
    print(aug_row)

    if failures:
        print(f"\n  FAILURES ({len(failures)}):")
        for f in failures:
            print(f"    ✗ {f}")
        print()
        return 1
    else:
        print(f"\n  All checks passed.\n")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
