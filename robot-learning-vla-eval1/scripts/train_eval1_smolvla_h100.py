#!/usr/bin/env python3
"""Fine-tune SmolVLA for Eval 1 on H100.

GPU          : H100 (80 GB)
dtype        : bf16  (H100 bf16 is ~3× faster than A100 bf16)
batch_size   : 128   (4× vs baseline 32; H100 80 GB handles it easily)
Steps budget :
  60 eps × ~300 frames = 18 000 raw frames
  18 000 / 128 = 140 steps/epoch
  Target 30 baseline epochs × 2.2 aug multiplier ≈ 67 effective epochs
  140 × 67 ≈ 9 380  →  rounded to 10 000 steps
Estimated time : ~30 – 55 min on H100

Augmentation (env-var controlled, all on by default):
  EVAL1_TASK_AUG=1            task-string rephrasing (canonical: "Put the banana in the X colored bowl.")
  EVAL1_BG_REPLACE=1          background replacement
  EVAL1_BG_REPLACE_P=0.3      per-frame probability
  EVAL1_BOWL_SHUFFLE=1        swap two non-target bowl regions
  EVAL1_BOWL_SHUFFLE_P=0.5    per-frame probability
  EVAL1_GAMMA_P=0.4           per-frame gamma correction probability
  EVAL1_MAX_FRAMES_PER_EP=300 episode truncation cap
  EVAL1_MASK_DIR=../outputs/eval1_masks
  EVAL1_BG_DIR=../outputs/eval3_backgrounds

Usage::

    python scripts/train_eval1_smolvla_h100.py
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset

# ── Add scripts dir to path (for eval3_lerobot_shim + eval1_dataset_prep) ────

_SCRIPT_DIR = Path(__file__).resolve().parent
_EVAL3_SCRIPTS = (
    Path(__file__).resolve().parents[2]
    / "robot-learning-vla"
    / "scripts"
)
for _p in (_SCRIPT_DIR, _EVAL3_SCRIPTS):
    if str(_p) not in sys.path and _p.exists():
        sys.path.insert(0, str(_p))

# ── Apply GROOT/transformers shim before any lerobot policy import ─────────────

from eval3_lerobot_shim import apply as _shim_apply  # noqa: E402

_shim_apply()

# ── Config ────────────────────────────────────────────────────────────────────

REPO_IDS = [
    "RobotLearningVLA/banana_green_bowl_eval1_v2",
    "RobotLearningVLA/banana_blue_bowl_eval1_v2",
    "RobotLearningVLA/banana_red_bowl_eval1_v2",
]

FRAMES_PER_EP   = 300
EPISODES_TOTAL  = len(REPO_IDS) * 20            # 60
FRAMES_TOTAL    = EPISODES_TOTAL * FRAMES_PER_EP # 18 000
BATCH_SIZE      = 128                            # H100: 4× baseline
STEPS_PER_EPOCH = FRAMES_TOTAL // BATCH_SIZE    # 140
AUG_MULTIPLIER  = 2.2
TOTAL_STEPS     = 10_000                        # ~67 effective epochs

print(f"""
╔══════════════════════════════════════════════════════════╗
║         SmolVLA eval1 · H100 · bf16 training plan       ║
╠══════════════════════════════════════════════════════════╣
║  GPU           : H100 (989 TFLOPS bf16)                  ║
║  dtype         : bf16                                    ║
║  Datasets      : 3 × 20 episodes = {EPISODES_TOTAL} eps              ║
║  Raw frames    : ~{FRAMES_TOTAL:,}                              ║
║  Batch size    : {BATCH_SIZE}                                   ║
║  Steps / epoch : ~{STEPS_PER_EPOCH}                                  ║
║  Aug multiplier: ×{AUG_MULTIPLIER}                                  ║
║  Total steps   : {TOTAL_STEPS:,}  (~67 effective epochs)       ║
║  Est. time     : ~30 – 55 min                            ║
║  HF upload     : every 500 steps                         ║
╚══════════════════════════════════════════════════════════╝
""")

# ── Preflight: validate mask + background paths before any heavy import ───────

def _preflight_check():
    import glob as _glob

    mask_dir = os.environ.get("EVAL1_MASK_DIR", "../outputs/eval1_masks")
    bg_dir   = os.environ.get("EVAL1_BG_DIR",   "../outputs/eval3_backgrounds")

    errors = []

    # background image pool must exist and be non-empty
    if not os.path.isdir(bg_dir):
        errors.append(f"  bg_dir not found          : {bg_dir}")
    else:
        pngs = _glob.glob(os.path.join(bg_dir, "*.png"))
        if not pngs:
            errors.append(f"  bg_dir has no .png files  : {bg_dir}")
        else:
            print(f"[preflight] bg_dir OK — {len(pngs)} backgrounds in {bg_dir}")

    # per-colour slug: three mask files each
    for repo_id in REPO_IDS:
        slug = next((c for c in ("green_bowl", "blue_bowl", "red_bowl") if c in repo_id.lower()), None)
        if slug is None:
            errors.append(f"  cannot detect colour slug from repo_id: {repo_id}")
            continue
        for fname in ("bg_mask.npy", "other1_mask.npy", "other2_mask.npy"):
            path = os.path.join(mask_dir, slug, fname)
            if not os.path.exists(path):
                errors.append(f"  mask not found [{slug:5s}] : {path}")
            else:
                print(f"[preflight] mask OK — {path}")

    if errors:
        print("\n[preflight] FATAL — missing augmentation assets:")
        for e in errors:
            print(e)
        print(
            "\nSet EVAL1_MASK_DIR / EVAL1_BG_DIR env vars to point at the "
            "correct directories, or generate the masks first.\n"
        )
        raise SystemExit(1)

    print("[preflight] All mask and background paths verified. Starting training.\n")

_preflight_check()

# ── Detect lerobot layout ─────────────────────────────────────────────────────

def _detect_lerobot():
    try:
        import lerobot.scripts.lerobot_train as train_mod
        from lerobot.datasets.factory import make_dataset
        print("[lerobot] NEW layout: lerobot.scripts.lerobot_train")
        return train_mod, make_dataset, "main"
    except ModuleNotFoundError:
        pass
    try:
        import lerobot.scripts.train as train_mod
        from lerobot.common.datasets.factory import make_dataset
        print("[lerobot] OLD layout: lerobot.scripts.train")
        return train_mod, make_dataset, "train_cli"
    except ModuleNotFoundError:
        pass
    raise ImportError("Cannot find lerobot. Run: pip install -e '.[smolvla]'")

_train_mod, _orig_make_dataset, _entry_fn = _detect_lerobot()

# ── Patched make_dataset ──────────────────────────────────────────────────────

from eval1_dataset_prep import (
    make_task_augmenter,
    build_eval1_pipeline,
)

def _patched_make_dataset(cfg):
    # ── Env-var controls ───────────────────────────────────────────────────────
    max_frames = int(os.environ.get("EVAL1_MAX_FRAMES_PER_EP", "300"))
    gamma_p    = float(os.environ.get("EVAL1_GAMMA_P",         "0.4"))
    bg_p       = float(os.environ.get("EVAL1_BG_REPLACE_P",    "0.3"))
    bs_p       = float(os.environ.get("EVAL1_BOWL_SHUFFLE_P",  "0.5"))
    mask_dir   = os.environ.get("EVAL1_MASK_DIR", "../outputs/eval1_masks")
    bg_dir     = os.environ.get("EVAL1_BG_DIR",   "../outputs/eval3_backgrounds")

    task_aug = make_task_augmenter()

    # ── Load + wrap each dataset ───────────────────────────────────────────────
    datasets = []
    for repo_id in REPO_IDS:
        cfg_i = copy.deepcopy(cfg)
        cfg_i.dataset.repo_id = repo_id
        cfg_i.dataset.revision = "main"
        ds = _orig_make_dataset(cfg_i)
        print(f"[dataset] Loaded '{repo_id}' | {len(ds)} frames")

        ds = build_eval1_pipeline(
            ds, repo_id,
            mask_dir=mask_dir,
            bg_dir=bg_dir,
            max_frames=max_frames,
            gamma_p=gamma_p,
            bg_p=bg_p,
            bs_p=bs_p,
            task_aug_fn=task_aug,
            strict=True,
        )
        s = ds.truncation_summary()
        print(f"[eval1] {s['repo_id']}  before={s['original_num_frames']}  "
              f"after={s['kept_num_frames']}  kept={s['kept_fraction']*100:.1f}%  "
              f"task_aug=True  bg_aug=True  bowl_shuffle=True")

        datasets.append(ds)

    combined = ConcatDataset(datasets)
    combined.num_frames   = sum(len(d) for d in datasets)
    combined.num_episodes = sum(getattr(d, "num_episodes", 0) for d in datasets)
    combined.meta  = datasets[0].meta
    combined.stats = getattr(datasets[0], "stats", None)
    print(f"[dataset] ConcatDataset | {combined.num_frames:,} frames | {combined.num_episodes} episodes")
    return combined

_train_mod.make_dataset = _patched_make_dataset

# ── Patched save_checkpoint: upload every checkpoint to HF Hub ────────────────
# save_freq=500 controls both local saves AND hub uploads.
# Each checkpoint is committed with a git tag (step-000500, step-001000, …)
# so any revision can be loaded: SmolVLAPolicy.from_pretrained(repo_id, revision="step-000500")

from huggingface_hub import HfApi as _HfApi

_orig_save_checkpoint = _train_mod.save_checkpoint


def _patched_save_checkpoint(checkpoint_dir, step, cfg, policy, **kwargs):
    _orig_save_checkpoint(checkpoint_dir, step, cfg, policy, **kwargs)

    pretrained_dir = Path(checkpoint_dir) / "pretrained_model"
    repo_id = cfg.policy.repo_id
    tag = f"step-{step:06d}"

    print(f"[hub] Uploading checkpoint step {step}/{cfg.steps} → {repo_id} ...")
    api = _HfApi()
    api.create_repo(
        repo_id=repo_id,
        private=cfg.policy.private,
        exist_ok=True,
        repo_type="model",
    )
    commit_info = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(pretrained_dir),
        commit_message=f"Checkpoint step {step}/{cfg.steps}",
        allow_patterns=["*.safetensors", "*.json", "*.yaml", "*.md"],
    )
    api.create_tag(
        repo_id=repo_id,
        repo_type="model",
        tag=tag,
        revision="main",
        exist_ok=True,
    )
    print(f"[hub] ✓ step {step} → {commit_info.repo_url}  (tag: {tag})")


_train_mod.save_checkpoint = _patched_save_checkpoint

# ── CLI args ──────────────────────────────────────────────────────────────────

sys.argv = [
    "lerobot-train",

    # ── Policy ────────────────────────────────────────────────────────────────
    "--policy.path=lerobot/smolvla_base",
    "--policy.repo_id=yukk1/test_eval1_v2",
    "--policy.device=cuda",
    "--policy.push_to_hub=true",
    "--policy.empty_cameras=2",
    "--policy.freeze_vision_encoder=true",
    "--policy.train_expert_only=true",
    "--policy.use_amp=true",
    "--policy.dtype=bf16",              # H100: ~3× faster than A100 in bf16
    "--policy.compile_model=true",      # H100: max-autotune Triton kernels, ~15-25% speedup

    # ── Dataset ───────────────────────────────────────────────────────────────
    "--dataset.repo_id=RobotLearningVLA/banana_green_bowl_eval1_v2",

    # ── Camera rename ─────────────────────────────────────────────────────────
    '--rename_map={"observation.images.front":"observation.images.camera1"}',

    # ── Image transforms (ported from eval3 aug-train, + eval1-safe spatial augs)
    # brightness/contrast weight=2.0 mirrors eval3: strongest defence against
    # per-bowl lighting cues (analogous to the LeCun↔Obama luma gap in eval3).
    # Spatial augs (affine, perspective, resized_crop) kept mild so the arm
    # target position stays visually consistent with the action labels.
    # gaussian_blur + erase add robustness to partial occlusion / focus blur.
    "--dataset.image_transforms.enable=true",
    "--dataset.image_transforms.max_num_transforms=4",
    (
        "--dataset.image_transforms.tfs={"
        '"brightness":{"type":"ColorJitter","kwargs":{"brightness":[0.6,1.4]},"weight":2.0},'
        '"contrast":{"type":"ColorJitter","kwargs":{"contrast":[0.6,1.4]},"weight":2.0},'
        '"saturation":{"type":"ColorJitter","kwargs":{"saturation":[0.5,1.5]},"weight":1.0},'
        '"hue":{"type":"ColorJitter","kwargs":{"hue":[-0.02,0.02]},"weight":1.0},'
        '"sharpness":{"type":"SharpnessJitter","kwargs":{"sharpness":[0.5,1.5]},"weight":1.0},'
        '"affine":{"type":"RandomAffine","kwargs":{"degrees":[-3.0,3.0],"translate":[0.03,0.03]},"weight":1.0},'
        '"perspective":{"type":"RandomPerspective","kwargs":{"distortion_scale":0.2,"p":0.5},"weight":1.5},'
        '"resized_crop":{"type":"RandomResizedCrop","kwargs":{"size":[480,640],"scale":[0.75,1.0],"ratio":[0.95,1.05]},"weight":1.0},'
        '"gaussian_blur":{"type":"GaussianBlur","kwargs":{"kernel_size":[5,9],"sigma":[0.3,2.0]},"weight":0.5},'
        '"erase":{"type":"RandomErasing","kwargs":{"p":0.3,"scale":[0.02,0.1]},"weight":0.5}'
        "}"
    ),

    # ── Training budget ───────────────────────────────────────────────────────
    # batch 128 → 140 steps/epoch → 10 000 steps ≈ 67 effective epochs (×2.2 aug)
    f"--batch_size={BATCH_SIZE}",
    f"--steps={TOTAL_STEPS}",

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Linear scaling rule: larger batch → scale LR proportionally
    # base lr 1e-4 @ batch 32 → 1e-4 × (128/32) = 4e-4 @ batch 128
    "--policy.optimizer_lr=4e-4",
    "--policy.optimizer_betas=[0.9,0.95]",
    "--policy.optimizer_eps=1e-8",
    "--policy.optimizer_weight_decay=1e-10",
    "--policy.optimizer_grad_clip_norm=10.0",

    # ── LR scheduler ──────────────────────────────────────────────────────────
    "--policy.scheduler_warmup_steps=1000",     # SmolVLA default; 10% of 10k steps
    "--policy.scheduler_decay_steps=9000",      # total_steps - warmup: full cosine decay within budget
    "--policy.scheduler_decay_lr=2.5e-6",

    # ── Checkpointing + HF upload every 1 000 steps ──────────────────────────
    "--output_dir=outputs/train/eval1_smolvla_h100",
    "--save_freq=1000",
    "--log_freq=25",                            # log every 25 steps; runs are short
    "--num_workers=16",                         # H100 nodes typically have many CPUs

    # ── W&B ───────────────────────────────────────────────────────────────────
    "--wandb.enable=true",
    "--wandb.project=smolvla_banana_eval1",
    "--wandb.run_name=eval1_3bowls_h100_bf16",
]

# ── Launch ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    entry_fn = getattr(_train_mod, _entry_fn)
    entry_fn()
