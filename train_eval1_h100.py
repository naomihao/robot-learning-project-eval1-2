"""
SmolVLA fine-tuning — eval1  ·  H100 edition
=============================================

GPU          : H100 (80 GB)
dtype        : bf16  (H100 bf16 is ~3× faster than A100 bf16)
batch_size   : 128   (4× vs baseline 32; H100 80 GB handles it easily)
Steps budget :
  60 eps × ~200 frames = 12 000 raw frames
  12 000 / 128 = 93 steps/epoch
  Target 30 baseline epochs × 2.2 aug multiplier ≈ 67 effective epochs
  93 × 67 ≈ 6 250  →  rounded to 7 000 steps
Estimated time : ~20 – 40 min on H100
"""

import copy
import sys
import types
import torch
from torch.utils.data import ConcatDataset

# ── Config ────────────────────────────────────────────────────────────────────

REPO_IDS = [
    "RobotLearningVLA/banana_green_bowl_eval1",
    "RobotLearningVLA/banana_blue_bowl_eval1",
    "RobotLearningVLA/banana_red_bowl_eval1",
]

FRAMES_PER_EP   = 200
EPISODES_TOTAL  = len(REPO_IDS) * 20           # 60
FRAMES_TOTAL    = EPISODES_TOTAL * FRAMES_PER_EP    # 12 000
BATCH_SIZE      = 128                           # H100: 4× baseline
STEPS_PER_EPOCH = FRAMES_TOTAL // BATCH_SIZE   # 93
AUG_MULTIPLIER  = 2.2
TOTAL_STEPS     = 7_000                        # ~67 effective epochs

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
║  Total steps   : {TOTAL_STEPS:,}   (~67 effective epochs)      ║
║  Est. time     : ~20 – 40 min                            ║
║  HF upload     : every 1 000 steps                       ║
╚══════════════════════════════════════════════════════════╝
""")

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

# ── Gamma augmentation wrapper ────────────────────────────────────────────────

def _wrap_gamma(dataset, p: float = 0.4):
    _orig = dataset.__getitem__
    def __getitem__(self, idx):
        sample = _orig(idx)
        if torch.rand(1).item() < p:
            gamma = 0.8 + 0.4 * torch.rand(1).item()
            for k, v in sample.items():
                if "images" in k and isinstance(v, torch.Tensor) and v.is_floating_point():
                    sample[k] = v.clamp(0.0, 1.0).pow(gamma)
        return sample
    dataset.__getitem__ = types.MethodType(__getitem__, dataset)
    return dataset

# ── Patched make_dataset ──────────────────────────────────────────────────────

def _patched_make_dataset(cfg):
    datasets = []
    for repo_id in REPO_IDS:
        cfg_i = copy.deepcopy(cfg)
        cfg_i.dataset.repo_id = repo_id
        cfg_i.dataset.revision = "main"
        ds = _orig_make_dataset(cfg_i)
        ds = _wrap_gamma(ds, p=0.4)
        datasets.append(ds)
        print(f"[dataset] Loaded '{repo_id}' | {len(ds)} frames")

    combined = ConcatDataset(datasets)
    combined.num_frames   = sum(len(d) for d in datasets)
    combined.num_episodes = sum(getattr(d, "num_episodes", 0) for d in datasets)
    combined.meta  = datasets[0].meta
    combined.stats = getattr(datasets[0], "stats", None)
    print(f"[dataset] ConcatDataset | {combined.num_frames:,} frames | {combined.num_episodes} episodes")
    return combined

_train_mod.make_dataset = _patched_make_dataset

# ── CLI args ──────────────────────────────────────────────────────────────────

sys.argv = [
    "lerobot-train",

    # ── Policy ────────────────────────────────────────────────────────────────
    "--policy.path=lerobot/smolvla_base",
    "--policy.repo_id=yukk1/test_eval1",
    "--policy.device=cuda",
    "--policy.push_to_hub=true",
    "--policy.empty_cameras=2",
    "--policy.freeze_vision_encoder=true",
    "--policy.train_expert_only=true",
    "--policy.use_amp=true",
    "--policy.dtype=bf16",              # H100: ~3× faster than A100 in bf16

    # ── Dataset ───────────────────────────────────────────────────────────────
    "--dataset.repo_id=RobotLearningVLA/banana_green_bowl_eval1",

    # ── Camera rename ─────────────────────────────────────────────────────────
    '--rename_map={"observation.images.front":"observation.images.camera1"}',

    # ── Image transforms ──────────────────────────────────────────────────────
    "--dataset.image_transforms.enable=true",
    (
        "--dataset.image_transforms.tfs={"
        '"brightness":{"type":"ColorJitter","kwargs":{"brightness":[0.7,1.3]},"weight":1.0},'
        '"contrast":{"type":"ColorJitter","kwargs":{"contrast":[0.7,1.3]},"weight":1.0},'
        '"saturation":{"type":"ColorJitter","kwargs":{"saturation":[0.8,1.2]},"weight":1.0},'
        '"hue":{"type":"ColorJitter","kwargs":{"hue":[-0.02,0.02]},"weight":1.0},'
        '"sharpness":{"type":"SharpnessJitter","kwargs":{"sharpness":[0.5,1.5]},"weight":1.0}'
        "}"
    ),

    # ── Training budget ───────────────────────────────────────────────────────
    # batch 128 → 93 steps/epoch → 7 000 steps ≈ 67 effective epochs (×2.2 aug)
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
    "--policy.scheduler_warmup_steps=200",      # shorter: fewer total steps
    "--policy.scheduler_decay_steps=15000",
    "--policy.scheduler_decay_lr=2.5e-6",

    # ── Checkpointing + HF upload every 1 000 steps ───────────────────────────
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
