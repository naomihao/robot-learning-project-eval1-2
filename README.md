# VLA Eval 1 — SO-101 Banana-in-Bowl Manipulation

Vision-language-conditioned pick-and-place policy for the course's **Project 1: VLA** challenge, built on
[LeRobot](https://github.com/huggingface/lerobot) and fine-tuned **SmolVLA**.

A SO-101 follower arm must pick up a squishy toy banana and place it into one of three bowls (blue / red / green,
arranged in a semicircle) based on a natural-language prompt. This repo covers the **Eval 1** (direct color prompt)
and **Eval 2** (compositional/spatial instruction) setups.

## Task recap

| Eval | Prompt style | Example | Points |
|------|---|---|---|
| **Eval 1** | Direct color | "Put the banana in the blue colored bowl." | 50 (9 rollouts) |
| **Eval 2** | Compositional / spatial | "Put the banana into the 2nd bowl from the left (robot perspective)." / "...bowl that is not green and not blue." | 50 (rollouts split across easy/hard prompts) |

Constraints from the project spec: single camera, VLA-only policy (no auxiliary LLM/VLM/YOLO/face-recognition
models at inference time), 20 s per rollout, ±5 cm object placement tolerance. See the full brief in
(https://docs.google.com/document/d/1YsQ_Qe4vEwDp1dJdqn3l9vSt7oJBkc6JazjbmWLxAXg/edit?tab=t.0)

## How it works

```
raw instruction  →  prompt normalizer (deterministic, no model)  →  canonical color prompt  →  SmolVLA policy  →  robot
```

Eval 2 prompts are never fed to the policy as-is. A rule-based normalizer resolves negation, color-analogy, and
spatial phrasing (using an HSV-detected bowl layout from the camera) down to the same canonical string the policy
was trained on: `"Put the banana in the {color} colored bowl."` This keeps the policy itself a single-purpose,
color-conditioned VLA — satisfying the "no auxiliary foundation model" rule — while still handling Eval 2's harder
instructions.

## Repository layout

```
vla_eval1/
├── setup_env.sh                     # one-time conda + ffmpeg + lerobot[smolvla] + HF/W&B setup
├── run_train_h100.sh                # detached H100 training launcher (start/--status/--log/--kill)
├── train_eval1_h100.py              # v1 training entrypoint (gamma aug only)
│
├── robot-learning-vla-eval1/        # v2 training pipeline with full augmentation stack
│   ├── scripts/
│   │   ├── train_eval1_smolvla_h100.py   # v2 entrypoint: task/bg/bowl-shuffle/gamma aug + HF checkpoint upload
│   │   ├── eval1_dataset_prep.py         # augmentation wrappers (see below)
│   │   └── test_eval1_augmentations.py   # unit tests for the augmentation pipeline
│   ├── tools/
│   │   ├── eval1_extract_masks.py        # build per-bowl-color bg/other1/other2 masks
│   │   ├── show_eval1_masks.py           # visualize mask-based augmentations
│   │   ├── eval1_visualize_augmentation.py  # torch-free preview of all image transforms
│   │   └── inspect_lerobot_dataset.py    # QA gate: inspect a LeRobotDataset's metadata/tensors
│   └── outputs/                     # saved preview images from the tools above
│
├── task2_prompt/
│   ├── eval1_prompt_normalizer.py   # PromptNormalizer: negation / analogy / spatial → canonical prompt
│   └── test_prompt_normalizer.py    # unit tests
│
├── task1_task2_deployment/
│   ├── run_eval.py                  # eval-day CLI: normalizes prompt, drives the rollout, auto-resets pose
│   ├── run_eval.sh                  # env-locating wrapper around run_eval.py
│   ├── models.json                  # {"task1": <hf repo>, "task2": <hf repo>} pretrained checkpoint paths
│   ├── test_prompt_normalization.py # end-to-end prompt-normalization regression tests
│   ├── prompt_normalization_all_orders.txt  # expected outputs across all 6 bowl-order permutations
│   └── best_effort_cases_all_orders.txt     # fallback-heuristic regression cases
│
└── outputs/eval3_backgrounds/       # background-image pool used by BackgroundReplaceAugmenter
```

## Setup

```bash
chmod +x setup_env.sh && ./setup_env.sh
```

Creates a `lerobot` conda env (Python 3.12), installs `ffmpeg 7.1.1`, clones/installs `lerobot[smolvla]` from
source into `~/lerobot`, and stores `WANDB_API_KEY` / `HF_TOKEN` in `~/.bashrc`.

## Training

Two training entrypoints target `lerobot/smolvla_base`, freeze the vision encoder, train only the action expert,
and fine-tune on 3 per-color datasets (`banana_{color}_bowl_eval1[...]`) concatenated together:

```bash
./run_train_h100.sh            # launch train_eval1_h100.py detached (survives SSH/laptop disconnects)
./run_train_h100.sh --status   # check progress
./run_train_h100.sh --log      # tail training log
./run_train_h100.sh --kill     # stop
```

- **`train_eval1_h100.py`** (repo root) — baseline: bf16, batch 128, 7,000 steps, ColorJitter/Sharpness transforms
  only, gamma augmentation wrapped around each dataset.
- **`robot-learning-vla-eval1/scripts/train_eval1_smolvla_h100.py`** — extended v2 pipeline: adds
  `torch.compile`, spatial transforms (affine/perspective/resized-crop/blur/erase), and the
  `eval1_dataset_prep.py` augmentation stack below. Controlled via env vars (`EVAL1_TASK_AUG`, `EVAL1_BG_REPLACE`,
  `EVAL1_BOWL_SHUFFLE`, `EVAL1_GAMMA_P`, `EVAL1_MAX_FRAMES_PER_EP`, `EVAL1_MASK_DIR`, `EVAL1_BG_DIR`).

Both scripts push checkpoints to the Hugging Face Hub every N steps (tagged `step-XXXXXX`) and log to W&B
(project `smolvla_banana_eval1`).

### Augmentation pipeline (`eval1_dataset_prep.py`)

- **`TaskAugmenter`** — rephrases the task string while preserving the target color (canonical wording gets the
  highest sampling weight, since that's what the policy will actually see on eval day).
- **`BackgroundReplaceAugmenter`** — pastes a random background image (`outputs/eval3_backgrounds/*.png`) behind
  the scene, using a manually-extracted `bg_mask.npy` per bowl color.
- **`BowlShuffleAugmenter`** — swaps the two *non-target* bowl regions to stop the model from associating color
  with a fixed table position. The target bowl is never touched (it must stay aligned with the action labels).
- Per-sample gamma correction for lighting robustness.

Masks are produced once via `tools/eval1_extract_masks.py` (manual polygon annotation over a cached dataset
frame) and can be sanity-checked with `tools/show_eval1_masks.py` / `tools/eval1_visualize_augmentation.py`.

## Prompt normalization (Eval 2)

`task2_prompt/eval1_prompt_normalizer.py` resolves any Eval 1/2 instruction to
`"Put the banana in the {color} colored bowl."` with **no extra model** — purely deterministic rules layered as:

1. **Direct** — "in/into the red bowl", "the red one" → straight lookup.
2. **Negation** — "not red and not blue" → remaining color.
3. **Analogy** — "the color of grass" / "stop sign" → concept→color lookup.
4. **Spatial** — "2nd bowl from the left", "right of the red bowl" — resolved against a bowl layout detected via
   HSV color segmentation on a captured camera frame (the camera faces the robot, so image left/right is mirrored
   relative to the robot's own perspective).
5. **Best-effort fallback** — if strict parsing fails, a heuristic guesses the most likely target so the arm
   still attempts a rollout rather than stalling.

Run the regression suite:

```bash
python task1_task2_deployment/test_prompt_normalization.py
python task2_prompt/test_prompt_normalizer.py
```

## Deployment (eval day)

`task1_task2_deployment/run_eval.py` (invoke via `run_eval.sh` for automatic env/PATH discovery) wraps
`lerobot-rollout` / `lerobot-record`, with support for:

- **task1 / task2** subcommands — task2 additionally captures camera frames and runs the prompt normalizer.
- **`--interactive`** — TA types one instruction per round; `--backend persistent` keeps the policy loaded across
  rounds instead of respawning a process each rollout.
- **Automatic pose reset** — captures the arm's start pose before each rollout and drives back to it afterward
  (`--reset-mode auto|manual|command|off`).
- **`--preflight` / `--dry-run`** — validate ports, camera, and the resolved command without moving the robot.

```bash
# Task 1
python3 task1_task2_deployment/run_eval.py task1 \
  --interactive \
  --duration 20 \ 
  --camera-index 0 \
  --device cpu
command example "Put the banana in the red colored bowl."

# Task 2 — prompt resolved from a live camera frame each rollout
python3 task1_task2_deployment/run_eval.py task1 \
  --interactive \
  --duration 20 \ 
  --camera-index 0 \
  --device cpu

command example "Put the banana into the bowl that is not avacado colored and not sky colored bowl" \


# TA-facing interactive session, model loaded once
./task1_task2_deployment/run_eval.sh task2 --interactive --backend persistent --duration 20
```

Model checkpoints per task are configured in `task1_task2_deployment/models.json` (override with
`--pretrained-path`).

## Testing

```bash
pytest robot-learning-vla-eval1 -svv                        # augmentation pipeline unit tests
python task1_task2_deployment/test_prompt_normalization.py  # normalizer regression, all bowl-order permutations
python task2_prompt/test_prompt_normalizer.py               # normalizer unit tests
```
