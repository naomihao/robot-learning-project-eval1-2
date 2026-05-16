#!/usr/bin/env python3
"""Inspect LeRobot v3 Hub/local datasets (Eval 1 QA gate).

Usage:
  python tools/inspect_lerobot_dataset.py
  python tools/inspect_lerobot_dataset.py --repo-id RobotLearningVLA/banana_red_bowl_eval1_v2 --episodes 0 1
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect LeRobotDataset metadata and sample tensors.")
    parser.add_argument(
        "--repo-id",
        default="RobotLearningVLA/banana_red_bowl_eval1_v2",
        help="HF dataset repo id",
    )
    parser.add_argument(
        "--episodes",
        nargs="*",
        type=int,
        default=None,
        help="Optional episode indices to restrict (subset loading)",
    )
    parser.add_argument("--sample-index", type=int, default=0, help="Frame index for tensor shape preview")
    parser.add_argument("--no-videos", action="store_true", help="Skip downloading/decoding videos if supported")
    parser.add_argument(
        "--with-sample",
        action="store_true",
        help="Load LeRobotDataset and print one sample. Requires PyTorch to import cleanly.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use files already present in the Hugging Face cache.",
    )
    parser.add_argument(
        "--video-backend",
        default="pyav",
        help="Video decoder for frames (use pyav on macOS if torchcodec/FFmpeg fails)",
    )
    parser.add_argument(
        "--expected-task",
        default="Put the banana in the red colored bowel.",
        help="If set non-empty, warn when unique task strings do not match this instruction (Eval 1 QA)",
    )
    args = parser.parse_args()

    if not args.with_sample:
        inspect_metadata_only(args)
        return

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import torch

    ds = LeRobotDataset(
        args.repo_id,
        episodes=args.episodes if args.episodes else None,
        download_videos=not args.no_videos,
        video_backend=args.video_backend,
    )

    print("=== LeRobotDataset ===")
    print(f"repo_id:          {ds.repo_id}")
    print(f"revision:         {ds.revision}")
    print(f"num_episodes:     {ds.num_episodes}")
    print(f"num_frames:       {ds.num_frames}")
    print(f"fps:              {ds.meta.fps}")

    feats = ds.features
    print("\n=== features ===")
    for name, ft in sorted(feats.items()):
        shape = getattr(ft, "shape", None)
        print(f"  {name}: shape={shape}")

    tasks = ds.meta.tasks
    print("\n=== tasks (meta.tasks) ===")
    try:
        if hasattr(tasks, "to_string"):
            print(tasks.to_string())
        else:
            print(repr(tasks))
    except Exception as e:
        print(f"(could not print tasks: {e})")

    # Task histogram from hf_dataset if task column exists
    print("\n=== task string histogram (from episodes / parquet) ===")
    try:
        hf = ds.hf_dataset
        col = "task" if "task" in hf.column_names else None
        if col:
            counts = Counter(hf[col])
            for k, v in counts.most_common(20):
                print(f"  {v:5d}  {k!r}")
            if len(counts) > 20:
                print(f"  ... ({len(counts)} unique)")
            if args.expected_task:
                bad = [t for t in counts if _normalize_task(str(t)) != _normalize_task(args.expected_task)]
                if bad:
                    print(
                        f"\n  WARNING Eval 1: {len(bad)} task string(s) do not match "
                        f"{args.expected_task!r}"
                    )
                    for t in bad[:10]:
                        print(f"      {t!r}")
                    if len(bad) > 10:
                        print("      ...")
                elif counts:
                    print(f"\n  OK: All sampled tasks match {args.expected_task!r}")
        else:
            print("  (no 'task' column in hf_dataset)")
    except Exception as e:
        print(f"  (skipped: {e})")

    idx = max(0, min(args.sample_index, len(ds) - 1))
    print(f"\n=== sample[{idx}] tensor shapes ===")
    row = ds[idx]
    for k, v in sorted(row.items()):
        if isinstance(v, torch.Tensor):
            print(f"  {k}: Tensor dtype={v.dtype} shape={tuple(v.shape)}")
        else:
            preview = repr(v)
            if len(preview) > 120:
                preview = preview[:117] + "..."
            print(f"  {k}: {type(v).__name__} {preview}")

    info_path = ds.root / "meta" / "info.json"
    if info_path.is_file():
        print("\n=== meta/info.json codebase_version ===")
        info = json.loads(info_path.read_text())
        print(f"  codebase_version: {info.get('codebase_version')}")


def _normalize_task(task: str) -> str:
    """Compare task strings without sensitivity to casing or final punctuation."""
    return task.strip().lower().rstrip(".")


def inspect_metadata_only(args: argparse.Namespace) -> None:
    """Inspect dataset metadata without importing torch/lerobot native libraries."""
    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            allow_patterns=["meta/**"],
            local_files_only=args.local_files_only,
        )
    )
    meta = root / "meta"

    print("=== Hugging Face dataset metadata ===")
    print(f"repo_id:          {args.repo_id}")
    print(f"snapshot_root:    {root}")

    info_path = meta / "info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text())
        print(f"codebase_version: {info.get('codebase_version')}")
        print(f"fps:              {info.get('fps')}")
        print(f"total_episodes:   {info.get('total_episodes')}")
        print(f"total_frames:     {info.get('total_frames')}")

        features = info.get("features")
        if isinstance(features, dict):
            print("\n=== features (meta/info.json) ===")
            for name, ft in sorted(features.items()):
                shape = ft.get("shape") if isinstance(ft, dict) else None
                dtype = ft.get("dtype") if isinstance(ft, dict) else None
                print(f"  {name}: dtype={dtype} shape={shape}")
    else:
        print("meta/info.json:   missing")

    tasks = read_parquet(meta / "tasks.parquet")
    tasks_source = "meta/tasks.parquet"
    if not tasks:
        tasks = read_json_or_jsonl(meta / "tasks.jsonl")
        tasks_source = "meta/tasks.jsonl"
    if not tasks:
        tasks = read_json_or_jsonl(meta / "tasks.json")
        tasks_source = "meta/tasks.json"
    print("\n=== tasks (metadata) ===")
    if tasks:
        print(f"source: {tasks_source}")
        task_strings = []
        for row in tasks:
            if isinstance(row, dict):
                task_strings.append(str(row.get("task") or row.get("name") or row))
            else:
                task_strings.append(str(row))
        counts = Counter(task_strings)
        for task, count in counts.most_common(20):
            print(f"  {count:5d}  {task!r}")
        if args.expected_task:
            expected = _normalize_task(args.expected_task)
            bad = [task for task in counts if _normalize_task(task) != expected]
            if bad:
                print(f"\n  WARNING Eval 1: {len(bad)} metadata task string(s) do not match {args.expected_task!r}")
                for task in bad[:10]:
                    print(f"      {task!r}")
            else:
                print(f"\n  OK: All metadata tasks match {args.expected_task!r}")
    else:
        if (meta / "tasks.parquet").is_file():
            print("  found meta/tasks.parquet, but no parquet reader is installed")
            print("  install one of: pyarrow, pandas+pyarrow, polars, or duckdb")
            print("  recommended:")
            print("    /home/htx/miniforge3/envs/lerobot/bin/python -m pip install pyarrow")
        else:
            print("  no meta/tasks.parquet, meta/tasks.jsonl, or meta/tasks.json found")

    print("\nSample tensor preview skipped. Re-run with --with-sample after fixing PyTorch import.")


def read_json_or_jsonl(path: Path) -> list[object]:
    if not path.is_file():
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return list(data.values())
    return [data]


def read_parquet(path: Path) -> list[object]:
    if not path.is_file():
        return []

    try:
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    except ModuleNotFoundError:
        pass

    try:
        import pandas as pd

        return pd.read_parquet(path).to_dict("records")
    except ModuleNotFoundError:
        pass

    try:
        import polars as pl

        return pl.read_parquet(path).to_dicts()
    except ModuleNotFoundError:
        pass

    try:
        import duckdb

        con = duckdb.connect(database=":memory:")
        cur = con.execute("SELECT * FROM read_parquet(?)", [str(path)])
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]
    except ModuleNotFoundError:
        pass

    return []


if __name__ == "__main__":
    main()
