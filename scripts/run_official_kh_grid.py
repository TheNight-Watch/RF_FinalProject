#!/usr/bin/env python3
"""Evaluate fixed (k, h) settings with the official Diffusion Policy checkpoint.

This extends the official PushT reproduction by sweeping inference-time
denoising steps k and executed horizon h on the upstream low-dimensional PushT
runner. It is intentionally narrower than the surrogate grid: the goal is to
provide official-checkpoint evidence that equal-compute settings behave
differently, not to replace the full controlled scheduler study.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import dill
import hydra
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "official_reproduction" / "diffusion_policy_source"
CKPT = ROOT / "official_reproduction" / "data" / "epoch=0550-test_mean_score=0.969.ckpt"
OUT_ROOT = ROOT / "official_reproduction" / "pusht_official_kh_grid"
RESULTS = ROOT / "results"
sys.path.insert(0, str(SOURCE))


def parse_pair(text: str) -> tuple[int, int]:
    if "," in text:
        left, right = text.split(",", 1)
    elif ":" in text:
        left, right = text.split(":", 1)
    else:
        raise argparse.ArgumentTypeError("Pairs must be formatted as k,h or k:h")
    return int(left), int(right)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CKPT)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pairs", nargs="+", type=parse_pair, default=[(2, 1), (4, 2), (8, 4), (16, 8)])
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--n-envs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--test-start-seed", type=int, default=4300000)
    return parser.parse_args()


def run_pair(args: argparse.Namespace, k: int, h: int) -> dict:
    out_dir = args.output_root / f"k{k}_h{h}"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(open(args.checkpoint, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    cfg.task.env_runner.n_train = 0
    cfg.task.env_runner.n_train_vis = 0
    cfg.task.env_runner.n_test = args.n_test
    cfg.task.env_runner.n_test_vis = 0
    cfg.task.env_runner.n_envs = args.n_envs
    cfg.task.env_runner.max_steps = args.max_steps
    cfg.task.env_runner.test_start_seed = args.test_start_seed
    cfg.task.env_runner.n_action_steps = h

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=str(out_dir))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.num_inference_steps = k
    policy.n_action_steps = h

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    policy.to(device)
    policy.eval()

    env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=str(out_dir))
    try:
        runner_log = env_runner.run(policy)
    finally:
        if hasattr(env_runner, "env"):
            env_runner.env.close()

    seed_scores = {
        key.removeprefix("test/sim_max_reward_"): float(value)
        for key, value in runner_log.items()
        if key.startswith("test/sim_max_reward_")
    }
    scores = np.asarray(list(seed_scores.values()), dtype=np.float64)
    row = {
        "method": f"official_fixed_k{k}_h{h}",
        "k": k,
        "h": h,
        "budget": k / h,
        "n_test": args.n_test,
        "n_envs": args.n_envs,
        "max_steps": args.max_steps,
        "mean_score": float(scores.mean()),
        "std_score": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        "success_rate_095": float((scores >= 0.95).mean()),
        "min_score": float(scores.min()),
        "max_score": float(scores.max()),
        "policy_calls_per_episode": float(np.ceil(args.max_steps / h)),
        "nfe_per_episode": float(np.ceil(args.max_steps / h) * k),
        "log_path": str(out_dir / "eval_log.json"),
    }
    payload_out = {"summary": row, "seed_scores": seed_scores}
    (out_dir / "eval_log.json").write_text(json.dumps(payload_out, indent=2, sort_keys=True), encoding="utf-8")
    return row


def main() -> None:
    args = parse_args()
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("CUDA_DEVICE_MEMORY_SHARED_CACHE", "/tmp/finalproject-vgpu-cache.cache")
    args.output_root.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    rows = [run_pair(args, k, h) for k, h in args.pairs]
    out_csv = RESULTS / "official_kh_grid_results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": rows, "csv": str(out_csv)}, indent=2))


if __name__ == "__main__":
    main()
