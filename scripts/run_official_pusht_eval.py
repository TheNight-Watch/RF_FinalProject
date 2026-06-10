#!/usr/bin/env python3
"""Run the official Diffusion Policy PushT checkpoint with small overrides.

The upstream eval.py has no CLI overrides for the number of episodes. This
wrapper keeps the checkpoint, workspace, policy, and PushT runner from the
official repository, while allowing short smoke tests or a longer n-test run in
the current Python 3.12 container.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import dill
import hydra
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "official_reproduction" / "diffusion_policy_source"
CKPT = ROOT / "official_reproduction" / "data" / "epoch=0550-test_mean_score=0.969.ckpt"
OUT = ROOT / "official_reproduction" / "pusht_eval_official_smoke"
sys.path.insert(0, str(SOURCE))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CKPT)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-test", type=int, default=1)
    parser.add_argument("--n-train", type=int, default=0)
    parser.add_argument("--n-test-vis", type=int, default=0)
    parser.add_argument("--n-train-vis", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--test-start-seed", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("CUDA_DEVICE_MEMORY_SHARED_CACHE", "/tmp/finalproject-vgpu-cache.cache")

    payload = torch.load(open(args.checkpoint, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    cfg.task.env_runner.n_train = args.n_train
    cfg.task.env_runner.n_train_vis = args.n_train_vis
    cfg.task.env_runner.n_test = args.n_test
    cfg.task.env_runner.n_test_vis = args.n_test_vis
    cfg.task.env_runner.n_envs = args.n_envs
    cfg.task.env_runner.max_steps = args.max_steps
    if args.test_start_seed is not None:
        cfg.task.env_runner.test_start_seed = args.test_start_seed

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=str(args.output_dir))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    if args.num_inference_steps is not None:
        policy.num_inference_steps = args.num_inference_steps

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    policy.to(device)
    policy.eval()

    env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=str(args.output_dir))
    try:
        runner_log = env_runner.run(policy)
    finally:
        if hasattr(env_runner, "env"):
            env_runner.env.close()

    json_log = {}
    for key, value in runner_log.items():
        json_log[key] = getattr(value, "_path", value)
    metadata = {
        "checkpoint": str(args.checkpoint),
        "source": str(SOURCE),
        "device": str(device),
        "n_test": args.n_test,
        "n_train": args.n_train,
        "n_envs": args.n_envs,
        "max_steps": args.max_steps,
        "num_inference_steps": getattr(policy, "num_inference_steps", None),
        "runner_log": json_log,
    }
    out_path = args.output_dir / "eval_log.json"
    out_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
