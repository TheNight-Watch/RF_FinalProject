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
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "official_reproduction" / "diffusion_policy_source"
CKPT = ROOT / "official_reproduction" / "data" / "epoch=0550-test_mean_score=0.969.ckpt"
OUT_ROOT = ROOT / "official_reproduction" / "pusht_official_kh_grid"
RESULTS = ROOT / "results"
sys.path.insert(0, str(SOURCE))


class DeploymentWrapper:
    """Thin deployment wrapper for observation/action post-processing."""

    def __init__(
        self,
        policy,
        action_mode: str,
        obs_impute: str,
        max_target_delta: float,
        geom_alpha: float,
        policy_seed: int | None,
        occlude_masked_keypoints: bool,
    ):
        self.policy = policy
        self.action_mode = action_mode
        self.obs_impute = obs_impute
        self.max_target_delta = max_target_delta
        self.geom_alpha = geom_alpha
        self.policy_seed = policy_seed
        self.occlude_masked_keypoints = occlude_masked_keypoints
        self.call_idx = 0
        self.last_keypoints = None
        self.has_keypoints = None
        self.prev_keypoints = None
        self.has_prev_keypoints = None
        self.last_full_frame = None
        self.has_full_frame = None
        self.keypoint_mean = None

    def __getattr__(self, name: str):
        return getattr(self.policy, name)

    def reset(self):
        self.call_idx = 0
        self.last_keypoints = None
        self.has_keypoints = None
        self.prev_keypoints = None
        self.has_prev_keypoints = None
        self.last_full_frame = None
        self.has_full_frame = None
        if hasattr(self.policy, "reset"):
            return self.policy.reset()
        return None

    def _keypoint_prior(self, device, dtype):
        if self.keypoint_mean is None:
            stats = self.policy.normalizer["obs"].get_input_stats()
            self.keypoint_mean = stats["mean"][:18].detach().clone()
        return self.keypoint_mean.to(device=device, dtype=dtype)

    def _preprocess_obs(self, obs_dict):
        if self.obs_impute == "none" and not self.occlude_masked_keypoints:
            return obs_dict
        modes = {"none", "carry_forward", "mean_prior", "frame_hold", "linear", "oracle"}
        if self.obs_impute not in modes:
            raise ValueError(f"Unknown observation imputation mode: {self.obs_impute}")
        if "obs_mask" not in obs_dict:
            return obs_dict

        obs = obs_dict["obs"].clone()
        mask = obs_dict["obs_mask"].to(device=obs.device, dtype=torch.bool)
        if obs.shape[-1] < 20:
            return obs_dict

        keypoints = obs[:, :, :18]
        keypoint_mask = mask[:, :, :18]
        if self.obs_impute == "oracle":
            out = dict(obs_dict)
            out["obs"] = obs
            return out

        if self.occlude_masked_keypoints:
            keypoints = torch.where(keypoint_mask, keypoints, torch.zeros_like(keypoints))
            obs[:, :, :18] = keypoints
        if self.obs_impute == "none":
            out = dict(obs_dict)
            out["obs"] = obs
            return out

        batch = obs.shape[0]
        mean_prior = self._keypoint_prior(obs.device, obs.dtype).expand(batch, -1)
        if self.last_keypoints is None or self.last_keypoints.shape[0] != batch:
            self.last_keypoints = mean_prior.clone()
            self.has_keypoints = torch.zeros((batch, 18), device=obs.device, dtype=torch.bool)
            self.prev_keypoints = mean_prior.clone()
            self.has_prev_keypoints = torch.zeros((batch, 18), device=obs.device, dtype=torch.bool)
            self.last_full_frame = mean_prior.clone()
            self.has_full_frame = torch.zeros((batch, 1), device=obs.device, dtype=torch.bool)
        else:
            self.last_keypoints = self.last_keypoints.to(device=obs.device, dtype=obs.dtype)
            self.has_keypoints = self.has_keypoints.to(device=obs.device)
            self.prev_keypoints = self.prev_keypoints.to(device=obs.device, dtype=obs.dtype)
            self.has_prev_keypoints = self.has_prev_keypoints.to(device=obs.device)
            self.last_full_frame = self.last_full_frame.to(device=obs.device, dtype=obs.dtype)
            self.has_full_frame = self.has_full_frame.to(device=obs.device)

        for t_idx in range(obs.shape[1]):
            visible = keypoint_mask[:, t_idx]
            current = keypoints[:, t_idx]
            if self.obs_impute == "mean_prior":
                fill_value = mean_prior
            elif self.obs_impute == "carry_forward":
                fill_value = torch.where(self.has_keypoints, self.last_keypoints, mean_prior)
            elif self.obs_impute == "frame_hold":
                fill_value = torch.where(self.has_full_frame, self.last_full_frame, mean_prior)
            elif self.obs_impute == "linear":
                extrapolated = self.last_keypoints + (self.last_keypoints - self.prev_keypoints)
                extrapolated = torch.clamp(extrapolated, min=0.0, max=512.0)
                has_linear = self.has_keypoints & self.has_prev_keypoints
                fill_value = torch.where(has_linear, extrapolated, torch.where(self.has_keypoints, self.last_keypoints, mean_prior))
            else:
                raise ValueError(f"Unexpected observation imputation mode: {self.obs_impute}")

            current = torch.where(visible, current, fill_value)
            keypoints[:, t_idx] = current
            self.prev_keypoints = torch.where(visible, self.last_keypoints, self.prev_keypoints)
            self.has_prev_keypoints = self.has_prev_keypoints | (visible & self.has_keypoints)
            self.last_keypoints = torch.where(visible, current, self.last_keypoints)
            self.has_keypoints = self.has_keypoints | visible
            full_visible = visible.all(dim=1, keepdim=True)
            self.last_full_frame = torch.where(full_visible, current, self.last_full_frame)
            self.has_full_frame = self.has_full_frame | full_visible

        obs[:, :, :18] = keypoints
        out = dict(obs_dict)
        out["obs"] = obs
        return out

    def predict_action(self, obs_dict):
        obs_dict = self._preprocess_obs(obs_dict)
        if self.policy_seed is not None:
            seed = int(self.policy_seed + self.call_idx)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            self.call_idx += 1
        result = self.policy.predict_action(obs_dict)
        if self.action_mode == "none":
            return result
        if self.action_mode not in {"clip", "rate_limit", "geom_fallback"}:
            raise ValueError(f"Unknown action postprocess mode: {self.action_mode}")

        action = result["action"]
        filtered = torch.clamp(action.clone(), min=0.0, max=512.0)
        if self.action_mode == "clip":
            result = dict(result)
            result["action_raw"] = action
            result["action"] = filtered
            return result

        if self.action_mode == "geom_fallback":
            obs = obs_dict["obs"][:, -1].to(device=action.device, dtype=action.dtype)
            block_kps = obs[:, :18].reshape(obs.shape[0], 9, 2)
            agent = obs[:, -2:]
            block_center = block_kps.mean(dim=1)
            goal = torch.tensor([256.0, 256.0], device=action.device, dtype=action.dtype).expand_as(block_center)
            to_goal = goal - block_center
            to_goal_norm = torch.linalg.norm(to_goal, dim=-1, keepdim=True).clamp_min(torch.finfo(action.dtype).eps)
            goal_dir = to_goal / to_goal_norm
            behind = torch.clamp(block_center - 72.0 * goal_dir, min=0.0, max=512.0)
            push_target = torch.clamp(block_center + 48.0 * goal_dir, min=0.0, max=512.0)
            agent_to_behind = torch.linalg.norm(behind - agent, dim=-1, keepdim=True)
            fallback = torch.where(agent_to_behind > 56.0, behind, push_target)
            raw_dir = action[:, :1, :] - agent[:, None, :]
            fallback_dir = fallback[:, None, :] - agent[:, None, :]
            alignment = (raw_dir * fallback_dir).sum(dim=-1, keepdim=True)
            should_filter = alignment < 0.0
            alpha = float(self.geom_alpha)
            proposal = (1.0 - alpha) * action + alpha * fallback[:, None, :]
            filtered = torch.where(should_filter, proposal, action)

        prev = obs_dict["obs"][:, -1, -2:].to(device=action.device, dtype=action.dtype)
        limit = float(self.max_target_delta)
        eps = torch.finfo(action.dtype).eps
        for step_idx in range(action.shape[1]):
            delta = filtered[:, step_idx] - prev
            norm = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(eps)
            scale = torch.clamp(limit / norm, max=1.0)
            prev = prev + delta * scale
            filtered[:, step_idx] = prev

        result = dict(result)
        result["action_raw"] = action
        result["action"] = filtered
        return result


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
    parser.add_argument("--csv-out", type=Path, default=RESULTS / "official_kh_grid_results.csv")
    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddpm")
    parser.add_argument("--postprocess", choices=["none", "clip", "rate_limit", "geom_fallback"], default="none")
    parser.add_argument(
        "--obs-impute",
        choices=["none", "carry_forward", "mean_prior", "frame_hold", "linear", "oracle"],
        default="none",
    )
    parser.add_argument("--max-target-delta", type=float, default=64.0)
    parser.add_argument("--geom-alpha", type=float, default=0.75)
    parser.add_argument("--policy-seed", type=int, default=None)
    parser.add_argument("--keypoint-visible-rate", type=float, default=None)
    parser.add_argument("--occlude-masked-keypoints", action="store_true")
    parser.add_argument(
        "--ddim-eta",
        type=float,
        default=0.0,
        help="DDIM stochasticity parameter. Only used with --sampler ddim.",
    )
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
    if args.keypoint_visible_rate is not None:
        cfg.task.env_runner.keypoint_visible_rate = args.keypoint_visible_rate

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=str(out_dir))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    if args.sampler == "ddim":
        policy.noise_scheduler = DDIMScheduler.from_config(policy.noise_scheduler.config)
        policy.kwargs["eta"] = args.ddim_eta
    policy.num_inference_steps = k
    policy.n_action_steps = h

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    policy.to(device)
    policy.eval()
    eval_policy = DeploymentWrapper(
        policy,
        args.postprocess,
        args.obs_impute,
        args.max_target_delta,
        args.geom_alpha,
        args.policy_seed,
        args.occlude_masked_keypoints,
    )

    env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=str(out_dir))
    try:
        runner_log = env_runner.run(eval_policy)
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
        "method": f"official_{args.sampler}_fixed_k{k}_h{h}",
        "sampler": args.sampler,
        "postprocess": args.postprocess,
        "obs_impute": args.obs_impute,
        "max_target_delta": float(args.max_target_delta) if args.postprocess in {"rate_limit", "geom_fallback"} else None,
        "geom_alpha": float(args.geom_alpha) if args.postprocess == "geom_fallback" else None,
        "policy_seed": args.policy_seed,
        "keypoint_visible_rate": args.keypoint_visible_rate,
        "occlude_masked_keypoints": bool(args.occlude_masked_keypoints),
        "k": k,
        "h": h,
        "budget": k / h,
        "n_test": args.n_test,
        "n_envs": args.n_envs,
        "max_steps": args.max_steps,
        "mean_score": float(scores.mean()),
        "std_score": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        "success_rate_095": float((scores >= 0.95).mean()),
        "ddim_eta": float(args.ddim_eta) if args.sampler == "ddim" else None,
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
    out_csv = args.csv_out
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": rows, "csv": str(out_csv)}, indent=2))


if __name__ == "__main__":
    main()
