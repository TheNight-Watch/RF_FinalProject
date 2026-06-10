#!/usr/bin/env python3
"""Run a small real PushT rollout with the LeRobot Diffusion Policy checkpoint.

This is a near-official reproduction path for the same PushT task used by the
Diffusion Policy paper. It uses gym-pusht directly and manually applies the
normalization buffers stored in the older LeRobot checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import gym_pusht  # noqa: F401
import imageio.v2 as imageio
import numpy as np
import torch
from safetensors import safe_open

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "official_reproduction" / "hf_lerobot_diffusion_pusht"
RESULTS_DIR = ROOT / "results"
VIDEO_DIR = ROOT / "artifacts" / "videos"


@dataclass
class EpisodeResult:
    episode: int
    seed: int
    steps: int
    total_reward: float
    max_reward: float
    success: bool
    elapsed_s: float
    inference_steps: int
    mean_action_x: float
    mean_action_y: float
    mean_action_delta: float


def _load_norm_buffers(model_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    keys = {
        "image_mean": "normalize_inputs.buffer_observation_image.mean",
        "image_std": "normalize_inputs.buffer_observation_image.std",
        "state_min": "normalize_inputs.buffer_observation_state.min",
        "state_max": "normalize_inputs.buffer_observation_state.max",
        "action_min": "unnormalize_outputs.buffer_action.min",
        "action_max": "unnormalize_outputs.buffer_action.max",
    }
    buffers: dict[str, torch.Tensor] = {}
    with safe_open(model_path, framework="pt", device="cpu") as f:
        for out_key, tensor_key in keys.items():
            buffers[out_key] = f.get_tensor(tensor_key).to(device=device, dtype=torch.float32)
    return buffers


def _minmax_normalize(x: torch.Tensor, min_value: torch.Tensor, max_value: torch.Tensor) -> torch.Tensor:
    denom = torch.where(max_value == min_value, torch.ones_like(max_value), max_value - min_value)
    return 2.0 * (x - min_value) / denom - 1.0


def _minmax_unnormalize(x: torch.Tensor, min_value: torch.Tensor, max_value: torch.Tensor) -> torch.Tensor:
    return (x + 1.0) * 0.5 * (max_value - min_value) + min_value


def _make_batch(obs: dict[str, np.ndarray], buffers: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    image = torch.as_tensor(obs["pixels"], device=device, dtype=torch.float32).permute(2, 0, 1) / 255.0
    image = (image - buffers["image_mean"]) / (buffers["image_std"] + 1e-8)
    state = torch.as_tensor(obs["agent_pos"], device=device, dtype=torch.float32)
    state = _minmax_normalize(state, buffers["state_min"], buffers["state_max"])
    return {
        "observation.image": image.unsqueeze(0),
        "observation.state": state.unsqueeze(0),
    }


def _run_episode(
    policy: DiffusionPolicy,
    buffers: dict[str, torch.Tensor],
    seed: int,
    episode: int,
    max_steps: int,
    save_video: bool,
    video_fps: int,
) -> EpisodeResult:
    device = next(policy.parameters()).device
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=96,
        observation_height=96,
        visualization_width=384,
        visualization_height=384,
    )
    obs, _ = env.reset(seed=seed)
    policy.reset()

    frames: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    total_reward = 0.0
    max_reward = -math.inf
    start = time.perf_counter()

    for step in range(max_steps):
        batch = _make_batch(obs, buffers, device)
        with torch.no_grad():
            action_norm = policy.select_action(batch)
        action = _minmax_unnormalize(
            action_norm.squeeze(0).detach().to(device=device, dtype=torch.float32),
            buffers["action_min"],
            buffers["action_max"],
        )
        action_np = action.detach().cpu().numpy().astype(np.float32)
        action_np = np.clip(action_np, env.action_space.low, env.action_space.high)

        obs, reward, terminated, truncated, _ = env.step(action_np)
        total_reward += float(reward)
        max_reward = max(max_reward, float(reward))
        actions.append(action_np)

        if save_video:
            frames.append(env.render())

        if terminated or truncated:
            step += 1
            break
    else:
        step = max_steps

    elapsed = time.perf_counter() - start
    env.close()

    if save_video and frames:
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(VIDEO_DIR / f"lerobot_pusht_episode_{episode:02d}.gif", frames, fps=video_fps)

    action_arr = np.asarray(actions, dtype=np.float32) if actions else np.zeros((1, 2), dtype=np.float32)
    deltas = np.linalg.norm(np.diff(action_arr, axis=0), axis=1) if len(action_arr) > 1 else np.asarray([0.0])
    return EpisodeResult(
        episode=episode,
        seed=seed,
        steps=int(step),
        total_reward=float(total_reward),
        max_reward=float(max_reward),
        success=bool(max_reward >= 0.95),
        elapsed_s=float(elapsed),
        inference_steps=int(policy.diffusion.num_inference_steps),
        mean_action_x=float(action_arr[:, 0].mean()),
        mean_action_y=float(action_arr[:, 1].mean()),
        mean_action_delta=float(deltas.mean()),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--num-inference-steps", type=int, default=32)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    model_path = args.model_dir / "model.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {model_path}")

    policy = DiffusionPolicy.from_pretrained(args.model_dir)
    policy.diffusion.num_inference_steps = args.num_inference_steps
    device = next(policy.parameters()).device
    buffers = _load_norm_buffers(model_path, device)

    results = [
        _run_episode(
            policy=policy,
            buffers=buffers,
            seed=args.seed + idx,
            episode=idx,
            max_steps=args.max_steps,
            save_video=args.save_video and idx == 0,
            video_fps=args.video_fps,
        )
        for idx in range(args.episodes)
    ]

    csv_path = RESULTS_DIR / "lerobot_pusht_smoke_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in results)

    summary = {
        "model_dir": str(args.model_dir),
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed_start": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "mean_total_reward": float(np.mean([r.total_reward for r in results])),
        "mean_max_reward": float(np.mean([r.max_reward for r in results])),
        "success_rate": float(np.mean([r.success for r in results])),
        "mean_steps": float(np.mean([r.steps for r in results])),
        "mean_elapsed_s": float(np.mean([r.elapsed_s for r in results])),
        "note": (
            "Checkpoint was downloaded via hf-mirror after official large-file transfers timed out; "
            "safetensors structure and model loading were verified, but the assembled file SHA did not "
            "match the HF LFS metadata."
        ),
        "episodes_detail": [asdict(row) for row in results],
    }
    json_path = RESULTS_DIR / "lerobot_pusht_smoke_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
