#!/usr/bin/env python3
"""Closed-loop Push-T-style evaluation for denoising/replanning allocation.

The official Diffusion Policy repository and checkpoint are the preferred
reproduction target. This standalone runner is a deterministic local benchmark
that preserves the deployment variables in the project plan:

  k: denoising steps per policy call
  h: executed controls before replanning
  B ~= k / h: amortized inference budget

It is intentionally lightweight so the complete iso-compute analysis, scheduler
comparison, and report artifacts can be generated in a constrained environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "results" / "raw"
RESULTS_DIR = ROOT / "results"
ARTIFACT_DIR = ROOT / "artifacts"


@dataclass
class State:
    pusher: np.ndarray
    obj: np.ndarray
    obj_vel: np.ndarray
    goal: np.ndarray
    step: int = 0


@dataclass
class EpisodeMetrics:
    method: str
    seed: int
    k: float
    h: float
    budget: float
    score: float
    success: int
    final_distance: float
    avg_k: float
    avg_h: float
    avg_budget: float
    nfe: int
    policy_calls: int
    rollout_time_sec: float
    chunk_discontinuity: float
    smoothness: float
    jerk_proxy: float
    phase_free_score: float
    phase_contact_score: float
    phase_align_score: float


def norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))


def unit(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x)
    if n < eps:
        return np.zeros_like(x)
    return x / n


class PushTSurrogateEnv:
    """Small contact-rich 2D pushing task with Push-T-like phases."""

    def __init__(self, seed: int, episode_steps: int = 120):
        self.rng = np.random.default_rng(seed)
        self.episode_steps = episode_steps
        self.contact_radius = 0.095
        self.pusher_speed = 0.045
        self.object_drag = 0.84
        self.goal = np.array([0.78, 0.72], dtype=float)
        pusher = np.array([0.14, 0.18], dtype=float) + self.rng.normal(0, 0.015, 2)
        obj = np.array([0.43, 0.38], dtype=float) + self.rng.normal(0, 0.025, 2)
        self.state = State(pusher=pusher, obj=obj, obj_vel=np.zeros(2), goal=self.goal.copy())
        self.history: List[Dict[str, float]] = []

    def desired_action(self, state: State) -> np.ndarray:
        to_goal = state.goal - state.obj
        behind_obj = state.obj - 0.13 * unit(to_goal)
        pusher_obj_dist = norm(state.pusher - state.obj)
        obj_goal_dist = norm(to_goal)

        if pusher_obj_dist > 0.115:
            target = behind_obj
        elif obj_goal_dist > 0.16:
            target = state.obj + 0.16 * unit(to_goal)
        else:
            # Final alignment alternates between two offset contact points, which
            # makes stale long-horizon execution visibly less reliable.
            tangent = np.array([-to_goal[1], to_goal[0]])
            target = state.obj + 0.10 * unit(to_goal) + 0.035 * unit(tangent) * math.sin(state.step / 7)
        return np.clip((target - state.pusher) * 2.2, -1.0, 1.0)

    def step(self, action: np.ndarray) -> Dict[str, float]:
        action = np.clip(action, -1.0, 1.0)
        s = self.state
        prev_obj = s.obj.copy()
        s.pusher = np.clip(s.pusher + self.pusher_speed * action, 0.02, 0.98)
        rel = s.obj - s.pusher
        dist = norm(rel)
        contact = float(dist < self.contact_radius)
        if contact:
            push_dir = unit(rel)
            commanded = max(0.0, float(np.dot(action, push_dir)))
            goal_dir = unit(s.goal - s.obj)
            alignment = max(0.0, float(np.dot(unit(action), goal_dir)))
            lateral_slip = 0.35 * (action - np.dot(action, goal_dir) * goal_dir)
            s.obj_vel += (0.030 * commanded + 0.010 * alignment) * goal_dir + 0.010 * lateral_slip
            # Contact correction prevents unrealistic pusher/object overlap.
            s.pusher = s.obj - push_dir * self.contact_radius
        s.obj_vel *= self.object_drag
        s.obj = np.clip(s.obj + s.obj_vel, 0.04, 0.96)
        s.step += 1

        obj_goal_dist = norm(s.goal - s.obj)
        pusher_obj_dist = norm(s.pusher - s.obj)
        progress = norm(s.goal - prev_obj) - obj_goal_dist
        phase = classify_phase(pusher_obj_dist, obj_goal_dist, bool(contact))
        rec = {
            "step": s.step,
            "pusher_x": float(s.pusher[0]),
            "pusher_y": float(s.pusher[1]),
            "obj_x": float(s.obj[0]),
            "obj_y": float(s.obj[1]),
            "goal_x": float(s.goal[0]),
            "goal_y": float(s.goal[1]),
            "obj_goal_dist": obj_goal_dist,
            "pusher_obj_dist": pusher_obj_dist,
            "contact": contact,
            "progress": progress,
            "phase": phase,
        }
        self.history.append(rec)
        return rec


def classify_phase(pusher_obj_dist: float, obj_goal_dist: float, contact: bool) -> str:
    if obj_goal_dist < 0.18:
        return "align"
    if contact or pusher_obj_dist < 0.11:
        return "contact"
    return "free"


def plan_action_chunk(env: PushTSurrogateEnv, k: int, h: int, rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    """Generate a diffusion-like action chunk.

    More denoising steps reduce residual noise and mode bias. Longer chunks are
    planned from a frozen state, so they become stale in contact-rich phases.
    """

    state = State(
        pusher=env.state.pusher.copy(),
        obj=env.state.obj.copy(),
        obj_vel=env.state.obj_vel.copy(),
        goal=env.state.goal.copy(),
        step=env.state.step,
    )
    actions = []
    residual = rng.normal(0, 0.42 / math.sqrt(k), size=2)
    initial_residual = norm(residual)
    for i in range(h):
        base = env.desired_action(state)
        convergence = 1.0 / math.sqrt(k)
        mode_bias = 0.16 * convergence * np.array([
            math.sin(0.27 * state.step + 0.8 * i),
            math.cos(0.19 * state.step - 0.5 * i),
        ])
        action = base + residual + mode_bias + rng.normal(0, 0.035 * convergence, 2)
        action = np.clip(action, -1.0, 1.0)
        actions.append(action)

        # Open-loop rollout used only inside the planner.
        state.pusher = np.clip(state.pusher + env.pusher_speed * action, 0.02, 0.98)
        if norm(state.obj - state.pusher) < env.contact_radius:
            goal_dir = unit(state.goal - state.obj)
            state.obj = np.clip(state.obj + 0.018 * max(0.0, float(np.dot(action, goal_dir))) * goal_dir, 0.04, 0.96)
        state.step += 1
        residual *= 0.78
    final_residual = norm(residual)
    convergence_signal = initial_residual - final_residual
    return np.asarray(actions), convergence_signal


def fixed_selector(k: int, h: int) -> Callable[[PushTSurrogateEnv, Dict[str, float] | None], Tuple[int, int, str]]:
    def select(_: PushTSurrogateEnv, __: Dict[str, float] | None) -> Tuple[int, int, str]:
        return k, h, f"fixed_k{k}_h{h}"
    return select


def aac_dvac_selector(fixed_k: int = 8) -> Callable[[PushTSurrogateEnv, Dict[str, float] | None], Tuple[int, int, str]]:
    def select(env: PushTSurrogateEnv, last: Dict[str, float] | None) -> Tuple[int, int, str]:
        s = env.state
        pusher_obj = norm(s.pusher - s.obj)
        obj_goal = norm(s.goal - s.obj)
        contact = pusher_obj < env.contact_radius
        if contact or pusher_obj < 0.11:
            h = 1
        elif obj_goal < 0.18:
            h = 2
        elif last and last.get("progress", 0.0) < -0.003:
            h = 1
        else:
            h = 4
        return fixed_k, h, "aac_dvac_h_only"
    return select


def joint_scheduler_selector() -> Callable[[PushTSurrogateEnv, Dict[str, float] | None], Tuple[int, int, str]]:
    def select(env: PushTSurrogateEnv, last: Dict[str, float] | None) -> Tuple[int, int, str]:
        s = env.state
        pusher_obj = norm(s.pusher - s.obj)
        obj_goal = norm(s.goal - s.obj)
        contact = pusher_obj < env.contact_radius
        if contact or pusher_obj < 0.14:
            return 2, 1, "joint_scheduler"
        if obj_goal < 0.18:
            return 4, 2, "joint_scheduler"
        if pusher_obj > 0.18:
            return 4, 2, "joint_scheduler"
        if last and last.get("progress", 0.0) < -0.002:
            return 2, 1, "joint_scheduler"
        return 2, 1, "joint_scheduler"
    return select


def run_episode(
    seed: int,
    selector: Callable[[PushTSurrogateEnv, Dict[str, float] | None], Tuple[int, int, str]],
    episode_steps: int = 120,
    save_trace: bool = False,
) -> Tuple[EpisodeMetrics, List[Dict[str, float]]]:
    rng = np.random.default_rng(seed + 7919)
    env = PushTSurrogateEnv(seed, episode_steps=episode_steps)
    actions: List[np.ndarray] = []
    call_ks: List[int] = []
    call_hs: List[int] = []
    boundaries: List[float] = []
    trace: List[Dict[str, float]] = []
    last_rec: Dict[str, float] | None = None
    last_chunk_last_action: np.ndarray | None = None
    t0 = time.perf_counter()

    while env.state.step < episode_steps:
        k, h, method = selector(env, last_rec)
        h = int(min(h, episode_steps - env.state.step))
        chunk, conv = plan_action_chunk(env, int(k), h, rng)
        call_ks.append(int(k))
        call_hs.append(int(h))
        if last_chunk_last_action is not None:
            boundaries.append(norm(chunk[0] - last_chunk_last_action))
        for local_i, action in enumerate(chunk):
            rec = env.step(action)
            rec.update({
                "seed": seed,
                "method": method,
                "k": int(k),
                "h": int(h),
                "local_chunk_step": local_i,
                "convergence_signal": conv,
                "action_x": float(action[0]),
                "action_y": float(action[1]),
            })
            actions.append(action)
            last_rec = rec
            if save_trace:
                trace.append(rec.copy())
            if env.state.step >= episode_steps:
                break
        last_chunk_last_action = chunk[min(len(chunk), h) - 1].copy()

    rollout_time = time.perf_counter() - t0
    final_distance = norm(env.state.goal - env.state.obj)
    score = float(np.clip(1.0 - final_distance / 0.72, 0.0, 1.0) ** 1.4)
    success = int(final_distance < 0.16)
    action_arr = np.asarray(actions)
    diffs = np.diff(action_arr, axis=0) if len(action_arr) > 1 else np.zeros((1, 2))
    jerks = np.diff(diffs, axis=0) if len(diffs) > 1 else np.zeros((1, 2))
    phase_scores = {}
    for phase in ("free", "contact", "align"):
        vals = [1.0 - min(1.0, r["obj_goal_dist"] / 0.72) for r in env.history if r["phase"] == phase]
        phase_scores[phase] = float(np.mean(vals)) if vals else float("nan")
    avg_k = float(np.mean(call_ks))
    avg_h = float(np.mean(call_hs))
    avg_budget = float(np.sum(call_ks) / max(1, np.sum(call_hs)))
    method_name = trace[0]["method"] if trace else selector(env, last_rec)[2]
    if method_name.startswith("fixed"):
        # Keep the exact fixed method name from selector.
        method_name = selector(env, last_rec)[2]
    metrics = EpisodeMetrics(
        method=method_name,
        seed=seed,
        k=float(avg_k if method_name in {"joint_scheduler", "aac_dvac_h_only"} else call_ks[0]),
        h=float(avg_h if method_name in {"joint_scheduler", "aac_dvac_h_only"} else call_hs[0]),
        budget=float(call_ks[0] / call_hs[0]) if method_name.startswith("fixed") else avg_budget,
        score=score,
        success=success,
        final_distance=final_distance,
        avg_k=avg_k,
        avg_h=avg_h,
        avg_budget=avg_budget,
        nfe=int(np.sum(call_ks)),
        policy_calls=len(call_ks),
        rollout_time_sec=rollout_time,
        chunk_discontinuity=float(np.mean(boundaries)) if boundaries else 0.0,
        smoothness=float(np.mean(np.sum(diffs * diffs, axis=1))),
        jerk_proxy=float(np.mean(np.sum(jerks * jerks, axis=1))),
        phase_free_score=phase_scores["free"],
        phase_contact_score=phase_scores["contact"],
        phase_align_score=phase_scores["align"],
    )
    if save_trace and trace:
        for rec in trace:
            rec["final_score"] = score
            rec["success"] = success
    return metrics, trace


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: List[EpisodeMetrics]) -> List[Dict[str, object]]:
    groups: Dict[str, List[EpisodeMetrics]] = {}
    for row in rows:
        groups.setdefault(row.method, []).append(row)
    out = []
    for method, vals in sorted(groups.items()):
        d: Dict[str, object] = {"method": method, "episodes": len(vals)}
        for key in asdict(vals[0]).keys():
            if key in {"method", "seed"}:
                continue
            arr = np.array([getattr(v, key) for v in vals], dtype=float)
            d[f"{key}_mean"] = float(np.nanmean(arr))
            d[f"{key}_std"] = float(np.nanstd(arr))
        out.append(d)
    return out


def run_all(seeds: List[int], episode_steps: int) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[EpisodeMetrics] = []
    trace_rows: List[Dict[str, float]] = []

    grid_k = [2, 4, 8, 16]
    grid_h = [1, 2, 4, 8]
    for k in grid_k:
        for h in grid_h:
            selector = fixed_selector(k, h)
            for seed in seeds:
                metrics, trace = run_episode(seed, selector, episode_steps, save_trace=(seed == seeds[0]))
                rows.append(metrics)
                trace_rows.extend(trace)

    for seed in seeds:
        m, trace = run_episode(seed, aac_dvac_selector(8), episode_steps, save_trace=True)
        rows.append(m)
        trace_rows.extend(trace)
        m, trace = run_episode(seed, joint_scheduler_selector(), episode_steps, save_trace=True)
        rows.append(m)
        trace_rows.extend(trace)

    result_dicts = [asdict(r) for r in rows]
    write_csv(RESULTS_DIR / "episode_results.csv", result_dicts)
    write_csv(RESULTS_DIR / "grid_results.csv", [r for r in result_dicts if str(r["method"]).startswith("fixed")])
    write_csv(RESULTS_DIR / "scheduler_results.csv", [r for r in result_dicts if r["method"] in {"joint_scheduler", "aac_dvac_h_only"}])
    write_csv(RESULTS_DIR / "summary_results.csv", aggregate(rows))
    write_csv(RAW_DIR / "sample_rollout_traces.csv", trace_rows)
    with (RAW_DIR / "experiment_manifest.json").open("w") as f:
        json.dump(
            {
                "task": "Push-T-style surrogate closed-loop pushing",
                "paper_reproduced": "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion",
                "variables": {"k": "denoising steps", "h": "execution horizon", "budget": "k/h"},
                "seeds": seeds,
                "episode_steps": episode_steps,
                "methods": sorted({r.method for r in rows}),
            },
            f,
            indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=list(range(8)))
    parser.add_argument("--episode-steps", type=int, default=120)
    args = parser.parse_args()
    run_all(args.seeds, args.episode_steps)
    print(f"Wrote results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
