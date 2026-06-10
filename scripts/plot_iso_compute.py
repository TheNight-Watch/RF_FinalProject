#!/usr/bin/env python3
"""Generate figures and a compact rollout video for the project report."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
VIDEOS = ROOT / "artifacts" / "videos"


def parse_kh(method: str) -> tuple[int, int] | None:
    if not method.startswith("fixed_k"):
        return None
    left, h = method.replace("fixed_k", "").split("_h")
    return int(left), int(h)


def save_heatmap(summary: pd.DataFrame) -> None:
    fixed = summary[summary.method.str.startswith("fixed")].copy()
    fixed[["k_val", "h_val"]] = fixed["method"].apply(lambda m: pd.Series(parse_kh(m)))
    table = fixed.pivot(index="k_val", columns="h_val", values="score_mean").sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    im = ax.imshow(table.values, cmap="viridis", vmin=0.25, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set_xlabel("Execution horizon h")
    ax.set_ylabel("Denoising steps k")
    ax.set_title("Push-T Surrogate Score over (k, h)")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, f"{table.values[i, j]:.2f}", ha="center", va="center", color="white", fontsize=10)
    fig.colorbar(im, ax=ax, label="Mean score")
    fig.tight_layout()
    fig.savefig(FIGURES / "kh_score_heatmap.png", dpi=220)
    plt.close(fig)


def save_iso_curves(summary: pd.DataFrame) -> None:
    fixed = summary[summary.method.str.startswith("fixed")].copy()
    fixed[["k_val", "h_val"]] = fixed["method"].apply(lambda m: pd.Series(parse_kh(m)))
    fixed["B"] = fixed.k_val / fixed.h_val
    frontiers = [1, 2, 4]
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for b in frontiers:
        sub = fixed[np.isclose(fixed.B, b)].sort_values("h_val")
        ax.plot(sub.h_val, sub.score_mean, marker="o", label=f"B={b:g}")
        for _, row in sub.iterrows():
            ax.annotate(f"({int(row.k_val)},{int(row.h_val)})", (row.h_val, row.score_mean),
                        textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    for method, color, label in [
        ("joint_scheduler", "#d62728", "Joint scheduler"),
        ("joint_scheduler_safe", "#7f3c8d", "Safe joint scheduler"),
    ]:
        if method in set(summary.method):
            joint = summary[summary.method == method].iloc[0]
            ax.scatter([joint.h_mean], [joint.score_mean], s=120, marker="*", color=color, label=label)
    ax.set_xlabel("Execution horizon h")
    ax.set_ylabel("Mean score")
    ax.set_title("Iso-Compute Frontiers: Denoising vs Replanning")
    ax.set_ylim(0.25, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "iso_compute_curves.png", dpi=220)
    plt.close(fig)


def save_pareto(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    fixed = summary[summary.method.str.startswith("fixed")].copy()
    ax.scatter(fixed.avg_budget_mean, fixed.score_mean, s=70, color="#4c78a8", label="Fixed (k,h)")
    for _, row in fixed.iterrows():
        if abs(row.avg_budget_mean - 2.0) < 1e-6 or row.score_mean > 0.88:
            kh = parse_kh(row.method)
            ax.annotate(f"{kh}", (row.avg_budget_mean, row.score_mean), textcoords="offset points", xytext=(4, 5), fontsize=8)
    for method, color, marker in [
        ("aac_dvac_h_only", "#f58518", "s"),
        ("joint_scheduler", "#d62728", "*"),
        ("joint_scheduler_safe", "#7f3c8d", "P"),
    ]:
        row = summary[summary.method == method].iloc[0]
        ax.scatter(row.avg_budget_mean, row.score_mean, s=150 if marker == "*" else 90, color=color, marker=marker, label=method)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Amortized compute B = k/h")
    ax.set_ylabel("Mean score")
    ax.set_title("Score vs Amortized Inference Budget")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "pareto_score_compute.png", dpi=220)
    plt.close(fig)


def save_smoothness(summary: pd.DataFrame) -> None:
    methods = ["fixed_k2_h1", "fixed_k4_h2", "fixed_k8_h4", "fixed_k16_h8", "aac_dvac_h_only", "joint_scheduler", "joint_scheduler_safe"]
    labels = ["(2,1)", "(4,2)", "(8,4)", "(16,8)", "h-only", "joint", "safe"]
    sub = summary.set_index("method").loc[methods]
    x = np.arange(len(methods))
    fig, ax1 = plt.subplots(figsize=(8.2, 4.8))
    ax1.bar(x - 0.18, sub.score_mean, width=0.36, color="#4c78a8", label="Score")
    ax1.set_ylabel("Mean score")
    ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, sub.smoothness_mean, width=0.36, color="#54a24b", label="Smoothness cost")
    ax2.set_ylabel("Action smoothness cost")
    ax1.set_xticks(x, labels)
    ax1.set_title("Budget B=2 Comparison")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "smoothness_vs_compute.png", dpi=220)
    plt.close(fig)


def save_main_comparison_bars(summary: pd.DataFrame) -> None:
    methods = [
        "fixed_k8_h4",
        "fixed_k2_h1",
        "fixed_k4_h2",
        "aac_dvac_h_only",
        "joint_scheduler",
        "joint_scheduler_safe",
    ]
    labels = ["default\n(8,4)", "best fixed\n(2,1)", "denoise-only\n(4,2)", "h-only", "joint", "safe\njoint"]
    sub = summary.set_index("method").loc[methods]
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    axes[0].bar(x - 0.18, sub.score_mean, width=0.36, color="#4c78a8", label="Score")
    axes[0].bar(x + 0.18, sub.success_mean, width=0.36, color="#72b7b2", label="Success")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xticks(x, labels)
    axes[0].set_title("Task Performance")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(x - 0.18, sub.policy_calls_mean, width=0.36, color="#f58518", label="Policy calls")
    axes[1].bar(x + 0.18, sub.nfe_mean, width=0.36, color="#e45756", label="NFE")
    axes[1].set_xticks(x, labels)
    axes[1].set_yscale("log")
    axes[1].set_title("Inference Work")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.2)

    fig.suptitle("Main Baseline Comparison under B ~= 2")
    fig.tight_layout()
    fig.savefig(FIGURES / "main_comparison_bars.png", dpi=220)
    plt.close(fig)


def save_phase_plot(summary: pd.DataFrame) -> None:
    methods = ["fixed_k2_h1", "fixed_k4_h2", "fixed_k8_h4", "joint_scheduler", "joint_scheduler_safe"]
    labels = ["(2,1)", "(4,2)", "(8,4)", "joint", "safe"]
    sub = summary.set_index("method").loc[methods]
    phases = ["phase_free_score_mean", "phase_contact_score_mean", "phase_align_score_mean"]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for i, phase in enumerate(phases):
        ax.bar(x + (i - 1) * width, sub[phase], width=width, label=phase.replace("phase_", "").replace("_score_mean", ""))
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Phase score proxy")
    ax.set_title("Phase-Wise Allocation Effects")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "phase_wise_scores.png", dpi=220)
    plt.close(fig)


def save_official_frontier() -> None:
    path = RESULTS / "official_kh_grid_results.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    labels = [f"({int(r.k)},{int(r.h)})" for _, r in df.iterrows()]
    x = np.arange(len(df))
    fig, ax1 = plt.subplots(figsize=(8.2, 4.6))
    ax1.bar(x - 0.18, df.mean_score, width=0.36, color="#4c78a8", label="Mean score")
    ax1.bar(x + 0.18, df.success_rate_095, width=0.36, color="#72b7b2", label="Success >= 0.95")
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x, labels)
    ax1.set_ylabel("Official PushT metric")
    ax1.set_xlabel("(k, h) on high-budget frontier")
    ax1.set_title("Official Checkpoint Frontier: Denoising Dominates Low-NFE Replanning")
    ax1.grid(axis="y", alpha=0.2)
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(x, df.policy_calls_per_episode, color="#f58518", marker="o", label="Policy calls")
    ax2.set_ylabel("Policy calls per episode")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "official_kh_frontier.png", dpi=220)
    plt.close(fig)


def save_rollout_video(trace: pd.DataFrame) -> None:
    joint = trace[(trace.method == "joint_scheduler") & (trace.seed == 0)].copy()
    if joint.empty:
        return
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title("Joint Scheduler Rollout (seed 0)")
    ax.grid(True, alpha=0.2)
    goal = joint.iloc[0][["goal_x", "goal_y"]].to_numpy(dtype=float)
    p_line, = ax.plot([], [], color="#4c78a8", lw=1.5, label="pusher path")
    o_line, = ax.plot([], [], color="#f58518", lw=1.5, label="object path")
    p_dot, = ax.plot([], [], "o", color="#4c78a8", ms=8)
    o_dot, = ax.plot([], [], "s", color="#f58518", ms=10)
    ax.plot(goal[0], goal[1], "*", color="#d62728", ms=16, label="goal")
    txt = ax.text(0.02, 0.97, "", va="top", transform=ax.transAxes)
    ax.legend(loc="lower right")

    px = joint.pusher_x.to_numpy()
    py = joint.pusher_y.to_numpy()
    ox = joint.obj_x.to_numpy()
    oy = joint.obj_y.to_numpy()
    ks = joint.k.to_numpy()
    hs = joint.h.to_numpy()

    def update(i: int):
        end = min(len(joint), i + 1)
        p_line.set_data(px[:end], py[:end])
        o_line.set_data(ox[:end], oy[:end])
        p_dot.set_data([px[i]], [py[i]])
        o_dot.set_data([ox[i]], [oy[i]])
        txt.set_text(f"t={i:03d}, k={int(ks[i])}, h={int(hs[i])}")
        return p_line, o_line, p_dot, o_dot, txt

    anim = FuncAnimation(fig, update, frames=range(0, len(joint), 2), interval=80, blit=True)
    VIDEOS.mkdir(parents=True, exist_ok=True)
    try:
        anim.save(VIDEOS / "joint_scheduler_rollout.gif", writer=PillowWriter(fps=12))
    except Exception:
        fig.savefig(VIDEOS / "joint_scheduler_rollout.png", dpi=180)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(RESULTS / "summary_results.csv")
    trace = pd.read_csv(RESULTS / "raw" / "sample_rollout_traces.csv")
    save_heatmap(summary)
    save_iso_curves(summary)
    save_pareto(summary)
    save_smoothness(summary)
    save_main_comparison_bars(summary)
    save_phase_plot(summary)
    save_official_frontier()
    save_rollout_video(trace)
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
