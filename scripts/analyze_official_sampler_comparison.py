#!/usr/bin/env python3
"""Analyze DDPM vs DDIM sampler results for the official Push-T checkpoint."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DDPM_ROOT = ROOT / "official_reproduction" / "pusht_official_kh_grid_highB_n20"
DDIM_ROOT = ROOT / "official_reproduction" / "pusht_official_sampler_ddim_n20"
PAIRS = [(25, 2), (50, 4), (100, 8)]


def load_seed_scores(root: Path, k: int, h: int) -> dict[int, float]:
    path = root / f"k{k}_h{h}" / "eval_log.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(seed): float(score) for seed, score in payload["seed_scores"].items()}


def bootstrap_ci(delta: np.ndarray, n_boot: int = 20000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    means = delta[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    seed_rows = []
    summary_rows = []

    for k, h in PAIRS:
        ddpm = load_seed_scores(DDPM_ROOT, k, h)
        ddim = load_seed_scores(DDIM_ROOT, k, h)
        seeds = sorted(set(ddpm) & set(ddim))
        if not seeds:
            raise RuntimeError(f"No matched seeds for k={k}, h={h}")
        ddpm_scores = np.asarray([ddpm[s] for s in seeds], dtype=np.float64)
        ddim_scores = np.asarray([ddim[s] for s in seeds], dtype=np.float64)
        delta = ddim_scores - ddpm_scores
        ci_low, ci_high = bootstrap_ci(delta, seed=1000 + k + h)
        for seed_id, a, b, d in zip(seeds, ddpm_scores, ddim_scores, delta):
            seed_rows.append({
                "seed": seed_id,
                "k": k,
                "h": h,
                "budget": k / h,
                "ddpm_score": a,
                "ddim_score": b,
                "delta_ddim_minus_ddpm": d,
            })
        summary_rows.append({
            "k": k,
            "h": h,
            "budget": k / h,
            "n_matched": len(seeds),
            "ddpm_mean": float(ddpm_scores.mean()),
            "ddpm_success_095": float((ddpm_scores >= 0.95).mean()),
            "ddim_mean": float(ddim_scores.mean()),
            "ddim_success_095": float((ddim_scores >= 0.95).mean()),
            "delta_mean": float(delta.mean()),
            "delta_bootstrap_ci_low": ci_low,
            "delta_bootstrap_ci_high": ci_high,
        })

    seed_csv = RESULTS / "official_sampler_comparison_seed_scores.csv"
    summary_csv = RESULTS / "official_sampler_comparison_summary.csv"
    with seed_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(seed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(seed_rows)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    df = pd.DataFrame(summary_rows)
    labels = [f"({int(r.k)},{int(r.h)})" for _, r in df.iterrows()]
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    axes[0].bar(x - 0.18, df.ddpm_mean, width=0.36, color="#4c78a8", label="DDPM")
    axes[0].bar(x + 0.18, df.ddim_mean, width=0.36, color="#f58518", label="DDIM")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Mean score")
    axes[0].set_title("Official Push-T Score")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(x - 0.18, df.ddpm_success_095, width=0.36, color="#72b7b2", label="DDPM")
    axes[1].bar(x + 0.18, df.ddim_success_095, width=0.36, color="#e45756", label="DDIM")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Success >= 0.95")
    axes[1].set_title("Official Push-T Success")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Plug-and-Play Sampler Replacement on Official Checkpoint")
    fig.tight_layout()
    fig.savefig(FIGURES / "official_sampler_comparison.png", dpi=220)
    plt.close(fig)

    print(json.dumps({
        "seed_scores": str(seed_csv),
        "summary": str(summary_csv),
        "figure": str(FIGURES / "official_sampler_comparison.png"),
        "rows": summary_rows,
    }, indent=2))


if __name__ == "__main__":
    main()
