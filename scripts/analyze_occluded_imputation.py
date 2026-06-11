#!/usr/bin/env python3
"""Analyze masked-keypoint imputation results for official Push-T rollouts."""

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
RUN_ROOT = ROOT / "official_reproduction" / "pusht_official_occluded_impute_n20"
VISIBLE_RATES = [0.5, 0.25]
PAIR = (100, 8)
MODES = ["none", "mean_prior", "frame_hold", "carry_forward", "linear", "oracle"]
MODE_LABELS = {
    "none": "zero-fill",
    "mean_prior": "mean prior",
    "frame_hold": "frame hold",
    "carry_forward": "carry-forward",
    "linear": "linear",
    "oracle": "oracle",
}
DEPLOYABLE = {
    "none": True,
    "mean_prior": True,
    "frame_hold": True,
    "carry_forward": True,
    "linear": True,
    "oracle": False,
}


def load_seed_scores(visible_rate: float, impute: str) -> dict[int, float]:
    k, h = PAIR
    path = RUN_ROOT / f"vr_{visible_rate}" / impute / f"k{k}_h{h}" / "eval_log.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing rollout log: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(seed): float(score) for seed, score in payload["seed_scores"].items()}


def bootstrap_ci(delta: np.ndarray, n_boot: int = 20000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    means = delta[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_delta(
    scores: dict[str, np.ndarray],
    mode: str,
    reference: str,
    seed: int,
) -> tuple[float, float, float]:
    delta = scores[mode] - scores[reference]
    ci_low, ci_high = bootstrap_ci(delta, seed=seed)
    return float(delta.mean()), ci_low, ci_high


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    seed_rows: list[dict[str, object]] = []
    wide_seed_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for visible_rate in VISIBLE_RATES:
        raw_scores = {mode: load_seed_scores(visible_rate, mode) for mode in MODES}
        common_seeds = sorted(set.intersection(*(set(v) for v in raw_scores.values())))
        if not common_seeds:
            raise RuntimeError(f"No common seeds for visible_rate={visible_rate}")

        scores = {
            mode: np.asarray([raw_scores[mode][seed] for seed in common_seeds], dtype=np.float64)
            for mode in MODES
        }

        for idx, seed_id in enumerate(common_seeds):
            wide_row: dict[str, object] = {"seed": seed_id, "visible_rate": visible_rate}
            for mode in MODES:
                score = float(scores[mode][idx])
                seed_rows.append({
                    "seed": seed_id,
                    "visible_rate": visible_rate,
                    "obs_impute": mode,
                    "label": MODE_LABELS[mode],
                    "deployable": DEPLOYABLE[mode],
                    "score": score,
                    "success_095": score >= 0.95,
                })
                wide_row[f"{mode}_score"] = score
            wide_seed_rows.append(wide_row)

        for mode in MODES:
            row: dict[str, object] = {
                "visible_rate": visible_rate,
                "obs_impute": mode,
                "label": MODE_LABELS[mode],
                "deployable": DEPLOYABLE[mode],
                "n_matched": len(common_seeds),
                "mean_score": float(scores[mode].mean()),
                "std_score": float(scores[mode].std(ddof=1)),
                "success_rate_095": float((scores[mode] >= 0.95).mean()),
            }
            if mode == "none":
                row.update({
                    "delta_vs_zero": 0.0,
                    "delta_vs_zero_ci_low": 0.0,
                    "delta_vs_zero_ci_high": 0.0,
                })
            else:
                delta, ci_low, ci_high = paired_delta(
                    scores,
                    mode=mode,
                    reference="none",
                    seed=int(100000 * visible_rate) + MODES.index(mode),
                )
                row.update({
                    "delta_vs_zero": delta,
                    "delta_vs_zero_ci_low": ci_low,
                    "delta_vs_zero_ci_high": ci_high,
                })

            if mode == "carry_forward":
                row.update({
                    "delta_vs_carry_forward": 0.0,
                    "delta_vs_carry_forward_ci_low": 0.0,
                    "delta_vs_carry_forward_ci_high": 0.0,
                })
            else:
                delta, ci_low, ci_high = paired_delta(
                    scores,
                    mode=mode,
                    reference="carry_forward",
                    seed=int(200000 * visible_rate) + MODES.index(mode),
                )
                row.update({
                    "delta_vs_carry_forward": delta,
                    "delta_vs_carry_forward_ci_low": ci_low,
                    "delta_vs_carry_forward_ci_high": ci_high,
                })
            summary_rows.append(row)

    seed_csv = RESULTS / "official_occluded_imputation_all_seed_scores.csv"
    wide_seed_csv = RESULTS / "official_occluded_imputation_all_seed_scores_wide.csv"
    summary_csv = RESULTS / "official_occluded_imputation_all_summary.csv"

    with seed_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(seed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(seed_rows)
    with wide_seed_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(wide_seed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(wide_seed_rows)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_df = pd.DataFrame(summary_rows)
    plot_modes = MODES
    colors = {
        "none": "#4c78a8",
        "mean_prior": "#f58518",
        "frame_hold": "#eeca3b",
        "carry_forward": "#54a24b",
        "linear": "#b279a2",
        "oracle": "#bab0ac",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.4))
    width = 0.12
    x = np.arange(len(VISIBLE_RATES))
    for offset, mode in enumerate(plot_modes):
        sub = summary_df[summary_df.obs_impute == mode].set_index("visible_rate").loc[VISIBLE_RATES]
        pos = x + (offset - (len(plot_modes) - 1) / 2.0) * width
        hatch = "//" if not DEPLOYABLE[mode] else None
        axes[0].bar(pos, sub.mean_score, width=width, color=colors[mode], label=MODE_LABELS[mode], hatch=hatch)
        axes[1].bar(pos, sub.success_rate_095, width=width, color=colors[mode], label=MODE_LABELS[mode], hatch=hatch)

    labels = [f"{int(v * 100)}%" for v in VISIBLE_RATES]
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xlabel("Visible keypoint rate")
    axes[0].set_ylabel("Mean score")
    axes[0].set_title("Occluded Push-T Score")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Visible keypoint rate")
    axes[1].set_ylabel("Success rate, score >= 0.95")
    axes[1].set_title("Occluded Push-T Success")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(ncol=3, loc="upper center", bbox_to_anchor=(-0.1, -0.18))

    fig.suptitle("Mask-Aware Temporal Keypoint Imputation on Official Push-T Checkpoint")
    fig.tight_layout()
    fig_path = FIGURES / "official_occluded_imputation_all.png"
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({
        "seed_scores": str(seed_csv),
        "wide_seed_scores": str(wide_seed_csv),
        "summary": str(summary_csv),
        "figure": str(fig_path),
        "rows": summary_rows,
    }, indent=2))


if __name__ == "__main__":
    main()
