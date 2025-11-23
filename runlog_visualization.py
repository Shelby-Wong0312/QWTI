#!/usr/bin/env python3
"""
Visualize key experiments recorded in RUNLOG_OPERATIONS.md.
Outputs a few PNGs under output/visualizations for quick scanning.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


plt.switch_backend("Agg")  # keep generation headless
OUTPUT_DIR = Path("output/visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _annotate_bars(ax, bars, offset: float = 1.2, fmt: str = "{:.1f}%") -> None:
    """Attach value labels on top of bars."""
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def save_data_quality() -> Path:
    """Data readiness snapshots (price coverage + bucket coverage)."""
    price_coverage = [
        ("Before price rebuild\n(old pipeline)", 1.3),
        ("After price rebuild\n(Capital.com)", 65.2),
    ]

    bucket_coverage = [
        ("Pre-backfill\n(single month)", 8.3),
        ("After backfill\n(13 months)", 100.0),
        ("Precision v1\n(filtered)", 54.2),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    price_labels, price_values = zip(*price_coverage)
    bars = axes[0].bar(range(len(price_values)), price_values, color="#4e79a7")
    axes[0].set_title("Price ret_1h non-zero coverage")
    axes[0].set_ylabel("% of hours")
    axes[0].set_xticks(range(len(price_labels)), price_labels, rotation=12)
    axes[0].set_ylim(0, 110)
    _annotate_bars(axes[0], bars, offset=2.5)

    bucket_labels, bucket_values = zip(*bucket_coverage)
    bars = axes[1].bar(range(len(bucket_values)), bucket_values, color="#f28e2b")
    axes[1].set_title("GDELT bucket coverage")
    axes[1].set_ylabel("% of hours with mapped buckets")
    axes[1].set_xticks(range(len(bucket_labels)), bucket_labels, rotation=12)
    axes[1].set_ylim(0, 110)
    _annotate_bars(axes[1], bars, offset=2.5)

    fig.suptitle("Data Quality Improvements (from RUNLOG_OPERATIONS.md)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    output_path = OUTPUT_DIR / "runlog_data_quality.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _sanitize(values: List[Optional[float]]) -> np.ndarray:
    """Convert Nones to NaN for plotting."""
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def save_h1_progression() -> Path:
    """Model performance progression for H=1 horizon."""
    experiments: List[Dict[str, Optional[float]]] = [
        {"name": "Ridge baseline\n(post price fix)", "ic": -0.0212, "ir": -56.87, "pmr": 0.0},
        {"name": "Precision v1\n(keyword tighten)", "ic": -0.0199, "ir": None, "pmr": 0.0},
        {"name": "Precision v2\n(tone + cooccur)", "ic": -0.0136, "ir": -0.51, "pmr": 0.33},
        {"name": "Lasso grid\n(α=0.01)", "ic": 0.0190, "ir": 3.73, "pmr": 1.00},
        {"name": "Lasso fine\n(α=0.005)", "ic": 0.01206, "ir": 0.343, "pmr": 0.60},
        {"name": "Feature filter\n(MACRO only)", "ic": 0.01206, "ir": 0.343, "pmr": 0.60},
        {"name": "LightGBM grid\n(depth=5, lr=0.1)", "ic": 0.026439, "ir": 0.99, "pmr": 0.83},
        {"name": "LightGBM stability\n(12 windows)", "ic": 0.018222, "ir": 0.26, "pmr": 0.67},
        {"name": "Regime ensemble\n(high/low vol)", "ic": 0.0206, "ir": 0.2786, "pmr": 0.619},
        {"name": "Stacking ensemble\n(3×LGB)", "ic": 0.0308, "ir": 0.3823, "pmr": 0.667},
        {"name": "Temporal features 65\n(stacking)", "ic": -0.0006, "ir": -0.0069, "pmr": 0.4762},
    ]

    labels = [item["name"] for item in experiments]
    x = np.arange(len(labels))
    ic = _sanitize([item["ic"] for item in experiments])
    ir_raw = _sanitize([item["ir"] for item in experiments])
    pmr = _sanitize([item["pmr"] for item in experiments])

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    hard_ic, hard_ir, hard_pmr = 0.02, 0.5, 0.55

    axes[0].plot(x, ic, marker="o", color="#4e79a7")
    axes[0].axhline(hard_ic, color="red", linestyle="--", linewidth=1, label="Hard IC=0.02")
    axes[0].set_ylabel("IC")
    axes[0].legend(loc="lower right")
    axes[0].set_title("H=1 performance progression (IC / IR / PMR)")
    for idx, val in enumerate(ic):
        if np.isnan(val):
            continue
        axes[0].text(idx, val + 0.0015, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    clip_min, clip_max = -1.5, 4.0
    ir_display = ir_raw.copy()
    for i, val in enumerate(ir_display):
        if np.isnan(val):
            continue
        if val < clip_min:
            ir_display[i] = clip_min
        elif val > clip_max:
            ir_display[i] = clip_max

    axes[1].plot(x, ir_display, marker="o", color="#f28e2b")
    axes[1].axhline(hard_ir, color="red", linestyle="--", linewidth=1, label="Hard IR=0.5")
    axes[1].set_ylabel("IR (clipped for scale)")
    axes[1].legend(loc="lower right")
    axes[1].set_ylim(clip_min - 0.3, clip_max + 0.5)
    for idx, (raw, disp) in enumerate(zip(ir_raw, ir_display)):
        if np.isnan(raw):
            continue
        label = f"{raw:.2f}" if raw == disp else f"{raw:.1f} (clipped)"
        axes[1].text(idx, disp + 0.12, label, ha="center", va="bottom", fontsize=8, rotation=0)

    axes[2].plot(x, pmr, marker="o", color="#59a14f")
    axes[2].axhline(hard_pmr, color="red", linestyle="--", linewidth=1, label="Hard PMR=0.55")
    axes[2].set_ylabel("PMR")
    axes[2].legend(loc="lower right")
    axes[2].set_ylim(0, 1.05)
    for idx, val in enumerate(pmr):
        if np.isnan(val):
            continue
        axes[2].text(idx, val + 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    # Highlight first Hard IC hit (LightGBM grid search)
    hard_hit_idx = 6
    for ax in axes:
        ax.axvspan(hard_hit_idx - 0.4, hard_hit_idx + 0.4, color="#cfcfff", alpha=0.5, label="First Hard IC pass")
        ax.set_xticks(x, labels, rotation=32, ha="right")
    axes[0].legend(loc="lower right")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.suptitle("RUNLOG experiments → visual snapshot (H=1, lag=1h)")

    output_path = OUTPUT_DIR / "runlog_h1_progression.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def save_h3_ic() -> Path:
    """IC progression for the longer horizon (H=3)."""
    experiments = [
        ("Ridge baseline", -0.0155),
        ("Precision v1", -0.0076),
        ("Precision v2", -0.0005),
        ("Lasso grid (α=0.01)", 0.00456),
        ("Lasso fine (α=0.005)", 0.00633),
        ("Nonlinear best (XGB)", 0.010244),
    ]

    names, ic_values = zip(*experiments)
    x = np.arange(len(names))

    colors = ["#59a14f" if v >= 0 else "#d37295" for v in ic_values]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x, ic_values, color=colors)
    ax.axhline(0.02, color="red", linestyle="--", linewidth=1, label="Hard IC=0.02")
    ax.set_xticks(x, names, rotation=22, ha="right")
    ax.set_ylabel("IC (H=3)")
    ax.set_title("H=3 IC trajectory (still below Hard threshold)")
    for idx, val in enumerate(ic_values):
        ax.text(idx, val + (0.001 if val >= 0 else -0.0018), f"{val:.3f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8)
    ax.legend(loc="upper left")
    ax.set_ylim(min(ic_values) - 0.005, max(ic_values) + 0.01)

    fig.tight_layout()
    output_path = OUTPUT_DIR / "runlog_h3_ic.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def save_linear_vs_nonlinear_scatter() -> Path:
    """
    Compare linear vs non-linear runs on IC vs IR, bubble-sized by PMR.
    Points come from RUNLOG entries to highlight correlation strength differences.
    """
    runs = [
        {"family": "Linear", "label": "Ridge baseline (raw GKG v2)", "ic": -0.0212, "ir": -56.87, "pmr": 0.0},
        {"family": "Linear", "label": "Lasso grid best α=0.01", "ic": 0.0190, "ir": 3.73, "pmr": 1.00},
        {"family": "Linear", "label": "Lasso fine α=0.005", "ic": 0.01206, "ir": 0.343, "pmr": 0.60},
        {"family": "Linear", "label": "Lasso feature filter (MACRO)", "ic": 0.01206, "ir": 0.343, "pmr": 0.60},
        {"family": "Non-linear", "label": "LightGBM grid depth=5 lr=0.1", "ic": 0.026439, "ir": 0.99, "pmr": 0.83},
        {"family": "Non-linear", "label": "LightGBM stability (12w)", "ic": 0.018222, "ir": 0.26, "pmr": 0.67},
        {"family": "Non-linear", "label": "Regime ensemble (vol split)", "ic": 0.0206, "ir": 0.2786, "pmr": 0.619},
        {"family": "Non-linear", "label": "Stacking ensemble (3×LGB)", "ic": 0.0308, "ir": 0.3823, "pmr": 0.667},
        {"family": "Non-linear", "label": "Temporal feature stack (65)", "ic": -0.0006, "ir": -0.0069, "pmr": 0.4762},
    ]

    colors = {"Linear": "#4e79a7", "Non-linear": "#f28e2b"}
    fig, ax = plt.subplots(figsize=(10, 6))
    for family in ("Linear", "Non-linear"):
        subset = [r for r in runs if r["family"] == family]
        ic = [r["ic"] for r in subset]
        ir = [r["ir"] for r in subset]
        pmr = [r["pmr"] for r in subset]
        labels = [r["label"] for r in subset]
        sizes = [max(40, 400 * p) for p in pmr]  # ensure visibility even when PMR=0
        scatter = ax.scatter(ic, ir, s=sizes, alpha=0.7, label=f"{family} (size=PMR)", c=colors[family], edgecolors="k", linewidths=0.5)
        for x, y, text in zip(ic, ir, labels):
            ax.text(x, y, text, fontsize=8, ha="left", va="center")

    ax.axvline(0.02, color="red", linestyle="--", linewidth=1, label="Hard IC=0.02")
    ax.axhline(0.5, color="purple", linestyle="--", linewidth=1, label="Hard IR=0.5")
    ax.set_xlabel("IC")
    ax.set_ylabel("IR")
    ax.set_title("Linear vs Non-linear: correlation strength & stability (H=1)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(-0.03, 0.05)
    # Clip extreme IR for readability but keep points visible
    ax.set_ylim(-5.0, 4.5)
    fig.tight_layout()

    output_path = OUTPUT_DIR / "runlog_linear_vs_nonlinear.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    outputs = [
        save_data_quality(),
        save_h1_progression(),
        save_h3_ic(),
        save_linear_vs_nonlinear_scatter(),
    ]
    for path in outputs:
        print(f"[saved] {path}")


if __name__ == "__main__":
    main()
