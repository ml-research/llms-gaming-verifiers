"""
Training dynamics plots: isomorphic vs non-isomorphic RLVR training (SLR model).

Non-isomorphic training optimises the extensional (base) verifier → model learns
to hack it → extensional reward diverges from isomorphic reward → gap grows.
Isomorphic training optimises the isomorphic verifier → both rewards track each
other → gap stays near zero.

Run this file directly:
    python shortcuts/training_dynamics_plots.py
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ── Paper-ready style ─────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

ISO_COLOR    = "#2ca02c"   # green  – isomorphic training
NONISO_COLOR = "#d62728"   # red    – non-isomorphic training
FILL_ALPHA   = 0.12

HISTORY_DIR = Path(__file__).parent.parent / "output" / "wandb_histories"
OUT_DIR     = Path(__file__).parent.parent / "output" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COL_STEP = "training_step"
COL_BASE = "objective/slr_bench_base_reward"       # extensional reward
COL_ISO  = "objective/slr_bench_isomorphic_reward" # isomorphic reward

SMOOTH = 30


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=lambda c: c in [COL_STEP, COL_BASE, COL_ISO])
    return df.dropna(subset=[COL_STEP]).sort_values(COL_STEP).reset_index(drop=True)


def _smooth(s: pd.Series) -> np.ndarray:
    return s.rolling(SMOOTH, min_periods=max(1, SMOOTH // 5), center=True).mean().to_numpy()


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.6)
    ax.grid(axis="x", color="#e8e8e8", linewidth=0.4, linestyle=":")


# ── Data ──────────────────────────────────────────────────────────────────────

iso    = _load(str(HISTORY_DIR / "run_RLVR-SLR-IsomorphicRL__1__1772631809_history.csv"))
noniso = _load(str(HISTORY_DIR / "run_RLVR-SLR-Non-IsomorphicRL__1__1772603080_history.csv"))


# =============================================================================
# FIGURE 1: Reward curves — extensional vs isomorphic for both training types
# =============================================================================

def plot_reward_curves(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)

    step_n = noniso[COL_STEP].to_numpy()
    base_n = _smooth(noniso[COL_BASE])
    iso_n  = _smooth(noniso[COL_ISO])

    step_i = iso[COL_STEP].to_numpy()
    base_i = _smooth(iso[COL_BASE])
    iso_i  = _smooth(iso[COL_ISO])

    ax.plot(step_n, base_n, color=NONISO_COLOR, linewidth=1.8,
            label="Non-Iso: extensional", zorder=4)
    ax.plot(step_n, iso_n,  color=NONISO_COLOR, linewidth=1.8,
            linestyle="--", alpha=0.65, label="Non-Iso: isomorphic", zorder=3)
    ax.plot(step_i, base_i, color=ISO_COLOR, linewidth=1.8,
            label="Iso: extensional", zorder=4)
    ax.plot(step_i, iso_i,  color=ISO_COLOR, linewidth=1.8,
            linestyle="--", alpha=0.65, label="Iso: isomorphic", zorder=3)

    mn = min(len(step_n), len(base_n), len(iso_n))
    ax.fill_between(step_n[:mn], iso_n[:mn], base_n[:mn],
                    where=(base_n[:mn] > iso_n[:mn]),
                    color=NONISO_COLOR, alpha=FILL_ALPHA, label="Hacking gap")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Reward")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _clean(ax)

    for ext in (".pdf", ".png"):
        fig.savefig(out_dir / f"training_reward_curves{ext}")
    plt.close(fig)
    print("  Saved: training_reward_curves")


# =============================================================================
# FIGURE 2: Hacking gap  (extensional − isomorphic)
# =============================================================================

def plot_reward_gap(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)

    for df, color, label in [
        (noniso, NONISO_COLOR, "Non-Iso training"),
        (iso,    ISO_COLOR,    "Iso training"),
    ]:
        steps = df[COL_STEP].to_numpy()
        gap   = _smooth(df[COL_BASE] - df[COL_ISO])
        ax.plot(steps, gap, color=color, linewidth=1.8, label=label, zorder=4)
        ax.fill_between(steps, 0, gap, where=(gap > 0),
                        color=color, alpha=FILL_ALPHA, zorder=2)

    ax.axhline(0, color="#aaaaaa", linewidth=0.8, zorder=1)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Hacking Gap (ext. − iso)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _clean(ax)

    for ext in (".pdf", ".png"):
        fig.savefig(out_dir / f"training_reward_gap{ext}")
    plt.close(fig)
    print("  Saved: training_reward_gap")


# =============================================================================
# FIGURE 3: Combined — curves left, gap right  (paper figure)
# =============================================================================

def plot_combined(out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.2), constrained_layout=True)

    # Left: reward curves
    step_n = noniso[COL_STEP].to_numpy()
    base_n = _smooth(noniso[COL_BASE])
    iso_n  = _smooth(noniso[COL_ISO])
    step_i = iso[COL_STEP].to_numpy()
    base_i = _smooth(iso[COL_BASE])
    iso_i  = _smooth(iso[COL_ISO])

    ax1.plot(step_n, base_n, color=NONISO_COLOR, linewidth=1.8,
             label="Non-Iso: extensional", zorder=4)
    ax1.plot(step_n, iso_n,  color=NONISO_COLOR, linewidth=1.8,
             linestyle="--", alpha=0.65, label="Non-Iso: isomorphic", zorder=3)
    ax1.plot(step_i, base_i, color=ISO_COLOR, linewidth=1.8,
             label="Iso: extensional", zorder=4)
    ax1.plot(step_i, iso_i,  color=ISO_COLOR, linewidth=1.8,
             linestyle="--", alpha=0.65, label="Iso: isomorphic", zorder=3)

    mn = min(len(step_n), len(base_n), len(iso_n))
    ax1.fill_between(step_n[:mn], iso_n[:mn], base_n[:mn],
                     where=(base_n[:mn] > iso_n[:mn]),
                     color=NONISO_COLOR, alpha=FILL_ALPHA)

    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Reward")
    ax1.legend(frameon=False, fontsize=8.5, loc="upper left")
    _clean(ax1)

    # Right: hacking gap
    for df, color, label in [
        (noniso, NONISO_COLOR, "Non-Iso training"),
        (iso,    ISO_COLOR,    "Iso training"),
    ]:
        steps = df[COL_STEP].to_numpy()
        gap   = _smooth(df[COL_BASE] - df[COL_ISO])
        ax2.plot(steps, gap, color=color, linewidth=1.8, label=label, zorder=4)
        ax2.fill_between(steps, 0, gap, where=(gap > 0),
                         color=color, alpha=FILL_ALPHA, zorder=2)

    ax2.axhline(0, color="#aaaaaa", linewidth=0.8, zorder=1)
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Hacking Gap (ext. − iso)")
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left")
    _clean(ax2)

    for ext in (".pdf", ".png"):
        fig.savefig(out_dir / f"training_dynamics_combined{ext}")
    plt.close(fig)
    print("  Saved: training_dynamics_combined")


# =============================================================================
# FIGURES 4 & 5: Per-run panels with shared y-axis scale
# =============================================================================

def _shared_ylim(margin: float = 0.08):
    """Compute y-limits covering both runs so the two panels are comparable."""
    all_vals = np.concatenate([
        _smooth(noniso[COL_BASE]), _smooth(noniso[COL_ISO]),
        _smooth(iso[COL_BASE]),    _smooth(iso[COL_ISO]),
    ])
    all_vals = all_vals[np.isfinite(all_vals)]
    span = all_vals.max() - all_vals.min()
    return all_vals.min() - margin * span, all_vals.max() + margin * span


def _plot_run_gap(df: pd.DataFrame, ax, ylim, ylabel: bool = True) -> None:
    steps  = df[COL_STEP].to_numpy()
    base_s = _smooth(df[COL_BASE])
    iso_s  = _smooth(df[COL_ISO])

    ax.plot(steps, base_s, color=NONISO_COLOR, linewidth=1.8,
            label="Extensional", zorder=4)
    ax.plot(steps, iso_s,  color=ISO_COLOR, linewidth=1.8,
            label="Isomorphic", zorder=3)

    mn = min(len(steps), len(base_s), len(iso_s))
    ax.fill_between(steps[:mn], iso_s[:mn], base_s[:mn],
                    where=(base_s[:mn] > iso_s[:mn]),
                    color=NONISO_COLOR, alpha=FILL_ALPHA, label="Hacking gap")

    ax.set_ylim(*ylim)
    ax.set_xlabel("Training Step")
    if ylabel:
        ax.set_ylabel("Reward")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _clean(ax)


def plot_noniso_gap(out_dir: Path) -> None:
    ylim = _shared_ylim()
    fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
    _plot_run_gap(noniso, ax, ylim)
    for ext in (".pdf", ".png"):
        fig.savefig(out_dir / f"training_noniso_gap{ext}")
    plt.close(fig)
    print("  Saved: training_noniso_gap")


def plot_iso_gap(out_dir: Path) -> None:
    ylim = _shared_ylim()
    fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
    _plot_run_gap(iso, ax, ylim, ylabel=True)
    for ext in (".pdf", ".png"):
        fig.savefig(out_dir / f"training_iso_gap{ext}")
    plt.close(fig)
    print("  Saved: training_iso_gap")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print(f"Writing plots to: {OUT_DIR}")
    plot_reward_curves(OUT_DIR)
    plot_reward_gap(OUT_DIR)
    plot_combined(OUT_DIR)
    plot_noniso_gap(OUT_DIR)
    plot_iso_gap(OUT_DIR)
    print("Done.")
