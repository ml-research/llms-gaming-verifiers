import os
import shutil
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.transforms import Bbox
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Paper-ready defaults.
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

# -- Colors ----------------------------------------------------------------

RLVR_COLOR = "#d62728"
BASE_COLOR = "#7f7f7f"

# Consistent per-model palette for RLVR models across all plots.
_RLVR_PALETTE = {
    "gpt-5": "#1f77b4",
    "gpt-5-mini": "#ff7f0e",
    "gpt-5-mini-high": "#2ca02c",
    "gpt-5-mini-low": "#9467bd",
    "gpt-5-nano": "#8c564b",
}

_FIG_SIZE_SMALL = (4.8, 3.0)


# -- Helpers ---------------------------------------------------------------

def _model_group(model_name: str) -> str:
    name = str(model_name).lower()
    if "gpt-5" in name or "o3" in name or "o4" in name or "qwen3" in name:
        return "rlvr"
    if "gpt-4" in name or "ministral" in name:
        return "base"
    return "other"


def _short_label(model_name: str) -> str:
    name = str(model_name)
    for suffix in ["-2024-04-09", "-2024-08-06", "-2025-02-27"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _color_for(label: str) -> str:
    if label in _RLVR_PALETTE:
        return _RLVR_PALETTE[label]
    cmap = plt.get_cmap("tab20")
    return mcolors.to_hex(cmap(abs(hash(label)) % 20))


def _ensure_plots_dir(output_dir: str, plots_subdir: str) -> str:
    plots_dir = os.path.join(output_dir, plots_subdir)
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def _export_paper_plots(plots_dir: str) -> None:
    paper_dir = os.path.join(plots_dir, "paper")
    os.makedirs(paper_dir, exist_ok=True)

    paper_basenames = [
        "paper_complexity_short_pressure",
        "paper_effort_short_pressure",
        "paper_effort_model_scatter_solved_vs_bad_shortcuts",
        "paper_level_model_scatter_solved_vs_bad_shortcuts",
        "paper_complexity_effort_horizontal",
    ]

    for base in paper_basenames:
        for ext in (".png", ".pdf"):
            src = os.path.join(plots_dir, f"{base}{ext}")
            if os.path.exists(src):
                dst = os.path.join(paper_dir, f"{base}{ext}")
                shutil.copy2(src, dst)


def _clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.6)


def _save_fixed_canvas(fig, out_path: str) -> None:
    width, height = fig.get_size_inches()
    fixed_bbox = Bbox.from_extents(0.0, 0.0, float(width), float(height))
    fig.savefig(out_path, bbox_inches=fixed_bbox, pad_inches=0.0)


def _exclude_false_positive_models(summary: pd.DataFrame) -> pd.DataFrame:
    if summary is None or summary.empty:
        return summary
    mask = ~summary["label"].astype(str).str.startswith("gpt-4-turbo")
    return summary[mask].copy()


def _ordered_scatter_models(labels: list[str], acc_map: Optional[dict[str, float]] = None) -> list[str]:
    preferred = [
        "gpt-5-mini-low",
        "gpt-5-mini",
        "gpt-5-mini-high",
        "gpt-5",
        "gpt-5-nano",
    ]
    unique_labels = list(dict.fromkeys(labels))
    if acc_map is not None and unique_labels:
        pref_rank = {name: idx for idx, name in enumerate(preferred)}
        remaining = [name for name in unique_labels if name != "gpt-5"]
        remaining_sorted = sorted(
            remaining,
            key=lambda name: (
                -(float(acc_map.get(name, -1.0))),
                pref_rank.get(name, 999),
                name,
            ),
        )
        ordered = (["gpt-5"] if "gpt-5" in unique_labels else []) + remaining_sorted
        return ordered

    label_set = set(unique_labels)
    ordered = [name for name in preferred if name in label_set]
    ordered += sorted([name for name in unique_labels if name not in set(ordered)])
    return ordered


def _compute_delta_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = ["default_correct", "local_correct", "completion_tokens", "model_name", "complexity"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return pd.DataFrame()

    out = df.copy()
    out = out.dropna(subset=["default_correct", "local_correct", "model_name", "complexity"])
    if out.empty:
        return pd.DataFrame()

    out["default_correct"] = pd.to_numeric(out["default_correct"], errors="coerce").astype(float)
    out["local_correct"] = pd.to_numeric(out["local_correct"], errors="coerce").astype(float)
    out["completion_tokens"] = pd.to_numeric(out["completion_tokens"], errors="coerce")
    out = out.dropna(subset=["default_correct", "local_correct"])

    out["shortcut_delta"] = (
        (out["local_correct"] - out["default_correct"]).clip(lower=0.0, upper=1.0)
    )
    if "is_shortcut" in out.columns:
        out["shortcut_attempt"] = (
            pd.to_numeric(out["is_shortcut"], errors="coerce")
            .fillna(0.0)
            .clip(0.0, 1.0)
        )
    else:
        out["shortcut_attempt"] = (out["shortcut_delta"] > 0).astype(float)
    out["model_group"] = out["model_name"].map(_model_group)
    out["complexity"] = out["complexity"].astype(str).str.lower()
    return out


def _build_model_summary(delta_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-model: total N_S, accuracy %, avg tokens."""
    rows = []
    for model_name, g in delta_df.groupby("model_name"):
        label = str(model_name)
        n_s = int(g["shortcut_delta"].sum())
        acc = g["default_correct"].mean() * 100
        tokens = g["completion_tokens"].mean()
        rows.append({
            "model_name": label,
            "label": _short_label(label),
            "model_group": _model_group(label),
            "N_S": n_s,
            "acc": acc,
            "tokens": tokens,
            "solved_default": pd.to_numeric(g["default_correct"], errors="coerce").sum(),
            "shortcut_attempts": pd.to_numeric(g["shortcut_attempt"], errors="coerce").sum(),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["short_pressure_pct"] = np.where(
        out["solved_default"] > 0,
        (out["shortcut_attempts"] / out["solved_default"]) * 100.0,
        np.nan,
    )
    return out


# ==========================================================================
# PLOT 1:  Shortcuts vs Task Complexity  (line plot, per RLVR model)
# ==========================================================================

def plot_paper_complexity(delta_df: pd.DataFrame, output_dir: str) -> None:
    """Line plot: N_S per complexity tier, one line per RLVR model.
    Non-RLVR models (all zero) shown as a flat baseline."""
    if delta_df.empty:
        return

    order = ["Basic", "Easy", "Medium", "Hard"]
    data = delta_df.copy()
    data["complexity"] = data["complexity"].str.capitalize()
    data = data[data["complexity"].isin(order)]
    rlvr = data[data["model_group"] == "rlvr"].copy()
    if rlvr.empty:
        return

    counts = (
        rlvr.groupby(["model_name", "complexity"])["shortcut_delta"]
        .sum()
        .reset_index(name="N_S")
    )
    counts["N_S"] = counts["N_S"].astype(int)
    solved_counts = (
        rlvr.groupby(["model_name", "complexity"])["default_correct"]
        .sum()
        .reset_index(name="solved_default")
    )
    shortcut_attempt_counts = (
        rlvr.groupby(["model_name", "complexity"])["shortcut_attempt"]
        .sum()
        .reset_index(name="shortcut_attempts")
    )
    counts = counts.merge(solved_counts, on=["model_name", "complexity"], how="left")
    counts = counts.merge(shortcut_attempt_counts, on=["model_name", "complexity"], how="left")
    counts["solved_default"] = pd.to_numeric(counts["solved_default"], errors="coerce").fillna(0.0)
    counts["shortcut_attempts"] = pd.to_numeric(counts["shortcut_attempts"], errors="coerce").fillna(0.0)
    counts["shortcut_pressure_pct"] = np.where(
        counts["solved_default"] > 0,
        (counts["shortcut_attempts"] / counts["solved_default"]) * 100.0,
        np.nan,
    )
    counts["complexity"] = pd.Categorical(counts["complexity"], categories=order, ordered=True)
    counts["label"] = counts["model_name"].map(_short_label)

    # Sort models by total shortcuts descending.
    model_order = (
        counts.groupby("label")["N_S"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=_FIG_SIZE_SMALL)

    for model_label in model_order:
        md = counts[counts["label"] == model_label].set_index("complexity")
        ys = [int(md.loc[t, "N_S"]) if t in md.index else 0 for t in order]
        color = _color_for(model_label)
        ax.plot(order, ys, marker="o", linewidth=1.8, markersize=4.5,
                color=color, label=model_label, zorder=3)

    ax.axhline(0, color=BASE_COLOR, linewidth=0.9, linestyle="-",
               zorder=1, alpha=0.35)

    ax.set_xlabel("Task Complexity")
    ax.set_ylabel("Shortcuts ($N_S$)")
    ax.set_ylim(bottom=-1, top=counts["N_S"].max() * 1.20)
    ax.margins(x=0.06)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _clean_axes(ax)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper left")
    fig.subplots_adjust(left=0.14, right=0.99, bottom=0.20, top=0.96)
    fig.savefig(os.path.join(output_dir, "paper_complexity.pdf"))
    fig.savefig(os.path.join(output_dir, "paper_complexity.png"))
    fig.savefig(os.path.join(output_dir, "paper_complexity_abs.pdf"))
    fig.savefig(os.path.join(output_dir, "paper_complexity_abs.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=_FIG_SIZE_SMALL)
    x = np.arange(len(order), dtype=float)

    for model_label in model_order:
        md = counts[counts["label"] == model_label].set_index("complexity")
        ys = np.array([float(md.loc[t, "shortcut_pressure_pct"]) if t in md.index else np.nan for t in order], dtype=float)
        color = _color_for(model_label)
        # faded line, then solid dots on top
        ax.plot(x, ys, linewidth=1.4, linestyle="--", alpha=0.45,
                color=color, label=model_label, zorder=3)
        ax.plot(x, ys, marker="o", linestyle="none", markersize=4.5,
                color=color, alpha=1.0, zorder=4)

    ax.axhline(0, color=BASE_COLOR, linewidth=0.9, linestyle="-",
               zorder=1, alpha=0.35)
    ax.set_xlabel("Task Complexity")
    ax.set_ylabel("Shortcut Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    max_pressure = pd.to_numeric(counts["shortcut_pressure_pct"], errors="coerce").replace([np.inf, -np.inf], np.nan).max()
    if pd.isna(max_pressure):
        max_pressure = 10.0
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(bottom=-0.1, top=max_pressure * 1.5)
    ax.margins(x=0.06)
    
    _clean_axes(ax)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper left")
    fig.subplots_adjust(left=0.14, right=0.99, bottom=0.20, top=0.96)
    fig.savefig(os.path.join(output_dir, "paper_complexity_freq.pdf"))
    fig.savefig(os.path.join(output_dir, "paper_complexity_freq.png"))
    _save_fixed_canvas(fig, os.path.join(output_dir, "paper_complexity_short_pressure.pdf"))
    _save_fixed_canvas(fig, os.path.join(output_dir, "paper_complexity_short_pressure.png"))
    plt.close(fig)
    print("  Plot 1 (complexity abs + short pressure) saved.")


# ==========================================================================
# PLOT 2:  Shortcuts vs Model Capability  (scatter)
# ==========================================================================

def plot_paper_capability(delta_df: pd.DataFrame, output_dir: str) -> None:
    """Scatter: x = accuracy (%), y = N_S.  RLVR vs non-RLVR."""
    summary = _build_model_summary(delta_df)
    if summary.empty:
        return

    summary = _exclude_false_positive_models(summary)
    rlvr = summary[summary["model_group"] == "rlvr"]
    base = summary[summary["model_group"] == "base"]

    fig, ax = plt.subplots(figsize=(4.2, 2.55))

    # Non-RLVR: all at y=0, green squares.
    if not base.empty:
        ax.scatter(
            base["acc"], base["N_S"],
            s=45, c=BASE_COLOR, marker="s", alpha=0.85,
            edgecolors="white", linewidths=0.5, zorder=3, label="Non-RLVR",
        )
        base_sorted = base.sort_values("acc").reset_index(drop=True)
        base_offsets = [-12, 2, 12, -18, 18]
        for i, (_, row) in enumerate(base_sorted.iterrows()):
            y_offset = base_offsets[i % len(base_offsets)]
            ax.annotate(
                row["label"], (row["acc"], row["N_S"]),
                xytext=(2, y_offset), textcoords="offset points",
                fontsize=7, color="#555", ha="center",
            )

    # RLVR: colored by model.
    if not rlvr.empty:
        for _, row in rlvr.iterrows():
            color = _color_for(row["label"])
            ax.scatter(
                row["acc"], row["N_S"],
                s=55, c=color, marker="o", alpha=0.9,
                edgecolors="white", linewidths=0.5, zorder=4,
            )
            ax.annotate(
                row["label"], (row["acc"], row["N_S"]),
                xytext=(5, 2), textcoords="offset points",
                fontsize=8, color="#333",
            )

    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("Shortcuts ($N_S$)")
    ax.margins(x=0.08, y=0.12)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _clean_axes(ax)
    ax.legend(
        handles=[
            Patch(color=RLVR_COLOR, label="RLVR"),
            Patch(color=BASE_COLOR, label="Non-RLVR"),
        ],
        frameon=False, fontsize=9, loc="upper left",
    )
    fig.subplots_adjust(left=0.14, right=0.99, bottom=0.20, top=0.96)
    fig.savefig(os.path.join(output_dir, "paper_capability.pdf"))
    fig.savefig(os.path.join(output_dir, "paper_capability.png"))
    plt.close(fig)
    print("  Plot 2 (capability) saved.")


# ==========================================================================
# PLOT 3:  Shortcuts vs Reasoning Effort  (scatter + mini-family line)
# ==========================================================================

def plot_paper_effort(delta_df: pd.DataFrame, output_dir: str) -> None:
    """Scatter: x = avg tokens, y = short Rate.
    Connects gpt-5-mini-{low,default,high} with a line to show scaling."""
    summary = _build_model_summary(delta_df)
    if summary.empty:
        return

    summary = _exclude_false_positive_models(summary)
    rlvr = summary[summary["model_group"] == "rlvr"].copy()
    rlvr_mini = rlvr[rlvr["label"].astype(str).str.contains("gpt-5-mini")].copy()

    fig, ax = plt.subplots(figsize=_FIG_SIZE_SMALL)

    # RLVR mini-family only (matches horizontal effort panel).
    if not rlvr_mini.empty:
        max_tokens = pd.to_numeric(rlvr_mini["tokens"], errors="coerce").max()
        label_offsets = {
            "gpt-5-mini": (10, 2, "left"),
            "gpt-5-mini-high": (-8, 8, "right"),
            "gpt-5-mini-low": (8, 8, "left"),
        }
        for _, row in rlvr_mini.iterrows():
            color = _color_for(row["label"])
            ax.scatter(
                row["tokens"], row["short_pressure_pct"],
                s=55, c=color, marker="o", alpha=0.9,
                edgecolors="white", linewidths=0.5, zorder=4,
            )
            label = str(row["label"])
            if label in label_offsets:
                xoff, yoff, ha = label_offsets[label]
            else:
                align_right = pd.notna(max_tokens) and float(row["tokens"]) > 0.90 * float(max_tokens)
                xoff, yoff, ha = (-6, 4, "right") if align_right else (6, 4, "left")
            ax.annotate(
                row["label"], (row["tokens"], row["short_pressure_pct"]),
                xytext=(xoff, yoff), textcoords="offset points",
                fontsize=8, color="#333", ha=ha,
            )

    # Connect gpt-5-mini family with a dashed line.
    mini_family = rlvr_mini.sort_values("tokens")
    if len(mini_family) >= 2:
        x_vals = pd.to_numeric(mini_family["tokens"], errors="coerce").to_numpy(dtype=float)
        y_vals = pd.to_numeric(mini_family["short_pressure_pct"], errors="coerce").to_numpy(dtype=float)
        ax.plot(
            x_vals, y_vals,
            linestyle="--", linewidth=1.2, color="#999", zorder=2,
        )

    ax.set_xlabel("Avg. Reasoning Effort (Tokens)")
    ax.set_ylabel("Shortcut Rate")
    max_pressure = pd.to_numeric(rlvr_mini["short_pressure_pct"], errors="coerce").replace([np.inf, -np.inf], np.nan).max()
    if pd.isna(max_pressure):
        max_pressure = 10.0
    ax.set_ylim(bottom=-0.3, top=max(5.0, float(max_pressure) * 1.22))
    # set x axis to log
    ax.set_xscale("log")
    # ax.set_yscale("log")
    # x axis starts at 1 value to 10^5, but zoom in to the data range with some padding
    # ax.set_xlim(1.2, 1.5 * max(rlvr_mini["tokens"].max(), 10))
    token_values = pd.to_numeric(rlvr_mini["tokens"], errors="coerce")
    token_min = token_values.min()
    token_max = token_values.max()
    if pd.notna(token_min) and pd.notna(token_max):
        span = max(1.0, float(token_max - token_min))
        ax.set_xlim(float(token_min) - 0.06 * span, float(token_max) + 0.08 * span)
    ax.margins(y=0.10)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    _clean_axes(ax)
    model_labels = rlvr_mini["label"].tolist() if not rlvr_mini.empty else []
    legend_handles = [
        Line2D([0], [0], marker="o", color=_color_for(lbl), markerfacecolor=_color_for(lbl),
               markeredgecolor="white", markersize=5.5, linewidth=1.2, label=lbl)
        for lbl in model_labels
    ]
    # ax.legend(handles=legend_handles, frameon=False, fontsize=9, ncol=2, loc="upper left")
    fig.subplots_adjust(left=0.14, right=0.99, bottom=0.20, top=0.96)
    fig.savefig(os.path.join(output_dir, "paper_effort.pdf"))
    fig.savefig(os.path.join(output_dir, "paper_effort.png"))
    _save_fixed_canvas(fig, os.path.join(output_dir, "paper_effort_short_pressure.pdf"))
    _save_fixed_canvas(fig, os.path.join(output_dir, "paper_effort_short_pressure.png"))
    plt.close(fig)
    print("  Plot 3 (effort) saved.")


def plot_paper_complexity_effort_stacked(delta_df: pd.DataFrame, output_dir: str) -> None:
    """Combined horizontal figure with three subplots:
    1) complexity short pressure
    2) effort short pressure
    3) solved vs shortcut effort scatter"""
    if delta_df.empty:
        return

    summary = _build_model_summary(delta_df)
    if summary.empty:
        return
    summary = _exclude_false_positive_models(summary)
    rlvr_summary = summary[summary["model_group"] == "rlvr"].copy()
    base_summary = summary[summary["model_group"] == "base"].copy()

    order = ["Basic", "Easy", "Medium", "Hard"]
    data = delta_df.copy()
    data["complexity"] = data["complexity"].str.capitalize()
    data = data[data["complexity"].isin(order)]
    data["label"] = data["model_name"].map(_short_label)
    data = data[~data["label"].astype(str).str.startswith("gpt-4-turbo")]
    rlvr = data[data["model_group"] == "rlvr"].copy()
    if rlvr.empty:
        return

    counts = (
        rlvr.groupby(["model_name", "complexity"])
        .agg(
            N_S=("shortcut_delta", "sum"),
            solved_default=("default_correct", "sum"),
            shortcut_attempts=("shortcut_attempt", "sum"),
        )
        .reset_index()
    )
    counts["N_S"] = pd.to_numeric(counts["N_S"], errors="coerce").fillna(0.0)
    counts["solved_default"] = pd.to_numeric(counts["solved_default"], errors="coerce").fillna(0.0)
    counts["shortcut_attempts"] = pd.to_numeric(counts["shortcut_attempts"], errors="coerce").fillna(0.0)
    counts["shortcut_rate"] = np.where(
        counts["solved_default"] > 0,
        counts["N_S"] / counts["solved_default"],
        np.nan,
    )
    counts["complexity"] = pd.Categorical(counts["complexity"], categories=order, ordered=True)
    counts["label"] = counts["model_name"].map(_short_label)

    model_order = (
        counts.groupby("label")["N_S"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # Only show tiers where at least one model has shortcuts
    tier_has_data1 = counts.groupby("complexity")["N_S"].sum() > 0
    order_rate1 = [t for t in order if tier_has_data1.get(t, False)]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.4, 2.75))

    rate_vals1 = counts["shortcut_rate"].replace([np.inf, -np.inf], np.nan).dropna()
    pos_vals1 = rate_vals1[rate_vals1 > 0]
    rate_min1 = float(pos_vals1.min()) if not pos_vals1.empty else 0.01
    rate_max1 = float(pos_vals1.max()) if not pos_vals1.empty else 1.0
    FLOOR_RATE1 = 10 ** (np.floor(np.log10(rate_min1)) - 0.5)

    x1 = np.arange(len(order_rate1), dtype=float)
    for model_label in model_order:
        md = counts[counts["label"] == model_label].set_index("complexity")
        ys_raw = np.array(
            [float(md.loc[t, "shortcut_rate"]) if t in md.index else np.nan for t in order_rate1],
            dtype=float,
        )
        ys = np.where(np.isfinite(ys_raw) & (ys_raw == 0), FLOOR_RATE1,
                      np.where(ys_raw > 0, ys_raw, np.nan))
        color = _color_for(model_label)
        nz = np.isfinite(ys_raw) & (ys_raw > 0)
        if nz.sum() >= 2:
            ax1.plot(x1[nz], ys_raw[nz], linewidth=1.4, color=color,
                     alpha=0.55, linestyle="--", zorder=3)
        ax1.plot(x1, ys, marker="o", linestyle="none",
                 markersize=4.2, color=color, label=model_label, zorder=4)

    ax1.set_xlabel("Task Complexity")
    ax1.set_ylabel("Shortcut Rate ($N_S$ / solved)")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(order_rate1)
    ax1.set_yscale("log")
    ax1.set_ylim(bottom=FLOOR_RATE1 * 0.7,
                 top=10 ** (np.ceil(np.log10(rate_max1)) + 0.3))
    ax1.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=6))
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax1.yaxis.set_minor_locator(mticker.NullLocator())
    ax1.margins(x=0.06)
    _clean_axes(ax1)
    ax1.legend(frameon=False, fontsize=9, ncol=2, loc="upper left")

    rlvr_mini = rlvr_summary[rlvr_summary["label"].astype(str).str.contains("gpt-5-mini")].copy()

    if not rlvr_mini.empty:
        max_tokens = pd.to_numeric(rlvr_mini["tokens"], errors="coerce").max()
        label_offsets = {
            "gpt-5-mini": (10, 2, "left"),
            "gpt-5-mini-high": (-8, 8, "right"),
            "gpt-5-mini-low": (8, 8, "left"),
        }
        for _, row in rlvr_mini.iterrows():
            color = _color_for(row["label"])
            ax3.scatter(
                row["tokens"], row["short_pressure_pct"],
                s=55, c=color, marker="o", alpha=0.9,
                edgecolors="white", linewidths=0.5, zorder=4,
            )
            label = str(row["label"])
            if label in label_offsets:
                xoff, yoff, ha = label_offsets[label]
            else:
                align_right = pd.notna(max_tokens) and float(row["tokens"]) > 0.90 * float(max_tokens)
                xoff, yoff, ha = (-6, 4, "right") if align_right else (6, 4, "left")
            ax3.annotate(
                row["label"], (row["tokens"], row["short_pressure_pct"]),
                xytext=(xoff, yoff), textcoords="offset points",
                fontsize=8, color="#333", ha=ha,
            )

        mini_family = rlvr_mini.sort_values("tokens")
        if len(mini_family) >= 2:
            x_vals = pd.to_numeric(mini_family["tokens"], errors="coerce").to_numpy(dtype=float)
            y_vals = pd.to_numeric(mini_family["short_pressure_pct"], errors="coerce").to_numpy(dtype=float)
            ax3.plot(
                x_vals, y_vals,
                linestyle="--", linewidth=1.2, color="#999", zorder=2,
            )

    ax3.set_xlabel("Reasoning Effort (Tokens)")
    ax3.set_ylabel("Shortcut Rate")
    max_pressure_e = pd.to_numeric(rlvr_mini["short_pressure_pct"], errors="coerce").replace([np.inf, -np.inf], np.nan).max()
    if pd.isna(max_pressure_e):
        max_pressure_e = 10.0
    ax3.set_ylim(bottom=-0.3, top=max(5.0, float(max_pressure_e) * 1.22))
    token_values = pd.to_numeric(rlvr_mini["tokens"], errors="coerce")
    token_min = token_values.min()
    token_max = token_values.max()
    if pd.notna(token_min) and pd.notna(token_max):
        span = max(1.0, float(token_max - token_min))
        ax3.set_xlim(float(token_min) - 0.12 * span, float(token_max) + 0.14 * span)
    ax3.margins(x=0.08, y=0.15)
    ax3.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    _clean_axes(ax3)

    scatter_data = delta_df.copy()
    scatter_data = scatter_data.dropna(subset=["model_name", "completion_tokens", "default_correct"]) 
    scatter_data["completion_tokens"] = pd.to_numeric(scatter_data["completion_tokens"], errors="coerce")
    scatter_data["default_correct"] = pd.to_numeric(scatter_data["default_correct"], errors="coerce")
    if "is_shortcut" in scatter_data.columns:
        scatter_data["is_shortcut"] = pd.to_numeric(scatter_data["is_shortcut"], errors="coerce").fillna(0.0)
    else:
        scatter_data["is_shortcut"] = 0.0
    scatter_data = scatter_data.dropna(subset=["completion_tokens", "default_correct"])
    solved = scatter_data[scatter_data["default_correct"] >= 1.0].copy()
    bad_shortcuts = scatter_data[(scatter_data["default_correct"] <= 0.0) & (scatter_data["is_shortcut"] >= 1.0)].copy()
    bad_shortcuts = bad_shortcuts[
        ~bad_shortcuts["model_name"].astype(str).str.startswith("gpt-4-turbo")
    ].copy()

    if not solved.empty or not bad_shortcuts.empty:
        keep_models = pd.concat([
            solved[["model_name"]],
            bad_shortcuts[["model_name"]],
        ]).drop_duplicates()
        keep_models["label"] = keep_models["model_name"].map(_short_label)
        raw_acc_map = (
            scatter_data.assign(label=scatter_data["model_name"].map(_short_label))
            .groupby("label")["default_correct"]
            .mean()
            .to_dict()
        )
        acc_map: dict[str, float] = {str(k): float(v) for k, v in raw_acc_map.items()}
        model_order_scatter = _ordered_scatter_models(keep_models["label"].tolist(), acc_map=acc_map)
        y_map = {label: idx for idx, label in enumerate(model_order_scatter)}

        solved["label"] = solved["model_name"].map(_short_label)
        bad_shortcuts["label"] = bad_shortcuts["model_name"].map(_short_label)
        solved = solved[solved["label"].isin(y_map)].copy()
        bad_shortcuts = bad_shortcuts[bad_shortcuts["label"].isin(y_map)].copy()

        rng = np.random.default_rng(42)
        solved_y = solved["label"].map(y_map).astype(float).to_numpy()
        bad_y = bad_shortcuts["label"].map(y_map).astype(float).to_numpy()
        solved_jitter = rng.uniform(-0.16, -0.02, size=len(solved_y)) if len(solved_y) else np.array([])
        bad_jitter = rng.uniform(0.02, 0.16, size=len(bad_y)) if len(bad_y) else np.array([])

        if len(solved):
            ax2.scatter(
                solved["completion_tokens"].to_numpy(dtype=float),
                solved_y + solved_jitter,
                s=10,
                c="#1f77b4",
                alpha=0.35,
                edgecolors="none",
                label="Solved",
                zorder=3,
            )
        if len(bad_shortcuts):
            ax2.scatter(
                bad_shortcuts["completion_tokens"].to_numpy(dtype=float),
                bad_y + bad_jitter,
                s=12,
                c="#d62728",
                alpha=0.55,
                edgecolors="none",
                label="Shortcut",
                zorder=4,
            )

        ax2.set_xlabel("Reasoning Effort (Tokens)")
        ax2.set_ylabel("")
        ax2.set_yticks(np.arange(len(model_order_scatter)))
        ax2.set_yticklabels(model_order_scatter, rotation=35, ha="right", rotation_mode="anchor")
        ax2.tick_params(axis="y", pad=2)
        ax2.set_ylim(-0.6, len(model_order_scatter) - 0.4)
        ax2.invert_yaxis()
        ax2.legend(
            handles=[
                Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f77b4", markersize=5, alpha=0.6, label="Solved"),
                Line2D([0], [0], marker="o", color="none", markerfacecolor="#d62728", markersize=5, alpha=0.7, label="Shortcut"),
            ],
            frameon=False,
            fontsize=9,
            loc="lower right",
        )
        _clean_axes(ax2)

    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.22, top=0.96, wspace=0.42)

    fig.savefig(os.path.join(output_dir, "paper_complexity_effort_stacked.pdf"))
    fig.savefig(os.path.join(output_dir, "paper_complexity_effort_stacked.png"))
    fig.savefig(os.path.join(output_dir, "paper_complexity_effort_horizontal.pdf"))
    fig.savefig(os.path.join(output_dir, "paper_complexity_effort_horizontal.png"))
    plt.close(fig)
    print("  Plot 4 (complexity+effort+scatter horizontal) saved.")


# ==========================================================================
# Legacy helpers (kept for CSV exports)
# ==========================================================================

def _mean(series: pd.Series) -> float:
    return pd.to_numeric(series, errors="coerce").mean()


def _wilson_interval(
    p: float,
    n: int,
    z: float = 1.96,
) -> Tuple[Optional[float], Optional[float]]:
    if n is None or n <= 0 or p is None or pd.isna(p):
        return None, None
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * ((p * (1.0 - p) + (z * z) / (4.0 * n)) / n) ** 0.5) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _compute_model_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, g in df.groupby("model_name"):
        rows.append({
            "model_name": model_name,
            "shortcut_rate": _mean(g["is_shortcut"]),
            "default_acc": _mean(g["default_correct"]),
            "local_acc": _mean(g["local_correct"]),
            "reasoning_effort": _mean(g["completion_tokens"]),
        })
    return pd.DataFrame(rows)


def _compute_complexity_metrics(df: pd.DataFrame) -> pd.DataFrame:
    order = ["basic", "easy", "medium", "hard"]
    subset = df[df["complexity"].isin(order)].copy()
    rows = []
    for complexity, g in subset.groupby("complexity"):
        rows.append({
            "complexity": complexity,
            "shortcut_rate": _mean(g["is_shortcut"]),
            "default_acc": _mean(g["default_correct"]),
            "local_acc": _mean(g["local_correct"]),
            "reasoning_effort": _mean(g["completion_tokens"]),
            "n": len(g),
        })
    out = pd.DataFrame(rows)
    out["complexity"] = pd.Categorical(out["complexity"], categories=order, ordered=True)
    if not out.empty:
        ci_bounds = [
            _wilson_interval(row["shortcut_rate"], int(row["n"]))
            for _, row in out.iterrows()
        ]
        out["ci_low"] = [c[0] for c in ci_bounds]
        out["ci_high"] = [c[1] for c in ci_bounds]
    return out.sort_values("complexity")


def _compute_level_metrics(df: pd.DataFrame) -> pd.DataFrame:
    subset = df.dropna(subset=["level", "is_shortcut"]).copy()
    subset["level"] = pd.to_numeric(subset["level"], errors="coerce")
    subset = subset.dropna(subset=["level"])
    if subset.empty:
        return pd.DataFrame()

    rows = []
    for level, g in subset.groupby("level"):
        level_value = pd.to_numeric(pd.Series([level]), errors="coerce").iloc[0]
        if pd.isna(level_value):
            continue
        rows.append({
            "level": int(level_value),
            "shortcut_rate": _mean(g["is_shortcut"]),
            "n": len(g),
        })
    out = pd.DataFrame(rows).sort_values("level")
    if not out.empty:
        ci_bounds = [
            _wilson_interval(row["shortcut_rate"], int(row["n"]))
            for _, row in out.iterrows()
        ]
        out["ci_low"] = [c[0] for c in ci_bounds]
        out["ci_high"] = [c[1] for c in ci_bounds]
    return out


def plot_reasoning_effort_solved_vs_bad_shortcuts(df: pd.DataFrame, output_dir: str) -> None:
    """Scatter plot: x=reasoning effort (completion tokens), y=models.

    Shows only:
    - solved instances (default_correct == 1)
    - incorrect shortcut instances (default_correct == 0 and is_shortcut == 1)
    """
    required = ["model_name", "completion_tokens", "default_correct", "is_shortcut"]
    if any(col not in df.columns for col in required):
        return

    data = df.copy()
    data = data.dropna(subset=["model_name", "completion_tokens", "default_correct"]) 
    if data.empty:
        return

    data["completion_tokens"] = pd.to_numeric(data["completion_tokens"], errors="coerce")
    data["default_correct"] = pd.to_numeric(data["default_correct"], errors="coerce")
    data["is_shortcut"] = pd.to_numeric(data["is_shortcut"], errors="coerce").fillna(0.0)
    data = data.dropna(subset=["completion_tokens", "default_correct"])
    if data.empty:
        return

    solved = data[data["default_correct"] >= 1.0].copy()
    bad_shortcuts = data[(data["default_correct"] <= 0.0) & (data["is_shortcut"] >= 1.0)].copy()
    bad_shortcuts = bad_shortcuts[
        ~bad_shortcuts["model_name"].astype(str).str.startswith("gpt-4-turbo")
    ].copy()
    if solved.empty and bad_shortcuts.empty:
        return

    keep_models = pd.concat([
        solved[["model_name"]],
        bad_shortcuts[["model_name"]],
    ]).drop_duplicates()
    keep_models["label"] = keep_models["model_name"].map(_short_label)
    raw_acc_map = (
        data.assign(label=data["model_name"].map(_short_label))
        .groupby("label")["default_correct"]
        .mean()
        .to_dict()
    )
    acc_map: dict[str, float] = {str(k): float(v) for k, v in raw_acc_map.items()}
    model_order = _ordered_scatter_models(keep_models["label"].tolist(), acc_map=acc_map)
    y_map = {label: idx for idx, label in enumerate(model_order)}

    solved["label"] = solved["model_name"].map(_short_label)
    bad_shortcuts["label"] = bad_shortcuts["model_name"].map(_short_label)
    solved = solved[solved["label"].isin(y_map)].copy()
    bad_shortcuts = bad_shortcuts[bad_shortcuts["label"].isin(y_map)].copy()

    rng = np.random.default_rng(42)
    solved_y = solved["label"].map(y_map).astype(float).to_numpy()
    bad_y = bad_shortcuts["label"].map(y_map).astype(float).to_numpy()
    solved_jitter = rng.uniform(-0.16, -0.02, size=len(solved_y)) if len(solved_y) else np.array([])
    bad_jitter = rng.uniform(0.02, 0.16, size=len(bad_y)) if len(bad_y) else np.array([])

    fig, ax = plt.subplots(figsize=_FIG_SIZE_SMALL)

    if len(solved):
        ax.scatter(
            solved["completion_tokens"].to_numpy(dtype=float),
            solved_y + solved_jitter,
            s=12,
            c="#1f77b4",
            alpha=0.35,
            edgecolors="none",
            label="Solved",
            zorder=3,
        )

    if len(bad_shortcuts):
        ax.scatter(
            bad_shortcuts["completion_tokens"].to_numpy(dtype=float),
            bad_y + bad_jitter,
            s=14,
            c="#d62728",
            alpha=0.55,
            edgecolors="none",
            label="Shortcut",
            zorder=4,
        )

    ax.set_xlabel("Reasoning Effort (Tokens)")
    ax.set_ylabel("")
    ax.set_yticks(np.arange(len(model_order)))
    ax.set_yticklabels(model_order, rotation=35, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="y", pad=2)
    ax.set_xscale("log")
    ax.set_ylim(-0.6, len(model_order) - 0.4)
    ax.invert_yaxis()
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f77b4", markersize=5, alpha=0.6, label="Solved"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#d62728", markersize=5, alpha=0.7, label="Shortcut"),
        ],
        frameon=False,
        fontsize=10,
        loc="lower right",
    )
    _clean_axes(ax)
    fig.subplots_adjust(left=0.24, right=0.99, bottom=0.18, top=0.96)

    _save_fixed_canvas(fig, os.path.join(output_dir, "paper_effort_model_scatter_solved_vs_bad_shortcuts.pdf"))
    _save_fixed_canvas(fig, os.path.join(output_dir, "paper_effort_model_scatter_solved_vs_bad_shortcuts.png"))
    plt.close(fig)
    print("  Plot 5 (model effort scatter: solved vs bad shortcuts) saved.")


def plot_level_complexity_solved_vs_bad_shortcuts(df: pd.DataFrame, output_dir: str) -> None:
    """Scatter plot: x=task level, y=models.

    Shows only:
    - solved instances (default_correct == 1)
    - incorrect shortcut instances (default_correct == 0 and is_shortcut == 1)
    """
    required = ["model_name", "level", "default_correct", "is_shortcut"]
    if any(col not in df.columns for col in required):
        return

    data = df.copy()
    data = data.dropna(subset=["model_name", "level", "default_correct"])
    if data.empty:
        return

    data["level"] = pd.to_numeric(data["level"], errors="coerce")
    data["default_correct"] = pd.to_numeric(data["default_correct"], errors="coerce")
    data["is_shortcut"] = pd.to_numeric(data["is_shortcut"], errors="coerce").fillna(0.0)
    data = data.dropna(subset=["level", "default_correct"])
    if data.empty:
        return

    solved = data[data["default_correct"] >= 1.0].copy()
    bad_shortcuts = data[(data["default_correct"] <= 0.0) & (data["is_shortcut"] >= 1.0)].copy()
    bad_shortcuts = bad_shortcuts[
        ~bad_shortcuts["model_name"].astype(str).str.startswith("gpt-4-turbo")
    ].copy()
    if solved.empty and bad_shortcuts.empty:
        return

    keep_models = pd.concat([
        solved[["model_name"]],
        bad_shortcuts[["model_name"]],
    ]).drop_duplicates()
    keep_models["label"] = keep_models["model_name"].map(_short_label)
    raw_acc_map = (
        data.assign(label=data["model_name"].map(_short_label))
        .groupby("label")["default_correct"]
        .mean()
        .to_dict()
    )
    acc_map: dict[str, float] = {str(k): float(v) for k, v in raw_acc_map.items()}
    model_order = _ordered_scatter_models(keep_models["label"].tolist(), acc_map=acc_map)
    y_map = {label: idx for idx, label in enumerate(model_order)}

    solved["label"] = solved["model_name"].map(_short_label)
    bad_shortcuts["label"] = bad_shortcuts["model_name"].map(_short_label)
    solved = solved[solved["label"].isin(y_map)].copy()
    bad_shortcuts = bad_shortcuts[bad_shortcuts["label"].isin(y_map)].copy()

    rng = np.random.default_rng(42)
    solved_y = solved["label"].map(y_map).astype(float).to_numpy()
    bad_y = bad_shortcuts["label"].map(y_map).astype(float).to_numpy()
    solved_y_jitter = rng.uniform(-0.16, -0.02, size=len(solved_y)) if len(solved_y) else np.array([])
    bad_y_jitter = rng.uniform(0.02, 0.16, size=len(bad_y)) if len(bad_y) else np.array([])

    solved_x = solved["level"].to_numpy(dtype=float)
    bad_x = bad_shortcuts["level"].to_numpy(dtype=float)
    solved_x_jitter = rng.uniform(-0.22, 0.22, size=len(solved_x)) if len(solved_x) else np.array([])
    bad_x_jitter = rng.uniform(-0.22, 0.22, size=len(bad_x)) if len(bad_x) else np.array([])

    fig, ax = plt.subplots(figsize=_FIG_SIZE_SMALL)

    if len(solved):
        ax.scatter(
            solved_x + solved_x_jitter,
            solved_y + solved_y_jitter,
            s=12,
            c="#1f77b4",
            alpha=0.35,
            edgecolors="none",
            label="Solved",
            zorder=3,
        )

    if len(bad_shortcuts):
        ax.scatter(
            bad_x + bad_x_jitter,
            bad_y + bad_y_jitter,
            s=14,
            c="#d62728",
            alpha=0.55,
            edgecolors="none",
            label="Shortcut",
            zorder=4,
        )

    level_min = int(np.floor(data["level"].min()))
    level_max = int(np.ceil(data["level"].max()))
    if level_max <= 10:
        tick_step = 1
    elif level_max <= 20:
        tick_step = 2
    else:
        tick_step = max(1, int(np.ceil((level_max - level_min) / 10)))

    ax.set_xlabel("Task Complexity")
    ax.set_ylabel("")
    ax.set_yticks(np.arange(len(model_order)))
    ax.set_yticklabels(model_order, rotation=35, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="y", pad=2)
    ax.set_xticks(np.arange(level_min+1, level_max + 1, tick_step))
    ax.set_xlim(level_min - 0.6, level_max + 0.6)
    ax.set_ylim(-0.6, len(model_order) - 0.4)
    ax.invert_yaxis()
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f77b4", markersize=5, alpha=0.6, label="Solved"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#d62728", markersize=5, alpha=0.7, label="Shortcut"),
        ],
        frameon=False,
        fontsize=10,
        loc="lower right",
    )
    _clean_axes(ax)
    fig.subplots_adjust(left=0.24, right=0.99, bottom=0.18, top=0.96)

    _save_fixed_canvas(fig, os.path.join(output_dir, "paper_level_model_scatter_solved_vs_bad_shortcuts.pdf"))
    _save_fixed_canvas(fig, os.path.join(output_dir, "paper_level_model_scatter_solved_vs_bad_shortcuts.png"))
    plt.close(fig)
    print("  Plot 6 (model complexity-level scatter: solved vs bad shortcuts) saved.")


# ==========================================================================
# Entry point
# ==========================================================================

def generate_shortcut_correlation_plots(
    df: pd.DataFrame,
    output_dir: str,
    plots_subdir: str = "shortcut_plots",
) -> None:
    if df is None or df.empty:
        print("No data available for plotting.")
        return

    plots_dir = _ensure_plots_dir(output_dir, plots_subdir)

    # CSV exports.
    metrics = _compute_model_metrics(df)
    complexity_df = _compute_complexity_metrics(df)
    level_df = _compute_level_metrics(df)
    delta_df = _compute_delta_frame(df)

    metrics.to_csv(os.path.join(plots_dir, "model_metrics.csv"), index=False)
    complexity_df.to_csv(os.path.join(plots_dir, "complexity_metrics.csv"), index=False)
    if not level_df.empty:
        level_df.to_csv(os.path.join(plots_dir, "level_metrics.csv"), index=False)
    if not delta_df.empty:
        delta_df.to_csv(os.path.join(plots_dir, "delta_metrics_long.csv"), index=False)

    # Paper plots.
    if not delta_df.empty:
        plot_paper_complexity(delta_df, plots_dir)
        plot_paper_capability(delta_df, plots_dir)
        plot_paper_effort(delta_df, plots_dir)
        plot_paper_complexity_effort_stacked(delta_df, plots_dir)
    plot_reasoning_effort_solved_vs_bad_shortcuts(df, plots_dir)
    plot_level_complexity_solved_vs_bad_shortcuts(df, plots_dir)
    _export_paper_plots(plots_dir)

    print(f"Plots saved to: {plots_dir}")
