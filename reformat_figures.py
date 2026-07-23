#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-processing only: reformat Policy Design manuscript figures from saved outputs.

IMPORTANT:
- This script does NOT fit models.
- This script does NOT run redesign/design frontier search.
- This script only reads already-saved CSV files and redraws three PDF figures:

    scatter.pdf
    focal.pdf
    policy.pdf
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "policy_design_manuscript_outputs"
OUTPUT_DIR = RESULTS_DIR / "reformatted_figures"

FOCAL_TEAMS = [5100, 7270]
CMAP_GREEN_TO_RED = "RdYlGn_r"
FIG_DPI = 300

SCATTER_XLIM = (4.0, 7.08)
SCATTER_YLIM = (4.0, 7.08)
POLICY_VMIN = 1.0
POLICY_VMAX = 7.0


def ensure_results_dir() -> Path:
    if RESULTS_DIR.exists():
        return RESULTS_DIR
    candidates = sorted(SCRIPT_DIR.glob("policy_design_manuscript_outputs*.zip"))
    if candidates:
        with zipfile.ZipFile(candidates[0], "r") as zf:
            zf.extractall(SCRIPT_DIR)
        if RESULTS_DIR.exists():
            return RESULTS_DIR
    raise FileNotFoundError("Could not find policy_design_manuscript_outputs/ or matching zip file.")


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_profiles(results_dir: Path) -> pd.DataFrame:
    df = read_csv_required(results_dir / "tables_csv" / "figure_2_all_team_baseline_profiles.csv")
    df["team_id"] = df["team_id"].astype(str)
    return df


def read_frontier(results_dir: Path, setting: str, team: int) -> pd.DataFrame:
    df = read_csv_required(results_dir / "raw_frontiers" / f"{setting}_frontier_team_{team}.csv")
    df["team_id"] = str(team)
    df["setting"] = setting
    return df


def read_saved_selected(results_dir: Path) -> pd.DataFrame:
    df = read_csv_required(results_dir / "selected_policies" / "selected_policy_rows_for_figure_4.csv")
    df["team_id"] = df["team_id"].astype(str)
    return df


def policy_feature_names_from_df(df: pd.DataFrame) -> List[str]:
    return [c.replace("policy__", "") for c in df.columns if c.startswith("policy__")]


def policy_cols(policy_feats: Sequence[str]) -> List[str]:
    return [f"policy__{feat}" for feat in policy_feats]


def hrm_activity_labels_single_line(policy_feats: Sequence[str]) -> List[str]:
    return list(policy_feats)


def row_signature(row: pd.Series, pcols: Sequence[str], ndigits: int = 8) -> Tuple[float, ...]:
    return tuple(round(float(row[c]), ndigits) for c in pcols if c in row and pd.notna(row[c]))


def add_unique_policy(rows: List[Dict], label: str, row: pd.Series, policy_feats: Sequence[str], setting: str) -> None:
    pcols = policy_cols(policy_feats)
    sig = row_signature(row, pcols)
    for existing in rows:
        if existing.get("_signature") == sig:
            return
    d = {
        "display_label": label,
        "setting": setting,
        "solution_id": row.get("solution_id", row.get("pattern_id", "")),
        "cost": float(row["cost"]),
        "mean_performance": float(row["mean_performance"]),
        "mean_job_satisfaction": float(row["mean_job_satisfaction"]),
        "_signature": sig,
    }
    for feat in policy_feats:
        d[f"policy__{feat}"] = float(row[f"policy__{feat}"])
    rows.append(d)


def select_initial_or_benchmark(saved_selected: pd.DataFrame, team: int, setting: str) -> pd.Series:
    sub = saved_selected[(saved_selected["team_id"].astype(str) == str(team)) & (saved_selected["setting"].astype(str) == setting)].copy()
    if sub.empty:
        raise ValueError(f"No selected policy rows for team={team}, setting={setting}")
    target = sub[sub["display_label"].astype(str).str.contains("Initial|Observed benchmark", case=False, na=False)]
    return target.iloc[0] if not target.empty else sub.iloc[0]


def select_min_cost(frontier: pd.DataFrame) -> pd.Series:
    return frontier.sort_values(["cost", "mean_performance", "mean_job_satisfaction"], ascending=[True, False, False]).iloc[0]


def select_threshold(frontier: pd.DataFrame, thr: float) -> pd.Series:
    ok = frontier[(frontier["mean_performance"].astype(float) >= thr - 1e-9) & (frontier["mean_job_satisfaction"].astype(float) >= thr - 1e-9)].copy()
    if not ok.empty:
        return ok.sort_values(["cost", "mean_performance", "mean_job_satisfaction"], ascending=[True, False, False]).iloc[0]
    work = frontier.copy()
    short_perf = np.maximum(0.0, thr - work["mean_performance"].to_numpy(dtype=float))
    short_sat = np.maximum(0.0, thr - work["mean_job_satisfaction"].to_numpy(dtype=float))
    work["shortfall"] = short_perf + short_sat
    return work.sort_values(["shortfall", "cost", "mean_performance", "mean_job_satisfaction"], ascending=[True, True, False, False]).iloc[0]


def select_redesign_budget5_max_combined(frontier: pd.DataFrame) -> pd.Series:
    work = frontier[frontier["cost"].astype(float) <= 5.0 + 1e-9].copy()
    if work.empty:
        work = frontier.copy()
    work["combined"] = work["mean_performance"].astype(float) + work["mean_job_satisfaction"].astype(float)
    return work.sort_values(["combined", "cost"], ascending=[False, True]).iloc[0]


def build_selected_policy_rows(frontier: pd.DataFrame, saved_selected: pd.DataFrame, team: int, setting: str, policy_feats: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict] = []
    initial = select_initial_or_benchmark(saved_selected, team, setting)
    add_unique_policy(rows, "Initial", initial, policy_feats, setting)
    if setting == "redesign":
        add_unique_policy(rows, "Budget 5", select_redesign_budget5_max_combined(frontier), policy_feats, setting)
    else:
        add_unique_policy(rows, "Min cost", select_min_cost(frontier), policy_feats, setting)
    add_unique_policy(rows, "6/6", select_threshold(frontier, 6.0), policy_feats, setting)
    add_unique_policy(rows, "6.5/6.5", select_threshold(frontier, 6.5), policy_feats, setting)
    add_unique_policy(rows, "7/7", select_threshold(frontier, 7.0), policy_feats, setting)
    out = pd.DataFrame(rows).drop(columns=["_signature"], errors="ignore")
    out.insert(0, "team_id", str(team))
    return out


def load_selected_map(results_dir: Path, saved_selected: pd.DataFrame) -> Tuple[Dict[Tuple[int, str], pd.DataFrame], List[str]]:
    sample_frontier = read_frontier(results_dir, "redesign", FOCAL_TEAMS[0])
    policy_feats = policy_feature_names_from_df(sample_frontier)
    selected_map: Dict[Tuple[int, str], pd.DataFrame] = {}
    for team in FOCAL_TEAMS:
        for setting in ["redesign", "design"]:
            frontier = read_frontier(results_dir, setting, team)
            selected_map[(team, setting)] = build_selected_policy_rows(frontier, saved_selected, team, setting, policy_feats)
    return selected_map, policy_feats


def compute_focal_limits(selected_map: Dict[Tuple[int, str], pd.DataFrame]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    perf_vals = []
    sat_vals = []
    for df in selected_map.values():
        perf_vals.extend(pd.to_numeric(df["mean_performance"], errors="coerce").tolist())
        sat_vals.extend(pd.to_numeric(df["mean_job_satisfaction"], errors="coerce").tolist())
    perf_vals = np.array([v for v in perf_vals if pd.notna(v)], dtype=float)
    sat_vals = np.array([v for v in sat_vals if pd.notna(v)], dtype=float)
    xlow = max(4.0, float(perf_vals.min()) - 0.08)
    ylow = max(4.0, float(sat_vals.min()) - 0.08)
    xhigh = min(7.08, max(7.02, float(perf_vals.max()) + 0.04))
    yhigh = min(7.08, max(7.02, float(sat_vals.max()) + 0.04))
    return (xlow, xhigh), (ylow, yhigh)


# ---------------------- scatter.pdf ----------------------

def plot_scatter_pdf(profiles: pd.DataFrame) -> None:
    panels = [
        ("Observed outcomes", "observed_mean_performance", "observed_mean_job_satisfaction"),
        ("Mixed model predictions (redesign)", "redesign_pred_baseline_performance", "redesign_pred_baseline_job_satisfaction"),
        ("Decision tree predictions (design)", "design_pred_benchmark_performance", "design_pred_benchmark_job_satisfaction"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.6), sharex=False, sharey=False)
    cmap = plt.get_cmap(CMAP_GREEN_TO_RED)
    cvals = profiles["baseline_policy_implementation_cost"].astype(float)
    size_map = {1: 35, 2: 70, 3: 115}
    sizes = profiles["n_employees"].map(size_map).fillna(70).astype(float)

    label_offsets = {
        0: {"5100": (-24, 8), "7270": (8, -12)},
        1: {"5100": (-22, -12), "7270": (6, 6)},
        2: {"5100": (-18, 8), "7270": (6, 6)},
    }

    last_scatter = None
    for i, (ax, (title, xcol, ycol)) in enumerate(zip(axes, panels)):
        x_plot = np.clip(pd.to_numeric(profiles[xcol], errors="coerce").to_numpy(dtype=float), *SCATTER_XLIM)
        y_plot = np.clip(pd.to_numeric(profiles[ycol], errors="coerce").to_numpy(dtype=float), *SCATTER_YLIM)
        last_scatter = ax.scatter(
            x_plot, y_plot, s=sizes, c=cvals, cmap=cmap, alpha=0.78, edgecolor="white", linewidth=0.45,
        )

        for team in FOCAL_TEAMS:
            row = profiles.loc[profiles["team_id"].astype(str) == str(team)]
            if row.empty:
                continue
            n_emp = int(row["n_employees"].iloc[0])
            s = float(size_map.get(n_emp, 85))
            x0 = float(np.clip(row[xcol].iloc[0], *SCATTER_XLIM))
            y0 = float(np.clip(row[ycol].iloc[0], *SCATTER_YLIM))
            ax.scatter([x0], [y0], s=s + 90, facecolors="none", edgecolors="black", linewidths=1.8, zorder=5)
            dx, dy = label_offsets[i].get(str(team), (6, 6))
            ax.annotate(str(team), (x0, y0), xytext=(dx, dy), textcoords="offset points", fontsize=9, fontweight="bold")

        ax.set_xlim(*SCATTER_XLIM)
        ax.set_ylim(*SCATTER_YLIM)
        ax.set_title(title, fontsize=12.5, fontweight="bold")
        ax.set_xlabel("Mean performance", fontsize=11)
        ax.set_ylabel("Mean job satisfaction", fontsize=11)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(alpha=0.25)

    handles, labels = [], []
    for n, s in size_map.items():
        handles.append(plt.scatter([], [], s=s, facecolors="lightgray", edgecolors="black", linewidths=0.5))
        labels.append(f"team size: {n}")
    axes[0].legend(handles, labels, loc="lower left", fontsize=9, frameon=True)

    cbar = fig.colorbar(last_scatter, ax=axes.ravel().tolist(), shrink=0.90, pad=0.02)
    cbar.set_label("Baseline policy cost", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    fig.savefig(OUTPUT_DIR / "scatter.pdf", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------- focal.pdf ----------------------

def circle_selected_points(ax: plt.Axes, selected: pd.DataFrame) -> None:
    ax.scatter(selected["mean_performance"], selected["mean_job_satisfaction"], s=175, facecolors="none", edgecolors="black", linewidths=1.5, zorder=6, clip_on=True)


def plot_frontier_panel(ax: plt.Axes, frontier: pd.DataFrame, selected: pd.DataFrame, title: str, xlim: Tuple[float, float], ylim: Tuple[float, float]) -> plt.cm.ScalarMappable:
    sc = ax.scatter(frontier["mean_performance"], frontier["mean_job_satisfaction"], c=frontier["cost"], cmap=CMAP_GREEN_TO_RED, s=42, alpha=0.88, edgecolor="black", linewidth=0.25)
    circle_selected_points(ax, selected)
    ax.set_title(title, fontsize=12.5, fontweight="bold")
    ax.set_xlabel("Mean performance", fontsize=11)
    ax.set_ylabel("Mean job satisfaction", fontsize=11)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(alpha=0.25)
    return sc


def plot_focal_pdf(results_dir: Path, selected_map: Dict[Tuple[int, str], pd.DataFrame]) -> None:
    xlim, ylim = compute_focal_limits(selected_map)
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2), sharex=True, sharey=True)
    # rows=settings, cols=teams
    layout = [(0, 0, "redesign", 5100), (0, 1, "redesign", 7270), (1, 0, "design", 5100), (1, 1, "design", 7270)]
    for r, c, setting, team in layout:
        ax = axes[r, c]
        frontier = read_frontier(results_dir, setting, team)
        selected = selected_map[(team, setting)]
        sc = plot_frontier_panel(ax, frontier, selected, f"Team {team} — {setting}", xlim, ylim)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.035)
        cbar.set_label("Cost", fontsize=10)
        cbar.ax.tick_params(labelsize=9)
    fig.savefig(OUTPUT_DIR / "focal.pdf", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------- policy.pdf ----------------------

def plot_policy_pdf(selected_map: Dict[Tuple[int, str], pd.DataFrame], policy_feats: Sequence[str]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(20.0, 8.2), sharex=True)
    cmap = plt.get_cmap(CMAP_GREEN_TO_RED)
    xlabels = hrm_activity_labels_single_line(policy_feats)
    pcols = policy_cols(policy_feats)
    last_img = None
    layout = [(0, 0, "redesign", 5100), (0, 1, "redesign", 7270), (1, 0, "design", 5100), (1, 1, "design", 7270)]

    for r, c, setting, team in layout:
        ax = axes[r, c]
        selected = selected_map[(team, setting)].copy()
        arr = selected[pcols].to_numpy(dtype=float)
        ylabels = selected["display_label"].astype(str).tolist()
        last_img = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=POLICY_VMIN, vmax=POLICY_VMAX)
        ax.set_title(f"Team {team} — {setting}", fontsize=15, fontweight="bold")
        ax.set_xticks(np.arange(len(policy_feats)))
        ax.set_xticklabels(xlabels, fontsize=8, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=13)
        ax.set_ylabel("")
        ax.set_xticks(np.arange(-0.5, len(policy_feats), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(ylabels), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(axis="both", labelsize=10)

    cbar = fig.colorbar(last_img, ax=axes.ravel().tolist(), shrink=0.82, pad=0.012)
    cbar.set_label("Policy Level", fontsize=14)
    cbar.ax.tick_params(labelsize=13)
    fig.savefig(OUTPUT_DIR / "policy.pdf", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------- main ----------------------

def main() -> None:
    results_dir = ensure_results_dir()
    ensure_output_dir()
    print(f"Reading saved outputs from: {results_dir}")
    print("No model fitting and no frontier search will be run.")

    profiles = read_profiles(results_dir)
    plot_scatter_pdf(profiles)
    print(f"Written: {OUTPUT_DIR / 'scatter.pdf'}")

    saved_selected = read_saved_selected(results_dir)
    selected_map, policy_feats = load_selected_map(results_dir, saved_selected)
    selected_all = pd.concat([df.assign(setting_key=f"{team}_{setting}") for (team, setting), df in selected_map.items()], ignore_index=True)
    selected_all.to_csv(OUTPUT_DIR / "selected_policy_rows_policy_composition_reformatted.csv", index=False)
    print(f"Written: {OUTPUT_DIR / 'selected_policy_rows_policy_composition_reformatted.csv'}")

    plot_focal_pdf(results_dir, selected_map)
    print(f"Written: {OUTPUT_DIR / 'focal.pdf'}")

    plot_policy_pdf(selected_map, policy_feats)
    print(f"Written: {OUTPUT_DIR / 'policy.pdf'}")
    print("Done.")


if __name__ == "__main__":
    main()
