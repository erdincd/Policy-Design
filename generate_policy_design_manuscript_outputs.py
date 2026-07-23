#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the revised manuscript-ready tables and figures for the Policy Design paper.

This script implements the current decisions:

Empirical setting
-----------------
Figure 2. All-team baseline outcome profiles
    - observed means
    - redesign-predicted baseline means from clipped mixed-model predictions
    - design-benchmark means from the decision tree
    - focal teams are highlighted with black outlines, not X markers
    - color encodes baseline policy implementation cost using a green-to-red scale
    - marker size encodes team size

Results
-------
Table 1. All-team redesign summary statistics
    - all observed teams
    - no fixed redesign budget cap; finite sign-restricted redesign domain
    - mixed-model predictions are clipped to [1, 7]
    - records threshold costs, candidate upper bound, visited states, pruned states,
      frontier size, and runtime

Table 2. All-team design summary statistics
    - all observed teams
    - decision-tree leaf-pattern + global hull design search
    - records matching threshold/computational statistics

Figure 3. Focal-team frontiers
    - 2 x 2 layout: rows = teams (5100, 7270), columns = redesign/design

Figure 4. Focal-team policy heatmaps
    - same 2 x 2 layout as Figure 3
    - selected policy solutions shown as absolute 1--7 policy levels
    - same fixed green-to-red colorbar from 1 to 7 in all panels

Expected input files in the same folder:
    - Data - Policy Redesign.csv
    - generate_policy_design_results.py

Run:
    python generate_policy_design_manuscript_outputs.py

Outputs:
    policy_design_manuscript_outputs/
"""

from __future__ import annotations

import math
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Make imports work when this script is run from any folder
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

try:
    import generate_policy_design_results as base
except ImportError as exc:
    raise ImportError(
        "This script imports helper functions from generate_policy_design_results.py. "
        "Please put both scripts in the same folder."
    ) from exc

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
RN = 42
DATA_PATH = Path("Data - Policy Redesign.csv")
OUT_DIR = Path("policy_design_manuscript_outputs")
FIG_DPI = 300
FIG_FORMATS = ["png", "pdf"]

FOCAL_TEAMS = [5100, 7270]

# Toggle these if runtime becomes an issue during development.
RUN_ALL_TEAM_REDESIGN = True
RUN_ALL_TEAM_DESIGN = True
WRITE_PER_TEAM_FRONTIERS = True

# Prediction bounds used for mixed-model redesign predictions.
OUTCOME_LOWER_BOUND = 1.0
OUTCOME_UPPER_BOUND = 7.0

# Thresholds recorded in Tables 1 and 2.
THRESHOLDS = [6.0, 6.5, 7.0]

# Design search uses the same decision-tree parameters as the existing script.
TREE_PARAMS = dict(base.TREE_PARAMS)

# Color scale: low values green, high values red.
CMAP_GREEN_TO_RED = "RdYlGn_r"

# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def ensure_outdirs() -> None:
    for sub in ["figures", "tables_csv", "raw_frontiers", "selected_policies"]:
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, stem: str) -> None:
    for fmt in FIG_FORMATS:
        fig.savefig(OUT_DIR / "figures" / f"{stem}.{fmt}", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def as_team_label(g: Any) -> str:
    try:
        gf = float(g)
        if np.isfinite(gf) and gf.is_integer():
            return str(int(gf))
    except Exception:
        pass
    return str(g)


def sort_group_values(values: Iterable[Any]) -> List[Any]:
    def key(v: Any) -> Tuple[int, Any]:
        try:
            return (0, int(float(v)))
        except Exception:
            return (1, str(v))
    return sorted(list(values), key=key)


def clip_outcome(x: float) -> float:
    return float(np.clip(float(x), OUTCOME_LOWER_BOUND, OUTCOME_UPPER_BOUND))


def python_int_product(values: Iterable[int]) -> int:
    out = 1
    for v in values:
        out *= int(v)
    return int(out)


def lookup_random_intercept(result: Any, group_value: Any) -> float:
    re_dict = getattr(result, "random_effects", {})
    candidates = [group_value, str(group_value), as_team_label(group_value)]
    try:
        gf = float(group_value)
        if np.isfinite(gf):
            candidates.append(gf)
            if gf.is_integer():
                candidates.append(int(gf))
    except Exception:
        pass

    for key in candidates:
        if key in re_dict:
            val = re_dict[key]
            if isinstance(val, pd.Series):
                return float(val.iloc[0])
            if isinstance(val, dict):
                return float(list(val.values())[0])
            arr = np.asarray(val).reshape(-1)
            return float(arr[0]) if arr.size else 0.0

    target = as_team_label(group_value)
    for key, val in re_dict.items():
        if as_team_label(key) == target:
            if isinstance(val, pd.Series):
                return float(val.iloc[0])
            if isinstance(val, dict):
                return float(list(val.values())[0])
            arr = np.asarray(val).reshape(-1)
            return float(arr[0]) if arr.size else 0.0
    return 0.0


def metric_from_meta(meta: pd.DataFrame, name: str, default: Any = np.nan) -> Any:
    try:
        d = meta.set_index("metric")["value"].to_dict()
        return d.get(name, default)
    except Exception:
        return default


def summarize_for_manuscript(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    stats = [
        ("Mean", lambda s: s.mean()),
        ("SD", lambda s: s.std(ddof=1)),
        ("Min", lambda s: s.min()),
        ("Median", lambda s: s.median()),
        ("Max", lambda s: s.max()),
    ]
    for stat_name, fn in stats:
        row: Dict[str, Any] = {"Statistic": stat_name}
        for col in columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            row[col] = float(fn(s)) if len(s) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def cost_to_threshold(frontier: pd.DataFrame, perf_thr: Optional[float] = None, sat_thr: Optional[float] = None) -> float:
    if frontier.empty:
        return np.nan
    mask = np.ones(len(frontier), dtype=bool)
    if perf_thr is not None:
        mask &= frontier["mean_performance"].to_numpy(dtype=float) >= perf_thr - 1e-9
    if sat_thr is not None:
        mask &= frontier["mean_job_satisfaction"].to_numpy(dtype=float) >= sat_thr - 1e-9
    if not np.any(mask):
        return np.nan
    return float(frontier.loc[mask, "cost"].min())


def parse_numeric_meta(meta: pd.DataFrame, key: str, default: float = np.nan) -> float:
    val = metric_from_meta(meta, key, default)
    try:
        return float(val)
    except Exception:
        return default

# -----------------------------------------------------------------------------
# All-team baseline profiles and Figure 1
# -----------------------------------------------------------------------------


def predict_mlm_employee_rows_clipped(data: base.DataBundle,
                                      mlm_results: Dict[str, Any],
                                      row_index: Sequence[Any]) -> pd.DataFrame:
    X = data.x.loc[row_index, data.all_feats].copy()
    groups = data.group_id.loc[row_index]
    out_df = pd.DataFrame(index=X.index)
    for out in data.outcomes:
        res = mlm_results[out]
        fe = res.fe_params.copy()
        vals = np.full(len(X), base.get_intercept(fe), dtype=float)
        for feat in data.all_feats:
            vals += base.get_coef(fe, feat) * X[feat].astype(float).to_numpy()
        vals += np.array([lookup_random_intercept(res, g) for g in groups], dtype=float)
        vals = np.clip(vals, OUTCOME_LOWER_BOUND, OUTCOME_UPPER_BOUND)
        out_df[out] = vals
    return out_df


def make_all_team_baseline_profiles(data: base.DataBundle,
                                    mlm_results: Dict[str, Any],
                                    tree_model: DecisionTreeRegressor) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for g in sort_group_values(data.group_id.unique()):
        team_mask = data.group_id == g
        idx = data.x.index[team_mask]
        team_x = data.x.loc[idx, data.all_feats].copy()
        team_y = data.y.loc[idx, data.outcomes].copy()
        baseline_policy_arr = team_x[data.policy_feats].iloc[0].to_numpy(dtype=float)
        baseline_cost = float(np.dot(data.cost_vector, baseline_policy_arr))

        redesign_pred = predict_mlm_employee_rows_clipped(data, mlm_results, idx)
        design_pred = tree_model.predict(team_x[data.all_feats])
        design_pred = np.clip(design_pred, OUTCOME_LOWER_BOUND, OUTCOME_UPPER_BOUND)

        rows.append({
            "team_id": as_team_label(g),
            "n_employees": int(team_mask.sum()),
            "observed_mean_performance": float(team_y[data.outcomes[0]].mean()),
            "observed_mean_job_satisfaction": float(team_y[data.outcomes[1]].mean()),
            "redesign_pred_baseline_performance": float(redesign_pred[data.outcomes[0]].mean()),
            "redesign_pred_baseline_job_satisfaction": float(redesign_pred[data.outcomes[1]].mean()),
            "design_pred_benchmark_performance": float(np.mean(design_pred[:, 0])),
            "design_pred_benchmark_job_satisfaction": float(np.mean(design_pred[:, 1])),
            "baseline_policy_implementation_cost": baseline_cost,
        })
    return pd.DataFrame(rows)


def plot_figure_2_all_team_baseline_profiles(profiles: pd.DataFrame) -> None:
    panels = [
        ("Observed outcomes", "observed_mean_performance", "observed_mean_job_satisfaction"),
        ("Redesign baseline predictions\n(clipped mixed model)", "redesign_pred_baseline_performance", "redesign_pred_baseline_job_satisfaction"),
        ("Design benchmark predictions\n(decision tree)", "design_pred_benchmark_performance", "design_pred_benchmark_job_satisfaction"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.8), sharex=False, sharey=False)
    cmap = plt.get_cmap(CMAP_GREEN_TO_RED)
    cvals = profiles["baseline_policy_implementation_cost"].astype(float)

    # Keep size legend interpretable: 1, 2, 3-person teams.
    size_map = {1: 35, 2: 70, 3: 110}
    sizes = profiles["n_employees"].map(size_map).fillna(70).astype(float)

    last_scatter = None
    for ax, (title, xcol, ycol) in zip(axes, panels):
        last_scatter = ax.scatter(
            profiles[xcol], profiles[ycol],
            s=sizes,
            c=cvals,
            cmap=cmap,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.45,
        )

        # Highlight focal teams by black outline only, no X marker.
        for team in FOCAL_TEAMS:
            row = profiles.loc[profiles["team_id"] == str(team)]
            if row.empty:
                continue
            team_size = float(size_map.get(int(row["n_employees"].iloc[0]), 90))
            ax.scatter(
                row[xcol], row[ycol],
                s=team_size + 75,
                facecolors="none",
                edgecolors="black",
                linewidths=1.7,
                zorder=5,
            )
            ax.annotate(
                str(team),
                (float(row[xcol].iloc[0]), float(row[ycol].iloc[0])),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8.5,
                fontweight="bold",
            )

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean performance")
        ax.set_ylabel("Mean job satisfaction")
        ax.set_xlim(OUTCOME_LOWER_BOUND - 0.15, OUTCOME_UPPER_BOUND + 0.15)
        ax.set_ylim(OUTCOME_LOWER_BOUND - 0.15, OUTCOME_UPPER_BOUND + 0.15)
        ax.grid(alpha=0.25)

    # Team-size legend.
    handles = []
    labels = []
    for n, s in size_map.items():
        handles.append(plt.scatter([], [], s=s, facecolors="lightgray", edgecolors="black", linewidths=0.5))
        labels.append(f"{n}-member team")
    axes[0].legend(handles, labels, title="Marker size = team size", loc="lower right", fontsize=8, title_fontsize=8, frameon=True)

    cbar = fig.colorbar(last_scatter, ax=axes.ravel().tolist(), shrink=0.88, pad=0.02)
    cbar.set_label("Baseline policy implementation cost", fontsize=9)
    fig.suptitle("Figure 1. All-team baseline outcome profiles", fontsize=14, fontweight="bold", y=1.02)
    save_fig(fig, "figure_2_all_team_baseline_profiles")

# -----------------------------------------------------------------------------
# Redesign frontiers with clipped outcomes and no fixed budget cap
# -----------------------------------------------------------------------------


def build_redesign_setup(data: base.DataBundle,
                         mlm_results: Dict[str, Any],
                         group_value: Any) -> Dict[str, Any]:
    team_mask = data.group_id == group_value
    if team_mask.sum() == 0:
        raise ValueError(f"No rows found for team {group_value}")

    team_x = data.x.loc[team_mask, data.all_feats].copy()
    team_fixed = team_x[data.fixed_feats].copy()
    baseline_policy = {feat: float(team_x[data.policy_feats].iloc[0][feat]) for feat in data.policy_feats}

    outcome_struct: Dict[str, Dict[str, Any]] = {}
    for out in data.outcomes:
        res = mlm_results[out]
        fe = res.fe_params.copy()
        fixed_mean_part = base.get_intercept(fe) + lookup_random_intercept(res, group_value)
        for feat in data.fixed_feats:
            fixed_mean_part += base.get_coef(fe, feat) * float(team_fixed[feat].mean())
        policy_coefs = {feat: base.get_coef(fe, feat) for feat in data.policy_feats}
        outcome_struct[out] = {
            "fixed_mean_part": float(fixed_mean_part),
            "policy_coefs": policy_coefs,
        }

    def evaluate_policy(policy: Dict[str, float]) -> Dict[str, float]:
        vals: Dict[str, float] = {}
        for out in data.outcomes:
            val = outcome_struct[out]["fixed_mean_part"]
            for feat in data.policy_feats:
                val += outcome_struct[out]["policy_coefs"][feat] * float(policy[feat])
            vals[out] = clip_outcome(val)
        cost = float(sum(base.COSTS[f] * abs(float(policy[f]) - baseline_policy[f]) for f in data.policy_feats))
        return {
            "mean_performance": vals[data.outcomes[0]],
            "mean_job_satisfaction": vals[data.outcomes[1]],
            "cost": cost,
        }

    domains: Dict[str, List[float]] = {}
    search_rows: List[Dict[str, Any]] = []
    max_possible_cost = 0.0
    for feat in data.policy_feats:
        beta_perf = outcome_struct[data.outcomes[0]]["policy_coefs"][feat]
        beta_sat = outcome_struct[data.outcomes[1]]["policy_coefs"][feat]
        direction = base.sign_category(beta_perf, beta_sat)
        current_val = baseline_policy[feat]
        observed_min = float(data.x[feat].min())
        observed_max = float(data.x[feat].max())

        if direction == "increase_only":
            low, high = current_val, observed_max
        elif direction == "decrease_only":
            low, high = observed_min, current_val
        else:
            low, high = observed_min, observed_max

        # No-effect movements are always dominated by baseline because they add cost only.
        if abs(beta_perf) <= 1e-12 and abs(beta_sat) <= 1e-12:
            domain_vals = [current_val]
            direction = "no_effect_baseline_only"
        elif base.USE_INTEGER_GRID and base.is_integer_like_series(data.x[feat]):
            domain_vals = [float(v) for v in range(int(round(low)), int(round(high)) + 1)]
        else:
            domain_vals = sorted([
                float(v) for v in data.x[feat].dropna().unique()
                if float(v) >= low - 1e-12 and float(v) <= high + 1e-12
            ])
            if current_val not in domain_vals:
                domain_vals = sorted(domain_vals + [current_val])

        if not domain_vals:
            domain_vals = [current_val]
        domains[feat] = domain_vals
        max_feature_cost = max(base.COSTS[feat] * abs(v - current_val) for v in domain_vals)
        max_possible_cost += float(max_feature_cost)
        search_rows.append({
            "team_id": as_team_label(group_value),
            "feature": feat,
            "baseline_value": current_val,
            "beta_performance": float(beta_perf),
            "beta_job_satisfaction": float(beta_sat),
            "direction_rule": direction,
            "domain_min": min(domain_vals),
            "domain_max": max(domain_vals),
            "n_domain_values": len(domain_vals),
            "domain_values": str(domain_vals),
            "cost_weight": base.COSTS[feat],
            "max_feature_cost": float(max_feature_cost),
        })

    candidate_space_upper_bound = python_int_product(len(domains[f]) for f in data.policy_feats)

    def priority(feat: str) -> Tuple[Any, ...]:
        beta_p = outcome_struct[data.outcomes[0]]["policy_coefs"][feat]
        beta_s = outcome_struct[data.outcomes[1]]["policy_coefs"][feat]
        impact = abs(beta_p) + abs(beta_s)
        max_move = max(abs(v - baseline_policy[feat]) for v in domains[feat])
        return (-impact, -base.COSTS[feat] * max_move, len(domains[feat]), feat)

    ordered_feats = sorted(data.policy_feats, key=priority)
    baseline_eval = evaluate_policy(baseline_policy)

    return {
        "team_id": group_value,
        "team_label": as_team_label(group_value),
        "baseline_policy": baseline_policy,
        "outcome_struct": outcome_struct,
        "domains": domains,
        "ordered_feats": ordered_feats,
        "search_info": pd.DataFrame(search_rows),
        "candidate_space_upper_bound": int(candidate_space_upper_bound),
        "max_possible_redesign_cost": float(max_possible_cost),
        "baseline_eval": baseline_eval,
        "evaluate_policy": evaluate_policy,
    }


def nondominated_prune_states(states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(states) <= 1:
        return states
    tmp_rows = []
    for i, st in enumerate(states):
        tmp_rows.append({
            "__state_id": i,
            "mean_performance": float(st["perf_gain"]),
            "mean_job_satisfaction": float(st["sat_gain"]),
            "cost": float(st["cost"]),
        })
    df = pd.DataFrame(tmp_rows)
    df = df.drop_duplicates(["mean_performance", "mean_job_satisfaction", "cost"], keep="first")
    nd = base.extract_global_nondominated_fast(df)
    keep_ids = nd["__state_id"].astype(int).tolist()
    return [states[i] for i in keep_ids]


def run_redesign_frontier_clipped_no_budget(data: base.DataBundle,
                                            mlm_results: Dict[str, Any],
                                            group_value: Any) -> base.FrontierResult:
    t0 = time.perf_counter()
    setup = build_redesign_setup(data, mlm_results, group_value)
    baseline = setup["baseline_policy"]
    outcome_struct = setup["outcome_struct"]
    domains = setup["domains"]
    ordered_feats = setup["ordered_feats"]
    base_perf = setup["baseline_eval"]["mean_performance"]
    base_sat = setup["baseline_eval"]["mean_job_satisfaction"]

    states: List[Dict[str, Any]] = [{"perf_gain": 0.0, "sat_gain": 0.0, "cost": 0.0, "policy": {}}]
    visited_partial_states = 0
    dominance_pruned_states = 0
    max_states_retained = 1
    level_rows: List[Dict[str, Any]] = []

    # This exact DP keeps only nondominated partial states after each feature. It does
    # not use a fixed budget cap; clipping at [1, 7] is applied at final evaluation.
    for k, feat in enumerate(ordered_feats, start=1):
        beta_perf = outcome_struct[data.outcomes[0]]["policy_coefs"][feat]
        beta_sat = outcome_struct[data.outcomes[1]]["policy_coefs"][feat]
        feat_states: List[Dict[str, Any]] = []
        current_val = baseline[feat]

        for st in states:
            for v in domains[feat]:
                visited_partial_states += 1
                delta = float(v) - current_val
                new_policy = dict(st["policy"])
                new_policy[feat] = float(v)
                feat_states.append({
                    "perf_gain": float(st["perf_gain"] + beta_perf * delta),
                    "sat_gain": float(st["sat_gain"] + beta_sat * delta),
                    "cost": float(st["cost"] + base.COSTS[feat] * abs(delta)),
                    "policy": new_policy,
                })

        before = len(feat_states)
        states = nondominated_prune_states(feat_states)
        after = len(states)
        dominance_pruned_states += max(0, before - after)
        max_states_retained = max(max_states_retained, after)
        level_rows.append({
            "team_id": setup["team_label"],
            "level": k,
            "feature": feat,
            "states_before_pruning": int(before),
            "states_after_pruning": int(after),
            "states_pruned_by_dominance": int(before - after),
        })

    # Convert retained complete states to policy rows and clip outcomes.
    rows: List[Dict[str, Any]] = []
    for idx, st in enumerate(states):
        full_policy = dict(baseline)
        full_policy.update(st["policy"])
        perf = clip_outcome(base_perf + st["perf_gain"])
        sat = clip_outcome(base_sat + st["sat_gain"])
        row: Dict[str, Any] = {
            "team_id": setup["team_label"],
            "solution_id": f"R{setup['team_label']}_S{idx + 1}",
            "mean_performance": perf,
            "mean_job_satisfaction": sat,
            "cost": float(st["cost"]),
            "delta_performance_from_baseline": float(perf - base_perf),
            "delta_job_satisfaction_from_baseline": float(sat - base_sat),
        }
        for feat in data.policy_feats:
            row[f"policy__{feat}"] = float(full_policy[feat])
            row[f"change__{feat}"] = float(full_policy[feat] - baseline[feat])
        rows.append(row)

    frontier = pd.DataFrame(rows)
    if frontier.empty:
        raise RuntimeError(f"No redesign frontier produced for team {setup['team_label']}")
    frontier = base.extract_global_nondominated_fast(frontier)
    frontier = frontier.sort_values(["cost", "mean_performance", "mean_job_satisfaction"], ascending=[True, False, False]).reset_index(drop=True)
    frontier["frontier_rank"] = np.arange(1, len(frontier) + 1)
    frontier["setting"] = "redesign"
    frontier["team_label"] = setup["team_label"]
    frontier["n_active_levers"] = 0
    for feat in data.policy_feats:
        frontier["n_active_levers"] += ((frontier[f"policy__{feat}"] - float(baseline[feat])).abs() > 1e-9).astype(int)

    # Effective cap statistics: once the clipped frontier reaches 7--7, higher cost is not substantively useful.
    cost_to_7_7 = cost_to_threshold(frontier, 7.0, 7.0)
    effective_budget_limit = cost_to_7_7 if np.isfinite(cost_to_7_7) else setup["max_possible_redesign_cost"]

    elapsed = time.perf_counter() - t0
    meta = pd.DataFrame([
        {"metric": "team_ids", "value": str([setup["team_label"]])},
        {"metric": "setting", "value": "redesign"},
        {"metric": "search_object", "value": "finite policy-lever assignments with clipped mixed-model outcomes"},
        {"metric": "candidate_space_upper_bound", "value": int(setup["candidate_space_upper_bound"])},
        {"metric": "visited_nodes", "value": int(visited_partial_states)},
        {"metric": "complete_policies_evaluated", "value": int(len(states))},
        {"metric": "unique_feasible_policies", "value": int(len(states))},
        {"metric": "global_nondominated", "value": int(len(frontier))},
        {"metric": "frontier_size", "value": int(len(frontier))},
        {"metric": "prune_infeasible", "value": 0},
        {"metric": "prune_cost", "value": 0},
        {"metric": "prune_dominated", "value": int(dominance_pruned_states)},
        {"metric": "pruned_nodes", "value": int(dominance_pruned_states)},
        {"metric": "max_states_retained", "value": int(max_states_retained)},
        {"metric": "max_possible_redesign_cost", "value": float(setup["max_possible_redesign_cost"])},
        {"metric": "effective_budget_limit_after_7_7", "value": float(effective_budget_limit)},
        {"metric": "cost_to_7_0_7_0", "value": float(cost_to_7_7) if np.isfinite(cost_to_7_7) else np.nan},
        {"metric": "runtime_seconds", "value": float(elapsed)},
    ])

    baseline_row = {
        "mean_performance": float(base_perf),
        "mean_job_satisfaction": float(base_sat),
        "cost": 0.0,
    }
    search_info = pd.concat([setup["search_info"], pd.DataFrame(level_rows)], ignore_index=True, sort=False)

    # all_solutions intentionally stores the final exact retained complete states, not an exhaustive cloud.
    return base.FrontierResult(
        setting="redesign",
        team_label=setup["team_label"],
        team_ids=[setup["team_id"]],
        all_solutions=frontier.copy(),
        pareto=frontier.copy(),
        comparison=pd.DataFrame(),
        search_info=search_info,
        meta=meta,
        baseline_policy=baseline,
        baseline_row=baseline_row,
        elapsed_seconds=elapsed,
    )

# -----------------------------------------------------------------------------
# Tables 4 and 5: per-team and aggregate summaries
# -----------------------------------------------------------------------------


def summarize_frontier_result(res: base.FrontierResult, n_employees: int) -> Dict[str, Any]:
    f = res.pareto.copy()
    visited = parse_numeric_meta(res.meta, "visited_nodes", np.nan)
    prune_infeasible = parse_numeric_meta(res.meta, "prune_infeasible", 0.0)
    prune_cost = parse_numeric_meta(res.meta, "prune_cost", 0.0)
    prune_dominated = parse_numeric_meta(res.meta, "prune_dominated", 0.0)
    pruned_nodes = prune_infeasible + prune_cost + prune_dominated

    out: Dict[str, Any] = {
        "team_id": res.team_label,
        "setting": res.setting,
        "n_employees": int(n_employees),
        "evaluated_complete_policies": parse_numeric_meta(res.meta, "complete_policies_evaluated", len(f)),
        "unique_feasible_policies": parse_numeric_meta(res.meta, "unique_feasible_policies", len(f)),
        "frontier_size": int(len(f)),
        "cost_to_6_0_6_0": cost_to_threshold(f, 6.0, 6.0),
        "cost_to_6_5_6_5": cost_to_threshold(f, 6.5, 6.5),
        "cost_to_7_0_7_0": cost_to_threshold(f, 7.0, 7.0),
        "cost_to_perf_7": cost_to_threshold(f, 7.0, None),
        "cost_to_sat_7": cost_to_threshold(f, None, 7.0),
        "candidate_upper_bound": parse_numeric_meta(res.meta, "candidate_space_upper_bound", np.nan),
        "visited_nodes": visited,
        "pruned_nodes": pruned_nodes,
        "pruned_pct": (pruned_nodes / visited) if np.isfinite(visited) and visited > 0 else np.nan,
        "frontier_max_cost": float(f["cost"].max()) if not f.empty else np.nan,
        "frontier_max_performance": float(f["mean_performance"].max()) if not f.empty else np.nan,
        "frontier_max_job_satisfaction": float(f["mean_job_satisfaction"].max()) if not f.empty else np.nan,
        "runtime_seconds": parse_numeric_meta(res.meta, "runtime_seconds", res.elapsed_seconds),
    }
    return out


def make_summary_tables(results: List[base.FrontierResult], data: base.DataBundle, setting: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    for res in results:
        # team_label may represent one observed team here.
        try:
            g_val = int(float(res.team_label))
            n_emp = int((data.group_id == g_val).sum())
            if n_emp == 0:
                n_emp = int((data.group_id.astype(str) == res.team_label).sum())
        except Exception:
            n_emp = int((data.group_id.astype(str) == res.team_label).sum())
        rows.append(summarize_frontier_result(res, n_emp))
    per_team = pd.DataFrame(rows)

    summary_cols = [
        "evaluated_complete_policies",
        "frontier_size",
        "cost_to_6_0_6_0",
        "cost_to_6_5_6_5",
        "cost_to_7_0_7_0",
        "cost_to_perf_7",
        "cost_to_sat_7",
        "candidate_upper_bound",
        "visited_nodes",
        "pruned_nodes",
        "pruned_pct",
        "frontier_max_cost",
        "frontier_max_performance",
        "frontier_max_job_satisfaction",
        "runtime_seconds",
    ]
    aggregate = summarize_for_manuscript(per_team, summary_cols)

    # Add reach counts as a compact extra row block for later text/table use.
    reach_rows = []
    n_teams = len(per_team)
    for label, col in [
        ("Teams reaching 6.0/6.0", "cost_to_6_0_6_0"),
        ("Teams reaching 6.5/6.5", "cost_to_6_5_6_5"),
        ("Teams reaching 7.0/7.0", "cost_to_7_0_7_0"),
        ("Teams reaching performance 7", "cost_to_perf_7"),
        ("Teams reaching satisfaction 7", "cost_to_sat_7"),
    ]:
        count = int(per_team[col].notna().sum()) if col in per_team.columns else 0
        reach_rows.append({"Statistic": label, "n_teams": count, "percent_teams": 100 * count / n_teams if n_teams else np.nan})
    reach = pd.DataFrame(reach_rows)
    reach.to_csv(OUT_DIR / "tables_csv" / f"table_{'1' if setting == 'redesign' else '2'}_{setting}_threshold_reach_counts.csv", index=False)
    return per_team, aggregate

# -----------------------------------------------------------------------------
# All-team frontier runners
# -----------------------------------------------------------------------------


def run_all_team_redesign(data: base.DataBundle, mlm_results: Dict[str, Any]) -> List[base.FrontierResult]:
    groups = sort_group_values(data.group_id.unique()) if RUN_ALL_TEAM_REDESIGN else FOCAL_TEAMS
    results: List[base.FrontierResult] = []
    print(f"Running all-team redesign frontiers for {len(groups)} teams...")
    for idx, g in enumerate(groups, start=1):
        label = as_team_label(g)
        print(f"  Redesign [{idx:03d}/{len(groups):03d}] team {label}", flush=True)
        res = run_redesign_frontier_clipped_no_budget(data, mlm_results, g)
        results.append(res)
        if WRITE_PER_TEAM_FRONTIERS:
            res.pareto.to_csv(OUT_DIR / "raw_frontiers" / f"redesign_frontier_team_{label}.csv", index=False)
    return results


def run_all_team_design(data: base.DataBundle, tree_model: DecisionTreeRegressor) -> List[base.FrontierResult]:
    groups = sort_group_values(data.group_id.unique()) if RUN_ALL_TEAM_DESIGN else FOCAL_TEAMS
    results: List[base.FrontierResult] = []
    print(f"Running all-team design frontiers for {len(groups)} teams...")
    for idx, g in enumerate(groups, start=1):
        label = as_team_label(g)
        print(f"  Design   [{idx:03d}/{len(groups):03d}] team {label}", flush=True)
        # base.run_design_frontier expects a sequence of team ids; passing the original group value keeps type matching.
        res = base.run_design_frontier(data, tree_model, [g])
        # Normalize labels for downstream plotting/tables.
        res.team_label = label
        res.setting = "design"
        results.append(res)
        if WRITE_PER_TEAM_FRONTIERS:
            res.pareto.to_csv(OUT_DIR / "raw_frontiers" / f"design_frontier_team_{label}.csv", index=False)
    return results

# -----------------------------------------------------------------------------
# Figure 3: focal-team frontier plots
# -----------------------------------------------------------------------------


def get_result_by_team(results: List[base.FrontierResult], team: int) -> base.FrontierResult:
    label = str(team)
    for res in results:
        if res.team_label == label:
            return res
    raise KeyError(f"Could not find result for team {team}")


def plot_frontier_axis(ax: plt.Axes, res: base.FrontierResult, title: str) -> Any:
    f = res.pareto.copy()
    sc = ax.scatter(
        f["mean_performance"],
        f["mean_job_satisfaction"],
        c=f["cost"],
        cmap=CMAP_GREEN_TO_RED,
        s=42,
        alpha=0.88,
        edgecolor="black",
        linewidth=0.25,
    )

    # Baseline/benchmark as outlined circle, not X.
    ax.scatter(
        [res.baseline_row["mean_performance"]],
        [res.baseline_row["mean_job_satisfaction"]],
        s=150,
        facecolors="none",
        edgecolors="black",
        linewidths=1.9,
        zorder=5,
    )
    ax.annotate("baseline" if res.setting == "redesign" else "benchmark",
                (res.baseline_row["mean_performance"], res.baseline_row["mean_job_satisfaction"]),
                xytext=(6, 6), textcoords="offset points", fontsize=8)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean performance")
    ax.set_ylabel("Mean job satisfaction")
    ax.set_xlim(OUTCOME_LOWER_BOUND - 0.1, OUTCOME_UPPER_BOUND + 0.1)
    ax.set_ylim(OUTCOME_LOWER_BOUND - 0.1, OUTCOME_UPPER_BOUND + 0.1)
    ax.grid(alpha=0.25)
    return sc


def plot_figure_3_focal_frontiers(redesign_results: List[base.FrontierResult],
                                  design_results: List[base.FrontierResult]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.4), sharex=True, sharey=True)
    for r, team in enumerate(FOCAL_TEAMS):
        red = get_result_by_team(redesign_results, team)
        des = get_result_by_team(design_results, team)
        sc1 = plot_frontier_axis(axes[r, 0], red, f"Team {team} — redesign frontier")
        sc2 = plot_frontier_axis(axes[r, 1], des, f"Team {team} — design frontier")
        cbar1 = fig.colorbar(sc1, ax=axes[r, 0], fraction=0.046, pad=0.035)
        cbar1.set_label("Redesign cost", fontsize=8)
        cbar2 = fig.colorbar(sc2, ax=axes[r, 1], fraction=0.046, pad=0.035)
        cbar2.set_label("Design cost", fontsize=8)
    fig.suptitle("Figure 3. Focal-team redesign and design frontiers", fontsize=14, fontweight="bold", y=1.01)
    save_fig(fig, "figure_3_focal_team_frontiers_2x2")

# -----------------------------------------------------------------------------
# Figure 5: selected-policy heatmaps
# -----------------------------------------------------------------------------


def nearest_to_threshold_row(p: pd.DataFrame, perf_thr: float, sat_thr: float) -> pd.Series:
    work = p.copy()
    # Penalize shortfalls; if all fail, pick the smallest shortfall, then lowest cost.
    short_perf = np.maximum(0.0, perf_thr - work["mean_performance"].to_numpy(dtype=float))
    short_sat = np.maximum(0.0, sat_thr - work["mean_job_satisfaction"].to_numpy(dtype=float))
    work["threshold_shortfall"] = short_perf + short_sat
    work = work.sort_values(["threshold_shortfall", "cost", "mean_performance", "mean_job_satisfaction"], ascending=[True, True, False, False])
    return work.iloc[0]


def select_min_cost_threshold_row(p: pd.DataFrame, perf_thr: float, sat_thr: float) -> Tuple[str, pd.Series]:
    ok = p[(p["mean_performance"] >= perf_thr - 1e-9) & (p["mean_job_satisfaction"] >= sat_thr - 1e-9)].copy()
    if not ok.empty:
        ok = ok.sort_values(["cost", "mean_performance", "mean_job_satisfaction"], ascending=[True, False, False])
        return f"Min cost ≥{perf_thr:g}/≥{sat_thr:g}", ok.iloc[0]
    return f"Closest to {perf_thr:g}/{sat_thr:g}", nearest_to_threshold_row(p, perf_thr, sat_thr)


def select_balanced_knee_row(p: pd.DataFrame) -> pd.Series:
    work = p.copy()
    eps = 1e-12
    perf_n = (work["mean_performance"] - work["mean_performance"].min()) / (work["mean_performance"].max() - work["mean_performance"].min() + eps)
    sat_n = (work["mean_job_satisfaction"] - work["mean_job_satisfaction"].min()) / (work["mean_job_satisfaction"].max() - work["mean_job_satisfaction"].min() + eps)
    cost_n = (work["cost"] - work["cost"].min()) / (work["cost"].max() - work["cost"].min() + eps)
    dist = np.sqrt((1 - perf_n) ** 2 + (1 - sat_n) ** 2 + cost_n ** 2)
    return work.loc[dist.idxmin()]


def build_initial_or_benchmark_row(res: base.FrontierResult, data: base.DataBundle) -> pd.Series:
    row: Dict[str, Any] = {
        "solution_id": "Initial" if res.setting == "redesign" else "Observed benchmark",
        "cost": float(res.baseline_row["cost"]),
        "mean_performance": float(res.baseline_row["mean_performance"]),
        "mean_job_satisfaction": float(res.baseline_row["mean_job_satisfaction"]),
    }
    for feat in data.policy_feats:
        row[f"policy__{feat}"] = float(res.baseline_policy[feat])
    return pd.Series(row)


def select_policy_rows_for_heatmap(res: base.FrontierResult, data: base.DataBundle) -> pd.DataFrame:
    p = res.pareto.copy().reset_index(drop=True)
    rows: List[Tuple[str, pd.Series]] = []

    rows.append(("Initial" if res.setting == "redesign" else "Observed benchmark", build_initial_or_benchmark_row(res, data)))

    min_cost = p.sort_values(["cost", "mean_performance", "mean_job_satisfaction"], ascending=[True, False, False]).iloc[0]
    rows.append(("Min cost frontier", min_cost))

    for thr in [6.0, 6.5, 7.0]:
        rows.append(select_min_cost_threshold_row(p, thr, thr))

    max_perf = p.sort_values(["mean_performance", "cost", "mean_job_satisfaction"], ascending=[False, True, False]).iloc[0]
    rows.append(("Max performance\n(lowest cost tie)", max_perf))

    max_sat = p.sort_values(["mean_job_satisfaction", "cost", "mean_performance"], ascending=[False, True, False]).iloc[0]
    rows.append(("Max satisfaction\n(lowest cost tie)", max_sat))

    rows.append(("Balanced/knee", select_balanced_knee_row(p)))

    out_rows: List[Dict[str, Any]] = []
    for label, row in rows:
        d = {"display_label": label,
             "solution_id": row.get("solution_id", ""),
             "cost": float(row["cost"]),
             "mean_performance": float(row["mean_performance"]),
             "mean_job_satisfaction": float(row["mean_job_satisfaction"])}
        for feat in data.policy_feats:
            d[f"policy__{feat}"] = float(row[f"policy__{feat}"])
        out_rows.append(d)
    return pd.DataFrame(out_rows)


def abbreviate_policy_names(policy_feats: Sequence[str]) -> List[str]:
    mapping = {
        "Prehire Tests": "Prehire\nTests",
        "Structured Interviews": "Structured\nInterviews",
        "Employee Participation": "Employee\nParticipation",
        "Complaint Procedure": "Complaint\nProcedure",
        "Group Bonuses": "Group\nBonuses",
        "Individual Bonuses": "Individual\nBonuses",
        "Performance Evaluation": "Performance\nEvaluation",
        "Goals Communication": "Goals\nCommunication",
        "Suggestion Adoption": "Suggestion\nAdoption",
        "Performance Pay": "Performance\nPay",
        "Internal Promotion": "Internal\nPromotion",
        "Job Autonomy": "Job\nAutonomy",
        "Selective Hiring": "Selective\nHiring",
        "Pay Competitiveness": "Pay\nCompetitiveness",
        "Job Training": "Job\nTraining",
    }
    return [mapping.get(f, f.replace(" ", "\n")) for f in policy_feats]


def plot_figure_4_policy_heatmaps(redesign_results: List[base.FrontierResult],
                                  design_results: List[base.FrontierResult],
                                  data: base.DataBundle) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18.0, 10.2), sharex=True)
    vmin, vmax = 1.0, 7.0
    cmap = plt.get_cmap(CMAP_GREEN_TO_RED)
    xlabels = abbreviate_policy_names(data.policy_feats)
    last_img = None

    selected_tables: List[pd.DataFrame] = []
    for r, team in enumerate(FOCAL_TEAMS):
        for c, (setting, results) in enumerate([("redesign", redesign_results), ("design", design_results)]):
            res = get_result_by_team(results, team)
            selected = select_policy_rows_for_heatmap(res, data)
            selected.insert(0, "team_id", str(team))
            selected.insert(1, "setting", setting)
            selected_tables.append(selected)

            arr = selected[[f"policy__{feat}" for feat in data.policy_feats]].to_numpy(dtype=float)
            ylabels = [
                f"{lab}\n(c={cost:.0f}, p={perf:.2f}, s={sat:.2f})"
                for lab, cost, perf, sat in zip(
                    selected["display_label"],
                    selected["cost"],
                    selected["mean_performance"],
                    selected["mean_job_satisfaction"],
                )
            ]

            ax = axes[r, c]
            last_img = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"Team {team} — {setting}", fontsize=11, fontweight="bold")
            ax.set_xticks(np.arange(len(data.policy_feats)))
            ax.set_xticklabels(xlabels, fontsize=7, rotation=45, ha="right")
            ax.set_yticks(np.arange(len(ylabels)))
            ax.set_yticklabels(ylabels, fontsize=7)
            ax.set_ylabel("Selected policy solutions", fontsize=8)
            ax.set_xticks(np.arange(-.5, len(data.policy_feats), 1), minor=True)
            ax.set_yticks(np.arange(-.5, len(ylabels), 1), minor=True)
            ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
            ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(last_img, ax=axes.ravel().tolist(), shrink=0.82, pad=0.012)
    cbar.set_label("Policy level (1–7)", fontsize=9)
    fig.suptitle("Figure 4. Policy composition of selected frontier solutions", fontsize=14, fontweight="bold", y=1.01)
    save_fig(fig, "figure_4_policy_composition")

    selected_all = pd.concat(selected_tables, ignore_index=True)
    selected_all.to_csv(OUT_DIR / "selected_policies" / "selected_policy_rows_for_figure_4.csv", index=False)

# -----------------------------------------------------------------------------
# Output writer
# -----------------------------------------------------------------------------


def write_workbook(profiles: pd.DataFrame,
                   redesign_per_team: pd.DataFrame,
                   redesign_summary: pd.DataFrame,
                   design_per_team: pd.DataFrame,
                   design_summary: pd.DataFrame,
                   redesign_results: List[base.FrontierResult],
                   design_results: List[base.FrontierResult]) -> None:
    workbook = OUT_DIR / "policy_design_manuscript_outputs.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        profiles.to_excel(writer, sheet_name="figure1_team_profiles", index=False)
        redesign_summary.to_excel(writer, sheet_name="table4_redesign_summary", index=False)
        redesign_per_team.to_excel(writer, sheet_name="table4_redesign_per_team", index=False)
        design_summary.to_excel(writer, sheet_name="table5_design_summary", index=False)
        design_per_team.to_excel(writer, sheet_name="table5_design_per_team", index=False)

        # Focal frontiers only, to keep workbook manageable.
        for team in FOCAL_TEAMS:
            red = get_result_by_team(redesign_results, team)
            des = get_result_by_team(design_results, team)
            red.pareto.to_excel(writer, sheet_name=f"frontier_R_{team}", index=False)
            des.pareto.to_excel(writer, sheet_name=f"frontier_D_{team}", index=False)
            red.meta.to_excel(writer, sheet_name=f"meta_R_{team}", index=False)
            des.meta.to_excel(writer, sheet_name=f"meta_D_{team}", index=False)

    print(f"Workbook written to: {workbook.resolve()}")

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------


def main() -> None:
    ensure_outdirs()
    np.random.seed(RN)

    print("Loading data...")
    base.DATA_PATH = DATA_PATH
    data = base.load_data()
    print(f"Rows: {len(data.df)} | Teams: {data.group_id.nunique()} | Outcomes: {data.outcomes}")

    print("\nFitting full-sample predictive models...")
    mlm_results = base.fit_mixed_models(data)
    tree_model = base.fit_design_tree(data)

    print("\nCreating Figure 2 profiles and plot...")
    profiles = make_all_team_baseline_profiles(data, mlm_results, tree_model)
    profiles.to_csv(OUT_DIR / "tables_csv" / "figure_2_all_team_baseline_profiles.csv", index=False)
    plot_figure_2_all_team_baseline_profiles(profiles)

    print("\nRunning redesign frontiers and creating Table 1...")
    redesign_results = run_all_team_redesign(data, mlm_results)
    redesign_per_team, redesign_summary = make_summary_tables(redesign_results, data, "redesign")
    redesign_per_team.to_csv(OUT_DIR / "tables_csv" / "table_1_redesign_per_team.csv", index=False)
    redesign_summary.to_csv(OUT_DIR / "tables_csv" / "table_1_redesign_summary_statistics.csv", index=False)

    print("\nRunning design frontiers and creating Table 2...")
    design_results = run_all_team_design(data, tree_model)
    design_per_team, design_summary = make_summary_tables(design_results, data, "design")
    design_per_team.to_csv(OUT_DIR / "tables_csv" / "table_2_design_per_team.csv", index=False)
    design_summary.to_csv(OUT_DIR / "tables_csv" / "table_2_design_summary_statistics.csv", index=False)

    print("\nCreating Figure 3...")
    plot_figure_3_focal_frontiers(redesign_results, design_results)

    print("\nCreating Figure 4...")
    plot_figure_4_policy_heatmaps(redesign_results, design_results, data)

    print("\nWriting combined workbook...")
    write_workbook(profiles, redesign_per_team, redesign_summary, design_per_team, design_summary, redesign_results, design_results)

    print("\nDone. Outputs are in:")
    print(f"  {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
