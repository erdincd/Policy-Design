#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions for the Policy Design manuscript-output workflow.

This module contains reusable functions for data loading, predictive modeling,
frontier generation, and output construction. It is imported by
``generate_policy_design_manuscript_outputs.py``, which is the script used to
reproduce the final manuscript outputs. The older table/figure labels below are
kept only as part of the legacy helper documentation:

Tables
------
Table 1. Empirical setup and implementation weights
Table 2. Focal team baseline summary
Table 3. Predictive model diagnostics
Table 4. Representative frontier solutions
Table 5. Computational performance of branch-and-bound
Table 6. Targeted constrained policy queries (computed by filtering generated feasible solutions)

Figures
-------
Figure 2. Redesign frontiers for focal teams
Figure 3. Design vs. redesign frontier comparison for one team
Figure 4. Policy-composition heatmap for selected frontier solutions

Expected input file
-------------------
Data - Policy Redesign.csv

Expected column order
---------------------
First 15 columns  : shared policy features
Next 9 columns    : employee characteristics
Next 2 columns    : outcomes
Last column       : Group ID

Main dependencies
-----------------
pandas, numpy, matplotlib, scipy, scikit-learn, statsmodels, openpyxl

Notes
-----
1. Redesign cost = weighted L1 distance from the observed baseline policy.
2. Design cost   = weighted implementation burden of the selected policy itself.
3. The targeted-query table does not solve new Pyomo/Gurobi OCL models. It uses the
   already-generated feasible/frontier solution sets to answer common constrained
   queries. This keeps the script portable and avoids extra solver dependencies.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================

RN = 42
np.random.seed(RN)

DATA_PATH = Path("Data - Policy Redesign.csv")
OUT_DIR = Path("policy_design_results_bundle")

P = 15  # number of shared policy features
C = 9   # number of fixed employee features
O = 2   # number of outcomes

FOCAL_TEAMS = [5100, 7270]
COMPARISON_TEAM = 5100

# Redesign search budget: weighted L1 distance from baseline policy
REDESIGN_MAX_BUDGET = 30
REDESIGN_COST_STEP = 1

# Design search budget: weighted implementation burden of selected policy
DESIGN_SEARCH_COST_MIN = 0.0
DESIGN_SEARCH_COST_MAX = 210.0
DESIGN_BUDGET_STEP = 1.0

TREE_PARAMS = dict(
    random_state=RN,
    max_depth=10,
    min_samples_leaf=3,
)

USE_INTEGER_GRID = True
USE_TEAM_RANDOM_INTERCEPT = True
STRICT_EPS = 1e-6
ROUND_DIGITS = 8

# Outcome thresholds for selected examples / targeted queries.
# Adjust these if you use a different success definition.
LB = {"Performance": 6.0, "Job Satisfaction": 6.0}
SPARSE_MAX_CHANGED_LEVERS = 3

# Save figures in both formats for manuscript and appendix workflows.
FIG_FORMATS = ["png", "pdf"]
FIG_DPI = 300

# HRM implementation weights used in the paper draft.
COSTS = {
    "Prehire Tests": 1,
    "Structured Interviews": 1,
    "Employee Participation": 2,
    "Complaint Procedure": 1,
    "Group Bonuses": 2,
    "Individual Bonuses": 3,
    "Performance Evaluation": 2,
    "Goals Communication": 1,
    "Suggestion Adoption": 2,
    "Performance Pay": 3,
    "Internal Promotion": 2,
    "Job Autonomy": 3,
    "Selective Hiring": 3,
    "Pay Competitiveness": 3,
    "Job Training": 2,
}


# =============================================================================
# BASIC HELPERS
# =============================================================================


def ensure_outdir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "tables_csv").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "raw_outputs").mkdir(parents=True, exist_ok=True)


def safe_sheet_name(name: str) -> str:
    bad = ["/", "\\", "?", "*", "[", "]", ":"]
    out = str(name)
    for ch in bad:
        out = out.replace(ch, "_")
    return out[:31]


def save_fig(fig: plt.Figure, stem: str) -> None:
    for fmt in FIG_FORMATS:
        path = OUT_DIR / "figures" / f"{stem}.{fmt}"
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def round_float(x: float, ndigits: int = ROUND_DIGITS) -> float:
    return round(float(x), ndigits)


def canonical_bound(v: float) -> Any:
    if np.isneginf(v):
        return "-inf"
    if np.isposinf(v):
        return "inf"
    return round_float(v)


def bounds_key(lb: np.ndarray, ub: np.ndarray) -> Tuple[Any, ...]:
    key_parts: List[Any] = []
    for a, b in zip(lb, ub):
        key_parts.append(canonical_bound(float(a)))
        key_parts.append(canonical_bound(float(b)))
    return tuple(key_parts)


def policy_signature_from_row(row: pd.Series, policy_feats: Sequence[str]) -> Tuple[float, ...]:
    return tuple(round_float(float(row[f"policy__{feat}"])) for feat in policy_feats)


def exact_dominates(row_a: Dict[str, float], row_b: Dict[str, float]) -> bool:
    weak = (
        row_a["mean_performance"] >= row_b["mean_performance"] and
        row_a["mean_job_satisfaction"] >= row_b["mean_job_satisfaction"] and
        row_a["cost"] <= row_b["cost"]
    )
    strict = (
        row_a["mean_performance"] > row_b["mean_performance"] or
        row_a["mean_job_satisfaction"] > row_b["mean_job_satisfaction"] or
        row_a["cost"] < row_b["cost"]
    )
    return bool(weak and strict)


class FenwickMax:
    def __init__(self, n: int):
        self.n = n
        self.tree = np.full(n + 1, -np.inf, dtype=float)

    def update(self, idx: int, value: float) -> None:
        while idx <= self.n:
            if value > self.tree[idx]:
                self.tree[idx] = value
            idx += idx & -idx

    def query(self, idx: int) -> float:
        res = -np.inf
        while idx > 0:
            if self.tree[idx] > res:
                res = self.tree[idx]
            idx -= idx & -idx
        return float(res)


def extract_global_nondominated_fast(df_in: pd.DataFrame) -> pd.DataFrame:
    """Exact global nondominated filter for max perf, max sat, min cost."""
    if df_in.empty:
        return df_in.copy()

    df_work = df_in.copy().reset_index(drop=False).rename(columns={"index": "__orig_idx"})
    df_work = df_work.sort_values(
        ["cost", "mean_performance", "mean_job_satisfaction"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    perf_values = np.sort(df_work["mean_performance"].unique())
    n_perf = len(perf_values)
    perf_rank = {v: i + 1 for i, v in enumerate(perf_values)}

    def rev_rank(perf_val: float) -> int:
        return n_perf - perf_rank[perf_val] + 1

    bit = FenwickMax(n_perf)
    keep_orig_idx: List[int] = []

    for _, g in df_work.groupby("cost", sort=False):
        g = g.sort_values(
            ["mean_performance", "mean_job_satisfaction"],
            ascending=[False, False],
        ).copy()

        unique_pairs = (
            g[["mean_performance", "mean_job_satisfaction"]]
            .drop_duplicates()
            .sort_values(["mean_performance", "mean_job_satisfaction"], ascending=[False, False])
            .reset_index(drop=True)
        )

        within_group_nd_pairs: List[Tuple[float, float]] = []
        running_best_sat = -np.inf

        for _, pair_row in unique_pairs.iterrows():
            perf = float(pair_row["mean_performance"])
            sat = float(pair_row["mean_job_satisfaction"])
            if sat > running_best_sat:
                within_group_nd_pairs.append((perf, sat))
                running_best_sat = sat

        global_nd_pairs: List[Tuple[float, float]] = []
        for perf, sat in within_group_nd_pairs:
            idx = rev_rank(perf)
            cheaper_best_sat = bit.query(idx)
            if cheaper_best_sat < sat:
                global_nd_pairs.append((perf, sat))

        if global_nd_pairs:
            global_pair_set = set(global_nd_pairs)
            mask_keep = g.apply(
                lambda r: (
                    float(r["mean_performance"]),
                    float(r["mean_job_satisfaction"]),
                ) in global_pair_set,
                axis=1,
            )
            keep_orig_idx.extend(g.loc[mask_keep, "__orig_idx"].tolist())
            for perf, sat in global_nd_pairs:
                bit.update(rev_rank(perf), sat)

    out = df_in.loc[sorted(set(keep_orig_idx))].copy().reset_index(drop=True)
    out = out.sort_values(
        ["cost", "mean_performance", "mean_job_satisfaction"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    return out


def get_intercept(fe_params: pd.Series) -> float:
    for idx in fe_params.index:
        idx_l = str(idx).strip().lower()
        if idx_l in ["intercept", "const"]:
            return float(fe_params[idx])
    for idx in fe_params.index:
        if "intercept" in str(idx).lower():
            return float(fe_params[idx])
    raise ValueError("Could not detect intercept in fe_params.")


def get_coef(fe_params: pd.Series, feat: str) -> float:
    candidates = [feat, f'Q("{feat}")', f"Q('{feat}')"]
    for c in candidates:
        if c in fe_params.index:
            return float(fe_params[c])

    feat_norm = feat.replace(" ", "").replace("_", "").replace('"', "").replace("'", "").lower()
    for idx in fe_params.index:
        idx_norm = str(idx).replace(" ", "").replace("_", "").replace('"', "").replace("'", "").lower()
        if feat_norm == idx_norm or feat_norm in idx_norm:
            return float(fe_params[idx])
    return 0.0


def get_random_intercept(result: Any, team_id_value: int) -> float:
    if not USE_TEAM_RANDOM_INTERCEPT:
        return 0.0
    try:
        re = result.random_effects
    except Exception:
        return 0.0

    candidate_keys = [team_id_value, str(team_id_value), float(team_id_value)]
    found_key = None
    for k in candidate_keys:
        if k in re:
            found_key = k
            break
    if found_key is None:
        return 0.0

    val = re[found_key]
    if isinstance(val, pd.Series):
        return float(val.iloc[0])
    if isinstance(val, dict):
        return float(list(val.values())[0])
    arr = np.asarray(val).ravel()
    return float(arr[0]) if len(arr) > 0 else 0.0


def sign_category(beta_perf: float, beta_sat: float, tol: float = 1e-12) -> str:
    s1 = 0 if abs(beta_perf) <= tol else (1 if beta_perf > 0 else -1)
    s2 = 0 if abs(beta_sat) <= tol else (1 if beta_sat > 0 else -1)
    if s1 >= 0 and s2 >= 0 and (s1 > 0 or s2 > 0):
        return "increase_only"
    if s1 <= 0 and s2 <= 0 and (s1 < 0 or s2 < 0):
        return "decrease_only"
    return "both_directions"


def is_integer_like_series(s: pd.Series, tol: float = 1e-9) -> bool:
    vals = s.dropna().astype(float).values
    if len(vals) == 0:
        return False
    return bool(np.all(np.abs(vals - np.round(vals)) <= tol))


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def success_count_from_employee_preds(pred_df: pd.DataFrame, outcomes: Sequence[str]) -> int:
    mask = np.ones(len(pred_df), dtype=bool)
    for out in outcomes:
        threshold = LB.get(out, -np.inf)
        mask &= pred_df[out].to_numpy(dtype=float) >= threshold
    return int(mask.sum())


# =============================================================================
# DATA AND MODEL FITTING
# =============================================================================


@dataclass
class DataBundle:
    df_raw: pd.DataFrame
    group_id: pd.Series
    df: pd.DataFrame
    x: pd.DataFrame
    y: pd.DataFrame
    outcomes: List[str]
    all_feats: List[str]
    policy_feats: List[str]
    fixed_feats: List[str]
    cost_vector: np.ndarray


def load_data() -> DataBundle:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Put this script in the same folder as the data file "
            "or edit DATA_PATH at the top of the script."
        )
    df_raw = pd.read_csv(DATA_PATH)
    if df_raw.shape[1] <= O:
        raise ValueError("Dataset must include a trailing Group ID column.")

    group_id = df_raw.iloc[:, -1].copy()
    df = df_raw.iloc[:, :-1].copy()
    outcomes = df.columns[-O:].tolist()
    all_feats = df.columns[:-O].tolist()
    policy_feats = all_feats[:P]
    fixed_feats = all_feats[P:P + C]

    missing_costs = [f for f in policy_feats if f not in COSTS]
    if missing_costs:
        raise ValueError(f"COSTS dictionary is missing policy features: {missing_costs}")

    x = df[all_feats].copy()
    y = df[outcomes].copy()
    cost_vector = np.array([float(COSTS[f]) for f in policy_feats], dtype=float)

    return DataBundle(df_raw, group_id, df, x, y, outcomes, all_feats, policy_feats, fixed_feats, cost_vector)


def fit_mixed_models(data: DataBundle) -> Dict[str, Any]:
    train_df = data.x.copy()
    for out in data.outcomes:
        train_df[out] = data.y[out].values
    train_df["GroupID"] = data.group_id.values

    mlm_results: Dict[str, Any] = {}
    for out in data.outcomes:
        formula = f'Q("{out}") ~ ' + " + ".join([f'Q("{c}")' for c in data.all_feats])
        model = smf.mixedlm(formula=formula, data=train_df, groups=train_df["GroupID"])
        try:
            result = model.fit(reml=False, method="lbfgs", maxiter=1000, disp=False)
        except Exception:
            result = model.fit(reml=False, method="nm", maxiter=1000, disp=False)
        mlm_results[out] = result
    return mlm_results


def fit_design_tree(data: DataBundle) -> DecisionTreeRegressor:
    tree_model = DecisionTreeRegressor(**TREE_PARAMS)
    tree_model.fit(data.x[data.all_feats], data.y[data.outcomes])
    return tree_model


def make_predictive_diagnostics(data: DataBundle,
                                mlm_results: Dict[str, Any],
                                tree_model: DecisionTreeRegressor) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    # Mixed model in-sample diagnostics with fitted values.
    for out in data.outcomes:
        y_true = data.y[out].to_numpy(dtype=float)
        y_pred = np.asarray(mlm_results[out].fittedvalues, dtype=float)
        m = metric_values(y_true, y_pred)
        rows.append({
            "setting": "Redesign",
            "predictive_model": "Random-intercept mixed model",
            "evaluation": "in-sample fitted values",
            "outcome": out,
            **m,
            "notes": "Includes estimated team random intercepts for observed groups.",
        })

    # Multi-output CART held-out diagnostics.
    X_train, X_test, y_train, y_test = train_test_split(
        data.x[data.all_feats], data.y[data.outcomes], test_size=0.20, random_state=RN
    )
    tree_eval = DecisionTreeRegressor(**TREE_PARAMS)
    tree_eval.fit(X_train, y_train)
    pred_test = tree_eval.predict(X_test)

    for j, out in enumerate(data.outcomes):
        y_true = y_test[out].to_numpy(dtype=float)
        y_pred = pred_test[:, j]
        m = metric_values(y_true, y_pred)
        rows.append({
            "setting": "Design",
            "predictive_model": "Multi-output decision tree",
            "evaluation": "80/20 held-out split",
            "outcome": out,
            **m,
            "notes": f"Tree params: {TREE_PARAMS}",
        })

    # Also add full-sample tree metadata.
    rows.append({
        "setting": "Design",
        "predictive_model": "Multi-output decision tree",
        "evaluation": "full-sample fitted model metadata",
        "outcome": "all outcomes",
        "RMSE": np.nan,
        "MAE": np.nan,
        "R2": np.nan,
        "notes": f"depth={tree_model.get_depth()}, leaves={tree_model.get_n_leaves()}",
    })

    return pd.DataFrame(rows)


# =============================================================================
# TABLE 1 AND TABLE 2
# =============================================================================


def make_empirical_setup_table(data: DataBundle) -> pd.DataFrame:
    cost_groups: Dict[int, List[str]] = {}
    for feat in data.policy_feats:
        cost_groups.setdefault(int(COSTS[feat]), []).append(feat)

    rows = [
        {
            "component": "Shared policy levers",
            "variables": "; ".join(data.policy_feats),
            "role_in_model": "Decision variables / shared team-level policy",
            "notes": "Same policy vector applies to all employees in the focal team.",
        },
        {
            "component": "Employee characteristics",
            "variables": "; ".join(data.fixed_feats),
            "role_in_model": "Fixed employee-level inputs",
            "notes": "Capture heterogeneity in responses to shared policies.",
        },
        {
            "component": "Outcomes",
            "variables": "; ".join(data.outcomes),
            "role_in_model": "Predicted outcomes / objective dimensions",
            "notes": "Reported as team-level mean predictions in frontier plots.",
        },
    ]
    for w in sorted(cost_groups):
        rows.append({
            "component": f"Implementation-cost weight {w}",
            "variables": "; ".join(cost_groups[w]),
            "role_in_model": "Policy-cost function",
            "notes": "Theory-informed implementation weight; not treated as a validated cost scale.",
        })
    return pd.DataFrame(rows)


def predict_mlm_team_means(data: DataBundle,
                           mlm_results: Dict[str, Any],
                           team_id: int,
                           policy: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    team_mask = data.group_id == team_id
    if team_mask.sum() == 0:
        raise ValueError(f"No rows found for TEAM_ID={team_id}")

    team_df = data.x.loc[team_mask].copy()
    team_fixed_df = team_df[data.fixed_feats].copy()
    if policy is None:
        policy = team_df[data.policy_feats].iloc[0].to_dict()

    out_means: Dict[str, float] = {}
    for out in data.outcomes:
        res = mlm_results[out]
        fe_params = res.fe_params.copy()
        val = get_intercept(fe_params) + get_random_intercept(res, team_id)
        for feat in data.fixed_feats:
            val += get_coef(fe_params, feat) * float(team_fixed_df[feat].mean())
        for feat in data.policy_feats:
            val += get_coef(fe_params, feat) * float(policy[feat])
        out_means[out] = float(val)
    return out_means


def make_focal_team_summary(data: DataBundle,
                            mlm_results: Dict[str, Any],
                            tree_model: DecisionTreeRegressor,
                            team_ids: Sequence[int]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for team_id in team_ids:
        team_mask = data.group_id == team_id
        if team_mask.sum() == 0:
            raise ValueError(f"No rows found for TEAM_ID={team_id}")
        team_x = data.x.loc[team_mask].copy()
        team_y = data.y.loc[team_mask].copy()
        baseline_policy_arr = team_x[data.policy_feats].iloc[0].to_numpy(dtype=float)
        baseline_policy = dict(zip(data.policy_feats, baseline_policy_arr))
        baseline_design_cost = float(np.dot(data.cost_vector, baseline_policy_arr))

        mlm_means = predict_mlm_team_means(data, mlm_results, team_id, baseline_policy)
        tree_pred = tree_model.predict(team_x[data.all_feats])

        row = {
            "team_id": team_id,
            "n_employees": int(team_mask.sum()),
            "observed_mean_performance": float(team_y[data.outcomes[0]].mean()),
            "observed_mean_job_satisfaction": float(team_y[data.outcomes[1]].mean()),
            "redesign_pred_baseline_performance": float(mlm_means[data.outcomes[0]]),
            "redesign_pred_baseline_job_satisfaction": float(mlm_means[data.outcomes[1]]),
            "design_pred_benchmark_performance": float(np.mean(tree_pred[:, 0])),
            "design_pred_benchmark_job_satisfaction": float(np.mean(tree_pred[:, 1])),
            "baseline_policy_implementation_cost": baseline_design_cost,
            "redesign_baseline_distance_cost": 0.0,
        }
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# REDESIGN FRONTIER
# =============================================================================


@dataclass
class FrontierResult:
    setting: str
    team_label: str
    team_ids: List[int]
    all_solutions: pd.DataFrame
    pareto: pd.DataFrame
    comparison: pd.DataFrame
    search_info: pd.DataFrame
    meta: pd.DataFrame
    baseline_policy: Dict[str, float]
    baseline_row: Dict[str, float]
    elapsed_seconds: float


def run_redesign_frontier(data: DataBundle,
                          mlm_results: Dict[str, Any],
                          team_id: int) -> FrontierResult:
    """
    Fast exact frontier generation for redesign.

    The original script uses branch-and-bound. For producing paper figures/tables quickly,
    this function uses an equivalent finite-domain multi-objective dynamic program that
    keeps only nondominated partial states after each policy lever. This is typically much
    faster for the current 15-lever, budget-limited redesign case.

    Important: the grey background cloud in the redesign figures is a sampled feasible
    cloud plus the exact frontier, not an exhaustive enumeration of all feasible policies.
    The exact non-dominated frontier is still computed from the DP states.
    """
    t0 = time.perf_counter()
    max_budget_steps = int(round(REDESIGN_MAX_BUDGET / REDESIGN_COST_STEP))

    team_mask = data.group_id == team_id
    if team_mask.sum() == 0:
        raise ValueError(f"No rows found for TEAM_ID={team_id}")
    team_df = data.x.loc[team_mask].copy()
    team_fixed_df = team_df[data.fixed_feats].copy()
    baseline_policy = team_df[data.policy_feats].iloc[0].to_dict()

    # Extract team-specific mean prediction function.
    outcome_struct: Dict[str, Dict[str, Any]] = {}
    for out in data.outcomes:
        res = mlm_results[out]
        fe_params = res.fe_params.copy()
        fixed_mean_part = get_intercept(fe_params) + get_random_intercept(res, team_id)
        for feat in data.fixed_feats:
            fixed_mean_part += get_coef(fe_params, feat) * float(team_fixed_df[feat].mean())
        policy_coefs = {feat: get_coef(fe_params, feat) for feat in data.policy_feats}
        outcome_struct[out] = {
            "fixed_mean_part": fixed_mean_part,
            "policy_coefs": policy_coefs,
        }

    def cost_of_policy(policy: Dict[str, float]) -> float:
        return float(sum(COSTS[f] * abs(float(policy[f]) - float(baseline_policy[f])) for f in data.policy_feats))

    def evaluate_policy(policy: Dict[str, float]) -> Dict[str, float]:
        row: Dict[str, float] = {}
        for out in data.outcomes:
            val = outcome_struct[out]["fixed_mean_part"]
            for feat in data.policy_feats:
                val += outcome_struct[out]["policy_coefs"][feat] * float(policy[feat])
            if out == data.outcomes[0]:
                row["mean_performance"] = float(val)
            else:
                row["mean_job_satisfaction"] = float(val)
        row["cost"] = cost_of_policy(policy)
        return row

    # Domains using sign-based direction rules.
    search_info_rows: List[Dict[str, Any]] = []
    domains: Dict[str, List[float]] = {}
    for feat in data.policy_feats:
        beta_perf = outcome_struct[data.outcomes[0]]["policy_coefs"][feat]
        beta_sat = outcome_struct[data.outcomes[1]]["policy_coefs"][feat]
        direction = sign_category(beta_perf, beta_sat)
        current_val = float(baseline_policy[feat])
        observed_min = float(data.x[feat].min())
        observed_max = float(data.x[feat].max())

        if direction == "increase_only":
            low, high = current_val, observed_max
        elif direction == "decrease_only":
            low, high = observed_min, current_val
        else:
            low, high = observed_min, observed_max

        if abs(beta_perf) <= 1e-12 and abs(beta_sat) <= 1e-12:
            domain_vals = [current_val]
            direction = "no_effect_baseline_only"
        elif USE_INTEGER_GRID and is_integer_like_series(data.x[feat]):
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
        search_info_rows.append({
            "feature": feat,
            "beta_performance": beta_perf,
            "beta_job_satisfaction": beta_sat,
            "direction_rule": direction,
            "baseline_value": current_val,
            "allowed_min": low,
            "allowed_max": high,
            "n_domain_values": len(domain_vals),
            "domain_values": str(domain_vals),
            "cost_weight": COSTS[feat],
        })
    search_info = pd.DataFrame(search_info_rows)

    def feature_priority(feat: str) -> Tuple[Any, ...]:
        beta_p = outcome_struct[data.outcomes[0]]["policy_coefs"][feat]
        beta_s = outcome_struct[data.outcomes[1]]["policy_coefs"][feat]
        max_move = max(abs(float(v) - float(baseline_policy[feat])) for v in domains[feat])
        max_cost = COSTS[feat] * max_move
        impact = abs(beta_p) + abs(beta_s)
        return (-impact, -max_cost, len(domains[feat]), feat)

    ordered_feats = sorted(data.policy_feats, key=feature_priority)
    baseline_outcomes = evaluate_policy(baseline_policy)
    baseline_perf = baseline_outcomes["mean_performance"]
    baseline_sat = baseline_outcomes["mean_job_satisfaction"]

    # Feature options in the ordered search space.
    feature_options: Dict[str, List[Dict[str, float]]] = {}
    for feat in ordered_feats:
        base_val = float(baseline_policy[feat])
        beta_perf = outcome_struct[data.outcomes[0]]["policy_coefs"][feat]
        beta_sat = outcome_struct[data.outcomes[1]]["policy_coefs"][feat]
        opts: List[Dict[str, float]] = []
        for v in domains[feat]:
            delta = float(v) - base_val
            add_cost = COSTS[feat] * abs(delta)
            add_steps = int(round(add_cost / REDESIGN_COST_STEP))
            opts.append({
                "value": float(v),
                "add_cost": float(add_cost),
                "add_steps": int(add_steps),
                "perf_gain": float(beta_perf * delta),
                "sat_gain": float(beta_sat * delta),
            })
        # Prefer efficient moves first; baseline remains available.
        opts = sorted(
            opts,
            key=lambda d: (1e18 if abs(d["add_cost"]) <= 1e-12 else (d["perf_gain"] + d["sat_gain"]) / d["add_cost"]),
            reverse=True,
        )
        feature_options[feat] = opts

    def nd_filter_states(states: List[Tuple[int, float, float, Tuple[float, ...]]]) -> List[Tuple[int, float, float, Tuple[float, ...]]]:
        if not states:
            return []
        df_states = pd.DataFrame(states, columns=["cost", "perf_gain", "sat_gain", "policy_tuple"])
        # Remove numerically duplicate outcome states while keeping one policy.
        df_states["perf_round"] = df_states["perf_gain"].round(12)
        df_states["sat_round"] = df_states["sat_gain"].round(12)
        df_states = df_states.drop_duplicates(["cost", "perf_round", "sat_round"])
        tmp = df_states.rename(columns={"perf_gain": "mean_performance", "sat_gain": "mean_job_satisfaction"})
        nd = extract_global_nondominated_fast(tmp[["cost", "mean_performance", "mean_job_satisfaction", "policy_tuple"]])
        nd = nd.rename(columns={"mean_performance": "perf_gain", "mean_job_satisfaction": "sat_gain"})
        return list(nd[["cost", "perf_gain", "sat_gain", "policy_tuple"]].itertuples(index=False, name=None))

    # Exact DP frontier over the finite redesign domain under budget.
    states: List[Tuple[int, float, float, Tuple[float, ...]]] = [(0, 0.0, 0.0, tuple())]
    transitions_considered = 0
    transitions_budget_feasible = 0
    dominated_pruned_total = 0
    max_states_retained = 1

    for feat in ordered_feats:
        new_states: List[Tuple[int, float, float, Tuple[float, ...]]] = []
        for c0, p0, s0, pol in states:
            for opt in feature_options[feat]:
                transitions_considered += 1
                c_new = int(c0 + opt["add_steps"])
                if c_new <= max_budget_steps:
                    transitions_budget_feasible += 1
                    new_states.append((
                        c_new,
                        p0 + float(opt["perf_gain"]),
                        s0 + float(opt["sat_gain"]),
                        pol + (float(opt["value"]),),
                    ))
        before = len(new_states)
        states = nd_filter_states(new_states)
        dominated_pruned_total += max(0, before - len(states))
        max_states_retained = max(max_states_retained, len(states))

    # Convert DP states to exact frontier dataframe.
    frontier_rows: List[Dict[str, Any]] = []
    for cost_steps, perf_gain, sat_gain, policy_tuple in states:
        policy_by_order = dict(zip(ordered_feats, policy_tuple))
        row = {
            "cost": float(cost_steps * REDESIGN_COST_STEP),
            "mean_performance": float(baseline_perf + perf_gain),
            "mean_job_satisfaction": float(baseline_sat + sat_gain),
        }
        for feat in data.policy_feats:
            row[f"policy__{feat}"] = float(policy_by_order[feat])
        frontier_rows.append(row)

    pareto = pd.DataFrame(frontier_rows)
    pareto = extract_global_nondominated_fast(pareto)
    pareto = pareto.sort_values(["cost", "mean_performance", "mean_job_satisfaction"], ascending=[True, False, False]).reset_index(drop=True)
    pareto["solution_id"] = [f"R{team_id}_S{i+1}" for i in range(len(pareto))]
    pareto["setting"] = "redesign"
    pareto["team_label"] = str(team_id)

    # Add active lever count to frontier.
    pareto["n_active_levers"] = 0
    for feat in data.policy_feats:
        pareto["n_active_levers"] += ((pareto[f"policy__{feat}"] - float(baseline_policy[feat])).abs() > 1e-9).astype(int)

    # Sample a feasible cloud for background plotting. This is not used to define the exact frontier.
    rng = np.random.default_rng(RN + int(team_id))
    sample_rows: List[Dict[str, Any]] = []
    n_samples = 20000
    for _ in range(n_samples):
        remaining = max_budget_steps
        policy: Dict[str, float] = {}
        for feat in ordered_feats:
            feasible_opts = [opt for opt in feature_options[feat] if int(opt["add_steps"]) <= remaining]
            if not feasible_opts:
                feasible_opts = [min(feature_options[feat], key=lambda opt: int(opt["add_steps"]))]
            opt = feasible_opts[int(rng.integers(0, len(feasible_opts)))]
            policy[feat] = float(opt["value"])
            remaining -= int(opt["add_steps"])
        row = evaluate_policy(policy)
        for feat in data.policy_feats:
            row[f"policy__{feat}"] = float(policy[feat])
        sample_rows.append(row)

    all_df = pd.concat([pd.DataFrame(sample_rows), pareto.drop(columns=["solution_id", "setting", "team_label", "n_active_levers"], errors="ignore")], ignore_index=True)
    policy_cols = [f"policy__{feat}" for feat in data.policy_feats]
    all_df = all_df.drop_duplicates(subset=policy_cols).reset_index(drop=True)
    all_df["n_active_levers"] = 0
    for feat in data.policy_feats:
        all_df["n_active_levers"] += ((all_df[f"policy__{feat}"] - float(baseline_policy[feat])).abs() > 1e-9).astype(int)

    comparison_rows = data.policy_feats + data.outcomes + ["cost", "n_active_levers"]
    comparison = pd.DataFrame(index=comparison_rows)
    comparison["baseline"] = np.nan
    for feat in data.policy_feats:
        comparison.loc[feat, "baseline"] = baseline_policy[feat]
    comparison.loc[data.outcomes[0], "baseline"] = baseline_perf
    comparison.loc[data.outcomes[1], "baseline"] = baseline_sat
    comparison.loc["cost", "baseline"] = 0.0
    comparison.loc["n_active_levers", "baseline"] = 0

    for _, row in pareto.iterrows():
        cname = row["solution_id"]
        for feat in data.policy_feats:
            comparison.loc[feat, cname] = row[f"policy__{feat}"]
        comparison.loc[data.outcomes[0], cname] = row["mean_performance"]
        comparison.loc[data.outcomes[1], cname] = row["mean_job_satisfaction"]
        comparison.loc["cost", cname] = row["cost"]
        comparison.loc["n_active_levers", cname] = row["n_active_levers"]
    comparison = comparison.reset_index().rename(columns={"index": "item"})

    elapsed = time.perf_counter() - t0
    candidate_space_ub = int(np.prod([max(1, len(domains[f])) for f in data.policy_feats]))
    meta = pd.DataFrame([
        {"metric": "team_ids", "value": str([team_id])},
        {"metric": "setting", "value": "redesign"},
        {"metric": "search_object", "value": "finite policy-lever assignments (fast multi-objective DP engine)"},
        {"metric": "candidate_space_upper_bound", "value": candidate_space_ub},
        {"metric": "visited_nodes", "value": transitions_considered},
        {"metric": "complete_policies_evaluated", "value": len(pareto)},
        {"metric": "unique_feasible_policies", "value": len(all_df)},
        {"metric": "global_nondominated", "value": len(pareto)},
        {"metric": "prune_cost", "value": transitions_considered - transitions_budget_feasible},
        {"metric": "prune_infeasible", "value": 0},
        {"metric": "prune_dominated", "value": dominated_pruned_total},
        {"metric": "max_states_retained", "value": max_states_retained},
        {"metric": "background_cloud_sample_size", "value": n_samples},
        {"metric": "runtime_seconds", "value": elapsed},
    ])

    baseline_row = {
        "mean_performance": baseline_perf,
        "mean_job_satisfaction": baseline_sat,
        "cost": 0.0,
    }
    return FrontierResult("redesign", str(team_id), [team_id], all_df, pareto, comparison, search_info, meta, baseline_policy, baseline_row, elapsed)

# =============================================================================
# DESIGN FRONTIER
# =============================================================================


def run_design_frontier(data: DataBundle,
                        tree_model: DecisionTreeRegressor,
                        team_ids: Sequence[int]) -> FrontierResult:
    t0 = time.perf_counter()
    team_ids = list(team_ids)
    team_label = "_".join(str(t) for t in team_ids)

    team_mask = data.group_id.isin(team_ids)
    if team_mask.sum() == 0:
        raise ValueError(f"No rows found for TEAM_IDS={team_ids}")

    team_df = data.x.loc[team_mask].copy()
    team_fixed_df = team_df[data.fixed_feats].copy()
    workers = team_fixed_df.index.tolist()
    n_workers = len(workers)

    y_pred_team = tree_model.predict(team_df[data.all_feats])
    baseline_policy_arr = team_df[data.policy_feats].iloc[0].to_numpy(dtype=float)
    baseline_policy = {feat: float(baseline_policy_arr[i]) for i, feat in enumerate(data.policy_feats)}
    baseline_cost = float(np.dot(data.cost_vector, baseline_policy_arr))
    baseline_mean_perf = float(np.mean(y_pred_team[:, 0]))
    baseline_mean_sat = float(np.mean(y_pred_team[:, 1]))

    # Global convex hull of observed policies.
    policy_unique_df = data.x[data.policy_feats].drop_duplicates().reset_index(drop=True)
    policy_unique_matrix = policy_unique_df[data.policy_feats].to_numpy(dtype=float)
    n_hull_points = policy_unique_matrix.shape[0]

    # Extract leaf path restrictions.
    tree_ = tree_model.tree_
    feature_name_by_index = {i: data.all_feats[i] for i in range(len(data.all_feats))}

    def leaf_pred_vector(node_id: int) -> np.ndarray:
        arr = np.asarray(tree_.value[node_id]).reshape(-1)
        if len(arr) < O:
            raise ValueError("Unexpected tree leaf value shape.")
        return arr[:O].astype(float)

    leaf_infos: List[Dict[str, Any]] = []

    def dfs_collect(node_id: int,
                    lower_bounds: Dict[str, float],
                    upper_bounds: Dict[str, float]) -> None:
        left = tree_.children_left[node_id]
        right = tree_.children_right[node_id]
        if left == -1 and right == -1:
            pred_vec = leaf_pred_vector(node_id)
            leaf_infos.append({
                "node_id": int(node_id),
                "lower_bounds": lower_bounds.copy(),
                "upper_bounds": upper_bounds.copy(),
                "pred_perf": float(pred_vec[0]),
                "pred_sat": float(pred_vec[1]),
            })
            return
        feat_idx = int(tree_.feature[node_id])
        thr = float(tree_.threshold[node_id])
        feat = feature_name_by_index[feat_idx]

        ub_left = upper_bounds.copy()
        ub_left[feat] = min(ub_left.get(feat, np.inf), thr)
        dfs_collect(left, lower_bounds.copy(), ub_left)

        lb_right = lower_bounds.copy()
        lb_right[feat] = max(lb_right.get(feat, -np.inf), thr)
        dfs_collect(right, lb_right, upper_bounds.copy())

    dfs_collect(0, {}, {})

    # Worker-specific reachable leaves.
    worker_fixed_values = {
        w: {feat: float(team_fixed_df.loc[w, feat]) for feat in data.fixed_feats}
        for w in workers
    }
    worker_leaf_options: Dict[int, List[Dict[str, Any]]] = {}
    search_info_rows: List[Dict[str, Any]] = []

    for w in workers:
        fixed_vals = worker_fixed_values[w]
        feasible_leaf_list: List[Dict[str, Any]] = []
        for leaf in leaf_infos:
            lower_bounds = leaf["lower_bounds"]
            upper_bounds = leaf["upper_bounds"]

            fixed_ok = True
            for feat in data.fixed_feats:
                v = fixed_vals[feat]
                lb = lower_bounds.get(feat, -np.inf)
                ub = upper_bounds.get(feat, np.inf)
                if not (v > lb + STRICT_EPS and v <= ub + STRICT_EPS):
                    fixed_ok = False
                    break
            if not fixed_ok:
                continue

            lb_arr = np.full(P, -np.inf, dtype=float)
            ub_arr = np.full(P, np.inf, dtype=float)
            for j, feat in enumerate(data.policy_feats):
                lb_arr[j] = float(lower_bounds.get(feat, -np.inf))
                ub_arr[j] = float(upper_bounds.get(feat, np.inf))
            feasible_leaf_list.append({
                "worker": int(w),
                "leaf_node_id": int(leaf["node_id"]),
                "lb": lb_arr,
                "ub": ub_arr,
                "pred_perf": float(leaf["pred_perf"]),
                "pred_sat": float(leaf["pred_sat"]),
            })

        if not feasible_leaf_list:
            raise ValueError(f"Worker {w} has no reachable leaves.")
        worker_leaf_options[w] = feasible_leaf_list
        search_info_rows.append({
            "worker_id": int(w),
            "n_reachable_leaves": len(feasible_leaf_list),
            "leaf_node_ids": str([leaf["leaf_node_id"] for leaf in feasible_leaf_list]),
        })

    search_info = pd.DataFrame(search_info_rows)
    ordered_workers = sorted(workers, key=lambda w: (len(worker_leaf_options[w]), w))

    milp_calls = 0
    milp_infeasible = 0

    def integerize_interval(lb: np.ndarray, ub: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        int_lb = np.empty(P, dtype=float)
        int_ub = np.empty(P, dtype=float)
        for j in range(P):
            lower = 1
            upper = 7
            if np.isfinite(lb[j]):
                # Left/right convention follows the original script:
                # lower path condition is strict, upper path condition is weak.
                lower = max(lower, int(np.floor(float(lb[j])) + 1))
            if np.isfinite(ub[j]):
                upper = min(upper, int(np.floor(float(ub[j]))))
            int_lb[j] = float(lower)
            int_ub[j] = float(upper)
            if lower > upper:
                return int_lb, int_ub, False
        return int_lb, int_ub, True

    @lru_cache(maxsize=None)
    def solve_interval_milp_cached(key: Tuple[Any, ...]) -> Dict[str, Any]:
        nonlocal milp_calls, milp_infeasible
        milp_calls += 1

        raw_lb = np.empty(P, dtype=float)
        raw_ub = np.empty(P, dtype=float)
        for j in range(P):
            a = key[2 * j]
            b = key[2 * j + 1]
            raw_lb[j] = -np.inf if a == "-inf" else float(a)
            raw_ub[j] = np.inf if b == "inf" else float(b)

        int_lb, int_ub, ok = integerize_interval(raw_lb, raw_ub)
        if not ok:
            milp_infeasible += 1
            return {"success": False, "min_cost": np.nan, "policy": None, "support_size": 0}

        n_vars = n_hull_points + P
        c_obj = np.concatenate([np.zeros(n_hull_points, dtype=float), data.cost_vector.copy()])
        integrality = np.concatenate([np.zeros(n_hull_points, dtype=int), np.ones(P, dtype=int)])
        lb_vars = np.concatenate([np.zeros(n_hull_points, dtype=float), int_lb.astype(float)])
        ub_vars = np.concatenate([np.full(n_hull_points, np.inf, dtype=float), int_ub.astype(float)])
        bounds = Bounds(lb_vars, ub_vars)

        Aeq = np.zeros((P + 1, n_vars), dtype=float)
        beq = np.zeros(P + 1, dtype=float)
        Aeq[0, :n_hull_points] = 1.0
        beq[0] = 1.0
        for j in range(P):
            Aeq[j + 1, :n_hull_points] = policy_unique_matrix[:, j].astype(float)
            Aeq[j + 1, n_hull_points + j] = -1.0
        constraints = [LinearConstraint(Aeq, beq, beq)]

        res = milp(
            c=c_obj,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options={"disp": False},
        )
        if (not res.success) or (res.x is None):
            milp_infeasible += 1
            return {"success": False, "min_cost": np.nan, "policy": None, "support_size": 0}

        sol = np.asarray(res.x, dtype=float)
        lambdas = sol[:n_hull_points]
        policy = sol[n_hull_points:n_hull_points + P]
        policy = np.rint(policy).astype(float)
        support_size = int(np.sum(lambdas > 1e-9))
        return {
            "success": True,
            "min_cost": float(np.dot(data.cost_vector, policy)),
            "policy": policy.astype(float),
            "support_size": support_size,
        }

    def evaluate_policy_exact(policy_arr: np.ndarray) -> Dict[str, float]:
        X_team = team_fixed_df.copy()
        for j, feat in enumerate(data.policy_feats):
            X_team[feat] = float(policy_arr[j])
        X_team = X_team[data.all_feats]
        preds = tree_model.predict(X_team)
        return {
            "mean_performance": float(np.mean(preds[:, 0])),
            "mean_job_satisfaction": float(np.mean(preds[:, 1])),
            "cost": float(np.dot(data.cost_vector, policy_arr)),
        }

    all_rows: List[Dict[str, Any]] = []
    global_frontier_for_pruning: List[Dict[str, float]] = []
    node_count = 0
    complete_patterns = 0
    prune_interval = 0
    prune_hull = 0
    prune_cost = 0
    prune_bound = 0

    def add_to_global_frontier(candidate: Dict[str, float]) -> bool:
        nonlocal global_frontier_for_pruning
        for f in global_frontier_for_pruning:
            if exact_dominates(f, candidate):
                return False
        global_frontier_for_pruning = [f for f in global_frontier_for_pruning if not exact_dominates(candidate, f)]
        global_frontier_for_pruning.append(candidate)
        return True

    def leaf_intersects_interval(leaf: Dict[str, Any], lb: np.ndarray, ub: np.ndarray) -> bool:
        new_lb = np.maximum(lb, leaf["lb"])
        new_ub = np.minimum(ub, leaf["ub"])
        return bool(np.all(new_lb + STRICT_EPS <= new_ub + STRICT_EPS))

    def optimistic_candidate(i: int,
                             sum_perf: float,
                             sum_sat: float,
                             lb: np.ndarray,
                             ub: np.ndarray,
                             min_cost_lb: float) -> Tuple[bool, Dict[str, float]]:
        max_perf_total = sum_perf
        max_sat_total = sum_sat
        for t in range(i, n_workers):
            w = ordered_workers[t]
            feasible_leaves = [leaf for leaf in worker_leaf_options[w] if leaf_intersects_interval(leaf, lb, ub)]
            if not feasible_leaves:
                return False, {}
            max_perf_total += max(float(leaf["pred_perf"]) for leaf in feasible_leaves)
            max_sat_total += max(float(leaf["pred_sat"]) for leaf in feasible_leaves)
        return True, {
            "mean_performance": max_perf_total / n_workers,
            "mean_job_satisfaction": max_sat_total / n_workers,
            "cost": float(min_cost_lb),
        }

    def recurse(i: int,
                current_lb: np.ndarray,
                current_ub: np.ndarray,
                sum_perf: float,
                sum_sat: float,
                chosen_leaf_nodes: List[int]) -> None:
        nonlocal node_count, complete_patterns, prune_interval, prune_hull, prune_cost, prune_bound
        node_count += 1

        key = bounds_key(current_lb, current_ub)
        milp_sol = solve_interval_milp_cached(key)
        if not milp_sol["success"]:
            prune_hull += 1
            return
        min_cost_lb = float(milp_sol["min_cost"])
        if min_cost_lb > DESIGN_SEARCH_COST_MAX + 1e-12:
            prune_cost += 1
            return

        optimistic_ok, opt_row = optimistic_candidate(i, sum_perf, sum_sat, current_lb, current_ub, min_cost_lb)
        if not optimistic_ok:
            prune_interval += 1
            return
        for f in global_frontier_for_pruning:
            if exact_dominates(f, opt_row):
                prune_bound += 1
                return

        if i == n_workers:
            complete_patterns += 1
            policy_arr = np.asarray(milp_sol["policy"], dtype=float)
            exact_row = evaluate_policy_exact(policy_arr)
            if exact_row["cost"] < DESIGN_SEARCH_COST_MIN - 1e-12 or exact_row["cost"] > DESIGN_SEARCH_COST_MAX + 1e-12:
                return

            row = {
                "pattern_id": f"P{complete_patterns}",
                "worker_leaf_nodes": "|".join(str(v) for v in chosen_leaf_nodes),
                "cost": float(exact_row["cost"]),
                "mean_performance": float(exact_row["mean_performance"]),
                "mean_job_satisfaction": float(exact_row["mean_job_satisfaction"]),
                "support_size": int(milp_sol["support_size"]),
            }
            for j, feat in enumerate(data.policy_feats):
                row[f"policy__{feat}"] = float(policy_arr[j])
            all_rows.append(row)
            add_to_global_frontier({
                "mean_performance": row["mean_performance"],
                "mean_job_satisfaction": row["mean_job_satisfaction"],
                "cost": row["cost"],
            })
            return

        w = ordered_workers[i]
        leaf_options = sorted(
            worker_leaf_options[w],
            key=lambda leaf: (-(leaf["pred_perf"] + leaf["pred_sat"]), leaf["leaf_node_id"]),
        )
        for leaf in leaf_options:
            new_lb = np.maximum(current_lb, leaf["lb"])
            new_ub = np.minimum(current_ub, leaf["ub"])
            if np.any(new_lb + STRICT_EPS > new_ub + STRICT_EPS):
                prune_interval += 1
                continue
            recurse(
                i + 1,
                new_lb,
                new_ub,
                sum_perf + float(leaf["pred_perf"]),
                sum_sat + float(leaf["pred_sat"]),
                chosen_leaf_nodes + [int(leaf["leaf_node_id"])],
            )

    recurse(
        i=0,
        current_lb=np.full(P, -np.inf, dtype=float),
        current_ub=np.full(P, np.inf, dtype=float),
        sum_perf=0.0,
        sum_sat=0.0,
        chosen_leaf_nodes=[],
    )

    solutions = pd.DataFrame(all_rows)
    if solutions.empty:
        raise ValueError(f"No feasible design leaf patterns produced a hull-feasible solution for teams {team_ids}.")

    solutions["policy_signature"] = solutions.apply(lambda row: policy_signature_from_row(row, data.policy_feats), axis=1)
    solutions = solutions.sort_values(
        ["cost", "mean_performance", "mean_job_satisfaction"],
        ascending=[True, False, False],
    ).drop_duplicates(subset=["policy_signature"], keep="first").reset_index(drop=True)
    solutions = solutions.drop(columns=["policy_signature"])

    pareto = extract_global_nondominated_fast(solutions)
    pareto = pareto.sort_values(["cost", "mean_performance", "mean_job_satisfaction"], ascending=[True, False, False]).reset_index(drop=True)
    pareto["solution_id"] = [f"D{team_label}_S{i+1}" for i in range(len(pareto))]
    pareto["setting"] = "design"
    pareto["team_label"] = team_label
    pareto["n_active_levers"] = np.nan

    initial_row = {
        "pattern_id": "observed benchmark",
        "worker_leaf_nodes": "observed_initial_policy",
        "cost": float(baseline_cost),
        "mean_performance": float(baseline_mean_perf),
        "mean_job_satisfaction": float(baseline_mean_sat),
        "support_size": np.nan,
    }
    for j, feat in enumerate(data.policy_feats):
        initial_row[f"policy__{feat}"] = float(baseline_policy_arr[j])

    comparison_rows = data.policy_feats + data.outcomes + ["cost"]
    comparison = pd.DataFrame(index=comparison_rows)
    comparison["observed benchmark"] = np.nan
    for feat in data.policy_feats:
        comparison.loc[feat, "observed benchmark"] = baseline_policy[feat]
    comparison.loc[data.outcomes[0], "observed benchmark"] = baseline_mean_perf
    comparison.loc[data.outcomes[1], "observed benchmark"] = baseline_mean_sat
    comparison.loc["cost", "observed benchmark"] = baseline_cost
    for _, row in pareto.iterrows():
        cname = row["solution_id"]
        for feat in data.policy_feats:
            comparison.loc[feat, cname] = row[f"policy__{feat}"]
        comparison.loc[data.outcomes[0], cname] = row["mean_performance"]
        comparison.loc[data.outcomes[1], cname] = row["mean_job_satisfaction"]
        comparison.loc["cost", cname] = row["cost"]
    comparison = comparison.reset_index().rename(columns={"index": "item"})

    elapsed = time.perf_counter() - t0
    candidate_space_ub = int(np.prod([max(1, len(worker_leaf_options[w])) for w in workers]))
    meta = pd.DataFrame([
        {"metric": "team_ids", "value": str(team_ids)},
        {"metric": "setting", "value": "design"},
        {"metric": "search_object", "value": "employee-specific reachable leaf combinations"},
        {"metric": "candidate_space_upper_bound", "value": candidate_space_ub},
        {"metric": "n_workers", "value": n_workers},
        {"metric": "n_observed_unique_policies_in_hull", "value": n_hull_points},
        {"metric": "tree_depth", "value": tree_model.get_depth()},
        {"metric": "tree_leaves", "value": tree_model.get_n_leaves()},
        {"metric": "visited_nodes", "value": node_count},
        {"metric": "complete_policies_evaluated", "value": complete_patterns},
        {"metric": "unique_feasible_policies", "value": len(solutions)},
        {"metric": "global_nondominated", "value": len(pareto)},
        {"metric": "prune_interval", "value": prune_interval},
        {"metric": "prune_infeasible", "value": prune_hull},
        {"metric": "prune_cost", "value": prune_cost},
        {"metric": "prune_dominated", "value": prune_bound},
        {"metric": "interval_milp_calls", "value": milp_calls},
        {"metric": "interval_milp_infeasible", "value": milp_infeasible},
        {"metric": "runtime_seconds", "value": elapsed},
    ])

    baseline_row = {
        "mean_performance": baseline_mean_perf,
        "mean_job_satisfaction": baseline_mean_sat,
        "cost": baseline_cost,
    }
    solutions_with_initial = pd.concat([pd.DataFrame([initial_row]), solutions], axis=0, ignore_index=True)
    return FrontierResult("design", team_label, team_ids, solutions_with_initial, pareto, comparison, search_info, meta, baseline_policy, baseline_row, elapsed)


# =============================================================================
# SELECTED SOLUTIONS, TARGETED QUERIES, AND PLOTS
# =============================================================================


def changed_levers_text(row: pd.Series,
                        baseline_policy: Dict[str, float],
                        policy_feats: Sequence[str],
                        max_items: int = 6) -> str:
    changes: List[Tuple[str, float]] = []
    for feat in policy_feats:
        val = float(row[f"policy__{feat}"])
        base = float(baseline_policy.get(feat, np.nan))
        if np.isfinite(base):
            delta = val - base
            if abs(delta) > 1e-9:
                changes.append((feat, delta))
    changes = sorted(changes, key=lambda x: abs(x[1]), reverse=True)
    if not changes:
        return "No change"
    txt = "; ".join([f"{f} ({d:+.0f})" if abs(d - round(d)) < 1e-9 else f"{f} ({d:+.2f})" for f, d in changes[:max_items]])
    if len(changes) > max_items:
        txt += f"; +{len(changes) - max_items} more"
    return txt


def active_levers_text(row: pd.Series,
                       policy_feats: Sequence[str],
                       max_items: int = 6) -> str:
    pairs = [(feat, float(row[f"policy__{feat}"])) for feat in policy_feats]
    # For design, show the strongest/highest-level levers first.
    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    txt = "; ".join([f"{f}={v:.0f}" if abs(v - round(v)) < 1e-9 else f"{f}={v:.2f}" for f, v in pairs[:max_items]])
    if len(pairs) > max_items:
        txt += f"; +{len(pairs) - max_items} more"
    return txt


def select_representative_solutions(pareto: pd.DataFrame,
                                    setting: str,
                                    max_rows: int = 5) -> pd.DataFrame:
    if pareto.empty:
        return pareto.copy()

    p = pareto.copy().reset_index(drop=True)
    chosen: Dict[str, int] = {}

    chosen["minimum cost"] = int(p["cost"].idxmin())
    chosen["maximum performance"] = int(p["mean_performance"].idxmax())
    chosen["maximum job satisfaction"] = int(p["mean_job_satisfaction"].idxmax())

    # Balanced/knee: nearest to the ideal point after min-max normalization.
    eps = 1e-12
    perf_n = (p["mean_performance"] - p["mean_performance"].min()) / (p["mean_performance"].max() - p["mean_performance"].min() + eps)
    sat_n = (p["mean_job_satisfaction"] - p["mean_job_satisfaction"].min()) / (p["mean_job_satisfaction"].max() - p["mean_job_satisfaction"].min() + eps)
    cost_n = (p["cost"] - p["cost"].min()) / (p["cost"].max() - p["cost"].min() + eps)
    dist_to_ideal = np.sqrt((1 - perf_n) ** 2 + (1 - sat_n) ** 2 + cost_n ** 2)
    chosen["balanced/knee"] = int(dist_to_ideal.idxmin())

    # Highest combined outcome, irrespective of cost.
    combined = perf_n + sat_n
    chosen["highest combined outcomes"] = int(combined.idxmax())

    rows: List[pd.Series] = []
    seen: set = set()
    for role, idx in chosen.items():
        if idx in seen:
            continue
        seen.add(idx)
        r = p.loc[idx].copy()
        r["representative_role"] = role
        rows.append(r)
        if len(rows) >= max_rows:
            break

    out = pd.DataFrame(rows).reset_index(drop=True)
    out["setting"] = setting
    return out


def make_representative_table(data: DataBundle,
                              results: Sequence[FrontierResult]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for res in results:
        rep = select_representative_solutions(res.pareto, res.setting)
        for _, row in rep.iterrows():
            if res.setting == "redesign":
                lever_summary = changed_levers_text(row, res.baseline_policy, data.policy_feats)
            else:
                lever_summary = active_levers_text(row, data.policy_feats)
            rows.append({
                "setting": res.setting,
                "team_label": res.team_label,
                "solution_id": row["solution_id"],
                "representative_role": row["representative_role"],
                "cost": float(row["cost"]),
                "mean_performance": float(row["mean_performance"]),
                "mean_job_satisfaction": float(row["mean_job_satisfaction"]),
                "n_active_levers": row.get("n_active_levers", np.nan),
                "policy_summary": lever_summary,
            })
    return pd.DataFrame(rows)


def make_computational_table(results: Sequence[FrontierResult]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for res in results:
        meta = res.meta.set_index("metric")["value"].to_dict()
        rows.append({
            "team_label": res.team_label,
            "setting": res.setting,
            "search_object": meta.get("search_object"),
            "candidate_space_upper_bound": meta.get("candidate_space_upper_bound"),
            "visited_nodes": meta.get("visited_nodes"),
            "pruned_infeasible": meta.get("prune_infeasible", 0),
            "pruned_by_cost": meta.get("prune_cost", 0),
            "pruned_by_dominance": meta.get("prune_dominated", 0),
            "complete_policies_evaluated": meta.get("complete_policies_evaluated"),
            "unique_feasible_policies": meta.get("unique_feasible_policies"),
            "frontier_size": meta.get("global_nondominated"),
            "runtime_seconds": meta.get("runtime_seconds"),
        })
    return pd.DataFrame(rows)


def make_targeted_query_table(data: DataBundle,
                              redesign_results: Sequence[FrontierResult]) -> pd.DataFrame:
    """
    Targeted queries derived from generated feasible redesign solutions.

    These are not independent Pyomo/Gurobi solves; they are post-hoc constrained queries
    over the generated feasible set. This is useful for paper tables and easy to audit.
    """
    rows: List[Dict[str, Any]] = []
    for res in redesign_results:
        df_all = res.all_solutions.copy()
        if df_all.empty:
            continue
        baseline_perf = res.baseline_row["mean_performance"]
        baseline_sat = res.baseline_row["mean_job_satisfaction"]

        # Ensure n_active_levers exists.
        if "n_active_levers" not in df_all.columns:
            df_all["n_active_levers"] = 0
            for feat in data.policy_feats:
                df_all["n_active_levers"] += ((df_all[f"policy__{feat}"] - float(res.baseline_policy[feat])).abs() > 1e-9).astype(int)

        queries = []

        # Q1: minimum-cost attainment.
        q1 = df_all[
            (df_all["mean_performance"] >= LB.get(data.outcomes[0], -np.inf)) &
            (df_all["mean_job_satisfaction"] >= LB.get(data.outcomes[1], -np.inf))
        ].copy()
        if not q1.empty:
            idx = q1.sort_values(["cost", "mean_performance", "mean_job_satisfaction"], ascending=[True, False, False]).index[0]
            queries.append(("minimum-cost attainment", f"{data.outcomes[0]} >= {LB.get(data.outcomes[0])}, {data.outcomes[1]} >= {LB.get(data.outcomes[1])}", "minimize cost", df_all.loc[idx]))
        else:
            queries.append(("minimum-cost attainment", f"{data.outcomes[0]} >= {LB.get(data.outcomes[0])}, {data.outcomes[1]} >= {LB.get(data.outcomes[1])}", "minimize cost", None))

        # Q2: do-no-harm, maximize balanced mean improvement.
        q2 = df_all[
            (df_all["mean_performance"] >= baseline_perf - 1e-12) &
            (df_all["mean_job_satisfaction"] >= baseline_sat - 1e-12)
        ].copy()
        if not q2.empty:
            q2["balanced_gain"] = (q2["mean_performance"] - baseline_perf) + (q2["mean_job_satisfaction"] - baseline_sat)
            idx = q2.sort_values(["balanced_gain", "cost"], ascending=[False, True]).index[0]
            queries.append(("do-no-harm balanced redesign", "no decrease in either predicted outcome relative to baseline", "maximize summed outcome gain", df_all.loc[idx]))
        else:
            queries.append(("do-no-harm balanced redesign", "no decrease in either predicted outcome relative to baseline", "maximize summed outcome gain", None))

        # Q3: sparse redesign, at most S changed levers.
        q3 = df_all[df_all["n_active_levers"] <= SPARSE_MAX_CHANGED_LEVERS].copy()
        if not q3.empty:
            q3["combined_outcome"] = q3["mean_performance"] + q3["mean_job_satisfaction"]
            idx = q3.sort_values(["combined_outcome", "cost"], ascending=[False, True]).index[0]
            queries.append(("sparse redesign", f"at most {SPARSE_MAX_CHANGED_LEVERS} changed levers", "maximize summed predicted outcomes", df_all.loc[idx]))
        else:
            queries.append(("sparse redesign", f"at most {SPARSE_MAX_CHANGED_LEVERS} changed levers", "maximize summed predicted outcomes", None))

        for q_name, constraint, objective, row in queries:
            if row is None:
                rows.append({
                    "team_label": res.team_label,
                    "query": q_name,
                    "main_constraint": constraint,
                    "objective": objective,
                    "feasible": False,
                    "cost": np.nan,
                    "mean_performance": np.nan,
                    "mean_job_satisfaction": np.nan,
                    "n_active_levers": np.nan,
                    "policy_summary": "No feasible solution found in generated feasible set.",
                })
            else:
                rows.append({
                    "team_label": res.team_label,
                    "query": q_name,
                    "main_constraint": constraint,
                    "objective": objective,
                    "feasible": True,
                    "cost": float(row["cost"]),
                    "mean_performance": float(row["mean_performance"]),
                    "mean_job_satisfaction": float(row["mean_job_satisfaction"]),
                    "n_active_levers": int(row["n_active_levers"]),
                    "policy_summary": changed_levers_text(row, res.baseline_policy, data.policy_feats),
                })
    return pd.DataFrame(rows)


def plot_frontier_panel(ax: plt.Axes,
                        res: FrontierResult,
                        title: str,
                        show_baseline: bool = True,
                        annotate_representatives: bool = True) -> Any:
    all_df = res.all_solutions.copy()
    pareto = res.pareto.copy()

    ax.scatter(
        all_df["mean_performance"],
        all_df["mean_job_satisfaction"],
        s=20,
        alpha=0.18,
        color="lightgray",
        label="Feasible policies",
    )
    sc = ax.scatter(
        pareto["mean_performance"],
        pareto["mean_job_satisfaction"],
        c=pareto["cost"],
        s=70,
        alpha=0.95,
        edgecolors="black",
        linewidths=0.6,
        label="Non-dominated policies",
    )
    if show_baseline:
        ax.scatter(
            res.baseline_row["mean_performance"],
            res.baseline_row["mean_job_satisfaction"],
            s=130,
            marker="X",
            edgecolors="black",
            linewidths=1.0,
            label="Baseline/benchmark",
        )
    if annotate_representatives:
        rep = select_representative_solutions(pareto, res.setting, max_rows=5)
        for _, row in rep.iterrows():
            ax.annotate(
                str(row["representative_role"]),
                (row["mean_performance"], row["mean_job_satisfaction"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
    ax.set_title(title)
    ax.set_xlabel("Mean predicted performance")
    ax.set_ylabel("Mean predicted job satisfaction")
    ax.grid(True, alpha=0.3)
    return sc


def make_frontier_figures(results_by_key: Dict[Tuple[str, str], FrontierResult]) -> None:
    # Figure 2: Redesign frontiers for focal teams.
    redesign_results = [results_by_key[("redesign", str(t))] for t in FOCAL_TEAMS]
    fig, axes = plt.subplots(1, len(redesign_results), figsize=(6.2 * len(redesign_results), 5.2), sharex=False, sharey=False)
    if len(redesign_results) == 1:
        axes = [axes]
    last_sc = None
    for ax, res in zip(axes, redesign_results):
        last_sc = plot_frontier_panel(ax, res, f"Redesign frontier — Team {res.team_label}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    if last_sc is not None:
        cb = fig.colorbar(last_sc, ax=axes, shrink=0.85)
        cb.set_label("Redesign cost: weighted distance from baseline")
    fig.suptitle("Figure 2. Redesign frontiers for existing teams", y=1.02)
    save_fig(fig, "figure_2_redesign_frontiers")

    # Figure 3: Design vs redesign for comparison team.
    red = results_by_key[("redesign", str(COMPARISON_TEAM))]
    des = results_by_key[("design", str(COMPARISON_TEAM))]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), sharex=False, sharey=False)
    sc1 = plot_frontier_panel(axes[0], red, f"A. Redesign — Team {COMPARISON_TEAM}")
    sc2 = plot_frontier_panel(axes[1], des, f"B. Design — Team {COMPARISON_TEAM}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    cb1 = fig.colorbar(sc1, ax=axes[0], shrink=0.85)
    cb1.set_label("Redesign cost")
    cb2 = fig.colorbar(sc2, ax=axes[1], shrink=0.85)
    cb2.set_label("Design implementation cost")
    fig.suptitle("Figure 3. Design and redesign frontiers answer different questions", y=1.02)
    save_fig(fig, "figure_3_design_vs_redesign")


def make_policy_heatmap(data: DataBundle,
                        red: FrontierResult,
                        des: FrontierResult) -> None:
    red_rep = select_representative_solutions(red.pareto, "redesign", max_rows=5)
    des_rep = select_representative_solutions(des.pareto, "design", max_rows=5)

    red_mat = []
    red_labels = []
    for _, row in red_rep.iterrows():
        red_labels.append(f"{row['representative_role']}\n{row['solution_id']}")
        red_mat.append([float(row[f"policy__{feat}"]) - float(red.baseline_policy[feat]) for feat in data.policy_feats])

    des_mat = []
    des_labels = []
    for _, row in des_rep.iterrows():
        des_labels.append(f"{row['representative_role']}\n{row['solution_id']}")
        des_mat.append([float(row[f"policy__{feat}"]) for feat in data.policy_feats])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharex=False, sharey=False)

    # Redesign heatmap: changes from baseline.
    red_mat_arr = np.asarray(red_mat, dtype=float)
    vmax = max(1.0, float(np.nanmax(np.abs(red_mat_arr))) if red_mat_arr.size else 1.0)
    im0 = axes[0].imshow(red_mat_arr, aspect="auto", vmin=-vmax, vmax=vmax, cmap="coolwarm")
    axes[0].set_title(f"A. Redesign: change from baseline — Team {red.team_label}")
    axes[0].set_yticks(np.arange(len(red_labels)))
    axes[0].set_yticklabels(red_labels, fontsize=8)
    axes[0].set_xticks(np.arange(len(data.policy_feats)))
    axes[0].set_xticklabels(data.policy_feats, rotation=60, ha="right", fontsize=8)
    cb0 = fig.colorbar(im0, ax=axes[0], shrink=0.8)
    cb0.set_label("Policy-level change")

    # Design heatmap: selected policy levels.
    des_mat_arr = np.asarray(des_mat, dtype=float)
    im1 = axes[1].imshow(des_mat_arr, aspect="auto", vmin=1, vmax=7, cmap="viridis")
    axes[1].set_title(f"B. Design: selected policy levels — Team {des.team_label}")
    axes[1].set_yticks(np.arange(len(des_labels)))
    axes[1].set_yticklabels(des_labels, fontsize=8)
    axes[1].set_xticks(np.arange(len(data.policy_feats)))
    axes[1].set_xticklabels(data.policy_feats, rotation=60, ha="right", fontsize=8)
    cb1 = fig.colorbar(im1, ax=axes[1], shrink=0.8)
    cb1.set_label("Policy level")

    fig.suptitle("Figure 4. Policy composition across selected frontier solutions", y=1.02)
    fig.tight_layout()
    save_fig(fig, "figure_4_policy_composition_heatmap")


# =============================================================================
# EXPORTS
# =============================================================================


def write_outputs(data: DataBundle,
                  table_1: pd.DataFrame,
                  table_2: pd.DataFrame,
                  table_3: pd.DataFrame,
                  table_4: pd.DataFrame,
                  table_5: pd.DataFrame,
                  table_6: pd.DataFrame,
                  results: Sequence[FrontierResult]) -> None:
    tables = {
        "table_1_empirical_setup": table_1,
        "table_2_focal_team_summary": table_2,
        "table_3_predictive_diagnostics": table_3,
        "table_4_representative_frontier": table_4,
        "table_5_computational_performance": table_5,
        "table_6_targeted_queries": table_6,
    }

    for name, df in tables.items():
        df.to_csv(OUT_DIR / "tables_csv" / f"{name}.csv", index=False)

    workbook_path = OUT_DIR / "policy_design_results_tables_and_raw_outputs.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for name, df in tables.items():
            df.to_excel(writer, sheet_name=safe_sheet_name(name), index=False)

        for res in results:
            prefix = f"{res.setting}_{res.team_label}"
            res.pareto.to_excel(writer, sheet_name=safe_sheet_name(f"{prefix}_pareto"), index=False)
            res.all_solutions.to_excel(writer, sheet_name=safe_sheet_name(f"{prefix}_all"), index=False)
            res.comparison.to_excel(writer, sheet_name=safe_sheet_name(f"{prefix}_comparison"), index=False)
            res.search_info.to_excel(writer, sheet_name=safe_sheet_name(f"{prefix}_search"), index=False)
            res.meta.to_excel(writer, sheet_name=safe_sheet_name(f"{prefix}_meta"), index=False)

    # Also save raw outputs separately as CSV for easier debugging.
    raw_dir = OUT_DIR / "raw_outputs"
    for res in results:
        prefix = f"{res.setting}_{res.team_label}"
        res.pareto.to_csv(raw_dir / f"{prefix}_global_nondominated.csv", index=False)
        res.all_solutions.to_csv(raw_dir / f"{prefix}_all_solutions.csv", index=False)
        res.comparison.to_csv(raw_dir / f"{prefix}_comparison.csv", index=False)
        res.search_info.to_csv(raw_dir / f"{prefix}_search_info.csv", index=False)
        res.meta.to_csv(raw_dir / f"{prefix}_meta.csv", index=False)

    print("\n" + "=" * 80)
    print("OUTPUTS WRITTEN")
    print("=" * 80)
    print(f"Workbook : {workbook_path.resolve()}")
    print(f"CSV dir  : {(OUT_DIR / 'tables_csv').resolve()}")
    print(f"Figures  : {(OUT_DIR / 'figures').resolve()}")
    print(f"Raw data : {raw_dir.resolve()}")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    ensure_outdir()
    print("Loading data...")
    data = load_data()
    print(f"Rows: {len(data.df)} | Groups: {data.group_id.nunique()} | Outcomes: {data.outcomes}")

    print("\nFitting predictive models...")
    mlm_results = fit_mixed_models(data)
    tree_model = fit_design_tree(data)

    print("\nBuilding setup and diagnostic tables...")
    table_1 = make_empirical_setup_table(data)
    table_2 = make_focal_team_summary(data, mlm_results, tree_model, FOCAL_TEAMS)
    table_3 = make_predictive_diagnostics(data, mlm_results, tree_model)

    results: List[FrontierResult] = []
    results_by_key: Dict[Tuple[str, str], FrontierResult] = {}

    print("\nRunning redesign frontiers...")
    for team_id in FOCAL_TEAMS:
        print(f"  Redesign Team {team_id}...")
        res = run_redesign_frontier(data, mlm_results, team_id)
        results.append(res)
        results_by_key[("redesign", str(team_id))] = res
        print(f"    frontier={len(res.pareto)} | feasible={len(res.all_solutions)} | time={res.elapsed_seconds:.2f}s")

    print("\nRunning design frontiers...")
    for team_id in FOCAL_TEAMS:
        print(f"  Design Team {team_id}...")
        res = run_design_frontier(data, tree_model, [team_id])
        results.append(res)
        results_by_key[("design", str(team_id))] = res
        print(f"    frontier={len(res.pareto)} | feasible={len(res.all_solutions)} | time={res.elapsed_seconds:.2f}s")

    print("\nBuilding result tables...")
    table_4 = make_representative_table(data, results)
    table_5 = make_computational_table(results)
    table_6 = make_targeted_query_table(data, [r for r in results if r.setting == "redesign"])

    print("\nCreating figures...")
    make_frontier_figures(results_by_key)
    make_policy_heatmap(
        data,
        red=results_by_key[("redesign", str(COMPARISON_TEAM))],
        des=results_by_key[("design", str(COMPARISON_TEAM))],
    )

    write_outputs(data, table_1, table_2, table_3, table_4, table_5, table_6, results)


if __name__ == "__main__":
    raise SystemExit(
        "This file is a helper module. Run generate_policy_design_manuscript_outputs.py "
        "to reproduce the manuscript outputs."
    )
