from __future__ import annotations

import numpy as np
import pandas as pd

from credit_scoring.profit.cutoff_explore import optimize_cutoffs
from credit_scoring.profit.cutoff_explore import evaluate_cutoffs


ECONOMICS = {
    "ins": {"lgd": 0.45, "apr_annual": 0.01, "provision": 0.0},
    "css": {"lgd": 0.55, "apr_annual": 0.18, "provision": 0.0},
}


def _synthetic_scored_pnl() -> pd.DataFrame:
    # Minimal columns required by apply_strategy + evaluate_strategy.
    # Profit/income/el are mocked but consistent: profit = income - el.
    rows = [
        # CSS loans
        {
            "aid": "a1",
            "product": "css",
            "period": "198001",
            "pd": 0.10,
            "pr": np.nan,
            "cross_pd": np.nan,
            "act_cus_active": 1,
            "agr12_Max_CMaxA_Due": 0.0,
            "default12": 0,
            "profit": 100.0,
            "income": 100.0,
            "el": 0.0,
        },
        {
            "aid": "a2",
            "product": "css",
            "period": "198001",
            "pd": 0.30,
            "pr": np.nan,
            "cross_pd": np.nan,
            "act_cus_active": 1,
            "agr12_Max_CMaxA_Due": 0.0,
            "default12": 1,
            "profit": -50.0,
            "income": 0.0,
            "el": 50.0,
        },
        {
            "aid": "a3",
            "product": "css",
            "period": "198001",
            "pd": 0.70,
            "pr": np.nan,
            "cross_pd": np.nan,
            "act_cus_active": 1,
            "agr12_Max_CMaxA_Due": 0.0,
            "default12": 0,
            "profit": 30.0,
            "income": 30.0,
            "el": 0.0,
        },
        # INS loans (mid-band rows should react to pr_min and cross_pd_max)
        {
            "aid": "a4",
            "product": "ins",
            "period": "198001",
            "pd": 0.005,
            "pr": 0.02,
            "cross_pd": 0.10,
            "act_cus_active": 1,
            "agr12_Max_CMaxA_Due": 0.0,
            "default12": 0,
            "profit": 80.0,
            "income": 80.0,
            "el": 0.0,
        },
        {
            "aid": "a5",
            "product": "ins",
            "period": "198001",
            "pd": 0.015,
            "pr": 0.05,
            "cross_pd": 0.20,
            "act_cus_active": 1,
            "agr12_Max_CMaxA_Due": 0.0,
            "default12": 1,
            "profit": -40.0,
            "income": 0.0,
            "el": 40.0,
        },
        {
            "aid": "a6",
            "product": "ins",
            "period": "198001",
            "pd": 0.030,
            "pr": 0.001,
            "cross_pd": 0.50,
            "act_cus_active": 1,
            "agr12_Max_CMaxA_Due": 0.0,
            "default12": 0,
            "profit": 60.0,
            "income": 60.0,
            "el": 0.0,
        },
    ]
    df = pd.DataFrame(rows)
    # Ensure dtype stability for optimizer.
    df["pd"] = df["pd"].astype(float)
    df["pr"] = df["pr"].astype(float)
    df["cross_pd"] = df["cross_pd"].astype(float)
    return df


def test_optimize_cutoffs_returns_feasible_best():
    scored_pnl = _synthetic_scored_pnl()

    out = optimize_cutoffs(
        scored_pnl,
        window_start="198001",
        window_end="198001",
        burn_in_before="197501",
        economics=ECONOMICS,
        bad_customer={"enabled": False},
        pd_ins_low=0.01,
        constraints={"min_n_accept": 1},
        grid_sizes={"pd_css": 3, "pd_ins_high": 3, "pr_min": 3, "cross_pd_max": 3},
        top_n=5,
        near_opt_rel_tol=0.01,
    )

    assert out["best_is_feasible"] is True
    assert out["feasible_count"] > 0
    assert out["top_policies"], "Expected at least one feasible policy in top_policies"

    best_cuts = out["best_cutoffs"]
    ev = evaluate_cutoffs(
        scored_pnl,
        best_cuts,
        window_start="198001",
        window_end="198001",
        burn_in_before="197501",
        economics=ECONOMICS,
        bad_customer={"enabled": False},
    )

    assert ev["midband_empty"] is False
    assert int(ev["n_accept"]) >= 1
    assert float(ev["total_profit"]) == float(out["best_metrics"]["total_profit"])

    top0 = out["top_policies"][0]
    assert float(top0["total_profit"]) == float(out["best_metrics"]["total_profit"])

