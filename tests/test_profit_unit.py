"""Unit tests for credit_scoring.profit helpers."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from credit_scoring.profit.pnl import compute_pnl_table, installment_amount, loan_pnl
from credit_scoring.profit.rules import apply_strategy, compare_strategies, rules_from_params


ECONOMICS = {
    "ins": {"lgd": 0.45, "apr_annual": 0.01, "provision": 0.0},
    "css": {"lgd": 0.55, "apr_annual": 0.18, "provision": 0.0},
}

BASE_RULES = {
    "window_start": "197501",
    "window_end": "198712",
    "burn_in_before": "197501",
    "economics": ECONOMICS,
    "cutoffs": {"pd_css": 0.30, "pd_ins_high": 0.10},
    "bad_customer": {
        "enabled": True,
        "feature": "agr12_Max_CMaxA_Due",
        "threshold": 3,
    },
}


def test_installment_amount_annuity():
    inst = installment_amount(5000, 12, 0.18 / 12)
    assert inst > 5000 / 12


def test_loan_pnl_good_and_bad_css():
    good = loan_pnl(5000, 12, "css", default12=0, economics=ECONOMICS)
    bad = loan_pnl(5000, 12, "css", default12=1, economics=ECONOMICS)
    assert good["el"] == 0 and good["profit"] == good["income"] > 0
    assert bad["income"] == 0 and abs(bad["el"] - 5000 * 0.55) < 1e-9
    assert abs(bad["profit"] + bad["el"]) < 1e-9


def test_compute_pnl_table_adds_columns():
    scored = pd.DataFrame(
        {
            "aid": ["a1", "a2"],
            "product": ["css", "ins"],
            "app_loan_amount": [5000.0, 3000.0],
            "app_n_installments": [12, 24],
            "default12": [0, 1],
        }
    )
    out = compute_pnl_table(scored, ECONOMICS)
    assert {"income", "el", "profit", "installment"}.issubset(out.columns)
    assert np.allclose(out["profit"], out["income"] - out["el"])


def test_apply_strategy_priority_fixtures():
    fixtures = pd.DataFrame(
        [
            {
                "cid": "c1",
                "aid": "a1",
                "product": "css",
                "period": "198001",
                "app_loan_amount": 1000,
                "app_n_installments": 12,
                "pd": 0.05,
                "act_cus_active": 0,
                "agr12_Max_CMaxA_Due": 0,
            },
            {
                "cid": "c2",
                "aid": "a2",
                "product": "ins",
                "period": "198001",
                "app_loan_amount": 1000,
                "app_n_installments": 12,
                "pd": 0.01,
                "act_cus_active": 1,
                "agr12_Max_CMaxA_Due": 5,
            },
            {
                "cid": "c3",
                "aid": "a3",
                "product": "css",
                "period": "198001",
                "app_loan_amount": 1000,
                "app_n_installments": 12,
                "pd": 0.99,
                "act_cus_active": 1,
                "agr12_Max_CMaxA_Due": 0,
            },
            {
                "cid": "c4",
                "aid": "a4",
                "product": "ins",
                "period": "198001",
                "app_loan_amount": 1000,
                "app_n_installments": 12,
                "pd": 0.001,
                "act_cus_active": 1,
                "agr12_Max_CMaxA_Due": 0,
            },
            {
                "cid": "c5",
                "aid": "a5",
                "product": "css",
                "period": "197401",
                "app_loan_amount": 1000,
                "app_n_installments": 12,
                "pd": 0.99,
                "act_cus_active": 0,
                "agr12_Max_CMaxA_Due": 9,
            },
        ]
    )
    dec = apply_strategy(fixtures, BASE_RULES)
    expected = {
        "a1": ("N", "998 not active customer"),
        "a2": ("D", "1 bad customer"),
        "a3": ("D", "1 PD cut-off on css"),
        "a4": ("A", "999ok"),
        "a5": ("A", "999ok"),
    }
    for aid, (d, r) in expected.items():
        row = dec.loc[dec["aid"].eq(aid)].iloc[0]
        assert row["decision"] == d and row["decline_reason"] == r


def test_cutoffs_from_params_change_declines():
    fixtures = pd.DataFrame(
        [
            {
                "cid": "c1",
                "aid": "a1",
                "product": "css",
                "period": "198001",
                "app_loan_amount": 1000,
                "app_n_installments": 12,
                "pd": 0.25,
                "act_cus_active": 1,
                "agr12_Max_CMaxA_Due": 0,
            }
        ]
    )
    loose = deepcopy(BASE_RULES)
    loose["bad_customer"]["enabled"] = False
    loose["cutoffs"] = {"pd_css": 0.50, "pd_ins_high": 0.10}
    tight = deepcopy(loose)
    tight["cutoffs"] = {"pd_css": 0.20, "pd_ins_high": 0.10}

    d_loose = apply_strategy(fixtures, loose)
    d_tight = apply_strategy(fixtures, tight)
    assert d_loose.iloc[0]["decision"] == "A"
    assert d_tight.iloc[0]["decision"] == "D"


def test_rules_from_params():
    params = {
        "window_start": "197501",
        "window_end": "198712",
        "burn_in_before": "197501",
        "reference": 731882,
        "economics": ECONOMICS,
        "cutoffs": {"pd_css": 0.2, "pd_ins_high": 0.01},
        "bad_customer": {"enabled": True, "feature": "agr12_Max_CMaxA_Due", "threshold": 3},
    }
    rules = rules_from_params(params)
    assert rules["cutoffs"]["pd_css"] == 0.2
    assert rules["bad_customer"]["threshold"] == 3


def test_compare_strategies_appends_reference():
    results = [
        {
            "name": "St_yours",
            "eval": {
                "ar_css": 0.1,
                "ar_ins": 0.05,
                "total_profit": 1_000_000.0,
                "n_accept": 100,
            },
            "source": "offline",
        }
    ]
    table = compare_strategies(results, reference=731882)
    assert (table["strategy"] == "reference").any()
    assert table.loc[table["strategy"].eq("reference"), "total_profit"].iloc[0] == 731882


def test_normalize_calib_nested_product():
    from credit_scoring.profit.scoring import normalize_calib_params

    a, b = normalize_calib_params(
        {"ins": {"intercept": 1.0, "coef": -0.02}}, product="ins"
    )
    assert a == 1.0 and b == -0.02
    a2, b2 = normalize_calib_params({"a": 2.0, "b": -0.01})
    assert a2 == 2.0 and b2 == -0.01


def test_find_cutoff_at_ar_and_min_ar():
    from credit_scoring.profit.cutoff import (
        find_cutoff_at_ar,
        find_optimal_cutoff,
        profit_curve_by_pd,
    )

    df = pd.DataFrame(
        {
            "aid": [f"a{i}" for i in range(10)],
            "product": ["css"] * 10,
            "pd": [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50],
            "profit": [10, 10, 10, 5, 5, -2, -20, -30, -40, -50],
            "default12": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    curve = profit_curve_by_pd(df, "css")
    unconstrained = find_optimal_cutoff(curve)
    assert unconstrained["pd_cutoff"] == 0.05
    at_ar = find_cutoff_at_ar(curve, 0.5)
    assert at_ar["reachable"] is True
    assert at_ar["ar_at_peak"] >= 0.5
    constrained = find_optimal_cutoff(curve, min_ar=0.5)
    assert constrained["ar_at_peak"] >= 0.5
    assert constrained["peak_profit"] <= unconstrained["peak_profit"]
