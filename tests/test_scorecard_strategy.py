"""Tests for cross labels, mid-band strategy, and scorecard HTML report."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from credit_scoring.profit.rules import apply_strategy
from credit_scoring.scorecard.params import artifact_registry, load_product_scorecard_params
from credit_scoring.scorecard.reports import build_model_report, render_scorecard_html
from credit_scoring.scorecard.selection_asb import filter_candidates
from credit_scoring.scorecard.stability import build_feature_qc_table
from credit_scoring.simulation.cross_labels import attach_cross_labels, cross_label_summary


ECONOMICS = {
    "ins": {"lgd": 0.45, "apr_annual": 0.01, "provision": 0.0},
    "css": {"lgd": 0.55, "apr_annual": 0.18, "provision": 0.0},
}

ROOT = Path(__file__).resolve().parents[1]


def _toy_abt() -> pd.DataFrame:
    """cid=1: ins then css 2 months later; cid=2: ins only."""
    rows = [
        # cid 1
        {"cid": "1", "aid": "i1", "product": "ins", "period": "198001", "decision": "A",
         "default12": 0, "app_loan_amount": 1000, "app_n_installments": 12},
        {"cid": "1", "aid": "c1", "product": "css", "period": "198003", "decision": "A",
         "default12": 1, "app_loan_amount": 2000, "app_n_installments": 24},
        # cid 2
        {"cid": "2", "aid": "i2", "product": "ins", "period": "198001", "decision": "A",
         "default12": 0, "app_loan_amount": 1000, "app_n_installments": 12},
        # burn filler so period grid spans
        {"cid": "3", "aid": "i3", "product": "ins", "period": "198002", "decision": "A",
         "default12": 0, "app_loan_amount": 500, "app_n_installments": 6},
        {"cid": "3", "aid": "c3", "product": "css", "period": "198004", "decision": "A",
         "default12": 0, "app_loan_amount": 500, "app_n_installments": 6},
        {"cid": "3", "aid": "i4", "product": "ins", "period": "198005", "decision": "A",
         "default12": 0, "app_loan_amount": 500, "app_n_installments": 6},
    ]
    return pd.DataFrame(rows)


def test_attach_cross_labels_flags_responder():
    abt = _toy_abt()
    out = attach_cross_labels(abt, response_n_months=6)
    assert "cross_response" in out.columns
    i1 = out.loc[out["aid"].eq("i1")].iloc[0]
    assert int(i1["cross_response"]) == 1
    assert str(i1["cross_aid"]) == "c1"
    assert int(i1["default_cross12"]) == 1
    i2 = out.loc[out["aid"].eq("i2")].iloc[0]
    assert int(i2["cross_response"]) == 0
    summary = cross_label_summary(out)
    assert summary["n_responders"] >= 1
    assert summary["has_default_cross12"]


def test_midband_strategy_uses_pr_and_cross():
    rules = {
        "window_start": "197501",
        "window_end": "198712",
        "burn_in_before": "197501",
        "economics": ECONOMICS,
        "cutoffs": {
            "pd_css": 0.30,
            "pd_ins_high": 0.08,
            "pd_ins_low": 0.02,
            "pr_min": 0.028,
            "cross_pd_max": 0.27,
        },
        "bad_customer": {"enabled": False},
    }
    fixtures = pd.DataFrame(
        [
            {
                "cid": "c1",
                "aid": "a1",
                "product": "ins",
                "period": "198001",
                "app_loan_amount": 1000,
                "app_n_installments": 12,
                "pd": 0.05,
                "pr": 0.01,
                "cross_pd": 0.10,
                "act_cus_active": 1,
                "agr12_Max_CMaxA_Due": 0,
            },
            {
                "cid": "c2",
                "aid": "a2",
                "product": "ins",
                "period": "198001",
                "app_loan_amount": 1000,
                "app_n_installments": 12,
                "pd": 0.05,
                "pr": 0.10,
                "cross_pd": 0.10,
                "act_cus_active": 1,
                "agr12_Max_CMaxA_Due": 0,
            },
            {
                "cid": "c3",
                "aid": "a3",
                "product": "ins",
                "period": "198001",
                "app_loan_amount": 1000,
                "app_n_installments": 12,
                "pd": 0.05,
                "pr": 0.10,
                "cross_pd": 0.50,
                "act_cus_active": 1,
                "agr12_Max_CMaxA_Due": 0,
            },
        ]
    )
    dec = apply_strategy(fixtures, rules)
    assert dec.loc[dec["aid"].eq("a1"), "decision"].iloc[0] == "D"
    assert "3 PD" in dec.loc[dec["aid"].eq("a1"), "decline_reason"].iloc[0]
    assert dec.loc[dec["aid"].eq("a2"), "decision"].iloc[0] == "A"
    assert dec.loc[dec["aid"].eq("a3"), "decision"].iloc[0] == "D"


def test_render_scorecard_html_smoke(tmp_path):
    points = pd.DataFrame(
        {
            "feature": ["feat_a_WOE", "feat_a_WOE", "feat_b_WOE"],
            "bin": ["low", "high", "G01"],
            "woe": [-0.5, 0.5, 0.1],
            "points": [10.0, -10.0, 3.0],
        }
    )
    points.attrs["base_points"] = 600.0
    points.attrs["factor"] = 20.0
    points.attrs["offset"] = 600.0

    class _M:
        params = pd.Series({"const": 0.1, "feat_a_WOE": -0.2, "feat_b_WOE": -0.1})
        pvalues = pd.Series({"const": 0.5, "feat_a_WOE": 0.01, "feat_b_WOE": 0.02})

    pkg = {
        "product": "ins",
        "target": "default12",
        "features": ["feat_a_WOE", "feat_b_WOE"],
        "model": _M(),
        "metrics": {
            "gini_train": 0.4,
            "gini_valid": 0.35,
            "ar_diff": 0.05,
            "max_vif": 1.2,
            "max_pvalue": 0.01,
            "n_features": 2,
            "n_negative_betas": 2,
        },
        "effects": pd.DataFrame(
            [
                {"feature": "feat_a_WOE", "beta": -0.2, "pvalue": 0.01, "vif": 1.1},
                {"feature": "feat_b_WOE", "beta": -0.1, "pvalue": 0.02, "vif": 1.2},
            ]
        ),
    }
    big = pd.DataFrame(
        [
            {
                "variable": "feat_a",
                "bin": "low",
                "condition": "low",
                "n_train": 50,
                "bads_train": 10,
                "goods_train": 40,
                "share_train": 0.5,
                "woe": -0.5,
                "iv_component": 0.01,
                "bad_rate_train": 0.2,
            },
            {
                "variable": "feat_a",
                "bin": "high",
                "condition": "high",
                "n_train": 50,
                "bads_train": 20,
                "goods_train": 30,
                "share_train": 0.5,
                "woe": 0.5,
                "iv_component": 0.02,
                "bad_rate_train": 0.4,
            },
            {
                "variable": "feat_b",
                "bin": "G01",
                "condition": "G01",
                "n_train": 40,
                "bads_train": 5,
                "goods_train": 35,
                "share_train": 0.4,
                "woe": 0.1,
                "iv_component": 0.005,
                "bad_rate_train": 0.125,
            },
        ]
    )
    gini_time = pd.DataFrame(
        {"period": ["198601", "198602"], "n": [40, 40], "bad_rate": [0.2, 0.25], "gini": [0.3, 0.32]}
    )
    cal_table = pd.DataFrame(
        {
            "decile": [1, 2],
            "n": [10, 10],
            "mean_score": [400.0, 500.0],
            "bad_rate": [0.3, 0.1],
        }
    )
    calib = {
        "params": {"a": 0.0, "b": -0.01, "target": "default12", "score_col": "score"},
        "diagnostics": {
            "auc_before": 0.7,
            "auc_after": 0.71,
            "brier_before": 0.2,
            "brier_after": 0.19,
            "mean_pd_predicted": 0.2,
            "mean_pd_actual": 0.21,
        },
    }
    report = build_model_report(pkg, points, gini_time, calib, cal_table, None)
    assert "target" not in report["Calibration"].columns
    assert list(report["Calibration_params"].columns) == ["a", "b"]
    assert "mean_score" in report["Calibration_deciles"].columns

    qc = pd.DataFrame(
        [
            {
                "variable": "feat_a",
                "n_unstable_bins": 0,
                "max_bad_rate_swing": 0.05,
                "n_fail_static_gaps": 0,
                "n_fail_period_gaps": 0,
                "woe_direction": "increasing",
                "pass_swing": True,
                "pass_static_br_gap": True,
                "pass_period_br_gap": True,
                "pass_woe_monotonic": True,
                "safe_to_include": True,
            }
        ]
    )
    out = tmp_path / "scorecard_report_ins.html"
    html = render_scorecard_html(
        product="ins",
        model_package=pkg,
        points_table=points,
        big_scorecard=big,
        model_report=report,
        gini_time=gini_time,
        calibration=calib,
        cal_table=cal_table,
        qc_table=qc,
        train_scores=pd.Series([400.0, 500.0, 600.0]),
        profit_one={
            "product": "ins",
            "total_profit": 1000.0,
            "n_accept": 10,
            "ar": 0.1,
            "bad_rate": 0.05,
            "cutoff": 0.01,
            "reference": 100.0,
            "beats_reference": True,
        },
        profit_all=None,
        output_path=out,
    )
    assert out.exists()
    assert ">Scorecard</h2>" not in html
    assert "points_round" in html
    assert "feat_a" in html
    assert "Feature QC" not in html
    assert "train_score_min" in html
    scale_chunk = html.split("<h2>Scale of scorecard</h2>")[1].split("</section>")[0]
    assert "<table" in scale_chunk
    assert "One product (ins)" in html
    assert "All products (by product)" in html
    assert "Collinearity" not in html
    assert "Not run yet" in html
    assert html.count('class="var-name"') == 2
    assert '<p class="hint">' in html
    assert "Brier" in html or "brier" in html.lower() or "mean squared" in html
    cal_chunk = html.split("<h2>Calibration</h2>")[1].split("</section>")[0]
    assert "<th>target</th>" not in cal_chunk
    assert "<th>default12</th>" not in cal_chunk
    assert "<th>a</th>" in cal_chunk
    assert "Score deciles" in cal_chunk
    assert "1000.0000" in html or "1000" in html


def test_load_product_scorecard_params_from_yml():
    yml = yaml.safe_load((ROOT / "conf" / "base" / "parameters.yml").read_text())
    css = load_product_scorecard_params(yml, "css")
    assert css["target"] == "default12"
    assert css["ncategories_int"] == 3
    assert css["minimum_share_int"] == 0.20
    assert css["ncategories_nom"] == 3
    assert css["prefixes"] == ("app", "act")
    assert css["blocked"] == ("agr", "ags")
    assert "factor" in css and "offset" in css
    assert css["soft_drop_qc_fails"] is True
    assert css["max_bad_rate_swing"] == 0.75
    assert css["min_static_br_gap"] == 0.01
    assert css["min_period_br_gap"] == 0.01
    assert css["min_bin_n_period"] == 5
    ins = load_product_scorecard_params(yml, "ins")
    assert ins["ncategories_int"] == 4
    assert ins["minimum_share_int"] == 0.20
    assert ins["ncategories_nom"] == 4
    assert ins["min_static_br_gap"] == 0.01
    assert ins["min_period_br_gap"] == 0.01
    assert ins["min_bin_n_period"] == 1
    assert ins["max_bad_rate_swing"] == 1.0
    assert ins["soft_drop_qc_fails"] is True
    pr = load_product_scorecard_params(yml, "pr")
    assert pr["target"] == "cross_response"
    assert pr["ncategories_int"] == 3
    assert pr["minimum_share_int"] == 0.12
    assert pr["ncategories_nom"] == 3
    assert pr["min_period_br_gap"] == 0.01
    assert pr["soft_drop_qc_fails"] is True
    assert "soft_drop_qc_fails" not in (yml.get("scorecard") or {}).get("css", {})
    cross = load_product_scorecard_params(yml, "cross")
    assert cross["target"] == "default_cross12"
    assert cross["ncategories_int"] == 3
    assert cross["minimum_share_int"] == 0.20
    assert cross["ncategories_nom"] == 3
    assert cross["min_period_br_gap"] == 0.001
    assert cross["min_bin_n_period"] == 1
    assert cross["max_bad_rate_swing"] == 1.0
    assert cross["soft_drop_qc_fails"] is False
    assert (yml.get("scorecard") or {}).get("cross", {}).get("soft_drop_qc_fails") is False


def test_filter_candidates_blocks_agr_ags_keeps_act_ccss():
    """No robust_css path: CSS keeps act_ccss_*, still blocks agr/ags."""
    n = 40
    abt = pd.DataFrame(
        {
            "product": ["css"] * n,
            "decision": ["A"] * n,
            "decline_reason": ["999ok"] * n,
            "app_income": np.linspace(1000, 5000, n),
            "act_ccss_maxdue": np.linspace(0, 10, n),
            "act_ccss_n_loan": np.linspace(0, 5, n),
            "act_cins_n_loan": np.linspace(0, 3, n),
            "agr12_Max_CMaxA_Due": np.linspace(0, 6, n),
            "ags3_Mean_CMaxA_Due": np.linspace(0, 4, n),
            "default12": ([0, 1] * (n // 2)),
        }
    )
    feats = filter_candidates(abt, "css")
    assert "app_income" in feats
    assert "act_ccss_maxdue" in feats
    assert "act_ccss_n_loan" in feats
    assert "act_cins_n_loan" in feats
    assert "agr12_Max_CMaxA_Due" not in feats
    assert "ags3_Mean_CMaxA_Due" not in feats
    sig = inspect.signature(filter_candidates)
    assert "robust_css" not in sig.parameters
    assert "allow_ccss" not in sig.parameters


def test_build_feature_qc_table_respects_gaps_and_swing():
    """Toy mono feature passes; collapsing BR gaps fail min_static_br_gap."""
    n = 200
    periods = np.array(["198601", "198602", "198603", "198604"] * (n // 4))
    x = np.concatenate([np.full(n // 2, 1.0), np.full(n // 2, 10.0)])
    y = np.concatenate([np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)])
    train_b = pd.DataFrame(
        {
            "feat_ok_GRP": np.where(x < 5, "( -inf, 5.0]", "(5.0,  inf]"),
            "period": periods,
            "default12": y,
        }
    )
    big = pd.DataFrame(
        [
            {
                "variable": "feat_ok",
                "bin": "( -inf, 5.0]",
                "bad_rate_train": 0.05,
                "bad_rate_valid": 0.06,
                "woe": -0.8,
            },
            {
                "variable": "feat_ok",
                "bin": "(5.0,  inf]",
                "bad_rate_train": 0.55,
                "bad_rate_valid": 0.50,
                "woe": 0.9,
            },
        ]
    )
    maps = {
        "feat_ok": {
            "type": "numeric",
            "intervals": ["( -inf, 5.0]", "(5.0,  inf]"],
            "edges": [-np.inf, 5.0, np.inf],
        }
    }
    params_loose = {
        "max_bad_rate_swing": 0.4,
        "min_bin_n_period": 5,
        "min_bin_n_total": 10,
        "min_static_br_gap": 0.02,
        "min_period_br_gap": 0.001,
    }
    qc = build_feature_qc_table(
        train_b, big, ["feat_ok"], "default12", "period", params_loose, maps
    )
    assert len(qc) == 1
    assert bool(qc.iloc[0]["pass_static_br_gap"])
    assert bool(qc.iloc[0]["pass_woe_monotonic"])

    big_tight = big.copy()
    big_tight["bad_rate_train"] = [0.40, 0.41]
    big_tight["bad_rate_valid"] = [0.39, 0.40]
    params_strict = {**params_loose, "min_static_br_gap": 0.05}
    qc2 = build_feature_qc_table(
        train_b, big_tight, ["feat_ok"], "default12", "period", params_strict, maps
    )
    assert not bool(qc2.iloc[0]["pass_static_br_gap"])
    assert not bool(qc2.iloc[0]["safe_to_include"])


def test_artifact_registry_smoke():
    yml = yaml.safe_load((ROOT / "conf" / "base" / "parameters.yml").read_text())
    rows = artifact_registry(
        yml["profit"], models_dir=ROOT / "data" / "06_models", root=ROOT
    )
    by_prod = {r["product"]: r for r in rows}
    assert by_prod["ins"]["yaml_exists"] is True
    assert by_prod["css"]["yaml_exists"] is True
    assert "pr" in by_prod and "cross" in by_prod


def test_bin_condition_nominal_raw_levels():
    from credit_scoring.scorecard.big_scorecard import _bin_condition

    maps = {
        "app_char_gender": {
            "type": "nominal",
            "category_map": {"female": "G01", "male": "G02", "Missing": "G01"},
            "other_label": "<OTHERS>",
            "missing_label": "Missing",
        }
    }
    cond = _bin_condition(maps, "app_char_gender", "G01")
    assert "female" in cond
    assert cond != "G01"


def test_bin_condition_numeric_interval():
    from credit_scoring.scorecard.big_scorecard import _bin_condition

    maps = {
        "app_age": {
            "type": "numeric",
            "intervals": ["(-inf, 44.5]", "(44.5, inf]"],
            "edges": [float("-inf"), 44.5, float("inf")],
            "missing_label": "Missing",
        }
    }
    assert _bin_condition(maps, "app_age", "(-inf, 44.5]") == "(-inf, 44.5]"


def test_evaluate_cutoffs_by_product():
    from credit_scoring.profit.cutoff_explore import evaluate_cutoffs

    scored = pd.DataFrame(
        [
            {
                "cid": "1",
                "aid": "a1",
                "product": "ins",
                "period": "198001",
                "app_loan_amount": 1000,
                "app_n_installments": 12,
                "pd": 0.005,
                "pr": 0.1,
                "cross_pd": 0.1,
                "default12": 0,
                "act_cus_active": 1,
                "agr12_Max_CMaxA_Due": 0,
                "income": 50.0,
                "el": 0.0,
                "profit": 50.0,
            },
            {
                "cid": "2",
                "aid": "a2",
                "product": "css",
                "period": "198001",
                "app_loan_amount": 2000,
                "app_n_installments": 24,
                "pd": 0.10,
                "pr": 0.1,
                "cross_pd": 0.1,
                "default12": 0,
                "act_cus_active": 1,
                "agr12_Max_CMaxA_Due": 0,
                "income": 80.0,
                "el": 0.0,
                "profit": 80.0,
            },
        ]
    )
    ev = evaluate_cutoffs(
        scored,
        {
            "pd_css": 0.32,
            "pd_ins_high": 0.08,
            "pd_ins_low": 0.02,
            "pr_min": 0.028,
            "cross_pd_max": 0.27,
        },
        window_start="197501",
        window_end="198712",
        economics=ECONOMICS,
        bad_customer={"enabled": False},
    )
    assert ev["n_accept"] == 2
    assert isinstance(ev["by_product"], pd.DataFrame)
    assert set(ev["by_product"]["product"]) == {"ins", "css"}
    assert ev["midband_empty"] is False


def test_profit_html_by_product_rows():
    from credit_scoring.scorecard.reports import _profit_html

    html = _profit_html(
        "ins",
        None,
        {
            "total_profit": 100.0,
            "n_accept": 2,
            "ar_ins": 0.2,
            "ar_css": 0.1,
            "by_product": [
                {"product": "ins", "n_accept": 1, "total_profit": 40.0, "bad_rate": 0.0},
                {"product": "css", "n_accept": 1, "total_profit": 60.0, "bad_rate": 0.0},
            ],
        },
    )
    assert "ins" in html and "css" in html
    assert "by product" in html.lower()


def test_save_workbench_bundle_handles_qcut_intervals(tmp_path):
    from credit_scoring.profit.cutoff_explore import save_workbench_product_bundle

    scores = pd.Series(np.linspace(300, 700, 40))
    tmp = pd.DataFrame({"aid": range(40), "score": scores, "default12": [0, 1] * 20})
    tmp["decile"] = pd.qcut(tmp["score"], 4, duplicates="drop")
    cal = (
        tmp.groupby("decile", observed=False)
        .agg(n=("aid", "size"), mean_score=("score", "mean"), bad_rate=("default12", "mean"))
        .reset_index()
    )
    out = save_workbench_product_bundle(
        "ins",
        {
            "cal_table": cal,
            "points_table": pd.DataFrame(
                {"feature": ["a_WOE"], "bin": ["G01"], "points": [1.0]}
            ),
            "model_package": {
                "product": "ins",
                "target": "default12",
                "features": ["a_WOE"],
                "metrics": {},
            },
            "calibration": {"params": {"a": 0.0, "b": -0.01}},
        },
        bundle_dir=tmp_path,
    )
    loaded = pd.read_parquet(out / "cal_table.parquet")
    assert "decile" in loaded.columns
    assert loaded["decile"].dtype == object or str(loaded["decile"].dtype).startswith("str")
