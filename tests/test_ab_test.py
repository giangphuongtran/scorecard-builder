"""Tests for offline A/B champion vs challenger comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_scoring.pipelines.ab_test.nodes import compare_policies_node
from credit_scoring.pipelines.ab_test.pipeline import create_pipeline
from credit_scoring.profit.tradeoffs import (
    compare_to_benchmark,
    midband_funnel,
    policy_sensitivity,
)


ECONOMICS = {
    "ins": {"lgd": 0.45, "apr_annual": 0.01, "provision": 0.0},
    "css": {"lgd": 0.55, "apr_annual": 0.18, "provision": 0.0},
}


def _synthetic_scored_pnl() -> pd.DataFrame:
    rows = [
        {
            "aid": "a1",
            "product": "css",
            "period": "198001",
            "pd": 0.40,
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
            "pd": 0.52,
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
            "product": "ins",
            "period": "198001",
            "pd": 0.005,
            "pr": 0.05,
            "cross_pd": 0.10,
            "act_cus_active": 1,
            "agr12_Max_CMaxA_Due": 0.0,
            "default12": 0,
            "profit": 80.0,
            "income": 80.0,
            "el": 0.0,
        },
        {
            "aid": "a4",
            "product": "ins",
            "period": "198001",
            "pd": 0.02,
            "pr": 0.05,
            "cross_pd": 0.10,
            "act_cus_active": 1,
            "agr12_Max_CMaxA_Due": 0.0,
            "default12": 0,
            "profit": 40.0,
            "income": 40.0,
            "el": 0.0,
        },
        {
            "aid": "a5",
            "product": "ins",
            "period": "198001",
            "pd": 0.02,
            "pr": 0.01,
            "cross_pd": 0.40,
            "act_cus_active": 1,
            "agr12_Max_CMaxA_Due": 0.0,
            "default12": 1,
            "profit": -30.0,
            "income": 0.0,
            "el": 30.0,
        },
    ]
    return pd.DataFrame(rows)


def test_ab_test_pipeline_io():
    pipeline = create_pipeline()
    assert [n.name for n in pipeline.nodes] == ["compare_policies_node"]
    node = next(iter(pipeline.nodes))
    assert list(node.inputs) == ["params:ab_test", "params:profit"]
    assert list(node.outputs) == ["ab_test_summary", "ab_test_report"]


def test_compare_policies_delta(monkeypatch, tmp_path):
    scored = _synthetic_scored_pnl()
    path = tmp_path / "asif.parquet"
    scored.to_parquet(path, index=False)
    meta = tmp_path / "meta.json"
    meta.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "credit_scoring.pipelines.ab_test.nodes.log_policy_run",
        lambda **k: None,
    )

    champ = {
        "pd_css": 0.50,
        "pd_ins_high": 0.03,
        "pd_ins_low": 0.01,
        "pr_min": 0.028,
        "cross_pd_max": 0.2724,
    }
    chall = dict(champ)
    chall["pd_css"] = 0.55

    summary, html = compare_policies_node(
        {
            "asif_path": str(path),
            "meta_path": str(meta),
            "reference": 100,
            "champion": {"name": "cltv_production", "cutoffs": champ},
            "challenger": {"name": "looser_css", "cutoffs": chall},
        },
        {
            "window_start": "198001",
            "window_end": "198001",
            "burn_in_before": "197501",
            "economics": ECONOMICS,
            "bad_customer": {"enabled": False},
            "reference": 100,
        },
    )
    assert "delta" in summary
    assert summary["delta"]["delta_n_accept"] >= 0  # looser CSS accepts a2
    assert summary["challenger"]["metrics"]["n_accept"] >= summary["champion"]["metrics"]["n_accept"]
    assert "Winner" in html or "winner" in html.lower()


def test_tradeoffs_helpers():
    scored = _synthetic_scored_pnl()
    cuts = {
        "pd_css": 0.50,
        "pd_ins_high": 0.03,
        "pd_ins_low": 0.01,
        "pr_min": 0.028,
        "cross_pd_max": 0.2724,
    }
    sens = policy_sensitivity(
        scored,
        cuts,
        "pd_css",
        [0.0, 0.05],
        window_start="198001",
        window_end="198001",
        burn_in_before="197501",
        economics=ECONOMICS,
        bad_customer={"enabled": False},
    )
    assert len(sens) == 2
    assert "delta_profit" in sens.columns

    funnel = midband_funnel(
        scored,
        cuts,
        window_start="198001",
        window_end="198001",
        burn_in_before="197501",
        bad_customer={"enabled": False},
    )
    assert set(funnel["stage"]) >= {"auto_approve", "grey_zone", "cltv_save", "cltv_decline"}

    from credit_scoring.profit.cutoff_explore import evaluate_cutoffs

    ev = evaluate_cutoffs(
        scored,
        cuts,
        window_start="198001",
        window_end="198001",
        burn_in_before="197501",
        economics=ECONOMICS,
        bad_customer={"enabled": False},
    )
    cmp_ = compare_to_benchmark(ev, 50.0)
    assert cmp_["delta_profit"] == ev["total_profit"] - 50.0
    assert cmp_["label"] == "published benchmark"
