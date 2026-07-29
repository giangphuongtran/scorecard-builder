"""Node wiring tests for the profit pipeline."""

from __future__ import annotations

import pandas as pd

from credit_scoring.pipelines.profit.nodes import (
    apply_strategy_node,
    report_profit_node,
    score_application_abt_node,
)
from credit_scoring.pipelines.profit.pipeline import create_pipeline


def test_profit_pipeline_io():
    pipeline = create_pipeline()
    names = [node.name for node in pipeline.nodes]
    assert names == [
        "score_application_abt_node",
        "apply_strategy_node",
        "report_profit_node",
    ]
    by_name = {node.name: node for node in pipeline.nodes}
    assert list(by_name["score_application_abt_node"].inputs) == [
        "abt_app_cross",
        "params:profit",
    ]
    assert "params:profit" in list(by_name["apply_strategy_node"].inputs)
    assert list(by_name["report_profit_node"].outputs) == [
        "profit_by_loan",
        "profit_summary",
        "profit_report",
    ]


def test_score_application_abt_node_wraps(monkeypatch):
    captured = {}

    def fake_load(_cfg):
        return (
            {"ins": {"product": "ins"}, "css": {"product": "css"}},
            {"ins": pd.DataFrame(), "css": pd.DataFrame()},
            {"ins": {"a": 1.0, "b": -0.01}, "css": {"a": 1.0, "b": -0.01}},
            {"pr": {"package": {}, "points": pd.DataFrame(), "calib": {}}},
        )

    def fake_score(abt, packages, points_tables, calibrations, secondary=None):
        captured["packages"] = packages
        captured["secondary"] = secondary
        return pd.DataFrame({"aid": ["a1"], "pd": [0.1]})

    monkeypatch.setattr(
        "credit_scoring.pipelines.profit.nodes.load_profit_artifacts",
        fake_load,
    )
    monkeypatch.setattr(
        "credit_scoring.pipelines.profit.nodes.score_abt_application",
        fake_score,
    )
    out = score_application_abt_node(
        pd.DataFrame(),
        {"artifacts": {"ins": {}, "css": {}, "pr": {}}},
    )
    assert captured["packages"]["ins"]["product"] == "ins"
    assert "pr" in captured["secondary"]
    assert list(out.columns) == ["aid", "pd"]


def test_apply_strategy_node_uses_params(monkeypatch):
    captured = {}

    def fake_apply(scored, rules):
        captured["rules"] = rules
        return pd.DataFrame({"aid": ["a1"], "decision": ["A"]})

    monkeypatch.setattr(
        "credit_scoring.pipelines.profit.nodes.apply_strategy",
        fake_apply,
    )
    params = {
        "window_start": "197501",
        "window_end": "198712",
        "burn_in_before": "197501",
        "economics": {"ins": {}, "css": {}},
        "cutoffs": {"pd_css": 0.2, "pd_ins_high": 0.01},
        "bad_customer": {"enabled": True, "feature": "x", "threshold": 3},
    }
    out = apply_strategy_node(pd.DataFrame({"aid": ["a1"]}), params)
    assert captured["rules"]["cutoffs"]["pd_css"] == 0.2
    assert out.iloc[0]["decision"] == "A"


def test_report_profit_node_summary(monkeypatch):
    monkeypatch.setattr(
        "credit_scoring.pipelines.profit.nodes.export_asif_scored",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "credit_scoring.pipelines.profit.nodes.log_policy_run",
        lambda **k: None,
    )
    scored = pd.DataFrame(
        {
            "aid": ["a1"],
            "product": ["css"],
            "period": ["198001"],
            "app_loan_amount": [1000.0],
            "app_n_installments": [12],
            "default12": [0],
            "pd": [0.1],
        }
    )
    decisions = pd.DataFrame(
        {
            "aid": ["a1"],
            "decision": ["A"],
            "decline_reason": ["999ok"],
        }
    )
    params = {
        "window_start": "197501",
        "window_end": "198712",
        "burn_in_before": "197501",
        "reference": 731882,
        "economics": {
            "ins": {"lgd": 0.45, "apr_annual": 0.01, "provision": 0.0},
            "css": {"lgd": 0.55, "apr_annual": 0.18, "provision": 0.0},
        },
        "cutoffs": {"pd_css": 0.2, "pd_ins_high": 0.01},
        "bad_customer": {"enabled": False, "feature": "agr12_Max_CMaxA_Due", "threshold": 3},
    }

    by_loan, summary, report = report_profit_node(scored, decisions, params)
    assert "profit" in by_loan.columns
    assert summary["reference"] == 731882
    assert summary["source"] == "offline"
    assert summary["best_strategy"] == "CLTV mid-band (production)"
    assert "Offline" in report
