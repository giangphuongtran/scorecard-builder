"""Node wiring tests for the scorecard pipeline."""

import pandas as pd

from credit_scoring.pipelines.scorecard.nodes import (
    calibrate_node,
    partition_and_bin_node,
    prepare_abt_node,
    train_scorecard_node,
    woe_and_screen_node,
)


def test_prepare_abt_node_passes_arguments(monkeypatch):
    abt = pd.DataFrame({"aid": ["a1"]})
    decisions = pd.DataFrame({"aid": ["a1"], "decision": ["A"]})
    params = {"target": "default12"}
    captured = {}

    def fake_prepare_abt(abt_arg, decisions_arg, params_arg, accepted_only=True):
        captured["abt"] = abt_arg
        captured["decisions"] = decisions_arg
        captured["params"] = params_arg
        captured["accepted_only"] = accepted_only
        return pd.DataFrame({"aid": ["a1"], "decision": ["A"]})

    monkeypatch.setattr(
        "credit_scoring.pipelines.scorecard.nodes.prepare_abt",
        fake_prepare_abt,
    )

    out = prepare_abt_node(abt, decisions, params)
    assert captured["abt"] is abt
    assert captured["decisions"] is decisions
    assert captured["params"] is params
    assert captured["accepted_only"] is True
    assert list(out.columns) == ["aid", "decision"]


def test_partition_and_bin_node_merges_params(monkeypatch):
    captured = {}

    def fake_merge(binning, model):
        captured["merged"] = {"from_binning": binning, "from_model": model}
        return {"merged": True}

    def fake_partition(abt_model, params):
        captured["params"] = params
        return {"maps": 1}, {"train": 1}, {"valid": 1}

    monkeypatch.setattr(
        "credit_scoring.pipelines.scorecard.nodes.merge_scorecard_params",
        fake_merge,
    )
    monkeypatch.setattr(
        "credit_scoring.pipelines.scorecard.nodes.partition_and_bin",
        fake_partition,
    )

    out = partition_and_bin_node(
        pd.DataFrame(), {"woe_epsilon": 1e-4}, {"target": "default12"}
    )
    assert captured["params"] == {"merged": True}
    assert out == ({"maps": 1}, {"train": 1}, {"valid": 1})


def test_woe_and_screen_node_passes_frames(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        "credit_scoring.pipelines.scorecard.nodes.merge_scorecard_params",
        lambda b, m: {"ok": True},
    )

    def fake_woe(train_binned, valid_binned, binning_maps, params):
        captured["train"] = train_binned
        captured["valid"] = valid_binned
        captured["maps"] = binning_maps
        captured["params"] = params
        return 1, 2, 3, 4, 5

    monkeypatch.setattr(
        "credit_scoring.pipelines.scorecard.nodes.woe_and_screen",
        fake_woe,
    )

    out = woe_and_screen_node({"css": 1}, {"css": 2}, {"css": 3}, {}, {})
    assert out == (1, 2, 3, 4, 5)
    assert captured["params"] == {"ok": True}


def test_train_scorecard_node_splits_packages(monkeypatch):
    monkeypatch.setattr(
        "credit_scoring.pipelines.scorecard.nodes.merge_scorecard_params",
        lambda b, m: {},
    )
    monkeypatch.setattr(
        "credit_scoring.pipelines.scorecard.nodes.train_both_products",
        lambda *args: (
            {"ins": {"product": "ins"}, "css": {"product": "css"}},
            {"ins": pd.DataFrame(), "css": pd.DataFrame()},
            {"ins": {}, "css": {}},
            {"ins": pd.DataFrame(), "css": pd.DataFrame()},
        ),
    )

    pd_ins, pd_css, points, diag, scores = train_scorecard_node({}, {}, {}, {}, {}, {})
    assert pd_ins["product"] == "ins"
    assert pd_css["product"] == "css"
    assert set(points) == {"ins", "css"}
    assert set(scores) == {"ins", "css"}


def test_calibrate_node_passes_scores(monkeypatch):
    captured = {}

    def fake_calibrate(scores_valid, params):
        captured["scores"] = scores_valid
        captured["params"] = params
        return {"ins": {}}, {"ins": {}}

    monkeypatch.setattr(
        "credit_scoring.pipelines.scorecard.nodes.calibrate_both",
        fake_calibrate,
    )
    out = calibrate_node({"ins": pd.DataFrame()}, {"target": "default12"})
    assert captured["params"]["target"] == "default12"
    assert out == ({"ins": {}}, {"ins": {}})
