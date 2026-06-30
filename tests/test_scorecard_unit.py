import numpy as np
import pandas as pd

from credit_scoring.scorecard.binning import apply_bins, fit_bin_numeric, fit_bin_nominal
from credit_scoring.scorecard.fit import prepare_target
from credit_scoring.scorecard.partition import partition_abt
from credit_scoring.scorecard.selection import (
    ID_COLS,
    LEAKAGE_COLS,
    check_vif,
    compute_gini,
    compute_psi,
    get_candidate_features,
)
from credit_scoring.scorecard.woe import build_woe_table, compute_iv, encode_woe


def test_prepare_target_maps_sentinels():
    df = pd.DataFrame({"default12": [0, 1, ".i", ".d", np.nan]})
    out = prepare_target(df)
    assert set(out["default12"].unique()) <= {0, 1}
    assert len(out) == 4


def test_get_candidate_features_excludes_ids_and_leakage():
    df = pd.DataFrame(
        {
            "cid": [1, 2],
            "aid": ["a1", "a2"],
            "period": ["198401", "198501"],
            "product": ["css", "css"],
            "decision": ["A", "A"],
            "decline_reason": ["", ""],
            "default12": [0, 1],
            "default3": [0, 1],
            "app_income": [1000.0, 2000.0],
            "app_char_job_code": ["X", "Y"],
        }
    )
    cand = get_candidate_features(df, "css")
    assert not set(ID_COLS + LEAKAGE_COLS) & set(cand["all"])
    assert "app_income" in cand["numeric"]
    assert "app_char_job_code" in cand["nominal"]


def test_apply_bins_creates_grp_without_duplicates():
    train = pd.DataFrame({"x": [1.0, 2.0, np.nan], "default12": [0, 1, 0]})
    params = {"symbol_missing": "Missing", "symbol_other": "<OTHERS>", "tree_random_state": 1}
    spec = fit_bin_numeric(train, "x", "default12", params)
    maps = {"x": spec}
    out1 = apply_bins(train, maps)
    out2 = apply_bins(out1, maps)
    assert "x_GRP" in out2.columns
    assert out2.columns.duplicated().sum() == 0


def test_woe_iv_and_vif_helpers():
    df = pd.DataFrame(
        {
            "feat_GRP": ["A", "A", "B", "B"],
            "default12": [0, 1, 0, 1],
        }
    )
    table = build_woe_table(df, "feat_GRP", "default12", 1e-4)
    assert compute_iv(table) >= 0
    assert {"bin", "woe", "iv_component"} <= set(table.columns)

    woe_maps = {"feat_GRP": table}
    encoded = encode_woe(df, woe_maps)
    assert "feat_WOE" in encoded.columns

    vif = check_vif(encoded[["feat_WOE", "default12"]])
    assert (vif.dropna() >= 1.0).all()


def test_partition_abt_boundaries():
    df = pd.DataFrame({"period": ["198401", "198412", "198501"], "default12": [0, 1, 0]})
    train, valid = partition_abt(df, "198412", "198501")
    assert train["period"].max() == "198412"
    assert valid["period"].min() == "198501"


def test_metric_helpers_bounds():
    y = pd.Series([0, 0, 1, 1])
    score = pd.Series([0.1, 0.2, 0.8, 0.9])
    assert 0.0 <= compute_gini(y, score) <= 1.0
    assert compute_psi(pd.Series(["A", "A", "B"]), pd.Series(["A", "B", "B"]), 1e-4) >= 0.0
