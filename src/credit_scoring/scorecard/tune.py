"""End-to-end ASB tune context for the interactive notebook."""

from __future__ import annotations

from typing import Any

import pandas as pd

from credit_scoring.scorecard.big_scorecard import build_big_scorecard, enrich_binning_intervals
from credit_scoring.scorecard.binning import apply_bins, fit_binning_maps
from credit_scoring.scorecard.calibration import calibrate_pd
from credit_scoring.scorecard.freeze import build_model_package, refit_logit
from credit_scoring.scorecard.partition import partition_abt
from credit_scoring.scorecard.reports import build_model_report, build_variable_report
from credit_scoring.scorecard.scaling import scale_scorecard, score_applicants
from credit_scoring.scorecard.selection import assess_logit_model, prescreen_features
from credit_scoring.scorecard.selection_asb import (
    BLOCKED,
    PREFIXES,
    filter_candidates,
    filter_submodel_list,
    search_combos,
    shortlist_rfe,
    uni_table,
    woe_keep_list,
)
from credit_scoring.scorecard.stability import (
    build_feature_qc_table,
    gini_over_time,
    stability_gate_summary,
)
from credit_scoring.scorecard.woe import build_iv_table, build_woe_maps, encode_woe


MODEL_SPECS = {
    "ins": {
        "abt_product": "ins",
        "target": "default12",
        "population": "accepted_product",
        "category_order": False,
        "require_negative_betas": True,
        "gini_floor": 0.45,
    },
    "css": {
        "abt_product": "css",
        "target": "default12",
        "population": "accepted_product",
        "category_order": False,
        "require_negative_betas": True,
        "gini_floor": 0.45,
    },
    "pr": {
        "abt_product": "ins",
        "target": "cross_response",
        "population": "accepted_product",
        "category_order": True,
        "require_negative_betas": False,
        "gini_floor": 0.30,
    },
    "cross": {
        "abt_product": "ins",
        "target": "default_cross12",
        "population": "accepted_responders",
        "category_order": False,
        "require_negative_betas": True,
        "gini_floor": 0.25,
    },
}


def _slice_population(abt: pd.DataFrame, product: str, spec: dict) -> pd.DataFrame:
    abt_product = spec["abt_product"]
    out = abt.loc[abt["product"].eq(abt_product)].copy()
    if "decision" in out.columns:
        out = out.loc[out["decision"].eq("A")].copy()
    target = spec["target"]
    if target == "default_cross12":
        out = out.loc[out.get("cross_response", 0) == 1].copy()
        out = out.dropna(subset=[target])
        out[target] = out[target].astype(int)
    elif target == "cross_response":
        out[target] = out[target].fillna(0).astype(int)
    else:
        out[target] = out[target].map({".i": 0, ".d": 0, 0: 0, 1: 1})
        out = out.dropna(subset=[target])
        out[target] = out[target].astype(int)
    return out


def prepare_scorecard_context(
    abt: pd.DataFrame,
    product: str,
    params: dict,
) -> dict[str, Any]:
    """Bin → WOE → prescreen → QC context for one PRODUCT family."""
    if product not in MODEL_SPECS:
        raise ValueError(f"unknown product {product}")
    spec = dict(MODEL_SPECS[product])

    target = params.get("target", spec["target"])
    time_col = params.get("time_col", "period")
    pop = _slice_population(abt, product, {**spec, "target": target})

    train_end = params["train_end_period"]
    valid_start = params["valid_start_period"]
    train, valid = partition_abt(pop, train_end, valid_start, time_col=time_col)

    prefixes = tuple(params.get("prefixes", PREFIXES))
    blocked = tuple(params.get("blocked", BLOCKED))
    feats = filter_candidates(pop, product, prefixes=prefixes, blocked=blocked)
    maps = enrich_binning_intervals(fit_binning_maps(train, feats, target, params))
    train_b = apply_bins(train, maps)
    valid_b = apply_bins(valid, maps)
    grp_cols = [f"{f}_GRP" for f in maps if f"{f}_GRP" in train_b.columns]
    woe_maps = build_woe_maps(train_b, grp_cols, target, params.get("woe_epsilon", 1e-4))
    iv = build_iv_table(woe_maps)
    train_w = encode_woe(train_b, woe_maps)
    valid_w = encode_woe(valid_b, woe_maps)

    screen_params = {
        **params,
        "target": target,
        "woe_epsilon": params.get("woe_epsilon", 1e-4),
    }
    screen = prescreen_features(train_w, valid_w, iv, screen_params)
    woe_cols = woe_keep_list(screen, train_w)
    raw_kept = [c[: -len("_WOE")] if c.endswith("_WOE") else c for c in woe_cols]
    big = build_big_scorecard(train_b, valid_b, woe_maps, maps, target, screen_params)

    qc_table = build_feature_qc_table(
        train_b,
        big,
        raw_kept,
        target,
        time_col,
        screen_params,
        maps,
    )
    # Univariate table over full prescreen pool (before QC soft-drop).
    uni = uni_table(train_w, valid_w, woe_cols, target, time_col=time_col)
    if params.get("soft_drop_qc_fails", True) and len(qc_table):
        safe = set(qc_table.loc[qc_table["safe_to_include"], "variable"])
        woe_cols = [
            c
            for c in woe_cols
            if (c[: -len("_WOE")] if c.endswith("_WOE") else c) in safe
        ]

    return {
        "product": product,
        "spec": spec,
        "target": target,
        "params": screen_params,
        "feats": feats,
        "maps": maps,
        "woe_maps": woe_maps,
        "iv": iv,
        "train_b": train_b,
        "valid_b": valid_b,
        "train_w": train_w,
        "valid_w": valid_w,
        "screen": screen,
        "woe_cols": woe_cols,
        "uni": uni,
        "big_scorecard": big,
        "qc_table": qc_table,
        "pop": pop,
    }


def run_variable_search(
    ctx: dict,
    *,
    number_vars: int = 12,
    number_features=(5, 6, 7),
    seed: int = 1234,
) -> tuple[pd.DataFrame, pd.DataFrame, dict | None]:
    """RFE shortlist + combinatorial Model_list / subModel_list."""
    params = ctx["params"]
    target = ctx["target"]
    short = shortlist_rfe(
        ctx["train_w"], ctx["uni"], ctx["woe_cols"], target, n=number_vars, seed=seed
    )
    model_list, best = search_combos(
        ctx["train_w"],
        ctx["valid_w"],
        short,
        target,
        sizes=tuple(number_features),
        pvalue_max=params.get("pvalue_max", 0.05),
        vif_max=params.get("vif_max", 3.0),
        gini_floor=ctx["spec"].get("gini_floor", 0.45),
        require_negative_betas=ctx["spec"].get("require_negative_betas", True),
    )
    sub = filter_submodel_list(
        model_list,
        pvalue_max=params.get("pvalue_max", 0.05),
        vif_max=params.get("vif_max", 3.0),
        require_negative_betas=ctx["spec"].get("require_negative_betas", True),
    )
    return model_list, sub, best


def fit_selected_model(
    ctx: dict,
    features: list[str],
    *,
    factor: float,
    offset: float,
) -> dict[str, Any]:
    """Refit winner features, scale, calibrate, build package + scores."""
    target = ctx["target"]
    product = ctx["product"]
    train_w, valid_w = ctx["train_w"], ctx["valid_w"]
    model = refit_logit(train_w, features, target)
    pkg = build_model_package(
        product=product,
        features=features,
        model=model,
        train_subset=train_w,
        valid_subset=valid_w,
        target=target,
        binning_maps=ctx["maps"],
        woe_maps=ctx["woe_maps"],
    )
    points = scale_scorecard(pkg, factor, offset)
    scored_tr = score_applicants(train_w, pkg, points)
    scored_tr = scored_tr.merge(train_w[["aid", target, "period"]], on="aid", how="left")
    scored_va = score_applicants(valid_w, pkg, points)
    scored_va = scored_va.merge(valid_w[["aid", target, "period"]], on="aid", how="left")
    calib = calibrate_pd(scored_tr, target=target)

    # Decile calibration table for report
    tmp = scored_va.copy()
    tmp["decile"] = pd.qcut(tmp["score"], 10, duplicates="drop")
    cal_table = (
        tmp.groupby("decile", observed=False)
        .agg(n=("aid", "size"), mean_score=("score", "mean"), bad_rate=(target, "mean"))
        .reset_index()
    )
    gini_time = gini_over_time(scored_va, target)
    raw_kept = [
        (f[: -len("_WOE")] if f.endswith("_WOE") else f) for f in features
    ]
    min_safe_ratio = float(ctx.get("params", {}).get("stability_min_safe_ratio", 1.0))
    stability_gate = stability_gate_summary(
        ctx.get("qc_table") or pd.DataFrame(),
        raw_kept,
        min_safe_ratio=min_safe_ratio,
    )
    model_report = build_model_report(
        pkg,
        points,
        gini_time,
        calib,
        cal_table,
        ctx["uni"],
        stability_gate=stability_gate,
    )
    var_report = build_variable_report(
        ctx["train_b"],
        ctx["big_scorecard"],
        [f[: -len("_WOE")] for f in features],
        target,
        params.get("time_col", "period") if (params := ctx["params"]) else "period",
        ctx["params"],
        binning_maps=ctx["maps"],
    )
    return {
        "model_package": pkg,
        "points_table": points,
        "calibration": calib,
        "cal_table": cal_table,
        "gini_time": gini_time,
        "model_report": model_report,
        "variable_report": var_report,
        "scored_train": scored_tr,
        "scored_valid": scored_va,
        "metrics": assess_logit_model(model, train_w, valid_w, target),
    }
