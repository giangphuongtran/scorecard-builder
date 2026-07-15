"""End-to-end scorecard orchestration for notebook and Kedro."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

from credit_scoring.scorecard.binning import apply_bins, fit_binning_maps
from credit_scoring.scorecard.calibration import calibrate_pd
from credit_scoring.scorecard.fit import prepare_abt, train_pd_model
from credit_scoring.scorecard.params import merge_scorecard_params
from credit_scoring.scorecard.partition import partition_abt
from credit_scoring.scorecard.scaling import scale_scorecard, score_applicants
from credit_scoring.scorecard.selection import get_candidate_features, prescreen_features
from credit_scoring.scorecard.woe import build_iv_table, build_woe_maps, encode_woe

PRODUCTS = ("ins", "css")


def partition_and_bin(
    abt_model: pd.DataFrame,
    params: dict,
    products: tuple[str, ...] = PRODUCTS,
) -> tuple[dict, dict, dict]:
    """Partition the modeling ABT and fit/apply bins per product.

    Returns:
        binning_maps, train_binned, valid_binned — each keyed by product.
    """
    df_train, df_valid = partition_abt(
        abt_model,
        params["train_end_period"],
        params["valid_start_period"],
        time_col=params["time_col"],
    )

    binning_maps: dict[str, dict] = {}
    train_binned: dict[str, pd.DataFrame] = {}
    valid_binned: dict[str, pd.DataFrame] = {}

    for product in products:
        cand = get_candidate_features(abt_model, product)
        train_product = df_train[df_train["product"] == product].copy()
        valid_product = df_valid[df_valid["product"] == product].copy()
        maps = fit_binning_maps(
            train_product, cand["all"], params["target"], params
        )
        binning_maps[product] = maps
        train_binned[product] = apply_bins(train_product, maps)
        valid_binned[product] = apply_bins(valid_product, maps)

    return binning_maps, train_binned, valid_binned


def woe_and_screen(
    train_binned: dict[str, pd.DataFrame],
    valid_binned: dict[str, pd.DataFrame],
    binning_maps: dict[str, dict],
    params: dict,
) -> tuple[dict, dict, dict, dict, dict]:
    """Build WOE maps, encode frames, and prescreen features per product.

    Returns:
        woe_maps, train_woe, valid_woe, iv_table, feature_screen_report
        — each keyed by product (iv_table and screen report are DataFrames).
    """
    woe_maps: dict[str, dict] = {}
    train_woe: dict[str, pd.DataFrame] = {}
    valid_woe: dict[str, pd.DataFrame] = {}
    iv_table: dict[str, pd.DataFrame] = {}
    feature_screen_report: dict[str, pd.DataFrame] = {}

    for product, maps in binning_maps.items():
        features = list(maps.keys())
        grp_cols = [
            f"{f}_GRP"
            for f in features
            if f"{f}_GRP" in train_binned[product].columns
        ]
        product_woe_maps = build_woe_maps(
            train_binned[product],
            grp_cols,
            params["target"],
            params["woe_epsilon"],
        )
        product_iv = build_iv_table(product_woe_maps)
        tr_woe = encode_woe(train_binned[product], product_woe_maps)
        va_woe = encode_woe(valid_binned[product], product_woe_maps)
        screen = prescreen_features(tr_woe, va_woe, product_iv, params)

        woe_maps[product] = product_woe_maps
        train_woe[product] = tr_woe
        valid_woe[product] = va_woe
        iv_table[product] = product_iv
        feature_screen_report[product] = screen

    return woe_maps, train_woe, valid_woe, iv_table, feature_screen_report


def train_both_products(
    train_woe: dict[str, pd.DataFrame],
    valid_woe: dict[str, pd.DataFrame],
    woe_maps: dict[str, dict],
    feature_screen_report: dict[str, pd.DataFrame],
    params: dict,
) -> tuple[dict, dict, dict, dict]:
    """Train PD models, build points tables, and score validation sets.

    Returns:
        model_packages, points_tables, model_diagnostics, scores_valid
        (each keyed by product).
    """
    model_packages: dict[str, dict] = {}
    points_tables: dict[str, pd.DataFrame] = {}
    model_diagnostics: dict[str, dict] = {}
    scores_valid: dict[str, pd.DataFrame] = {}

    for product in train_woe:
        screen = feature_screen_report[product]
        selected = screen.loc[screen["status"] == "keep", "feature"].tolist()
        candidate_woe = [
            f"{f}_WOE"
            for f in selected
            if f"{f}_WOE" in train_woe[product].columns
        ]
        package = train_pd_model(
            product,
            train_woe[product],
            valid_woe[product],
            params,
            candidate_woe_features=candidate_woe,
            woe_maps=woe_maps[product],
        )
        points = scale_scorecard(package, params["factor"], params["offset"])
        scored = score_applicants(valid_woe[product], package, points)
        scored = scored.merge(
            valid_woe[product][[params["id_col"], params["target"]]].rename(
                columns={params["id_col"]: "aid"}
            ),
            on="aid",
            how="left",
        )

        metrics = package.get("metrics", {}) or {}
        serializable_metrics = {
            k: (float(v) if hasattr(v, "item") else v)
            for k, v in metrics.items()
            if isinstance(v, (int, float, str, bool, type(None)))
            or hasattr(v, "item")
        }
        model_packages[product] = package
        points_tables[product] = points
        model_diagnostics[product] = {
            "metrics": serializable_metrics,
            "features": list(package.get("features", [])),
            "n_candidates": len(candidate_woe),
        }
        scores_valid[product] = scored

    return model_packages, points_tables, model_diagnostics, scores_valid


def calibrate_both(
    scores_valid: dict[str, pd.DataFrame],
    params: dict,
) -> tuple[dict, dict]:
    """Calibrate PD per product from scored validation frames."""
    calibration_params: dict[str, dict] = {}
    calibration_diagnostics: dict[str, dict] = {}
    target = params["target"]

    for product, scored in scores_valid.items():
        calibration = calibrate_pd(scored, target=target)
        calibration_params[product] = calibration["params"]
        calibration_diagnostics[product] = calibration["diagnostics"]

    return calibration_params, calibration_diagnostics


def run_product_scorecard_pipeline(
    df_model: pd.DataFrame,
    product: str,
    params: dict,
) -> dict:
    """Run binning, WOE, screening, training, scaling, and calibration for one product.

    Actions:
    1. Select accepted-only candidates and partition train/valid.
    2. Fit bins on product train, encode WOE, and prescreen features.
    3. Train final logit, build points table, score valid, and calibrate PD.
    """
    binning_maps, train_binned, valid_binned = partition_and_bin(
        df_model, params, products=(product,)
    )
    woe_maps, train_woe, valid_woe, iv_table, screen_report = woe_and_screen(
        train_binned, valid_binned, binning_maps, params
    )
    model_packages, points_tables, _, scores_valid = train_both_products(
        train_woe, valid_woe, woe_maps, screen_report, params
    )
    calibration_params, calibration_diagnostics = calibrate_both(
        scores_valid, params
    )

    return {
        "product": product,
        "candidates": get_candidate_features(df_model, product),
        "binning_maps": binning_maps[product],
        "woe_maps": woe_maps[product],
        "iv_table": iv_table[product],
        "screen_report": screen_report[product],
        "train_woe": train_woe[product],
        "valid_woe": valid_woe[product],
        "model_package": model_packages[product],
        "points_table": points_tables[product],
        "valid_scores": scores_valid[product],
        "calibration": {
            "params": calibration_params[product],
            "diagnostics": calibration_diagnostics[product],
        },
    }


def run_full_scorecard_pipeline(
    abt_path: str | Path,
    decisions_path: str | Path,
    params: dict,
    output_dir: str | Path = "data/06_models",
) -> dict:
    """Run accepted-only PD scorecards for both products and save artifacts.

    Actions:
    1. Load and prepare modeling ABT.
    2. Train/score/calibrate ``ins`` and ``css`` separately.
    3. Persist model packages, WOE maps, points tables, and calibration JSON.
    """
    if isinstance(params, dict) and "binning" in params and "model" in params:
        flat = merge_scorecard_params(params["binning"], params["model"])
    else:
        flat = params

    abt = pd.read_parquet(abt_path)
    decisions = pd.read_parquet(decisions_path)
    df_model = prepare_abt(abt, decisions, flat, accepted_only=True)

    results = {
        product: run_product_scorecard_pipeline(df_model, product, flat)
        for product in PRODUCTS
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "woe_maps.pkl").open("wb") as fh:
        pickle.dump({p: results[p]["woe_maps"] for p in PRODUCTS}, fh)

    for product in PRODUCTS:
        with (out / f"pd_{product}.pkl").open("wb") as fh:
            pickle.dump(results[product]["model_package"], fh)

        results[product]["points_table"].to_parquet(
            out / f"points_table_{product}.parquet", index=False
        )
        results[product]["screen_report"].to_parquet(
            out / f"feature_screen_report_{product}.parquet", index=False
        )

    calibration_params = {
        product: results[product]["calibration"]["params"] for product in PRODUCTS
    }
    calibration_diagnostics = {
        product: results[product]["calibration"]["diagnostics"]
        for product in PRODUCTS
    }

    with (out / "calibration_params.json").open("w", encoding="utf-8") as fh:
        json.dump(calibration_params, fh, indent=2)

    with (out / "calibration_diagnostics.json").open("w", encoding="utf-8") as fh:
        json.dump(calibration_diagnostics, fh, indent=2)

    results["calibration_params"] = calibration_params
    results["calibration_diagnostics"] = calibration_diagnostics
    return results


__all__ = [
    "PRODUCTS",
    "calibrate_both",
    "merge_scorecard_params",
    "partition_and_bin",
    "prepare_abt",
    "run_full_scorecard_pipeline",
    "run_product_scorecard_pipeline",
    "train_both_products",
    "woe_and_screen",
]
