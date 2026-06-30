"""End-to-end scorecard orchestration for notebook and Kedro."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

from credit_scoring.scorecard.binning import apply_bins, fit_binning_maps
from credit_scoring.scorecard.calibration import calibrate_pd
from credit_scoring.scorecard.fit import load_and_prepare_abt, train_pd_model
from credit_scoring.scorecard.partition import partition_abt
from credit_scoring.scorecard.scaling import scale_scorecard, score_applicants
from credit_scoring.scorecard.selection import get_candidate_features, prescreen_features
from credit_scoring.scorecard.woe import build_iv_table, build_woe_maps, encode_woe


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
    cand = get_candidate_features(df_model, product)
    all_features = cand["all"]

    df_train, df_valid = partition_abt(
        df_model,
        params["train_end_period"],
        params["valid_start_period"],
        time_col=params["time_col"],
    )

    train_product = df_train[df_train["product"] == product].copy()
    valid_product = df_valid[df_valid["product"] == product].copy()

    binning_maps = fit_binning_maps(
        train_product, all_features, params["target"], params
    )
    train_binned = apply_bins(train_product, binning_maps)
    valid_binned = apply_bins(valid_product, binning_maps)

    grp_cols = [f"{f}_GRP" for f in all_features if f"{f}_GRP" in train_binned.columns]
    woe_maps = build_woe_maps(
        train_binned, grp_cols, params["target"], params["woe_epsilon"]
    )
    iv_table = build_iv_table(woe_maps)

    train_woe = encode_woe(train_binned, woe_maps)
    valid_woe = encode_woe(valid_binned, woe_maps)

    screen_report = prescreen_features(train_woe, valid_woe, iv_table, params)
    selected = screen_report.loc[screen_report["status"] == "keep", "feature"].tolist()
    candidate_woe = [f"{f}_WOE" for f in selected if f"{f}_WOE" in train_woe.columns]

    model_package = train_pd_model(
        product,
        train_woe,
        valid_woe,
        params,
        candidate_woe_features=candidate_woe,
        woe_maps=woe_maps,
    )
    points_table = scale_scorecard(
        model_package, params["factor"], params["offset"]
    )

    valid_scored = score_applicants(valid_woe, model_package, points_table)
    valid_scored = valid_scored.merge(
        valid_woe[[params["id_col"], params["target"]]].rename(columns={params["id_col"]: "aid"}),
        on="aid",
        how="left",
    )
    calibration = calibrate_pd(valid_scored, target=params["target"])

    return {
        "product": product,
        "candidates": cand,
        "binning_maps": binning_maps,
        "woe_maps": woe_maps,
        "iv_table": iv_table,
        "screen_report": screen_report,
        "train_woe": train_woe,
        "valid_woe": valid_woe,
        "model_package": model_package,
        "points_table": points_table,
        "valid_scores": valid_scored,
        "calibration": calibration,
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
    df_model = load_and_prepare_abt(abt_path, decisions_path)
    df_model = df_model[df_model["decision"] == "A"].copy()

    results = {
        product: run_product_scorecard_pipeline(df_model, product, params)
        for product in ("ins", "css")
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "woe_maps.pkl").open("wb") as fh:
        pickle.dump(
            {p: results[p]["woe_maps"] for p in results},
            fh,
        )

    for product in results:
        with (out / f"pd_{product}.pkl").open("wb") as fh:
            pickle.dump(results[product]["model_package"], fh)

        results[product]["points_table"].to_parquet(
            out / f"points_table_{product}.parquet", index=False
        )
        results[product]["screen_report"].to_parquet(
            out / f"feature_screen_report_{product}.parquet", index=False
        )

    calibration_params = {
        product: results[product]["calibration"]["params"] for product in results
    }
    calibration_diagnostics = {
        product: results[product]["calibration"]["diagnostics"] for product in results
    }

    with (out / "calibration_params.json").open("w", encoding="utf-8") as fh:
        json.dump(calibration_params, fh, indent=2)

    with (out / "calibration_diagnostics.json").open("w", encoding="utf-8") as fh:
        json.dump(calibration_diagnostics, fh, indent=2)

    results["calibration_params"] = calibration_params
    results["calibration_diagnostics"] = calibration_diagnostics
    return results
