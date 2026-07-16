"""Thin Kedro nodes for the scorecard pipeline."""

from __future__ import annotations

from credit_scoring.scorecard.orchestration import (
    calibrate_both,
    partition_and_bin,
    train_both_products,
    woe_and_screen,
)
from credit_scoring.scorecard.fit import prepare_abt
from credit_scoring.scorecard.params import merge_scorecard_params
from credit_scoring.mlflow_utils import maybe_log_run

import tempfile
import json
from pathlib import Path


def prepare_abt_node(abt_app, decisions, model_params):
    """Merge ABT + decisions and keep accepted applicants."""
    return prepare_abt(abt_app, decisions, model_params, accepted_only=True)


def partition_and_bin_node(abt_model, binning_params, model_params):
    """Partition train/valid and fit/apply bins for both products."""
    params = merge_scorecard_params(binning_params, model_params)
    return partition_and_bin(abt_model, params)


def woe_and_screen_node(train_binned, valid_binned, binning_maps, binning_params, model_params):
    """Encode WOE and prescreen features for both products."""
    params = merge_scorecard_params(binning_params, model_params)
    return woe_and_screen(train_binned, valid_binned, binning_maps, params)


def train_scorecard_node(
    train_woe,
    valid_woe,
    woe_maps,
    feature_screen_report,
    binning_params,
    model_params,
):
    """Train PD models, build points tables, and score validation."""
    params = merge_scorecard_params(binning_params, model_params)
    model_packages, points_tables, model_diagnostics, scores_valid = train_both_products(
        train_woe, valid_woe, woe_maps, feature_screen_report, params
    )

    # Optional MLflow logging for experiment tracking.
    for product in ("ins", "css"):
        md = model_diagnostics.get(product) or {}
        metrics = (md.get("metrics") or {}) if isinstance(md, dict) else {}
        if metrics:
            with tempfile.TemporaryDirectory() as td:
                pt_path = Path(td) / f"points_table_{product}.parquet"
                try:
                    points_tables[product].to_parquet(pt_path, index=False)
                except Exception:
                    pt_path = None
                artifacts = {"points_table.parquet": pt_path} if pt_path else {}
                maybe_log_run(
                    run_name=f"scorecard_train__{product}",
                    params={
                        "target": params.get("target"),
                        "product": product,
                        "gini_floor": params.get("gini_floor"),
                        "psi_max": params.get("psi_max"),
                        "ar_diff_max": params.get("ar_diff_max"),
                        "vif_max": params.get("vif_max"),
                        "max_features": params.get("max_features"),
                    },
                    metrics={f"{product}_{k}": v for k, v in metrics.items() if v is not None},
                    artifacts=artifacts,
                    tags={"pipeline": "scorecard", "node": "train_scorecard_node"},
                )

    return (
        model_packages.get("ins"),
        model_packages.get("css"),
        points_tables,
        model_diagnostics,
        scores_valid,
    )


def calibrate_node(scores_valid, model_params):
    """Calibrate PD per product from validation scores."""
    calibration_params, calibration_diagnostics = calibrate_both(scores_valid, model_params)

    for product in ("ins", "css"):
        params = (calibration_params or {}).get(product) or {}
        diag = (calibration_diagnostics or {}).get(product) or {}
        if params or diag:
            with tempfile.TemporaryDirectory() as td:
                calib_path = Path(td) / f"calibration_params_{product}.json"
                calib_path.write_text(
                    json.dumps({"params": params, "diagnostics": diag}, default=str, indent=2),
                    encoding="utf-8",
                )
                maybe_log_run(
                    run_name=f"scorecard_calibrate__{product}",
                    params={
                        "target": params.get("target", model_params.get("target")),
                        "product": product,
                    },
                    metrics={
                        **{f"{product}_{k}": v for k, v in (diag or {}).items() if isinstance(v, (int, float))},
                    },
                    artifacts={"calibration.json": calib_path},
                    tags={"pipeline": "scorecard", "node": "calibrate_node"},
                )

    return calibration_params, calibration_diagnostics
