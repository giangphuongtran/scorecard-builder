"""Freeze / persist scorecard model bundles for profit scoring."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import statsmodels.api as sm

from credit_scoring.scorecard.scaling import scale_scorecard, score_applicants
from credit_scoring.scorecard.selection import assess_logit_model, check_vif
from credit_scoring.scorecard.calibration import calibrate_pd


def build_model_package(
    *,
    product: str,
    features: list[str],
    model,
    train_subset: pd.DataFrame,
    valid_subset: pd.DataFrame,
    target: str,
    binning_maps: dict,
    woe_maps: dict[str, pd.DataFrame],
    id_col: str = "aid",
    extra: dict | None = None,
) -> dict:
    """Assemble a Gate-B-compatible model package with metrics and WOE tables."""
    metrics = assess_logit_model(model, train_subset, valid_subset, target)
    effects = pd.DataFrame(
        [
            {
                "feature": feat,
                "beta": float(model.params[feat]),
                "pvalue": float(model.pvalues[feat]),
                "vif": float(check_vif(train_subset[features]).get(feat, float("nan"))),
            }
            for feat in features
        ]
    )
    woe_tables = {}
    for woe_col in features:
        raw = woe_col[: -len("_WOE")] if woe_col.endswith("_WOE") else woe_col
        grp = f"{raw}_GRP"
        if grp in woe_maps:
            woe_tables[woe_col] = woe_maps[grp]

    selected_raw = {f[: -len("_WOE")] if f.endswith("_WOE") else f for f in features}
    # Keep edges/category_map only — scoring uses apply_bins (pd.cut), matching WOE labels
    maps = {feat: dict(binning_maps[feat]) for feat in selected_raw if feat in binning_maps}
    for spec in maps.values():
        spec.pop("intervals", None)
    pkg: dict[str, Any] = {
        "product": product,
        "target": target,
        "features": list(features),
        "model": model,
        "metrics": {
            **metrics,
            "nnegative_betas": metrics.get("n_negative_betas"),
        },
        "effects": effects,
        "train_subset": train_subset,
        "valid_subset": valid_subset,
        "woe_tables": woe_tables,
        "binning_maps": maps,
        "id_col": id_col,
    }
    if extra:
        pkg.update(extra)
    return pkg


def refit_logit(
    train_df: pd.DataFrame, features: list[str], target: str
):
    """Refit statsmodels Logit on fixed WOE features."""
    x = sm.add_constant(train_df[features], has_constant="add")
    return sm.Logit(train_df[target], x).fit(disp=0, maxiter=200)


def save_model_bundle(
    *,
    product: str,
    version: str,
    model_package: dict,
    points_table: pd.DataFrame | None = None,
    calibration: dict | None = None,
    output_dir: str | Path = "data/06_models",
    factor: float | None = None,
    offset: float | None = None,
    decision_log: dict | None = None,
    prefix: str | None = None,
) -> dict[str, Path]:
    """Write pkl + points parquet + calib json (+ optional decision log).

    Naming:
    - application PD: ``pd_{product}_{version}.*``
    - PR: ``pr_css_{version}.*`` when product=='pr'
    - Cross: ``cross_pd_css_{version}.*`` when product=='cross'
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if prefix is None:
        if product == "pr":
            prefix = f"pr_css_{version}"
            pkg_name = f"pr_css_{version}.pkl"
        elif product == "cross":
            prefix = f"cross_pd_css_{version}"
            pkg_name = f"cross_pd_css_{version}.pkl"
        else:
            prefix = f"pd_{product}_{version}"
            pkg_name = f"pd_{product}_{version}.pkl"
    else:
        pkg_name = f"{prefix}.pkl"

    # Drop large frames before pickle (scoring only needs maps/features/betas)
    slim = {
        k: v
        for k, v in model_package.items()
        if k not in ("train_subset", "valid_subset")
    }
    # Keep serializable effects
    if isinstance(slim.get("effects"), pd.DataFrame):
        slim["effects"] = slim["effects"].copy()

    if points_table is None:
        if factor is None or offset is None:
            raise ValueError("points_table or factor/offset required")
        points_table = scale_scorecard(model_package, factor, offset)

    if calibration is None:
        # Build scores on train for calib if available
        train = model_package.get("train_subset")
        target = model_package.get("target", "default12")
        if train is None:
            raise ValueError("calibration required when train_subset missing")
        scored = score_applicants(train, model_package, points_table)
        scored = scored.merge(train[["aid", target]], on="aid", how="left")
        calibration = calibrate_pd(scored, target=target)

    pkg_path = out / pkg_name
    with open(pkg_path, "wb") as f:
        pickle.dump(slim, f)

    points_path = out / f"points_table_{prefix.replace('pd_', '')}.parquet"
    # Normalize points filename to match profit.artifacts convention
    if product in ("ins", "css"):
        points_path = out / f"points_table_{product}_{version}.parquet"
        calib_path = out / f"calibration_params_{product}_{version}.json"
        log_path = out / f"decision_log_{product}_{version}.json"
    elif product == "pr":
        points_path = out / f"points_table_pr_css_{version}.parquet"
        calib_path = out / f"calibration_params_pr_css_{version}.json"
        log_path = out / f"decision_log_pr_css_{version}.json"
    else:
        points_path = out / f"points_table_cross_pd_css_{version}.parquet"
        calib_path = out / f"calibration_params_cross_pd_css_{version}.json"
        log_path = out / f"decision_log_cross_pd_css_{version}.json"

    # Preserve attrs via sidecar columns if needed
    pt = points_table.copy()
    pt.to_parquet(points_path, index=False)

    # Write calibration matching existing Gate B JSON shape
    raw = calibration.get("params", calibration)
    a = float(raw.get("a", raw.get("intercept", 0.0)))
    b = float(raw.get("b", raw.get("coef", 0.0)))
    target = raw.get("target", model_package.get("target", "default12"))
    flat = {"intercept": a, "coef": b, "target": target, "score_col": "score"}
    if product in ("ins", "css"):
        calib_payload = {product: flat}
    else:
        # PR / Cross scored without product-key nesting
        calib_payload = flat

    meta = {
        "base_points": float(points_table.attrs.get("base_points", 0.0)),
        "factor": float(points_table.attrs.get("factor", factor or 0.0)),
        "offset": float(points_table.attrs.get("offset", offset or 0.0)),
    }
    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump(calib_payload, f, indent=2)
    meta_path = out / f"{prefix}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "diagnostics": calibration.get("diagnostics")}, f, indent=2)

    if decision_log is not None:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(decision_log, f, indent=2, default=str)
    else:
        log_path = None

    paths = {
        "package": pkg_path,
        "points": points_path,
        "calib": calib_path,
        "meta": meta_path,
    }
    if log_path:
        paths["decision_log"] = log_path
    return paths


def load_calib_a_b(calib_obj: dict, product: str | None = None) -> tuple[float, float]:
    """Extract (a, b) from frozen calib JSON payload."""
    from credit_scoring.profit.scoring import normalize_calib_params

    if "params" in calib_obj and isinstance(calib_obj["params"], dict):
        return normalize_calib_params(calib_obj["params"], product=product)
    return normalize_calib_params(calib_obj, product=product)
