"""Score application ABT rows with frozen Gate B packages."""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_scoring.scorecard.scaling import score_applicants
from credit_scoring.scorecard.woe import encode_woe
from credit_scoring.scorecard.binning import apply_bins as apply_bins_pipeline


def normalize_calib_params(params: dict, product: str | None = None) -> tuple[float, float]:
    """Return (a, b) for PD = 1 / (1 + exp(-(a + b * score))).

    Accepts flat ``{a,b}`` / ``{intercept,coef}`` or nested ``{product: {...}}``
    as stored in ``calibration_params_*_v2.json``.
    """
    if product and product in params and isinstance(params[product], dict):
        params = params[product]
    elif len(params) == 1:
        only = next(iter(params.values()))
        if isinstance(only, dict) and (
            {"a", "b"} <= set(only) or {"intercept", "coef"} <= set(only)
        ):
            params = only

    if "a" in params and "b" in params:
        return float(params["a"]), float(params["b"])
    if "intercept" in params and "coef" in params:
        return float(params["intercept"]), float(params["coef"])
    raise KeyError(f"Calibration params need a/b or intercept/coef; got {sorted(params)}")


def score_to_pd(score, a: float, b: float) -> np.ndarray:
    """Platt-style PD from score."""
    score_arr = np.asarray(score, dtype=float)
    return 1.0 / (1.0 + np.exp(-(a + b * score_arr)))


def _assign_numeric_bin(x, edges, intervals, missing_label, missing_bin):
    """Assign a value to a notebook-style numeric interval label."""
    if pd.isna(x):
        return missing_label if missing_bin else intervals[0]
    for i in range(len(edges) - 1):
        left, right = edges[i], edges[i + 1]
        if left == -np.inf and right == np.inf:
            return intervals[i]
        if left == -np.inf and x < right:
            return intervals[i]
        if right == np.inf and x >= left:
            return intervals[i]
        if left <= x < right:
            return intervals[i]
    return intervals[-1]


def apply_bins_notebook(df: pd.DataFrame, binning_maps: dict) -> pd.DataFrame:
    """Apply ASB binning maps (interval labels) and add ``{feature}_GRP`` columns."""
    base = df.drop(columns=[c for c in df.columns if c.endswith("_GRP")], errors="ignore")
    new_cols = {}
    for feat, spec in binning_maps.items():
        grp_col = f"{feat}_GRP"
        if spec["type"] == "numeric":
            intervals = spec.get("intervals")
            if not intervals:
                raise ValueError(
                    f"{feat}: numeric binning map missing intervals (need Gate B package)"
                )
            new_cols[grp_col] = base[feat].apply(
                lambda x, s=spec: _assign_numeric_bin(
                    x,
                    s["edges"],
                    s["intervals"],
                    s["missing_label"],
                    s.get("missing_bin", False),
                )
            )
        elif spec["type"] == "nominal":
            s = base[feat]
            missing_mask = s.isna()
            mapped = s.astype("string").map(spec["category_map"]).fillna(spec["other_label"])
            mapped.loc[missing_mask] = spec["missing_label"]
            new_cols[grp_col] = mapped.astype("string")
        else:
            raise ValueError(f"Unknown binning type for {feat}: {spec['type']}")
    return pd.concat([base, pd.DataFrame(new_cols, index=base.index)], axis=1).copy()


def woe_tables_to_maps(woe_tables: dict) -> dict[str, pd.DataFrame]:
    """Convert model_package['woe_tables'] (*_WOE keys) to encode_woe maps (*_GRP keys)."""
    out = {}
    for key, table in woe_tables.items():
        feat = key[: -len("_WOE")] if key.endswith("_WOE") else key
        out[f"{feat}_GRP"] = table
    return out


def score_product_slice(
    abt_slice: pd.DataFrame,
    package: dict,
    points_table: pd.DataFrame,
    calib: dict,
) -> pd.DataFrame:
    """Score one product slice: bins -> WOE -> points score -> calibrated PD."""
    if abt_slice.empty:
        return pd.DataFrame(
            columns=[
                "aid",
                "cid",
                "product",
                "period",
                "app_loan_amount",
                "app_n_installments",
                "default12",
                "act_cus_active",
                "agr12_Max_CMaxA_Due",
                "score",
                "pd",
            ]
        )

    product = package["product"]
    binning_maps = package["binning_maps"]
    woe_maps = woe_tables_to_maps(package["woe_tables"])
    a, b = normalize_calib_params(calib, product=product)

    # Notebook Gate B packages store explicit interval labels; pipeline maps use edges only.
    needs_pipeline_bins = any(
        spec.get("type") == "numeric" and not spec.get("intervals")
        for spec in binning_maps.values()
    )
    if needs_pipeline_bins:
        binned = apply_bins_pipeline(abt_slice, binning_maps)
    else:
        binned = apply_bins_notebook(abt_slice, binning_maps)
    encoded = encode_woe(binned, woe_maps)
    missing = [f for f in package["features"] if f not in encoded.columns]
    if missing:
        raise ValueError(f"{product}: missing WOE columns after encode: {missing}")

    scored = score_applicants(encoded, package, points_table)
    keep_cols = [
        c
        for c in [
            "aid",
            "cid",
            "product",
            "period",
            "app_loan_amount",
            "app_n_installments",
            "default12",
            "act_cus_active",
            "agr12_Max_CMaxA_Due",
        ]
        if c in abt_slice.columns
    ]
    out = abt_slice[keep_cols].copy().reset_index(drop=True)
    out = out.merge(scored[["aid", "score"]], on="aid", how="left")
    out["pd"] = score_to_pd(out["score"].to_numpy(), a, b)
    out["product"] = product
    return out


def score_secondary_model(
    abt: pd.DataFrame,
    package: dict,
    points_table: pd.DataFrame,
    calib: dict,
    *,
    score_col: str,
    pd_col: str,
) -> pd.DataFrame:
    """Score all ABT rows with a secondary model (PR or Cross PD).

    Returns ``aid``, ``score_col``, ``pd_col``.
    """
    if abt.empty:
        return pd.DataFrame(columns=["aid", score_col, pd_col])

    product_key = package.get("product", "ins")
    binning_maps = package["binning_maps"]
    woe_maps = woe_tables_to_maps(package["woe_tables"])
    a, b = normalize_calib_params(calib, product=product_key)

    needs_pipeline_bins = any(
        spec.get("type") == "numeric" and not spec.get("intervals")
        for spec in binning_maps.values()
    )
    if needs_pipeline_bins:
        binned = apply_bins_pipeline(abt, binning_maps)
    else:
        binned = apply_bins_notebook(abt, binning_maps)
    encoded = encode_woe(binned, woe_maps)
    missing = [f for f in package["features"] if f not in encoded.columns]
    if missing:
        raise ValueError(f"{pd_col}: missing WOE columns after encode: {missing}")

    scored = score_applicants(encoded, package, points_table)
    out = scored[["aid", "score"]].rename(columns={"score": score_col}).copy()
    out[pd_col] = score_to_pd(out[score_col].to_numpy(), a, b)
    return out


def score_abt_application(
    abt: pd.DataFrame,
    packages: dict,
    points_tables: dict,
    calibrations: dict,
    *,
    secondary: dict | None = None,
) -> pd.DataFrame:
    """Score full ABT with per-product Gate B packages; row grain = aid.

    Optional ``secondary`` dict may include ``pr`` and/or ``cross`` each with
    ``package``, ``points``, ``calib`` to attach ``pr`` / ``cross_pd`` columns
    on every row (SAS decision-engine style).
    """
    frames = []
    for product in ("ins", "css"):
        if product not in packages:
            continue
        slice_ = abt.loc[abt["product"].eq(product)].copy()
        frames.append(
            score_product_slice(
                slice_,
                packages[product],
                points_tables[product],
                calibrations[product],
            )
        )
    if not frames:
        raise ValueError("no ins/css packages provided")
    out = pd.concat(frames, ignore_index=True)
    if not out["aid"].is_unique:
        raise ValueError("duplicate aid after scoring")

    if secondary:
        if "pr" in secondary:
            pr = score_secondary_model(
                abt,
                secondary["pr"]["package"],
                secondary["pr"]["points"],
                secondary["pr"]["calib"],
                score_col="pr_score",
                pd_col="pr",
            )
            out = out.merge(pr, on="aid", how="left")
        if "cross" in secondary:
            cr = score_secondary_model(
                abt,
                secondary["cross"]["package"],
                secondary["cross"]["points"],
                secondary["cross"]["calib"],
                score_col="cross_score",
                pd_col="cross_pd",
            )
            out = out.merge(cr, on="aid", how="left")
    return out
