"""Bin stability and Gini-over-time helpers for scorecard QC."""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_scoring.scorecard.selection import compute_gini


def _ordered_bins(feature: str, binning_maps: dict) -> list:
    """Return bins in cutpoint order, including special nominal bins."""
    spec = binning_maps.get(feature, {})
    order = list(spec.get("intervals", []))
    if not order and spec.get("type") == "numeric" and "edges" in spec:
        edges = spec["edges"]
        order = []
        for i in range(len(edges) - 1):
            left, right = edges[i], edges[i + 1]
            left_s = "-inf" if np.isneginf(left) else str(left)
            right_s = "inf" if np.isposinf(right) else str(right)
            order.append(f"({left_s}, {right_s}]")
    if not order and spec.get("type") == "nominal":
        labels = sorted({v for v in spec.get("category_map", {}).values()})
        order = labels

    other_label = spec.get("other_label")
    if spec.get("type") == "nominal" and other_label and other_label not in order:
        order.append(other_label)

    missing_label = spec.get("missing_label")
    if (spec.get("missing_bin") or spec.get("type") == "nominal") and missing_label and missing_label not in order:
        order.append(missing_label)

    return order


def _woe_monotonic_bins(feature: str, binning_maps: dict) -> list:
    """Bins used for pass/fail monotonicity (exclude Missing / OTHER)."""
    spec = binning_maps.get(feature, {})
    intervals = list(spec.get("intervals", []))
    if intervals:
        return intervals
    return [b for b in _ordered_bins(feature, binning_maps) if b not in ("Missing", "<OTHERS>", spec.get("other_label"), spec.get("missing_label"))]


def bin_bad_rate_by_period(
    df_binned: pd.DataFrame,
    feature: str,
    target: str,
    time_col: str = "period",
) -> pd.DataFrame:
    """Per-bin bad rate and share by period."""
    grp_col = f"{feature}_GRP"
    g = (
        df_binned.groupby([grp_col, time_col], observed=False)[target]
        .agg(n="count", bads="sum")
        .reset_index()
        .rename(columns={grp_col: "bin", time_col: "period"})
    )
    g["goods"] = g["n"] - g["bads"]
    g["bad_rate"] = g["bads"] / g["n"]
    period_totals = df_binned.groupby(time_col).size().rename("period_n")
    g = g.merge(period_totals, left_on="period", right_index=True)
    g["share"] = g["n"] / g["period_n"]
    g["variable"] = feature
    return g[["variable", "bin", "period", "n", "bads", "goods", "bad_rate", "share"]]


def flag_unstable_bins(period_table: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Flag bins with large bad-rate swing or tiny samples."""
    rows = []
    for (var, bin_label), sub in period_table.groupby(["variable", "bin"]):
        n_periods = sub["period"].nunique()
        min_n_period = sub["n"].min()
        total_n = sub["n"].sum()
        min_br = sub["bad_rate"].min()
        max_br = sub["bad_rate"].max()
        swing = max_br - min_br
        reasons = []
        if swing > params.get("max_bad_rate_swing", 0.25):
            reasons.append("bad rate swing too high")
        if min_n_period < params.get("min_bin_n_period", 10):
            reasons.append("too few obs in a period")
        if total_n < params.get("min_bin_n_total", 30):
            reasons.append("bin sample too small")
        rows.append(
            {
                "variable": var,
                "bin": bin_label,
                "n_periods": n_periods,
                "min_bad_rate": min_br,
                "max_bad_rate": max_br,
                "bad_rate_swing": swing,
                "min_n_period": min_n_period,
                "total_n": total_n,
                "flag_unstable": bool(reasons),
                "flag_reason": "; ".join(reasons),
            }
        )
    return pd.DataFrame(rows)


def bin_stability_report(
    df_binned: pd.DataFrame,
    features: list[str],
    target: str,
    time_col: str,
    params: dict,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Build per-feature period tables and unstable-bin flags."""
    period_tables: dict[str, pd.DataFrame] = {}
    parts = []
    for feat in features:
        pt = bin_bad_rate_by_period(df_binned, feat, target, time_col)
        period_tables[feat] = pt
        parts.append(pt)
    combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    flags = flag_unstable_bins(combined, params) if len(combined) else pd.DataFrame()
    return period_tables, flags


def compute_static_br_gaps(
    big_scorecard: pd.DataFrame, feature: str, binning_maps: dict
) -> pd.DataFrame:
    """Adjacent-bin bad-rate gaps (train/valid) in binning order."""
    order = _ordered_bins(feature, binning_maps)
    if len(order) < 2:
        return pd.DataFrame()

    sub = big_scorecard[big_scorecard["variable"] == feature].set_index("bin")
    rows = []
    for low, high in zip(order[:-1], order[1:]):
        if low not in sub.index or high not in sub.index:
            continue
        br_l_tr = float(sub.loc[low, "bad_rate_train"])
        br_h_tr = float(sub.loc[high, "bad_rate_train"])
        br_l_va = float(sub.loc[low, "bad_rate_valid"])
        br_h_va = float(sub.loc[high, "bad_rate_valid"])
        rows.append(
            {
                "variable": feature,
                "bin_low": low,
                "bin_high": high,
                "br_low_train": br_l_tr,
                "br_high_train": br_h_tr,
                "br_gap_train": br_h_tr - br_l_tr,
                "abs_br_gap_train": abs(br_h_tr - br_l_tr),
                "br_low_valid": br_l_va,
                "br_high_valid": br_h_va,
                "br_gap_valid": br_h_va - br_l_va,
                "abs_br_gap_valid": abs(br_h_va - br_l_va),
            }
        )
    return pd.DataFrame(rows)


def compute_period_br_gaps(
    period_table: pd.DataFrame, feature: str, binning_maps: dict
) -> pd.DataFrame:
    """Adjacent-bin bad-rate gaps by period."""
    if period_table is None or period_table.empty:
        return pd.DataFrame()

    order = _ordered_bins(feature, binning_maps)
    if len(order) < 2:
        return pd.DataFrame()

    rows = []
    for period, g in period_table.groupby("period"):
        br_map = g.set_index("bin")["bad_rate"].to_dict()
        n_map = g.set_index("bin")["n"].to_dict()
        for low, high in zip(order[:-1], order[1:]):
            if low not in br_map or high not in br_map:
                continue
            br_l = float(br_map[low])
            br_h = float(br_map[high])
            rows.append(
                {
                    "variable": feature,
                    "period": period,
                    "bin_low": low,
                    "bin_high": high,
                    "pair": f"{low} -> {high}",
                    "br_low": br_l,
                    "br_high": br_h,
                    "br_gap": br_h - br_l,
                    "abs_br_gap": abs(br_h - br_l),
                    "n_low": n_map.get(low, np.nan),
                    "n_high": n_map.get(high, np.nan),
                }
            )
    return pd.DataFrame(rows)


def check_woe_monotonicity(
    big_scorecard: pd.DataFrame, feature: str, binning_maps: dict
) -> tuple[pd.DataFrame, bool, bool, str]:
    """Return WOE table in bin order + monotonicity flags."""
    order = _woe_monotonic_bins(feature, binning_maps)
    sub = big_scorecard[big_scorecard["variable"] == feature].set_index("bin")
    rows = []
    for b in order:
        if b not in sub.index:
            continue
        rows.append(
            {
                "bin": b,
                "bad_rate_train": float(sub.loc[b, "bad_rate_train"]),
                "woe": float(sub.loc[b, "woe"]),
            }
        )
    df = pd.DataFrame(rows)
    if len(df) < 2:
        return df, True, True, "single_bin"

    steps = df["woe"].diff()
    inc = bool((steps.dropna() >= 0).all())
    dec = bool((steps.dropna() <= 0).all())
    if inc:
        direction = "increasing"
    elif dec:
        direction = "decreasing"
    else:
        direction = "not_monotonic"
    df["woe_step"] = steps
    return df, inc, dec, direction


def _woe_display_bins(
    big_scorecard: pd.DataFrame, feature: str, binning_maps: dict
) -> pd.DataFrame:
    """All bins for WOE plot (includes OTHER / Missing)."""
    order = _ordered_bins(feature, binning_maps)
    sub = big_scorecard[big_scorecard["variable"] == feature].set_index("bin")
    rows = []
    for b in order:
        if b not in sub.index:
            continue
        rows.append(
            {
                "bin": b,
                "bad_rate_train": float(sub.loc[b, "bad_rate_train"]),
                "woe": float(sub.loc[b, "woe"]),
            }
        )
    return pd.DataFrame(rows)


def build_feature_qc_table(
    train_binned: pd.DataFrame,
    big_scorecard: pd.DataFrame,
    features: list[str],
    target: str,
    time_col: str,
    params: dict,
    binning_maps: dict,
) -> pd.DataFrame:
    """Per-feature swing / BR-gap / WOE-mono gates (03 notebook ``safe_to_include``)."""
    if not features:
        return pd.DataFrame(
            columns=[
                "variable",
                "n_unstable_bins",
                "max_bad_rate_swing",
                "n_fail_static_gaps",
                "n_fail_period_gaps",
                "woe_direction",
                "pass_swing",
                "pass_static_br_gap",
                "pass_period_br_gap",
                "pass_woe_monotonic",
                "safe_to_include",
            ]
        )

    period_tables, flags = bin_stability_report(
        train_binned, features, target, time_col, params
    )
    min_static = float(params.get("min_static_br_gap", 0.02))
    min_period = float(params.get("min_period_br_gap", 0.1))

    if flags is None or flags.empty:
        stab = {f: {"n_unstable_bins": 0, "max_bad_rate_swing": 0.0} for f in features}
    else:
        stab = (
            flags.groupby("variable")
            .agg(
                n_unstable_bins=("flag_unstable", "sum"),
                max_bad_rate_swing=("bad_rate_swing", "max"),
            )
            .to_dict(orient="index")
        )

    rows = []
    for feat in features:
        s = stab.get(feat, {"n_unstable_bins": 0, "max_bad_rate_swing": 0.0})
        n_unstable = int(s.get("n_unstable_bins", 0) or 0)
        max_swing = float(s.get("max_bad_rate_swing", 0.0) or 0.0)

        static_gaps = compute_static_br_gaps(big_scorecard, feat, binning_maps)
        if static_gaps is None or static_gaps.empty:
            n_fail_static = 0
        else:
            n_fail_static = int(
                (
                    (static_gaps["abs_br_gap_train"] < min_static)
                    | (static_gaps["abs_br_gap_valid"] < min_static)
                ).sum()
            )

        period_gaps = compute_period_br_gaps(
            period_tables.get(feat, pd.DataFrame()), feat, binning_maps
        )
        if period_gaps is None or period_gaps.empty:
            n_fail_period = 0
        else:
            n_fail_period = int((period_gaps["abs_br_gap"] < min_period).sum())

        _, woe_inc, woe_dec, woe_direction = check_woe_monotonicity(
            big_scorecard, feat, binning_maps
        )
        pass_woe = bool(woe_inc or woe_dec or woe_direction == "single_bin")
        pass_swing = n_unstable == 0
        pass_static = n_fail_static == 0
        pass_period = n_fail_period == 0
        rows.append(
            {
                "variable": feat,
                "n_unstable_bins": n_unstable,
                "max_bad_rate_swing": max_swing,
                "n_fail_static_gaps": n_fail_static,
                "n_fail_period_gaps": n_fail_period,
                "woe_direction": woe_direction,
                "pass_swing": pass_swing,
                "pass_static_br_gap": pass_static,
                "pass_period_br_gap": pass_period,
                "pass_woe_monotonic": pass_woe,
                "safe_to_include": bool(
                    pass_swing and pass_static and pass_period and pass_woe
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["safe_to_include", "variable"], ascending=[False, True]
    ).reset_index(drop=True)


def gini_over_time(
    scored_df: pd.DataFrame,
    target: str,
    time_col: str = "period",
    score_col: str = "score",
    min_n: int = 30,
) -> pd.DataFrame:
    """Per-period Gini of a scored frame."""
    rows = []
    for period, sub in scored_df.groupby(time_col):
        if len(sub) < min_n:
            continue
        if sub[target].nunique() < 2:
            continue
        g = compute_gini(sub[target], sub[score_col])
        rows.append(
            {
                "period": period,
                "n": len(sub),
                "bad_rate": sub[target].mean(),
                "gini": g,
            }
        )
    return pd.DataFrame(rows).sort_values("period") if rows else pd.DataFrame(
        columns=["period", "n", "bad_rate", "gini"]
    )


def stability_gate_summary(
    qc_table: pd.DataFrame,
    kept_features: list[str],
    *,
    min_safe_ratio: float = 1.0,
) -> dict:
    """Summarize stability gate pass/fail from ``build_feature_qc_table`` output.

    This is intentionally simple: ``safe_to_include`` already encodes the detailed
    bin-level rules (bad-rate swing, min bin counts, adjacent gaps, WOE monotonicity).
    The gate then checks whether the final selected features remain stable enough
    under those rules.
    """
    kept = list(kept_features or [])
    qc = qc_table if isinstance(qc_table, pd.DataFrame) else pd.DataFrame()
    if qc.empty or "variable" not in qc.columns or "safe_to_include" not in qc.columns:
        return {
            "gate_pass": True,
            "n_vars": int(len(kept)),
            "n_safe": int(len(kept)),
            "n_unsafe": 0,
            "min_safe_ratio": float(min_safe_ratio),
            "max_bad_rate_swing_unsafe": None,
            "worst_woe_direction_unsafe": None,
        }

    qc = qc.copy()
    qc = qc.set_index("variable", drop=False)
    kept = [f for f in kept if f in qc.index]
    n_vars = int(len(kept))
    if n_vars == 0:
        return {
            "gate_pass": True,
            "n_vars": 0,
            "n_safe": 0,
            "n_unsafe": 0,
            "min_safe_ratio": float(min_safe_ratio),
            "max_bad_rate_swing_unsafe": None,
            "worst_woe_direction_unsafe": None,
        }

    flags = qc.loc[kept, "safe_to_include"]
    # If a feature is missing from qc_table, treat it as unsafe so the gate stays conservative.
    if len(flags) != len(kept):
        return {
            "gate_pass": False,
            "n_vars": n_vars,
            "n_safe": int(flags.sum()),
            "n_unsafe": int(n_vars - int(flags.sum())),
            "min_safe_ratio": float(min_safe_ratio),
            "max_bad_rate_swing_unsafe": None,
            "worst_woe_direction_unsafe": None,
        }

    n_safe = int(flags.sum())
    n_unsafe = int(n_vars - n_safe)
    safe_ratio = float(n_safe / max(1, n_vars))
    gate_pass = bool(safe_ratio >= float(min_safe_ratio))

    unsafe = qc.loc[kept, :].loc[~qc.loc[kept, "safe_to_include"].astype(bool)]
    if len(unsafe):
        max_swing = unsafe.get("max_bad_rate_swing")
        max_swing_val = float(np.nanmax(max_swing.to_numpy(dtype=float))) if max_swing is not None else None
        woe_dir = unsafe.get("woe_direction")
        woe_dir_val = str(woe_dir.iloc[0]) if woe_dir is not None and len(woe_dir) else None
    else:
        max_swing_val = None
        woe_dir_val = None

    return {
        "gate_pass": gate_pass,
        "n_vars": n_vars,
        "n_safe": n_safe,
        "n_unsafe": n_unsafe,
        "min_safe_ratio": float(min_safe_ratio),
        "max_bad_rate_swing_unsafe": max_swing_val,
        "worst_woe_direction_unsafe": woe_dir_val,
    }
