"""Big scorecard table builder (ASB Big_scorecard equivalent)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _bin_condition(binning_maps: dict, feature: str, bin_label) -> str:
    """Human-readable bin condition: raw categories (nominal) or interval edges (numeric)."""
    label = str(bin_label)
    spec = (binning_maps or {}).get(feature) or {}
    btype = spec.get("type")

    if btype == "nominal":
        missing_label = str(spec.get("missing_label", "Missing"))
        other_label = str(spec.get("other_label", "<OTHERS>"))
        if label == missing_label:
            return missing_label
        if label == other_label:
            return other_label
        cat_map = spec.get("category_map") or {}
        raw = sorted({str(k) for k, v in cat_map.items() if str(v) == label})
        if raw:
            return " | ".join(raw)
        return label

    if btype == "numeric":
        missing_label = str(spec.get("missing_label", "Missing"))
        if label == missing_label:
            return missing_label
        intervals = spec.get("intervals") or []
        if label in intervals:
            return label
        # pd.cut Interval string may differ slightly from enrich() labels — normalize
        for iv in intervals:
            if str(iv).replace(" ", "") == label.replace(" ", ""):
                return str(iv)
        return label

    return label


def enrich_binning_intervals(binning_maps: dict) -> dict:
    """Add pd.cut-style interval labels to numeric maps for QC / reporting."""
    out = {}
    for feat, spec in binning_maps.items():
        s = dict(spec)
        if s.get("type") == "numeric" and "edges" in s and not s.get("intervals"):
            edges = s["edges"]
            intervals = []
            for i in range(len(edges) - 1):
                left, right = edges[i], edges[i + 1]
                left_s = "-inf" if np.isneginf(left) else f"{left:g}"
                right_s = "inf" if np.isposinf(right) else f"{right:g}"
                intervals.append(f"({left_s}, {right_s}]")
            s["intervals"] = intervals
        out[feat] = s
    return out


def build_big_scorecard(
    train_binned: pd.DataFrame,
    valid_binned: pd.DataFrame,
    woe_maps: dict[str, pd.DataFrame],
    binning_maps: dict,
    target: str,
    params: dict,
) -> pd.DataFrame:
    """Per-bin WOE / IV / PSI table across train and valid."""
    eps = params.get("woe_epsilon", 1e-4)
    rows = []
    features = [grp[: -len("_GRP")] for grp in woe_maps]
    n_train = len(train_binned)
    n_valid = len(valid_binned)
    sum_bad_train = float(train_binned[target].sum())
    sum_bad_valid = float(valid_binned[target].sum())

    for feat in features:
        grp_col = f"{feat}_GRP"
        woe_tbl = woe_maps[grp_col]
        tr = (
            train_binned.groupby(grp_col, observed=False)[target]
            .agg(n_train="count", bads_train="sum")
            .reset_index()
            .rename(columns={grp_col: "bin"})
        )
        tr["goods_train"] = tr["n_train"] - tr["bads_train"]
        tr["bad_rate_train"] = tr["bads_train"] / tr["n_train"]
        tr["share_train"] = tr["n_train"] / max(n_train, 1)
        tr["bad_share_train"] = tr["bads_train"] / max(sum_bad_train, 1)

        va = (
            valid_binned.groupby(grp_col, observed=False)[target]
            .agg(n_valid="count", bads_valid="sum")
            .reset_index()
            .rename(columns={grp_col: "bin"})
        )
        va["goods_valid"] = va["n_valid"] - va["bads_valid"]
        va["bad_rate_valid"] = va["bads_valid"] / va["n_valid"].clip(lower=1)
        va["share_valid"] = va["n_valid"] / max(n_valid, 1)
        va["bad_share_valid"] = va["bads_valid"] / max(sum_bad_valid, 1)

        merged = tr.merge(va, on="bin", how="outer")
        woe_part = woe_tbl[["bin", "woe", "iv_component"]].copy()
        if "bad_rate" in woe_tbl.columns:
            woe_part = woe_part.merge(
                woe_tbl[["bin", "bad_rate"]].rename(columns={"bad_rate": "bad_rate_woe_train"}),
                on="bin",
                how="left",
            )
        merged = merged.merge(woe_part, on="bin", how="left")
        merged["variable"] = feat
        merged["condition"] = merged["bin"].map(lambda b: _bin_condition(binning_maps, feat, b))
        merged["psi_bin"] = (merged["share_train"] - merged["share_valid"]) * np.log(
            (merged["share_train"] + eps) / (merged["share_valid"] + eps)
        )
        merged["psi_bad"] = (merged["bad_share_train"] - merged["bad_share_valid"]) * np.log(
            (merged["bad_share_train"] + eps) / (merged["bad_share_valid"] + eps)
        )
        rows.append(merged)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    col_order = [
        "variable",
        "bin",
        "condition",
        "n_train",
        "bads_train",
        "goods_train",
        "bad_rate_train",
        "share_train",
        "n_valid",
        "bads_valid",
        "goods_valid",
        "bad_rate_valid",
        "share_valid",
        "woe",
        "iv_component",
        "psi_bin",
        "psi_bad",
    ]
    return out[[c for c in col_order if c in out.columns]]
