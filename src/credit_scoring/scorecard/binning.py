"""Decision-tree and nominal binning for scorecard features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def _bin_params(params: dict) -> dict:
    """Resolve shared binning label and tree keys from scorecard params."""
    return {
        "other_label": params.get("symbol_other", params.get("other_label", "<OTHERS>")),
        "missing_label": params.get("symbol_missing", params.get("missing_label", "Missing")),
        "max_bins": params.get("ncategories_int", params.get("max_bins", 5)),
        "min_bin_size": params.get("minimum_share_int", params.get("min_bin_size", 0.05)),
        "tree_random_state": params.get("tree_random_state", 1234),
        "rare_threshold": params.get("rare_threshold", 0.02),
        "max_groups": params.get("ncategories_nom", 4),
        "nominal_int_threshold": params.get("nominal_int_threshold", 10),
    }


def fit_bin_numeric(train: pd.DataFrame, feature: str, target: str, params: dict) -> dict:
    """Fit decision-tree bins for one numeric feature.

    Actions:
    1. Drop rows with missing feature or target.
    2. Fit a shallow decision tree on the numeric feature.
    3. Return sorted edges from ``-inf`` to ``+inf`` plus a missing-bin flag.
    """
    cfg = _bin_params(params)
    has_missing = train[feature].isna().any()
    clean = train.dropna(subset=[feature, target]).copy()
    x = clean[[feature]].values
    y = clean[target].values

    tree = DecisionTreeClassifier(
        max_leaf_nodes=cfg["max_bins"],
        min_samples_leaf=max(1, int(cfg["min_bin_size"] * len(clean))),
        random_state=cfg["tree_random_state"],
    )
    tree.fit(x, y)

    thresholds = np.unique(tree.tree_.threshold[tree.tree_.threshold != -2])
    edges = [-np.inf] + list(np.sort(thresholds)) + [np.inf]

    return {
        "type": "numeric",
        "feature": feature,
        "edges": edges,
        "missing_bin": bool(has_missing),
        "missing_label": cfg["missing_label"],
    }


def fit_bin_nominal(train: pd.DataFrame, feature: str, target: str, params: dict) -> dict:
    """Fit risk-ordered nominal groups with rare-category pooling.

    Actions:
    1. Pool rare categories below ``rare_threshold`` into ``symbol_other``.
    2. Sort remaining categories by event rate.
    3. Greedily merge adjacent groups until at most ``ncategories_nom`` remain.
    """
    cfg = _bin_params(params)
    other_label = cfg["other_label"]
    missing_label = cfg["missing_label"]
    rare_threshold = cfg["rare_threshold"]
    max_groups = cfg["max_groups"]

    clean = train.dropna(subset=[feature, target]).copy()
    clean["_cat_norm"] = clean[feature].astype(str)

    freq = clean["_cat_norm"].value_counts(normalize=True)
    rare_cats = set(freq[freq < rare_threshold].index)
    valid_cats = set(freq[freq >= rare_threshold].index)

    stats = (
        clean[clean["_cat_norm"].isin(valid_cats)]
        .groupby("_cat_norm")[target]
        .agg(["mean", "count"])
        .rename(columns={"mean": "event_rate", "count": "n"})
        .sort_values(by="event_rate", ascending=True)
    )

    groups: list[list[str]] = [[cat] for cat in stats.index]

    while len(groups) > max_groups:
        best_i, best_n = 0, float("inf")
        for i in range(len(groups) - 1):
            combined = sum(stats.loc[c, "n"] for c in groups[i] + groups[i + 1])
            if combined < best_n:
                best_n, best_i = combined, i
        groups[best_i] += groups[best_i + 1]
        groups.pop(best_i + 1)

    category_map: dict[str, str] = {}
    for idx, grp in enumerate(groups):
        label = f"G{idx + 1:02d}"
        for cat in grp:
            category_map[cat] = label

    for cat in rare_cats:
        category_map[str(cat)] = other_label

    return {
        "type": "nominal",
        "feature": feature,
        "category_map": category_map,
        "other_label": other_label,
        "missing_label": missing_label,
    }


def fit_binning_maps(
    train: pd.DataFrame, features: list[str], target: str, params: dict
) -> dict[str, dict]:
    """Dispatch numeric or nominal bin fitters for every candidate feature.

    Actions:
    1. Classify each feature as nominal or numeric from dtype and cardinality.
    2. Call the appropriate fitter and collect train-only bin specs.
    """
    cfg = _bin_params(params)
    nominal_int_threshold = cfg["nominal_int_threshold"]
    binning_maps: dict[str, dict] = {}

    for feat in features:
        col = train[feat]
        is_nominal = (
            pd.api.types.is_bool_dtype(col)
            or pd.api.types.is_object_dtype(col)
            or pd.api.types.is_string_dtype(col)
            or isinstance(col.dtype, pd.CategoricalDtype)
            or (
                pd.api.types.is_integer_dtype(col)
                and col.nunique(dropna=True) <= nominal_int_threshold
            )
        )
        if is_nominal:
            binning_maps[feat] = fit_bin_nominal(train, feat, target, params)
        else:
            binning_maps[feat] = fit_bin_numeric(train, feat, target, params)

    return binning_maps


def apply_bins(df: pd.DataFrame, binning_maps: dict[str, dict]) -> pd.DataFrame:
    """Apply fitted bin definitions and add ``{feature}_GRP`` columns.

    Actions:
    1. Drop any existing grouped columns to keep reruns idempotent.
    2. Map numeric values with ``pd.cut`` and nominal values with category maps.
    3. Concatenate all new grouped columns in one pass.
    """
    base = df.drop(columns=[c for c in df.columns if c.endswith("_GRP")], errors="ignore")
    new_cols: dict[str, pd.Series] = {}

    for feat, spec in binning_maps.items():
        grp_col = f"{feat}_GRP"

        if spec["type"] == "numeric":
            missing_label = spec.get("missing_label", "Missing")
            edges = spec["edges"]
            cut = pd.cut(base[feat], bins=edges, include_lowest=True, right=True)
            new_cols[grp_col] = cut.astype("string").fillna(missing_label)

        elif spec["type"] == "nominal":
            cat_map = spec["category_map"]
            other_label = spec["other_label"]
            missing_label = spec["missing_label"]

            s = base[feat]
            missing_mask = s.isna()
            norm = s.astype("string")
            mapped = norm.map(cat_map).fillna(other_label)
            mapped.loc[missing_mask] = missing_label
            new_cols[grp_col] = mapped.astype("string")

        else:
            raise ValueError(f"Unknown binning type for {feat}: {spec['type']}")

    grp_df = pd.DataFrame(new_cols, index=base.index)
    return pd.concat([base, grp_df], axis=1).copy()
