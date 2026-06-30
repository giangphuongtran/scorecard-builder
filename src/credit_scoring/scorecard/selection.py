"""Feature screening metrics and model-selection helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

LEAKAGE_COLS = [
    "default3",
    "default6",
    "default12",
    "decision",
    "decline_reason",
    "act3_n_arrears",
    "act3_n_arrears_days",
    "act3_n_good_days",
    "act6_n_arrears",
    "act6_n_arrears_days",
    "act6_n_good_days",
    "act9_n_arrears",
    "act9_n_arrears_days",
    "act9_n_good_days",
    "act12_n_arrears",
    "act12_n_arrears_days",
    "act12_n_good_days",
    "act_cus_active",
]

ID_COLS = ["cid", "aid", "period", "product"]


def get_candidate_features(df: pd.DataFrame, product: str) -> dict[str, list[str]]:
    """List accepted-only model candidates for one product.

    Actions:
    1. Filter to the product and accepted decisions.
    2. Drop identifiers, targets, and post-decision leakage columns.
    3. Split remaining columns into numeric and nominal feature lists.
    """
    work = df.copy()
    work = work[work["product"] == product]
    work = work[work["decision"] == "A"]
    work = work.dropna(subset=["decision", "decline_reason"])

    drop_cols = [c for c in LEAKAGE_COLS + ID_COLS if c in work.columns]
    work = work.drop(columns=drop_cols)

    numeric = [
        c
        for c in work.columns
        if pd.api.types.is_numeric_dtype(work[c]) and not pd.api.types.is_bool_dtype(work[c])
    ]
    nominal = [
        c
        for c in work.columns
        if c not in numeric
        and (
            pd.api.types.is_object_dtype(work[c])
            or pd.api.types.is_string_dtype(work[c])
            or isinstance(work[c].dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(work[c])
        )
    ]
    return {"numeric": numeric, "nominal": nominal, "all": numeric + nominal}


def compute_gini(y_true, y_score) -> float:
    """Compute non-negative Gini from a binary target and continuous score.

    Actions:
    1. Compute ROC AUC.
    2. Flip direction if needed so AUC is at least 0.5.
    3. Return ``2 * AUC - 1`` clipped to ``[0, 1]``.
    """
    auc = roc_auc_score(y_true, y_score)
    auc = max(auc, 1 - auc)
    return float(np.clip(2 * auc - 1, 0.0, 1.0))


def compute_psi(train_series: pd.Series, valid_series: pd.Series, epsilon: float) -> float:
    """Compute population stability index between train and validation bins.

    Actions:
    1. Build normalized category distributions on train and valid.
    2. Align categories and smooth with ``epsilon``.
    3. Sum PSI components and clip at zero.
    """
    train_dist = train_series.value_counts(normalize=True)
    valid_dist = valid_series.value_counts(normalize=True)

    all_bins = train_dist.index.union(valid_dist.index)
    train_pct = train_dist.reindex(all_bins, fill_value=0) + epsilon
    valid_pct = valid_dist.reindex(all_bins, fill_value=0) + epsilon

    psi_components = (train_pct - valid_pct) * np.log(train_pct / valid_pct)
    return float(max(psi_components.sum(), 0.0))


def check_vif(x: pd.DataFrame) -> pd.Series:
    """Compute variance inflation factor for numeric design-matrix columns.

    Actions:
    1. Keep only numeric predictor columns.
    2. Add an intercept column for VIF computation.
    3. Return one VIF value per feature.
    """
    x_num = x.select_dtypes(include=[np.number]).copy()
    x_ = x_num.copy()
    x_.insert(0, "_intercept", 1.0)

    vif_values = {}
    for i, col in enumerate(x_.columns):
        if col == "_intercept":
            continue
        vif_values[col] = variance_inflation_factor(x_.values, i)

    return pd.Series(vif_values, name="vif")


def prescreen_features(
    train_woe: pd.DataFrame,
    valid_woe: pd.DataFrame,
    iv_table: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Apply IV, Gini, PSI, and AR-diff thresholds to every candidate feature.

    Actions:
    1. Evaluate each feature in ``iv_table`` against screening thresholds.
    2. Record train/valid Gini, PSI, and AR difference.
    3. Return keep/reject status with semicolon-joined rejection reasons.
    """
    target = params["target"]
    iv_min = params["iv_min"]
    gini_min = params["gini_min"]
    psi_max = params["psi_max"]
    ar_diff_max = params["ar_diff_max"]
    epsilon = params["woe_epsilon"]

    rows = []
    for _, row in iv_table.iterrows():
        feature = row["feature"]
        iv = row["iv"]
        woe_col = f"{feature}_WOE"
        grp_col = f"{feature}_GRP"
        reasons: list[str] = []

        if iv < iv_min:
            reasons.append(f"IV < {iv_min}")

        gini_train = gini_valid = ar_diff = np.nan

        if woe_col in train_woe.columns:
            gini_train = compute_gini(train_woe[target], train_woe[woe_col])
        else:
            reasons.append("No WOE column (train)")

        if woe_col in valid_woe.columns:
            gini_valid = compute_gini(valid_woe[target], valid_woe[woe_col])
        else:
            reasons.append("No WOE column (valid)")

        if not np.isnan(gini_train) and gini_train < gini_min:
            reasons.append(f"Gini train < {gini_min}")
        if not np.isnan(gini_valid) and gini_valid < gini_min:
            reasons.append(f"Gini valid < {gini_min}")

        if not np.isnan(gini_train) and not np.isnan(gini_valid):
            ar_diff = abs(gini_train - gini_valid)
            if ar_diff > ar_diff_max:
                reasons.append(f"AR-diff > {ar_diff_max}")

        psi = np.nan
        if grp_col in train_woe.columns and grp_col in valid_woe.columns:
            psi = compute_psi(train_woe[grp_col], valid_woe[grp_col], epsilon)
            if psi > psi_max:
                reasons.append(f"PSI > {psi_max}")
        else:
            reasons.append("No GRP column")

        rows.append(
            {
                "feature": feature,
                "iv": iv,
                "gini_train": gini_train,
                "gini_valid": gini_valid,
                "ar_diff": ar_diff,
                "psi": psi,
                "status": "keep" if not reasons else "reject",
                "reason": "; ".join(reasons),
            }
        )

    return pd.DataFrame(rows)


def assess_logit_model(
    model, train_df: pd.DataFrame, valid_df: pd.DataFrame, target: str
) -> dict:
    """Bundle Gini, p-values, VIF, correlation, and beta-sign diagnostics.

    Actions:
    1. Score train/valid with the fitted logit model.
    2. Compute Gini and absolute AR difference.
    3. Summarise p-values, VIF, Pearson off-diagonal max, and beta signs.
    """
    feature_cols = [col for col in model.params.index if col != "const"]

    x_train = sm.add_constant(train_df[feature_cols], has_constant="add")
    x_valid = sm.add_constant(valid_df[feature_cols], has_constant="add")

    pred_train = model.predict(x_train)
    pred_valid = model.predict(x_valid)

    gini_train = compute_gini(train_df[target], pred_train)
    gini_valid = compute_gini(valid_df[target], pred_valid)
    ar_diff = abs(gini_train - gini_valid)

    pvalues = model.pvalues.drop(labels=["const"], errors="ignore").to_dict()
    max_pvalue = max(pvalues.values()) if pvalues else 0.0

    vif_series = check_vif(train_df[feature_cols])
    vif_dict = vif_series.to_dict()
    max_vif = max(vif_dict.values()) if vif_dict else 1.0

    pearson_corr = train_df[feature_cols].corr(method="pearson")
    off_diag = pearson_corr.where(~np.eye(len(pearson_corr), dtype=bool))
    max_pearson_offdiag = (
        float(off_diag.abs().max().max()) if len(feature_cols) > 1 else 0.0
    )

    betas = model.params.drop(labels=["const"], errors="ignore")
    beta_signs = {
        feat: ("positive" if b > 0 else "negative" if b < 0 else "zero")
        for feat, b in betas.items()
    }
    n_negative_betas = sum(1 for s in beta_signs.values() if s == "negative")

    return {
        "gini_train": gini_train,
        "gini_valid": gini_valid,
        "ar_diff": ar_diff,
        "pvalues": pvalues,
        "max_pvalue": max_pvalue,
        "vif": vif_dict,
        "max_vif": max_vif,
        "pearson_corr": pearson_corr,
        "max_pearson_offdiag": max_pearson_offdiag,
        "beta_signs": beta_signs,
        "n_negative_betas": n_negative_betas,
        "n_features": len(feature_cols),
    }


def forward_select_logit(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    features: list[str],
    target: str,
    params: dict,
) -> list[str]:
    """Greedy forward selection of WOE features under model constraints.

    Actions:
    1. Try adding each remaining WOE feature one at a time.
    2. Keep candidates that pass p-value, VIF, Pearson, AR-diff, and beta-sign rules.
    3. Add the best valid-Gini improver until no candidate remains or ``max_features``.
    """
    pvalue_max = params["pvalue_max"]
    vif_max = params["vif_max"]
    pearson_max = params["pearson_max"]
    ar_diff_max = params["ar_diff_model_max"]
    max_features = params["max_features"]
    epsilon = params["epsilon"]

    selected: list[str] = []
    remaining = list(features)
    best_gini_valid = -np.inf

    while remaining and len(selected) < max_features:
        candidates_results = []

        for feat in remaining:
            trial_features = selected + [feat]
            x_train = sm.add_constant(train_df[trial_features], has_constant="add")
            y_train = train_df[target]

            try:
                model = sm.Logit(y_train, x_train).fit(disp=0)
            except Exception:
                continue

            diagnostics = assess_logit_model(model, train_df, valid_df, target)
            if diagnostics["max_pvalue"] > pvalue_max:
                continue
            if diagnostics["max_vif"] > vif_max:
                continue
            if diagnostics["max_pearson_offdiag"] > pearson_max:
                continue
            if diagnostics["ar_diff"] > ar_diff_max:
                continue
            # WOE for default models is typically negatively related to default risk.
            # Reject mixed-sign coefficient sets; all-negative (or all-positive) is fine.

            signs = set(diagnostics["beta_signs"].values())
            if "zero" in signs and len(signs) > 1:
                continue
            if "positive" in signs and "negative" in signs:
                continue

            candidates_results.append(
                (feat, diagnostics["gini_valid"], diagnostics["max_pvalue"])
            )

        if not candidates_results:
            break

        def sort_key(item):
            feat, gini_valid, max_pvalue = item
            original_idx = features.index(feat)
            return (-gini_valid, max_pvalue, original_idx)

        candidates_results.sort(key=sort_key)
        best_feat, best_feat_gini, _ = candidates_results[0]

        if best_feat_gini <= best_gini_valid + epsilon:
            break

        selected.append(best_feat)
        remaining.remove(best_feat)
        best_gini_valid = best_feat_gini

    return selected
