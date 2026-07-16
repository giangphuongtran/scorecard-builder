"""ASB-style RFE + combinatorial logit selection."""

from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from credit_scoring.scorecard.selection import check_vif, compute_gini, get_candidate_features

PREFIXES = ("app", "act")
BLOCKED = ("agr", "ags")
LEAKAGE_EXTRA = {
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
}


def period_gini_stats(y, pred, periods, min_n: int = 20):
    """Min period Gini, slope, count, and detail table."""
    tmp = pd.DataFrame(
        {"y": np.asarray(y), "pred": np.asarray(pred), "period": np.asarray(periods)}
    )
    rows = []
    for period, grp in tmp.groupby("period", sort=True):
        if grp["y"].nunique() < 2 or len(grp) < min_n:
            continue
        rows.append(
            {
                "period": period,
                "n": len(grp),
                "bad_rate": float(grp["y"].mean()),
                "gini": compute_gini(grp["y"], grp["pred"]),
            }
        )
    gini_time = pd.DataFrame(rows)
    if gini_time.empty:
        return np.nan, np.nan, 0, gini_time
    vals = gini_time["gini"].to_numpy(dtype=float)
    slope = float(np.polyfit(np.arange(len(vals)), vals, 1)[0]) if len(vals) >= 2 else np.nan
    return float(vals.min()), slope, int(len(vals)), gini_time


def filter_candidates(
    abt: pd.DataFrame,
    product: str,
    *,
    prefixes: tuple[str, ...] = PREFIXES,
    blocked: tuple[str, ...] = BLOCKED,
    extra_drop: set[str] | None = None,
) -> list[str]:
    """Candidate raw features for one product / model family.

    Keep ``app*`` / ``act*``; always block ``agr*`` / ``ags*`` (strategy still
    uses agr12 for bad_customer on the ABT — not as scorecard inputs).
    """
    extra_drop = extra_drop or LEAKAGE_EXTRA
    # For pr/cross, candidacy product is still "ins" on the ABT filter.
    cand_product = "ins" if product in ("pr", "cross") else product
    cand = get_candidate_features(abt, cand_product)
    out = []
    for f in cand["all"]:
        fl = f.lower()
        if f in extra_drop:
            continue
        if fl.startswith(blocked):
            continue
        if f[:3].lower() in prefixes:
            out.append(f)
    return sorted(set(out))


def woe_keep_list(screen: pd.DataFrame, train_w: pd.DataFrame) -> list[str]:
    """Map prescreen keep status to *_WOE column names present in train."""
    keep = screen.loc[screen["status"] == "keep", "feature"].tolist()
    out = []
    for f in keep:
        if f.endswith("_WOE") and f in train_w.columns:
            out.append(f)
        elif f"{f}_WOE" in train_w.columns:
            out.append(f"{f}_WOE")
    return sorted(set(out))


def uni_table(
    train_w: pd.DataFrame,
    valid_w: pd.DataFrame,
    woe_cols: list[str],
    target: str,
    time_col: str = "period",
) -> pd.DataFrame:
    """Univariate Gini + period stability table."""
    cols = [
        "variable",
        "woe",
        "gini_train",
        "gini_valid",
        "min_period_gini",
        "period_gini_slope",
        "n_periods",
    ]
    rows = []
    for woe in woe_cols:
        raw = woe[: -len("_WOE")] if woe.endswith("_WOE") else woe
        min_g, slope, nper, _ = period_gini_stats(
            valid_w[target].values, valid_w[woe].values, valid_w[time_col].values
        )
        rows.append(
            {
                "variable": raw,
                "woe": woe,
                "gini_train": compute_gini(train_w[target], train_w[woe]),
                "gini_valid": compute_gini(valid_w[target], valid_w[woe]),
                "min_period_gini": min_g,
                "period_gini_slope": slope,
                "n_periods": nper,
            }
        )
    out = pd.DataFrame(rows, columns=cols)
    if out.empty:
        return out
    return out.sort_values(
        ["min_period_gini", "period_gini_slope", "gini_valid"], ascending=False
    ).reset_index(drop=True)


def assess_combo(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_set,
    target: str,
    time_col: str = "period",
    *,
    vif_max: float = 3.0,
    pvalue_max: float = 0.05,
    gini_floor: float = 0.45,
    require_period_gate: bool = False,
    require_negative_betas: bool = True,
) -> dict:
    """Fit logit on a WOE feature combo and return diagnostics."""
    cols = list(feature_set)
    x_train = sm.add_constant(train_df[cols], has_constant="add")
    y_train = train_df[target]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.Logit(y_train, x_train).fit(disp=0, maxiter=200)
    feature_cols = [c for c in model.params.index if c != "const"]
    x_valid = sm.add_constant(valid_df[feature_cols], has_constant="add")
    pred_train = model.predict(x_train)
    pred_valid = model.predict(x_valid)
    gini_train = compute_gini(y_train, pred_train)
    gini_valid = compute_gini(valid_df[target], pred_valid)
    pvalues = model.pvalues.drop(labels=["const"], errors="ignore")
    max_pvalue = float(pvalues.max()) if len(pvalues) else 0.0
    vif_series = check_vif(train_df[feature_cols])
    max_vif = float(vif_series.max()) if len(vif_series) else 1.0
    betas = model.params.drop(labels=["const"], errors="ignore")
    nnegative = int((betas < 0).sum())
    min_g, slope, nper, gini_time = period_gini_stats(
        valid_df[target].values, pred_valid.values, valid_df[time_col].values
    )
    sign_ok = (nnegative == len(feature_cols)) if require_negative_betas else True
    soft_ok = (
        gini_train >= gini_floor
        and gini_valid >= gini_floor
        and max_vif < vif_max
        and max_pvalue <= pvalue_max
        and sign_ok
    )
    gate_b = soft_ok and (
        (not require_period_gate)
        or (min_g >= 0.60 and (not np.isnan(slope) and slope > 0.0))
    )
    return {
        "Variables": ",".join(feature_cols),
        "n_features": len(feature_cols),
        "nnegative_betas": nnegative,
        "max_pvalue": max_pvalue,
        "max_vif": max_vif,
        "gini_train": gini_train,
        "gini_valid": gini_valid,
        "ar_diff": abs(gini_train - gini_valid),
        "min_period_gini": min_g,
        "period_gini_slope": slope,
        "n_gini_periods": nper,
        "gate_b_pass": bool(gate_b),
        "soft_ok": bool(soft_ok),
        "model": model,
        "gini_time": gini_time,
        "betas": betas,
        "feature_cols": feature_cols,
        "score": gini_valid - 0.5 * abs(gini_train - gini_valid)
        + 0.1 * (0.0 if np.isnan(min_g) else min_g),
    }


def shortlist_rfe(
    train_w: pd.DataFrame,
    uni: pd.DataFrame,
    woe_cols: list[str],
    target: str,
    n: int = 12,
    seed: int = 1234,
) -> list[str]:
    """RFE shortlist from univariate-preferred pool."""
    preferred = uni[(uni["period_gini_slope"] > 0) & (uni["gini_valid"] >= 0.05)].head(n)
    if len(preferred) < max(4, n // 2):
        preferred = uni.head(n)
    pool = preferred["woe"].tolist()
    use = [w for w in pool if w in train_w.columns]
    if len(use) < 4:
        use = [w for w in woe_cols if w in train_w.columns][:n]
    if not use:
        return []
    # RFE needs ≥2 features; with a tiny pool just return what we have.
    if len(use) < 2:
        return use[:n]
    X = train_w[use].astype(float)
    y = train_w[target].astype(int)
    n_sel = min(n, len(use))
    rfe = RFE(LogisticRegression(max_iter=800, random_state=seed), n_features_to_select=n_sel)
    rfe.fit(X, y)
    short = [c for c, k in zip(use, rfe.support_) if k]
    for w in uni.head(5)["woe"]:
        if w not in short and w in train_w.columns:
            short.append(w)
    return short[:n]


def search_combos(
    train_w: pd.DataFrame,
    valid_w: pd.DataFrame,
    shortlisted: list[str],
    target: str,
    sizes=(5, 6),
    *,
    time_col: str = "period",
    pvalue_max: float = 0.05,
    vif_max: float = 3.0,
    gini_floor: float = 0.45,
    require_negative_betas: bool = True,
    require_period_gate: bool = False,
) -> tuple[pd.DataFrame, dict | None]:
    """Exhaustive combinatorial logit search over shortlist sizes."""
    rows = []
    best = None
    for k in sizes:
        if k > len(shortlisted):
            continue
        for combo in combinations(shortlisted, k):
            try:
                res = assess_combo(
                    train_w,
                    valid_w,
                    combo,
                    target,
                    time_col=time_col,
                    vif_max=vif_max,
                    pvalue_max=pvalue_max,
                    gini_floor=gini_floor,
                    require_period_gate=require_period_gate,
                    require_negative_betas=require_negative_betas,
                )
            except Exception:
                continue
            row = {
                k_: v
                for k_, v in res.items()
                if k_ not in ("model", "gini_time", "betas", "feature_cols")
            }
            rows.append(row)
            if res["soft_ok"] and (
                best is None
                or res["score"] > best["score"]
                or (
                    res["gate_b_pass"]
                    and not best.get("gate_b_pass", False)
                )
            ):
                best = res
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(
            ["soft_ok", "score", "gini_valid"], ascending=[False, False, False]
        )
    return table, best


def filter_submodel_list(
    model_list: pd.DataFrame,
    *,
    pvalue_max: float = 0.05,
    vif_max: float = 3.0,
    require_negative_betas: bool = True,
) -> pd.DataFrame:
    """Filter Model_list like ASB subModel_list."""
    if model_list is None or model_list.empty:
        return pd.DataFrame()
    mask = (
        (model_list["max_pvalue"] <= pvalue_max)
        & (model_list["max_vif"] <= vif_max)
        & (model_list.get("soft_ok", True))
    )
    if require_negative_betas:
        mask = mask & (model_list["nnegative_betas"] == model_list["n_features"])
    out = model_list.loc[mask].copy()
    return out.sort_values(["score", "gini_valid"], ascending=[False, False])
