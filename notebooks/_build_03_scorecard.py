"""One-off script to rebuild 03_scorecard.ipynb with ASB workbench structure."""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "03_scorecard.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": text.splitlines(keepends=True),
    }


INTRO = """# Phase 4 — Scorecard ASB workbench (PD Ins + PD Css)

Notebook-only workflow mirroring `ASB_step_by_step.ipynb`: tune bins, review stability in charts,
then fit scorecard. **No Excel** — tables via `display()`, charts via matplotlib.

Work one product at a time (`PRODUCT`). Freeze bins before §3.

## Checklist
- [ ] §0 `SCORECARD_PARAMS` + `PRODUCT`
- [ ] §1 load + partition + slice
- [ ] §2 tuning loop until exit gate passes
- [ ] §3 model fit
- [ ] §4 scorecard + calibration
- [ ] §5 Gini over time
- [ ] §6 save parquet artifacts
- [ ] Repeat for `css`

| Prof reference | Notebook |
|----------------|----------|
| Big_scorecard.xlsx | `big_scorecard` + plots |
| Gini_vars.xlsx | `gini_vars` |
| Variable_report.xlsx | `variable_report` + `plot_bin_stability` |
| Model_report.xlsx | `model_report` + `display_model_report` |
"""

SECTION0 = '''
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor

pd.set_option("display.max_columns", None)

assert Path("../data/04_feature/abt_app.parquet").exists()
assert Path("../data/04_feature/decisions.parquet").exists()

PRODUCT = "ins"

SCORECARD_PARAMS = {
    "target": "default12",
    "time_col": "period",
    "id_col": "aid",
    "train_end_period": "198412",
    "valid_start_period": "198501",
    "ncategories_int": 4,
    "minimum_share_int": 0.03,
    "symbol_missing": "Missing",
    "symbol_other": "<OTHERS>",
    "ncategories_nom": 4,
    "iv_min": 0.02,
    "gini_min": 0.05,
    "psi_max": 0.1,
    "psi_tar_max": 0.1,
    "delta_gini_max": 0.2,
    "ar_diff_max": 0.20,
    "pvalue_max": 0.05,
    "vif_max": 5.0,
    "pearson_max": 0.7,
    "ar_diff_model_max": 0.10,
    "factor": 20 / np.log(2),
    "offset": 600 - (20 / np.log(2)) * np.log(50),
    "woe_epsilon": 1e-4,
    "tree_random_state": 1234,
    "max_features": 12,
    "epsilon": 1e-4,
    "min_bin_n_period": 10,
    "min_bin_n_total": 30,
    "max_bad_rate_swing": 0.15,
    "min_periods_for_stability": 6,
}
'''

SECTION1_DEFS = r'''
LEAKAGE_COLS = [
    "default3", "default6", "default12", "decision", "decline_reason",
    "act3_n_arrears", "act3_n_arrears_days", "act3_n_good_days",
    "act6_n_arrears", "act6_n_arrears_days", "act6_n_good_days",
    "act9_n_arrears", "act9_n_arrears_days", "act9_n_good_days",
    "act12_n_arrears", "act12_n_arrears_days", "act12_n_good_days",
    "act_cus_active",
]
ID_COLS = ["cid", "aid", "period", "product"]


def partition_abt(df, train_end, valid_start, time_col="period"):
    work = df.copy()
    df_train = work[work[time_col] <= train_end]
    df_valid = work[work[time_col] >= valid_start]
    return df_train, df_valid


def prepare_target(df, target="default12"):
    out = df.copy()
    out[target] = out[target].map({".i": 0, ".d": 0, 0: 0, 1: 1})
    out = out.dropna(subset=[target])
    out[target] = out[target].astype(int)
    return out


def load_and_prepare_abt(abt_path, decisions_path):
    abt = pd.read_parquet(abt_path)
    decisions = pd.read_parquet(decisions_path)
    merged = abt.merge(
        decisions[["aid", "decision", "decline_reason"]],
        on="aid",
        how="left",
    )
    return prepare_target(merged)


def slice_product(df, product, decision="A"):
    return df[(df["product"] == product) & (df["decision"] == decision)].copy()


def get_candidate_features(df, product):
    work = slice_product(df, product)
    work = work.dropna(subset=["decision", "decline_reason"])
    drop_cols = [c for c in LEAKAGE_COLS + ID_COLS if c in work.columns]
    work = work.drop(columns=drop_cols)
    numeric = [
        c for c in work.columns
        if pd.api.types.is_numeric_dtype(work[c]) and not pd.api.types.is_bool_dtype(work[c])
    ]
    nominal = [
        c for c in work.columns
        if c not in numeric
        and (
            pd.api.types.is_object_dtype(work[c])
            or pd.api.types.is_string_dtype(work[c])
            or isinstance(work[c].dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(work[c])
        )
    ]
    return {"numeric": numeric, "nominal": nominal, "all": numeric + nominal}
'''

SECTION1_EXEC = '''
df_model = load_and_prepare_abt(
    "../data/04_feature/abt_app.parquet",
    "../data/04_feature/decisions.parquet",
)
assert df_model["aid"].is_unique
print(df_model.groupby(["product", "decision"])["default12"].mean())
print("rows:", len(df_model), "bad rate:", f"{df_model['default12'].mean():.2%}")

cand = get_candidate_features(df_model, PRODUCT)
print(f"{PRODUCT} candidates:", len(cand["all"]))

df_accepted = df_model[df_model["decision"] == "A"].copy()
df_train, df_valid = partition_abt(
    df_accepted,
    SCORECARD_PARAMS["train_end_period"],
    SCORECARD_PARAMS["valid_start_period"],
    time_col=SCORECARD_PARAMS["time_col"],
)
train_product = slice_product(df_train, PRODUCT)
valid_product = slice_product(df_valid, PRODUCT)

for name, d in [("train", train_product), ("valid", valid_product)]:
    print(
        name,
        d["period"].min(), "→", d["period"].max(),
        "bad_rate", f"{d['default12'].mean():.2%}",
        "n", len(d),
    )
'''

SECTION2_BINNING = r'''
def _bin_params(params):
    return {
        "other_label": params.get("symbol_other", "<OTHERS>"),
        "missing_label": params.get("symbol_missing", "Missing"),
        "max_bins": params.get("ncategories_int", 5),
        "min_bin_size": params.get("minimum_share_int", 0.05),
        "tree_random_state": params.get("tree_random_state", 1234),
        "rare_threshold": params.get("rare_threshold", 0.02),
        "max_groups": params.get("ncategories_nom", 4),
        "nominal_int_threshold": params.get("nominal_int_threshold", 10),
    }


def fit_bin_numeric(train, feature, target, params):
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


def fit_bin_nominal(train, feature, target, params):
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
    groups = [[cat] for cat in stats.index]
    while len(groups) > max_groups:
        best_i, best_n = 0, float("inf")
        for i in range(len(groups) - 1):
            combined = sum(stats.loc[c, "n"] for c in groups[i] + groups[i + 1])
            if combined < best_n:
                best_n, best_i = combined, i
        groups[best_i] += groups[best_i + 1]
        groups.pop(best_i + 1)
    category_map = {}
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


def fit_binning_maps(train, features, target, params):
    cfg = _bin_params(params)
    nominal_int_threshold = cfg["nominal_int_threshold"]
    binning_maps = {}
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


def apply_bins(df, binning_maps):
    base = df.drop(columns=[c for c in df.columns if c.endswith("_GRP")], errors="ignore")
    new_cols = {}
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
'''

SECTION2_WOE = r'''
def build_woe_table(df_binned, feature_grp, target, epsilon):
    total_good = (df_binned[target] == 0).sum()
    total_bad = (df_binned[target] == 1).sum()
    grouped = (
        df_binned.groupby(feature_grp, observed=False)[target]
        .agg(n="count", bads="sum")
        .reset_index()
        .rename(columns={feature_grp: "bin"})
    )
    grouped["goods"] = grouped["n"] - grouped["bads"]
    grouped["dist_good"] = (grouped["goods"] + epsilon) / (total_good + epsilon)
    grouped["dist_bad"] = (grouped["bads"] + epsilon) / (total_bad + epsilon)
    grouped["woe"] = np.log(grouped["dist_good"] / grouped["dist_bad"])
    grouped["iv_component"] = (grouped["dist_good"] - grouped["dist_bad"]) * grouped["woe"]
    grouped["bad_rate"] = grouped["bads"] / grouped["n"]
    return grouped


def build_woe_maps(df_binned, grp_cols, target, epsilon):
    return {grp: build_woe_table(df_binned, grp, target, epsilon) for grp in grp_cols}


def compute_iv(woe_table):
    return float(max(woe_table["iv_component"].sum(), 0.0))


def build_iv_table(woe_maps):
    rows = []
    for grp, table in woe_maps.items():
        feature = grp[: -len("_GRP")]
        rows.append({"feature": feature, "iv": compute_iv(table)})
    return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)


def encode_woe(df_binned, woe_maps):
    base = df_binned.drop(columns=[c for c in df_binned.columns if c.endswith("_WOE")], errors="ignore")
    new_cols = {}
    for grp_col, woe_table in woe_maps.items():
        feature = grp_col[: -len("_GRP")]
        woe_col = f"{feature}_WOE"
        bin_to_woe = dict(zip(woe_table["bin"], woe_table["woe"]))
        new_cols[woe_col] = base[grp_col].map(bin_to_woe).fillna(0.0)
    woe_df = pd.DataFrame(new_cols, index=base.index)
    return pd.concat([base, woe_df], axis=1).copy()


def _bin_condition(binning_maps, feature, bin_label):
    spec = binning_maps.get(feature, {})
    if spec.get("type") == "numeric":
        return str(bin_label)
    if spec.get("type") == "nominal":
        inv = {v: k for k, v in spec.get("category_map", {}).items()}
        return inv.get(bin_label, str(bin_label))
    return str(bin_label)


def build_big_scorecard(train_binned, valid_binned, woe_maps, binning_maps, target, params):
    eps = params["woe_epsilon"]
    rows = []
    features = [grp[: -len("_GRP")] for grp in woe_maps]
    n_train = len(train_binned)
    n_valid = len(valid_binned)
    sum_bad_train = train_binned[target].sum()
    sum_bad_valid = valid_binned[target].sum()

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
        tr["share_train"] = tr["n_train"] / n_train
        tr["bad_share_train"] = tr["bads_train"] / max(sum_bad_train, 1)

        va = (
            valid_binned.groupby(grp_col, observed=False)[target]
            .agg(n_valid="count", bads_valid="sum")
            .reset_index()
            .rename(columns={grp_col: "bin"})
        )
        va["goods_valid"] = va["n_valid"] - va["bads_valid"]
        va["bad_rate_valid"] = va["bads_valid"] / va["n_valid"]
        va["share_valid"] = va["n_valid"] / max(n_valid, 1)
        va["bad_share_valid"] = va["bads_valid"] / max(sum_bad_valid, 1)

        merged = tr.merge(va, on="bin", how="outer")
        woe_part = woe_tbl[["bin", "woe", "iv_component", "bad_rate"]].rename(
            columns={"bad_rate": "bad_rate_woe_train"}
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

    out = pd.concat(rows, ignore_index=True)
    col_order = [
        "variable", "bin", "condition",
        "n_train", "bads_train", "goods_train", "bad_rate_train", "share_train",
        "n_valid", "bads_valid", "goods_valid", "bad_rate_valid", "share_valid",
        "woe", "iv_component", "psi_bin", "psi_bad",
    ]
    return out[[c for c in col_order if c in out.columns]]


def build_gini_vars_table(
    train_binned, valid_binned, train_woe, valid_woe, big_scorecard, cand_features, target, params
):
    rows = []
    for feat in cand_features:
        woe_col = f"{feat}_WOE"
        gini_train = compute_gini(train_woe[target], train_woe[woe_col]) if woe_col in train_woe else np.nan
        gini_valid = compute_gini(valid_woe[target], valid_woe[woe_col]) if woe_col in valid_woe else np.nan
        delta = np.nan
        if gini_train and gini_train > 0:
            delta = abs(gini_train - gini_valid) / gini_train
        sub = big_scorecard[big_scorecard["variable"] == feat]
        iv = sub["iv_component"].sum() if len(sub) else 0.0
        psi = sub["psi_bin"].sum() if len(sub) else np.nan
        psi_tar = sub["psi_bad"].sum() if len(sub) else np.nan
        pm = train_binned[feat].isnull().mean() if feat in train_binned else np.nan
        nuniq = train_binned[feat].nunique(dropna=True) if feat in train_binned else np.nan
        rows.append({
            "variable": feat,
            "gini_train": gini_train,
            "gini_valid": gini_valid,
            "delta_gini": delta,
            "iv": iv,
            "psi": psi,
            "psi_tar": psi_tar,
            "percent_missing": pm,
            "count_unique": nuniq,
        })
    return pd.DataFrame(rows).sort_values("gini_train", ascending=False).reset_index(drop=True)


def apply_prof_prescreen(gini_vars, params):
    return gini_vars.query(
        "gini_train > @params['gini_min']"
        " and delta_gini < @params['delta_gini_max']"
        " and psi_tar < @params['psi_tar_max']"
        " and psi < @params['psi_max']"
    )["variable"].tolist()
'''

SECTION2_SCREENING = r'''
def compute_gini(y_true, y_score):
    auc = roc_auc_score(y_true, y_score)
    auc = max(auc, 1 - auc)
    return float(np.clip(2 * auc - 1, 0.0, 1.0))


def compute_psi(train_series, valid_series, epsilon):
    train_dist = train_series.value_counts(normalize=True)
    valid_dist = valid_series.value_counts(normalize=True)
    all_bins = train_dist.index.union(valid_dist.index)
    train_pct = train_dist.reindex(all_bins, fill_value=0) + epsilon
    valid_pct = valid_dist.reindex(all_bins, fill_value=0) + epsilon
    psi_components = (train_pct - valid_pct) * np.log(train_pct / valid_pct)
    return float(max(psi_components.sum(), 0.0))


def check_vif(x):
    x_num = x.select_dtypes(include=[np.number]).copy()
    x_ = x_num.copy()
    x_.insert(0, "_intercept", 1.0)
    vif_values = {}
    for i, col in enumerate(x_.columns):
        if col == "_intercept":
            continue
        vif_values[col] = variance_inflation_factor(x_.values, i)
    return pd.Series(vif_values, name="vif")


def prescreen_features(train_woe, valid_woe, iv_table, params):
    target = params["target"]
    rows = []
    for _, row in iv_table.iterrows():
        feature = row["feature"]
        iv = row["iv"]
        woe_col = f"{feature}_WOE"
        grp_col = f"{feature}_GRP"
        reasons = []
        if iv < params["iv_min"]:
            reasons.append(f"IV < {params['iv_min']}")
        gini_train = gini_valid = ar_diff = np.nan
        if woe_col in train_woe.columns:
            gini_train = compute_gini(train_woe[target], train_woe[woe_col])
        else:
            reasons.append("No WOE column (train)")
        if woe_col in valid_woe.columns:
            gini_valid = compute_gini(valid_woe[target], valid_woe[woe_col])
        else:
            reasons.append("No WOE column (valid)")
        if not np.isnan(gini_train) and gini_train < params["gini_min"]:
            reasons.append(f"Gini train < {params['gini_min']}")
        if not np.isnan(gini_valid) and gini_valid < params["gini_min"]:
            reasons.append(f"Gini valid < {params['gini_min']}")
        if not np.isnan(gini_train) and not np.isnan(gini_valid):
            ar_diff = abs(gini_train - gini_valid)
            if ar_diff > params["ar_diff_max"]:
                reasons.append(f"AR-diff > {params['ar_diff_max']}")
        psi = np.nan
        if grp_col in train_woe.columns and grp_col in valid_woe.columns:
            psi = compute_psi(train_woe[grp_col], valid_woe[grp_col], params["woe_epsilon"])
            if psi > params["psi_max"]:
                reasons.append(f"PSI > {params['psi_max']}")
        else:
            reasons.append("No GRP column")
        rows.append({
            "feature": feature,
            "iv": iv,
            "gini_train": gini_train,
            "gini_valid": gini_valid,
            "ar_diff": ar_diff,
            "psi": psi,
            "status": "keep" if not reasons else "reject",
            "reason": "; ".join(reasons),
        })
    return pd.DataFrame(rows)
'''

SECTION2_STABILITY = r'''
def bin_bad_rate_by_period(df_binned, feature, target, time_col="period"):
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


def flag_unstable_bins(period_table, params):
    rows = []
    for (var, bin_label), sub in period_table.groupby(["variable", "bin"]):
        n_periods = sub["period"].nunique()
        min_n_period = sub["n"].min()
        total_n = sub["n"].sum()
        min_br = sub["bad_rate"].min()
        max_br = sub["bad_rate"].max()
        swing = max_br - min_br
        reasons = []
        if swing > params["max_bad_rate_swing"]:
            reasons.append(f"swing>{params['max_bad_rate_swing']}")
        if min_n_period < params["min_bin_n_period"]:
            reasons.append(f"min_n_period<{params['min_bin_n_period']}")
        if total_n < params["min_bin_n_total"]:
            reasons.append(f"total_n<{params['min_bin_n_total']}")
        rows.append({
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
        })
    return pd.DataFrame(rows)


def bin_stability_report(df_binned, features, target, time_col, params):
    period_tables = {}
    parts = []
    for feat in features:
        pt = bin_bad_rate_by_period(df_binned, feat, target, time_col)
        period_tables[feat] = pt
        parts.append(pt)
    combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    flags = flag_unstable_bins(combined, params) if len(combined) else pd.DataFrame()
    return period_tables, flags


def build_variable_report(train_binned, big_scorecard, kept, target, time_col, params):
    report = {}
    period_tables, _ = bin_stability_report(
        train_binned, kept, target, time_col, params
    )
    static_cols = [
        "bin", "condition", "bad_rate_train", "share_train",
        "n_train", "bads_train", "goods_train",
    ]
    for feat in kept:
        static = big_scorecard.loc[
            big_scorecard["variable"] == feat, static_cols
        ].copy()
        period = period_tables.get(feat, pd.DataFrame())
        report[feat] = {"static": static, "period": period}
    return report
'''

SECTION2_VIZ = r'''
def plot_bin_stability(period_table, feature, train_end_period=None, min_n=None):
    if period_table.empty:
        return
    min_n = min_n or SCORECARD_PARAMS["min_bin_n_period"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Bin stability — {feature}")
    for bin_label, sub in period_table.groupby("bin"):
        sub = sub.sort_values("period")
        if sub["n"].min() < min_n:
            continue
        axes[0].plot(sub["period"], sub["bad_rate"], marker="o", label=str(bin_label))
        axes[1].plot(sub["period"], sub["share"], marker="o", label=str(bin_label))
    if train_end_period:
        for ax in axes:
            ax.axvline(train_end_period, color="gray", linestyle="--", alpha=0.7)
    axes[0].set_title("Bad rate over time")
    axes[0].set_ylabel("bad_rate")
    axes[1].set_title("Bin share over time")
    axes[1].set_ylabel("share")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=7, loc="best")
    plt.tight_layout()
    plt.show()


def plot_woe_ladder(woe_maps, big_scorecard, feature):
    grp = f"{feature}_GRP"
    sub = big_scorecard[big_scorecard["variable"] == feature].copy()
    if sub.empty:
        return
    sub = sub.sort_values("bad_rate_train")
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(range(len(sub)), sub["bad_rate_train"].values, alpha=0.7, label="bad_rate_train")
    ax1.set_xticks(range(len(sub)))
    ax1.set_xticklabels(sub["bin"].astype(str), rotation=45, ha="right")
    ax1.set_ylabel("bad_rate_train")
    ax2 = ax1.twinx()
    ax2.plot(range(len(sub)), sub["woe"].values, color="red", marker="o", label="WOE")
    ax2.set_ylabel("WOE")
    plt.title(f"WOE ladder — {feature}")
    plt.tight_layout()
    plt.show()


def plot_bin_bad_rate_bar(big_scorecard, variable):
    sub = big_scorecard[big_scorecard["variable"] == variable].sort_values("bad_rate_train")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(sub["bin"].astype(str), sub["bad_rate_train"])
    ax.set_title(f"Train bad rate by bin — {variable}")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show()


def display_variable_report(variable_report, feature):
    block = variable_report[feature]
    print(f"### {feature}")
    print("Static bins (Big_scorecard left block)")
    display(block["static"])
    print("Period × bin (Variable_report right block)")
    display(block["period"])
    plot_bin_stability(
        block["period"],
        feature,
        train_end_period=SCORECARD_PARAMS["train_end_period"],
    )
'''

SECTION2_EXEC = '''
# §2 — Tuning loop (re-run after changing binning params until exit gate passes)

binning_maps = fit_binning_maps(
    train_product, cand["all"], SCORECARD_PARAMS["target"], SCORECARD_PARAMS
)
train_binned = apply_bins(train_product, binning_maps)
valid_binned = apply_bins(valid_product, binning_maps)

grp_cols = [f"{f}_GRP" for f in cand["all"] if f"{f}_GRP" in train_binned.columns]
woe_maps = build_woe_maps(
    train_binned, grp_cols, SCORECARD_PARAMS["target"], SCORECARD_PARAMS["woe_epsilon"]
)
iv_table = build_iv_table(woe_maps)
train_woe = encode_woe(train_binned, woe_maps)
valid_woe = encode_woe(valid_binned, woe_maps)

big_scorecard = build_big_scorecard(
    train_binned, valid_binned, woe_maps, binning_maps,
    SCORECARD_PARAMS["target"], SCORECARD_PARAMS,
)
print("big_scorecard rows:", len(big_scorecard))
display(big_scorecard.head(10))

sample_feat = cand["all"][0] if cand["all"] else None
if sample_feat:
    plot_woe_ladder(woe_maps, big_scorecard, sample_feat)
    plot_bin_bad_rate_bar(big_scorecard, sample_feat)

gini_vars = build_gini_vars_table(
    train_binned, valid_binned, train_woe, valid_woe,
    big_scorecard, cand["all"], SCORECARD_PARAMS["target"], SCORECARD_PARAMS,
)
display(gini_vars.head(15))

kept = apply_prof_prescreen(gini_vars, SCORECARD_PARAMS)
rejected = gini_vars[~gini_vars["variable"].isin(kept)]
print(f"kept: {len(kept)} of {len(gini_vars)}")
display(rejected.head(20))

screen_report = prescreen_features(train_woe, valid_woe, iv_table, SCORECARD_PARAMS)

variable_report = build_variable_report(
    train_binned, big_scorecard, kept,
    SCORECARD_PARAMS["target"], SCORECARD_PARAMS["time_col"], SCORECARD_PARAMS,
)
period_tables, stability_flags = bin_stability_report(
    train_binned, kept, SCORECARD_PARAMS["target"],
    SCORECARD_PARAMS["time_col"], SCORECARD_PARAMS,
)

for feat in kept:
    display_variable_report(variable_report, feat)
    unstable = stability_flags.query("variable == @feat & flag_unstable")
    if len(unstable):
        print("unstable bins:")
        display(unstable[["bin", "bad_rate_swing", "flag_reason"]])
'''

SECTION2_GATE = '''
n_unstable = int(stability_flags["flag_unstable"].sum()) if len(stability_flags) else 0
thin_bins = big_scorecard.loc[
    big_scorecard["variable"].isin(kept)
    & (big_scorecard["share_train"] < SCORECARD_PARAMS["minimum_share_int"])
]
print(f"§2 check — PRODUCT={PRODUCT} kept={len(kept)} unstable_bins={n_unstable} thin_bins={len(thin_bins)}")
if n_unstable == 0 and len(kept) >= 10 and thin_bins.empty:
    print("§2 PASS — proceed to §3")
else:
    print("§2 NOT PASS — tune SCORECARD_PARAMS and re-run §2 block above")
'''

SECTION3_DEFS = r'''
def assess_logit_model(model, train_df, valid_df, target):
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
    max_pearson_offdiag = float(off_diag.abs().max().max()) if len(feature_cols) > 1 else 0.0
    betas = model.params.drop(labels=["const"], errors="ignore")
    beta_signs = {
        feat: ("positive" if b > 0 else "negative" if b < 0 else "zero")
        for feat, b in betas.items()
    }
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
        "n_negative_betas": sum(1 for s in beta_signs.values() if s == "negative"),
        "n_features": len(feature_cols),
    }


def forward_select_logit(train_df, valid_df, features, target, params):
    pvalue_max = params["pvalue_max"]
    vif_max = params["vif_max"]
    pearson_max = params["pearson_max"]
    ar_diff_max = params["ar_diff_model_max"]
    max_features = params["max_features"]
    epsilon = params["epsilon"]
    selected = []
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
            signs = set(diagnostics["beta_signs"].values())
            if "zero" in signs and len(signs) > 1:
                continue
            if "positive" in signs and "negative" in signs:
                continue
            candidates_results.append((feat, diagnostics["gini_valid"], diagnostics["max_pvalue"]))
        if not candidates_results:
            break
        candidates_results.sort(key=lambda item: (-item[1], item[2], features.index(item[0])))
        best_feat, best_feat_gini, _ = candidates_results[0]
        if best_feat_gini <= best_gini_valid + epsilon:
            break
        selected.append(best_feat)
        remaining.remove(best_feat)
        best_gini_valid = best_feat_gini
    return selected


def train_pd_model(product, train_df, valid_df, params, *, candidate_woe_features=None, woe_maps=None):
    target = params["target"]
    train_subset = slice_product(train_df, product)
    valid_subset = slice_product(valid_df, product)
    if candidate_woe_features is None:
        candidate_woe_features = [c for c in train_subset.columns if c.endswith("_WOE")]
    selected = forward_select_logit(
        train_subset, valid_subset, candidate_woe_features, target, params
    )
    if not selected and candidate_woe_features:
        selected = [candidate_woe_features[0]]
    x_train = sm.add_constant(train_subset[selected], has_constant="add")
    y_train = train_subset[target]
    final_model = sm.Logit(y_train, x_train).fit(disp=0)
    metrics = assess_logit_model(final_model, train_subset, valid_subset, target)
    woe_tables = {}
    if woe_maps is not None:
        for woe_col in selected:
            raw_feat = woe_col[: -len("_WOE")]
            grp_key = f"{raw_feat}_GRP"
            if grp_key in woe_maps:
                woe_tables[woe_col] = woe_maps[grp_key]
    return {
        "product": product,
        "features": selected,
        "model": final_model,
        "metrics": metrics,
        "train_subset": train_subset,
        "valid_subset": valid_subset,
        "woe_tables": woe_tables,
        "id_col": params.get("id_col", "aid"),
    }


def summarize_model_effects(model_package):
    model = model_package["model"]
    train_df = model_package["train_subset"]
    feature_cols = model_package["features"]
    vif_series = check_vif(train_df[feature_cols])
    rows = []
    for feat in feature_cols:
        rows.append({
            "feature": feat,
            "beta": model.params[feat],
            "pvalue": model.pvalues[feat],
            "vif": vif_series.get(feat, np.nan),
        })
    return pd.DataFrame(rows).assign(abs_beta=lambda d: d["beta"].abs()).sort_values(
        "abs_beta", ascending=False
    ).drop(columns=["abs_beta"])
'''

SECTION3_EXEC = '''
candidate_woe = [f"{f}_WOE" for f in kept if f"{f}_WOE" in train_woe.columns]
model_package = train_pd_model(
    PRODUCT,
    train_woe,
    valid_woe,
    SCORECARD_PARAMS,
    candidate_woe_features=candidate_woe,
    woe_maps=woe_maps,
)
print("metrics:", model_package["metrics"])
effects = summarize_model_effects(model_package)
display(effects)
'''

SECTION4_DEFS = r'''
def scale_scorecard(model_package, factor, offset):
    model = model_package["model"]
    features = model_package["features"]
    intercept = model.params.get("const", 0.0)
    base_points = offset - factor * intercept
    rows = []
    for feat in features:
        beta = model.params[feat]
        woe_table = model_package.get("woe_tables", {}).get(feat)
        if woe_table is None:
            rows.append({"feature": feat, "bin": "<ALL>", "woe": np.nan, "points": np.nan})
            continue
        for _, row in woe_table.iterrows():
            rows.append({
                "feature": feat,
                "bin": row["bin"],
                "woe": row["woe"],
                "points": -beta * row["woe"] * factor,
            })
    points_table = pd.DataFrame(rows)
    points_table.attrs["base_points"] = base_points
    points_table.attrs["intercept"] = intercept
    points_table.attrs["factor"] = factor
    points_table.attrs["offset"] = offset
    return points_table


def score_applicants(df_woe, model_package, points_table=None):
    features = model_package["features"]
    id_col = model_package.get("id_col", "aid")
    out = pd.DataFrame({id_col: df_woe[id_col].values})
    if points_table is not None:
        base_points = points_table.attrs.get("base_points", 0.0)
        total = pd.Series(base_points, index=df_woe.index, dtype=float)
        for feat in features:
            raw_feat = feat[: -len("_WOE")]
            grp_col = f"{raw_feat}_GRP"
            feat_points_map = (
                points_table[points_table["feature"] == feat]
                .set_index("bin")["points"]
                .to_dict()
            )
            contrib = df_woe[grp_col].map(feat_points_map).fillna(0.0)
            out[f"{feat}_points"] = contrib.values
            total = total + contrib
        out["score"] = total.values
    else:
        model = model_package["model"]
        x = df_woe[features].copy()
        x.insert(0, "const", 1.0)
        out["score"] = x.values @ model.params[["const"] + features].values
    return out.rename(columns={id_col: "aid"}) if id_col != "aid" else out


def calibrate_pd(scores_df, target="default12"):
    y = scores_df[target].values
    score = scores_df["score"].values
    x = sm.add_constant(score, has_constant="add")
    calib_model = sm.Logit(y, x).fit(disp=0)
    if hasattr(calib_model.params, "index"):
        param_index = calib_model.params.index.tolist()
        a = float(calib_model.params["const"]) if "const" in param_index else float(calib_model.params.iloc[0])
        slope_cols = [c for c in param_index if c != "const"]
        b = float(calib_model.params[slope_cols[0]]) if slope_cols else 0.0
    else:
        a = float(calib_model.params[0])
        b = float(calib_model.params[1]) if len(calib_model.params) > 1 else 0.0
    pd_calibrated = calib_model.predict(x)
    return {
        "params": {"intercept": a, "coef": b, "target": target, "score_col": "score"},
        "diagnostics": {
            "auc_before": float(roc_auc_score(y, score)),
            "auc_after": float(roc_auc_score(y, pd_calibrated)),
            "brier_before": float(brier_score_loss(y, (score - score.min()) / (score.max() - score.min() + 1e-12))),
            "brier_after": float(brier_score_loss(y, pd_calibrated)),
            "mean_pd_predicted": float(pd_calibrated.mean()),
            "mean_pd_actual": float(y.mean()),
        },
    }


def variable_importance_from_points(points_table):
    scale = (
        points_table.groupby("feature")["points"]
        .agg(min_points="min", max_points="max")
        .reset_index()
    )
    scale["range"] = scale["max_points"] - scale["min_points"]
    total_range = scale["range"].sum()
    scale["importance_pct"] = scale["range"] / total_range if total_range else 0.0
    return scale.sort_values("importance_pct", ascending=False)


def plot_calibration_curve(scores_df, target, n_deciles=10):
    work = scores_df[[target, "score"]].dropna().copy()
    work["decile"] = pd.qcut(work["score"], n_deciles, duplicates="drop")
    cal = work.groupby("decile", observed=False).agg(
        n=("score", "count"),
        mean_score=("score", "mean"),
        bad_rate=(target, "mean"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cal["mean_score"], cal["bad_rate"], marker="o")
    ax.set_xlabel("mean score (decile)")
    ax.set_ylabel("actual bad rate")
    ax.set_title("Calibration curve")
    plt.tight_layout()
    plt.show()
    return cal
'''

SECTION4_EXEC = '''
points_table = scale_scorecard(
    model_package, SCORECARD_PARAMS["factor"], SCORECARD_PARAMS["offset"]
)
display(points_table.head(20))

importance = variable_importance_from_points(points_table)
display(importance)
importance.plot.barh(x="feature", y="importance_pct", legend=False, figsize=(8, 5))
plt.xlabel("importance_pct")
plt.tight_layout()
plt.show()

valid_scored = score_applicants(valid_woe, model_package, points_table)
valid_scored = valid_scored.merge(
    valid_product[[SCORECARD_PARAMS["id_col"], SCORECARD_PARAMS["target"]]].rename(
        columns={SCORECARD_PARAMS["id_col"]: "aid"}
    ),
    on="aid",
)
calibration = calibrate_pd(valid_scored, target=SCORECARD_PARAMS["target"])
print("calibration params:", calibration["params"])

cal_table = plot_calibration_curve(valid_scored, SCORECARD_PARAMS["target"])
display(cal_table)
'''

SECTION5_DEFS = r'''
def gini_over_time(scored_df, target, time_col="period", score_col="score", min_n=30):
    rows = []
    for period, sub in scored_df.groupby(time_col):
        if len(sub) < min_n:
            continue
        g = compute_gini(sub[target], sub[score_col])
        rows.append({
            "period": period,
            "n": len(sub),
            "bad_rate": sub[target].mean(),
            "gini": g,
        })
    return pd.DataFrame(rows).sort_values("period")


def plot_gini_over_time(gini_by_period, overall_gini=None):
    if gini_by_period.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gini_by_period["period"], gini_by_period["gini"], marker="o")
    median_g = gini_by_period["gini"].median()
    ax.axhline(median_g, color="gray", linestyle="--", label=f"median {median_g:.2%}")
    if overall_gini is not None:
        ax.axhline(overall_gini, color="green", linestyle=":", label=f"overall valid {overall_gini:.2%}")
    ax.set_title("Gini over time")
    ax.set_ylabel("Gini")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.show()
    flagged = gini_by_period[gini_by_period["gini"] < median_g - 0.10]
    if len(flagged):
        print("Warning: periods with Gini >10pp below median:")
        display(flagged)
'''

SECTION5_EXEC = '''
scored_with_period = valid_scored.merge(
    valid_product[[SCORECARD_PARAMS["id_col"], SCORECARD_PARAMS["time_col"]]].rename(
        columns={SCORECARD_PARAMS["id_col"]: "aid"}
    ),
    on="aid",
)
gini_time = gini_over_time(
    scored_with_period,
    SCORECARD_PARAMS["target"],
    SCORECARD_PARAMS["time_col"],
)
display(gini_time)
plot_gini_over_time(gini_time, overall_gini=model_package["metrics"]["gini_valid"])
'''

SECTION6_DEFS = r'''
def build_model_report(model_package, points_table, gini_time, calibration, cal_table, gini_vars):
    m = model_package["metrics"]
    main = pd.DataFrame({
        "Measure": [
            "gini_train", "gini_valid", "ar_diff", "max_vif",
            "max_pvalue", "n_features", "n_negative_betas",
        ],
        "Value": [
            m["gini_train"], m["gini_valid"], m["ar_diff"], m["max_vif"],
            m["max_pvalue"], m["n_features"], m["n_negative_betas"],
        ],
    })
    effects = summarize_model_effects(model_package)
    importance = variable_importance_from_points(points_table)
    scorecard = points_table.copy()
    scorecard["points_round"] = scorecard["points"].round(0)
    cal_df = pd.DataFrame([calibration["params"]])
    cal_df = pd.concat([cal_df, cal_table], ignore_index=True)
    final_vars = [f[: -len("_WOE")] for f in model_package["features"]]
    gini_final = gini_vars[gini_vars["variable"].isin(final_vars)]
    return {
        "Main_measures": main,
        "Effects": effects,
        "Gini_over_time": gini_time,
        "Scorecard": scorecard,
        "Variable importance": importance,
        "Calibration": cal_df,
        "Gini_vars_final": gini_final,
    }


def display_model_report(model_report):
    for sheet_name, df in model_report.items():
        print(f"## {sheet_name}")
        display(df)
        if sheet_name == "Gini_over_time":
            plot_gini_over_time(df)
        elif sheet_name == "Variable importance":
            df.plot.barh(x="feature", y="importance_pct", legend=False, figsize=(8, 4))
            plt.tight_layout()
            plt.show()


def save_product_artifacts(
    product, out_dir, big_scorecard, gini_vars, stability_flags, gini_time,
    model_package, points_table, screen_report, woe_maps, calibration,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    big_scorecard.to_parquet(out / f"big_scorecard_{product}.parquet", index=False)
    gini_vars.to_parquet(out / f"gini_vars_{product}.parquet", index=False)
    stability_flags.to_parquet(out / f"bin_stability_flags_{product}.parquet", index=False)
    gini_time.to_parquet(out / f"gini_by_period_{product}.parquet", index=False)
    points_table.to_parquet(out / f"points_table_{product}.parquet", index=False)
    screen_report.to_parquet(out / f"feature_screen_report_{product}.parquet", index=False)
    with (out / f"pd_{product}.pkl").open("wb") as fh:
        pickle.dump(model_package, fh)
    woe_path = out / "woe_maps.pkl"
    existing = {}
    if woe_path.exists():
        with woe_path.open("rb") as fh:
            existing = pickle.load(fh)
    existing[product] = woe_maps
    with woe_path.open("wb") as fh:
        pickle.dump(existing, fh)
    cal_path = out / "calibration_params.json"
    cal_existing = {}
    if cal_path.exists():
        with cal_path.open(encoding="utf-8") as fh:
            cal_existing = json.load(fh)
    cal_existing[product] = calibration["params"]
    with cal_path.open("w", encoding="utf-8") as fh:
        json.dump(cal_existing, fh, indent=2)
'''

SECTION6_EXEC = '''
model_report = build_model_report(
    model_package, points_table, gini_time, calibration, cal_table, gini_vars
)
display_model_report(model_report)

save_product_artifacts(
    PRODUCT,
    "../data/06_models",
    big_scorecard,
    gini_vars,
    stability_flags,
    gini_time,
    model_package,
    points_table,
    screen_report,
    woe_maps,
    calibration,
)

gini_valid = model_package["metrics"]["gini_valid"]
print(f"PD {PRODUCT}: {len(model_package['features'])} features, valid Gini {gini_valid:.1%}")
print("Saved parquet/pkl for", PRODUCT)
print("When ins is done, set PRODUCT='css' and re-run §2–§6")
'''

SECTION6_OPTIONAL = '''
# Optional regression pipeline — SKIPS §2 stability loop; use only after manual tuning.
# results = run_full_scorecard_pipeline(
#     "../data/04_feature/abt_app.parquet",
#     "../data/04_feature/decisions.parquet",
#     SCORECARD_PARAMS,
#     output_dir="../data/06_models",
# )
'''

cells = [
    md(INTRO),
    md("## §0 — Setup"),
    code(SECTION0),
    md("## §1 — Data prep (run once)"),
    code(SECTION1_DEFS),
    code(SECTION1_EXEC),
    md("## §2 — Tuning loop (repeat until stable)\n\nRe-run cells below after tuning `ncategories_int`, `minimum_share_int`, `ncategories_nom`, `psi_max`."),
    code(SECTION2_BINNING),
    code(SECTION2_SCREENING),
    code(SECTION2_WOE),
    code(SECTION2_STABILITY),
    code(SECTION2_VIZ),
    code(SECTION2_EXEC),
    code(SECTION2_GATE),
    md("## §3 — Model fit (after §2 stable)"),
    code(SECTION3_DEFS),
    code(SECTION3_EXEC),
    md("## §4 — Scorecard points & calibration"),
    code(SECTION4_DEFS),
    code(SECTION4_EXEC),
    md("## §5 — Model stability (Gini over time)"),
    code(SECTION5_DEFS),
    code(SECTION5_EXEC),
    md("## §6 — Save artifacts"),
    code(SECTION6_DEFS),
    code(SECTION6_EXEC),
    md("### Optional: full pipeline regression (skips §2 stability)"),
    code(SECTION6_OPTIONAL),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": ".venv", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {NB_PATH} ({len(cells)} cells)")
