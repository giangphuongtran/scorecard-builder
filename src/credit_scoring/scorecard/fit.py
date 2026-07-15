"""Load ABT and prepare target."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import statsmodels.api as sm

from credit_scoring.scorecard.selection import assess_logit_model, forward_select_logit


def prepare_target(df: pd.DataFrame, target: str = "default12") -> pd.DataFrame:
    """Map raw default labels to binary {0, 1} and drop missing targets.

    Actions:
    1. Map ``.i`` / ``.d`` sentinel values to 0 (non-default).
    2. Drop rows with missing target.
    3. Cast surviving target values to int.
    """
    out = df.copy()
    out[target] = out[target].map({".i": 0, ".d": 0, 0: 0, 1: 1})
    out = out.dropna(subset=[target])
    out[target] = out[target].astype(int)
    return out


def prepare_abt(
    abt: pd.DataFrame,
    decisions: pd.DataFrame,
    params: dict | None = None,
    *,
    accepted_only: bool = True,
) -> pd.DataFrame:
    """Merge ABT with decisions, prepare the target, optionally keep accepted only.

    Actions:
    1. Left-merge ``decision`` / ``decline_reason`` on ``aid``.
    2. Call ``prepare_target`` for ``default12`` (or ``params["target"]``).
    3. Optionally filter to ``decision == "A"``.
    4. Assert ``aid`` uniqueness on the modeling frame.
    """
    params = params or {}
    target = params.get("target", "default12")
    decision_cols = [c for c in ("aid", "decision", "decline_reason") if c in decisions.columns]
    merged = abt.merge(decisions[decision_cols], on="aid", how="left")
    out = prepare_target(merged, target=target)
    if accepted_only:
        out = out[out["decision"] == "A"].copy()
    if not out["aid"].is_unique:
        raise ValueError("Prepared ABT must have unique aid values")
    return out


def load_and_prepare_abt(abt_path: str | Path, decisions_path: str | Path) -> pd.DataFrame:
    """Load parquet paths, then merge and prepare the target (all decisions).

    Actions:
    1. Read ``abt_app`` and ``decisions`` parquet files.
    2. Delegate to ``prepare_abt`` without accepted-only filter.
    """
    abt = pd.read_parquet(abt_path)
    decisions = pd.read_parquet(decisions_path)
    return prepare_abt(abt, decisions, accepted_only=False)


def train_pd_model(
    product: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    params: dict,
    *,
    candidate_woe_features: list[str] | None = None,
    woe_maps: dict[str, pd.DataFrame] | None = None,
) -> dict:
    """Train one product PD model with forward selection and final logit fit.

    Actions:
    1. Restrict to accepted applicants for the product.
    2. Forward-select from prescreened WOE columns.
    3. Fit final logit and attach metrics plus WOE tables for scaling.
    """
    target = params["target"]
    train_subset = train_df[
        (train_df["product"] == product) & (train_df["decision"] == "A")
    ].copy()
    valid_subset = valid_df[
        (valid_df["product"] == product) & (valid_df["decision"] == "A")
    ].copy()

    if candidate_woe_features is None:
        candidate_woe_features = [
            col for col in train_subset.columns if col.endswith("_WOE")
        ]

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
