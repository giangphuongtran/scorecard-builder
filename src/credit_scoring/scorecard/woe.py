"""WOE tables, IV, and WOE encoding."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_woe_table(
    df_binned: pd.DataFrame, feature_grp: str, target: str, epsilon: float
) -> pd.DataFrame:
    """Compute goods, bads, WOE, and IV contribution for one grouped feature.

    Actions:
    1. Aggregate goods and bads by grouped bin on train.
    2. Smooth category shares with ``epsilon``.
    3. Compute WOE and per-bin IV contribution.
    """
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
    return grouped


def build_woe_maps(
    df_binned: pd.DataFrame,
    grp_cols: list[str],
    target: str,
    epsilon: float,
) -> dict[str, pd.DataFrame]:
    """Build one WOE table per grouped feature column.

    Actions:
    1. Loop over ``{feature}_GRP`` column names.
    2. Call ``build_woe_table`` for each column.
    3. Return a dict keyed by grouped column name.
    """
    return {
        grp: build_woe_table(df_binned, grp, target, epsilon) for grp in grp_cols
    }


def compute_iv(woe_table: pd.DataFrame) -> float:
    """Sum per-bin IV contributions for one variable.

    Actions:
    1. Sum ``iv_component`` across bins.
    2. Clip at zero for floating-point safety.
    """
    return float(max(woe_table["iv_component"].sum(), 0.0))


def build_iv_table(
    woe_maps: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Aggregate IV to one row per raw feature name.

    Actions:
    1. Convert each grouped key to its base feature name.
    2. Sum IV via ``compute_iv``.
    3. Sort descending by IV.
    """
    rows = []
    for grp, table in woe_maps.items():
        feature = grp[: -len("_GRP")]
        rows.append({"feature": feature, "iv": compute_iv(table)})
    return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)


def encode_woe(
    df_binned: pd.DataFrame, woe_maps: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Map grouped bins to numeric WOE columns.

    Actions:
    1. Drop any existing ``*_WOE`` columns to keep reruns idempotent.
    2. Map each ``{feature}_GRP`` label to its train WOE.
    3. Fill unseen bins with neutral WOE ``0.0``.
    """
    base = df_binned.drop(columns=[c for c in df_binned.columns if c.endswith("_WOE")], errors="ignore")
    new_cols: dict[str, pd.Series] = {}

    for grp_col, woe_table in woe_maps.items():
        feature = grp_col[: -len("_GRP")]
        woe_col = f"{feature}_WOE"
        bin_to_woe = dict(zip(woe_table["bin"], woe_table["woe"]))
        new_cols[woe_col] = base[grp_col].map(bin_to_woe).fillna(0.0)

    woe_df = pd.DataFrame(new_cols, index=base.index)
    return pd.concat([base, woe_df], axis=1).copy()
