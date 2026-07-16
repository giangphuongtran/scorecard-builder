"""As-if profit U-curves and optimal PD cut-offs."""

from __future__ import annotations

import pandas as pd


def profit_curve_by_pd(
    df: pd.DataFrame,
    product: str,
    pd_col: str = "pd",
) -> pd.DataFrame:
    """Cumulative profit / AR vs PD threshold for one product (sorted ascending PD)."""
    sub = df.loc[df["product"].eq(product)].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=[
                "pd",
                "n",
                "n_cum",
                "ar",
                "profit",
                "profit_cum",
                "bad_rate_cum",
                "defaults_cum",
            ]
        )
    sub = sub.sort_values(pd_col, ascending=True).reset_index(drop=True)
    g = (
        sub.groupby(pd_col, sort=True)
        .agg(n=("aid", "size"), profit=("profit", "sum"), defaults=("default12", "sum"))
        .reset_index()
        .rename(columns={pd_col: "pd"})
    )
    g["n_cum"] = g["n"].cumsum()
    g["profit_cum"] = g["profit"].cumsum()
    g["defaults_cum"] = g["defaults"].cumsum()
    g["ar"] = g["n_cum"] / g["n"].sum()
    g["bad_rate_cum"] = g["defaults_cum"] / g["n_cum"]
    return g[
        ["pd", "n", "n_cum", "ar", "profit", "profit_cum", "bad_rate_cum", "defaults_cum"]
    ]


def find_cutoff_at_ar(curve: pd.DataFrame, target_ar: float) -> dict:
    """Return first PD where cumulative AR reaches ``target_ar`` (0–1)."""
    if curve.empty:
        raise ValueError("empty profit curve")
    if not 0 < float(target_ar) <= 1:
        raise ValueError("target_ar must be in (0, 1]")
    hit = curve.loc[curve["ar"] >= float(target_ar)]
    if hit.empty:
        row = curve.iloc[-1]
        return {
            "pd_cutoff": float(row["pd"]),
            "peak_profit": float(row["profit_cum"]),
            "ar_at_peak": float(row["ar"]),
            "n_accepted": int(row["n_cum"]),
            "accepted_bad_rate": float(row["bad_rate_cum"]),
            "target_ar": float(target_ar),
            "reachable": False,
        }
    row = hit.iloc[0]
    return {
        "pd_cutoff": float(row["pd"]),
        "peak_profit": float(row["profit_cum"]),
        "ar_at_peak": float(row["ar"]),
        "n_accepted": int(row["n_cum"]),
        "accepted_bad_rate": float(row["bad_rate_cum"]),
        "target_ar": float(target_ar),
        "reachable": True,
    }


def find_optimal_cutoff(
    curve: pd.DataFrame,
    min_ar: float | None = None,
) -> dict:
    """Return cut-off at max profit_cum; ties -> lower PD.

    If ``min_ar`` is set, restrict to rows with ``ar >= min_ar``. When no row
    qualifies, fall back to :func:`find_cutoff_at_ar`.
    """
    if curve.empty:
        raise ValueError("empty profit curve")
    eligible = curve
    if min_ar is not None:
        eligible = curve.loc[curve["ar"] >= float(min_ar)]
        if eligible.empty:
            return find_cutoff_at_ar(curve, float(min_ar))
    peak = eligible["profit_cum"].max()
    at_peak = eligible.loc[eligible["profit_cum"] == peak]
    row = at_peak.nsmallest(1, "pd").iloc[0]
    out = {
        "pd_cutoff": float(row["pd"]),
        "peak_profit": float(row["profit_cum"]),
        "ar_at_peak": float(row["ar"]),
        "n_accepted": int(row["n_cum"]),
        "accepted_bad_rate": float(row["bad_rate_cum"]),
    }
    if min_ar is not None:
        out["min_ar"] = float(min_ar)
    return out
