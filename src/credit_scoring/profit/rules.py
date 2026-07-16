"""Decision strategy rules and evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .pnl import filter_profit_window


def rules_from_params(profit_params: dict) -> dict:
    """Build apply_strategy rules dict from params:profit."""
    return {
        "window_start": profit_params["window_start"],
        "window_end": profit_params["window_end"],
        "burn_in_before": profit_params.get("burn_in_before", "197501"),
        "economics": profit_params["economics"],
        "cutoffs": dict(profit_params.get("cutoffs") or {}),
        "bad_customer": dict(profit_params.get("bad_customer") or {}),
    }


def apply_strategy(scored: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Apply decision rules; first matching decline wins.

    Priority: burn-in keep A -> 998 inactive css -> bad customer -> PD css
    -> PD ins high -> INS mid-band (PR / Cross PD) -> done.

    Mid-band (Strategy 1 / st1_high style) when cutoffs provide
    ``pd_ins_low``, and optionally ``pr_min`` / ``cross_pd_max``:
    decline INS with ``pd_ins_low < pd <= pd_ins_high`` when
    ``pr < pr_min`` OR ``cross_pd > cross_pd_max``.
    """
    out_cols = [
        "cid",
        "aid",
        "product",
        "period",
        "decision",
        "decline_reason",
        "app_loan_amount",
        "app_n_installments",
        "pd",
    ]
    df = scored.copy()
    decision = pd.Series("A", index=df.index, dtype="string")
    reason = pd.Series("999ok", index=df.index, dtype="string")

    burn_in_before = rules.get("burn_in_before", "197501")
    period = df["period"].astype(str)
    burn = period < burn_in_before

    inactive_css = df["product"].eq("css") & (df["act_cus_active"] != 1) & (~burn)
    decision.loc[inactive_css] = "N"
    reason.loc[inactive_css] = "998 not active customer"

    bad_cfg = rules.get("bad_customer") or {}
    if bad_cfg.get("enabled", False):
        feat = bad_cfg["feature"]
        thr = bad_cfg["threshold"]
        if feat not in df.columns:
            raise KeyError(f"bad_customer feature missing: {feat}")
        bad = (~burn) & decision.eq("A") & df[feat].notna() & (df[feat] > thr)
        decision.loc[bad] = "D"
        reason.loc[bad] = "1 bad customer"

    cut = rules.get("cutoffs") or {}
    pd_css = cut.get("pd_css")
    pd_ins = cut.get("pd_ins_high")
    pd_ins_low = cut.get("pd_ins_low")
    pr_min = cut.get("pr_min")
    cross_pd_max = cut.get("cross_pd_max")

    if pd_css is not None:
        css_cut = (
            (~burn) & decision.eq("A") & df["product"].eq("css") & (df["pd"] > pd_css)
        )
        decision.loc[css_cut] = "D"
        reason.loc[css_cut] = "1 PD cut-off on css"

    if pd_ins is not None:
        ins_cut = (
            (~burn) & decision.eq("A") & df["product"].eq("ins") & (df["pd"] > pd_ins)
        )
        decision.loc[ins_cut] = "D"
        reason.loc[ins_cut] = "2 PD cut-off on ins"

    # Mid-band: keep only if PR high enough AND Cross PD low enough
    if pd_ins is not None and pd_ins_low is not None:
        mid = (
            (~burn)
            & decision.eq("A")
            & df["product"].eq("ins")
            & (df["pd"] <= pd_ins)
            & (df["pd"] > pd_ins_low)
        )
        fail_pr = False
        fail_cross = False
        if pr_min is not None and "pr" in df.columns:
            fail_pr = df["pr"].isna() | (df["pr"] < pr_min)
        if cross_pd_max is not None and "cross_pd" in df.columns:
            fail_cross = df["cross_pd"].isna() | (df["cross_pd"] > cross_pd_max)
        if pr_min is not None or cross_pd_max is not None:
            fail = mid & (fail_pr | fail_cross)
            decision.loc[fail] = "D"
            reason.loc[fail] = "3 PD,PDCross and PR cut-offs on ins"

    decision.loc[burn] = "A"
    reason.loc[burn] = "999ok"

    slim = df[
        [
            c
            for c in [
                "cid",
                "aid",
                "product",
                "period",
                "app_loan_amount",
                "app_n_installments",
                "pd",
                "pr",
                "cross_pd",
            ]
            if c in df.columns
        ]
    ].copy()
    slim["decision"] = decision.to_numpy()
    slim["decline_reason"] = reason.to_numpy()
    keep = [c for c in out_cols if c in slim.columns]
    return slim[keep]


def evaluate_strategy(
    scored_pnl: pd.DataFrame,
    decisions: pd.DataFrame,
    window_start: str,
    window_end: str,
) -> dict:
    """Profit metrics on decision=='A' inside window; AR on decisionable apps (excl N)."""
    base = scored_pnl.copy()
    if "decision" in base.columns:
        base = base.drop(columns=["decision", "decline_reason"], errors="ignore")
    m = base.merge(
        decisions[["aid", "decision", "decline_reason"]],
        on="aid",
        how="inner",
    )
    m = filter_profit_window(m, window_start, window_end)

    accepted = m.loc[m["decision"].eq("A")]
    decisionable = m.loc[m["decision"].isin(["A", "D"])]

    def _ar(product: str) -> float:
        d = decisionable.loc[decisionable["product"].eq(product)]
        if len(d) == 0:
            return float("nan")
        return float(d["decision"].eq("A").mean())

    def _bad(product: str) -> float:
        a = accepted.loc[accepted["product"].eq(product)]
        if len(a) == 0:
            return float("nan")
        return float(a["default12"].mean())

    by_product = (
        accepted.groupby("product")
        .agg(
            n_accept=("aid", "size"),
            total_profit=("profit", "sum"),
            total_income=("income", "sum"),
            total_el=("el", "sum"),
            bad_rate=("default12", "mean"),
        )
        .reset_index()
    )
    by_year = accepted.copy()
    by_year["year"] = by_year["period"].astype(str).str[:4]
    by_year = by_year.groupby(["year", "product"], as_index=False).agg(
        n=("aid", "size"), profit=("profit", "sum")
    )

    return {
        "total_profit": float(accepted["profit"].sum()),
        "total_income": float(accepted["income"].sum()),
        "total_el": float(accepted["el"].sum()),
        "ar_ins": _ar("ins"),
        "ar_css": _ar("css"),
        "bad_rate_ins": _bad("ins"),
        "bad_rate_css": _bad("css"),
        "n_apps": int(len(m)),
        "n_accept": int(len(accepted)),
        "n_N": int(m["decision"].eq("N").sum()),
        "by_product": by_product,
        "by_year": by_year,
    }


def compare_strategies(
    results: list[dict],
    *,
    reference: float | None = None,
    reference_css_ar: float = 0.0868,
    reference_ins_ar: float = 0.2600,
) -> pd.DataFrame:
    """Build strategy comparison table; append optional course reference row."""
    rows = []
    for r in results:
        rows.append(
            {
                "strategy": r["name"],
                "css_ar": r["eval"]["ar_css"],
                "ins_ar": r["eval"]["ar_ins"],
                "total_profit": r["eval"]["total_profit"],
                "n_accept": r["eval"]["n_accept"],
                "source": r.get("source", "offline"),
            }
        )
    if reference is not None:
        rows.append(
            {
                "strategy": "reference",
                "css_ar": reference_css_ar,
                "ins_ar": reference_ins_ar,
                "total_profit": float(reference),
                "n_accept": np.nan,
                "source": "course",
            }
        )
    return pd.DataFrame(rows)
