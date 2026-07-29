"""Business-facing policy trade-off helpers for Streamlit and reporting."""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from credit_scoring.profit.cutoff_explore import evaluate_cutoffs


def policy_sensitivity(
    scored_pnl: pd.DataFrame,
    base_cutoffs: dict,
    param: str,
    deltas: Iterable[float],
    *,
    window_start: str,
    window_end: str,
    burn_in_before: str = "197501",
    economics: dict | None = None,
    bad_customer: dict | None = None,
) -> pd.DataFrame:
    """Sweep one cutoff by relative deltas; return accept / bad / profit metrics.

    ``deltas`` are absolute additives applied to ``base_cutoffs[param]``
    (e.g. ``[-0.01, 0.0, 0.01]`` for a ±1pp PD shift).
    """
    if param not in base_cutoffs:
        raise KeyError(f"Unknown cutoff param: {param}")
    base_val = float(base_cutoffs[param])
    rows: list[dict[str, Any]] = []
    for delta in deltas:
        cuts = dict(base_cutoffs)
        new_val = base_val + float(delta)
        # Keep mid-band ordered when sweeping PD band edges.
        if param == "pd_ins_high" and "pd_ins_low" in cuts:
            new_val = max(new_val, float(cuts["pd_ins_low"]) + 1e-6)
        if param == "pd_ins_low" and "pd_ins_high" in cuts:
            new_val = min(new_val, float(cuts["pd_ins_high"]) - 1e-6)
        cuts[param] = new_val
        ev = evaluate_cutoffs(
            scored_pnl,
            cuts,
            window_start=window_start,
            window_end=window_end,
            burn_in_before=burn_in_before,
            economics=economics,
            bad_customer=bad_customer,
        )
        n_bad = 0
        decisions = ev.get("decisions")
        if isinstance(decisions, pd.DataFrame) and len(decisions) and "default12" in scored_pnl.columns:
            accepted_ids = set(decisions.loc[decisions["decision"].eq("A"), "aid"])
            accepted = scored_pnl.loc[scored_pnl["aid"].isin(accepted_ids)]
            n_bad = int(accepted["default12"].fillna(0).sum())
        rows.append(
            {
                "param": param,
                "delta": float(delta),
                "value": new_val,
                "n_accept": int(ev.get("n_accept") or 0),
                "n_apps": int(ev.get("n_apps") or 0),
                "n_bad": n_bad,
                "total_profit": float(ev.get("total_profit") or 0.0),
                "ar_ins": float(ev.get("ar_ins")) if ev.get("ar_ins") == ev.get("ar_ins") else float("nan"),
                "ar_css": float(ev.get("ar_css")) if ev.get("ar_css") == ev.get("ar_css") else float("nan"),
                "bad_rate_ins": float(ev.get("bad_rate_ins"))
                if ev.get("bad_rate_ins") == ev.get("bad_rate_ins")
                else float("nan"),
                "bad_rate_css": float(ev.get("bad_rate_css"))
                if ev.get("bad_rate_css") == ev.get("bad_rate_css")
                else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        base_row = out.loc[out["delta"].eq(0.0)]
        if len(base_row):
            base_profit = float(base_row.iloc[0]["total_profit"])
            base_accept = int(base_row.iloc[0]["n_accept"])
            out["delta_profit"] = out["total_profit"] - base_profit
            out["delta_n_accept"] = out["n_accept"] - base_accept
        else:
            out["delta_profit"] = float("nan")
            out["delta_n_accept"] = float("nan")
    return out


def compare_to_benchmark(ev: dict, reference: float) -> dict[str, Any]:
    """Delta profit / AR / bad rate vs a published benchmark profit figure."""
    profit = float(ev.get("total_profit") or 0.0)
    ref = float(reference)
    return {
        "total_profit": profit,
        "reference_profit": ref,
        "delta_profit": profit - ref,
        "ar_ins": ev.get("ar_ins"),
        "ar_css": ev.get("ar_css"),
        "bad_rate_ins": ev.get("bad_rate_ins"),
        "bad_rate_css": ev.get("bad_rate_css"),
        "n_accept": ev.get("n_accept"),
        "n_apps": ev.get("n_apps"),
        "label": "published benchmark",
        "note": (
            "Offline historical replay vs published benchmark profit. "
            "Not a live closed-loop portfolio result."
        ),
    }


def midband_funnel(
    scored_pnl: pd.DataFrame,
    cutoffs: dict,
    *,
    window_start: str,
    window_end: str,
    burn_in_before: str = "197501",
    bad_customer: dict | None = None,
) -> pd.DataFrame:
    """INS grey-zone funnel: auto-approve / CLTV-save / CLTV-decline counts.

    Stages:
    - ``auto_approve``: INS with ``pd <= pd_ins_low``
    - ``grey_zone``: ``pd_ins_low < pd <= pd_ins_high``
    - ``cltv_save``: grey-zone rows that pass PR and Cross PD gates
    - ``cltv_decline``: grey-zone rows that fail PR or Cross PD
    - ``outright_decline``: INS with ``pd > pd_ins_high``
    """
    from credit_scoring.profit.pnl import filter_profit_window

    df = filter_profit_window(scored_pnl.copy(), window_start, window_end)
    period = df["period"].astype(str)
    burn = period < str(burn_in_before)
    ins = df.loc[(~burn) & df["product"].eq("ins")].copy()

    lo = float(cutoffs.get("pd_ins_low", 0.0))
    hi = float(cutoffs.get("pd_ins_high", 0.0))
    pr_min = cutoffs.get("pr_min")
    cross_max = cutoffs.get("cross_pd_max")

    # Optional bad-customer pre-filter so funnel matches production priority.
    if bad_customer and bad_customer.get("enabled"):
        feat = bad_customer["feature"]
        thr = bad_customer["threshold"]
        if feat in ins.columns:
            ins = ins.loc[~(ins[feat].notna() & (ins[feat] > thr))].copy()

    auto = ins["pd"] <= lo
    grey = (ins["pd"] > lo) & (ins["pd"] <= hi)
    high = ins["pd"] > hi

    fail_pr = pd.Series(False, index=ins.index)
    fail_cross = pd.Series(False, index=ins.index)
    if pr_min is not None and "pr" in ins.columns:
        fail_pr = ins["pr"].isna() | (ins["pr"] < float(pr_min))
    if cross_max is not None and "cross_pd" in ins.columns:
        fail_cross = ins["cross_pd"].isna() | (ins["cross_pd"] > float(cross_max))
    cltv_fail = grey & (fail_pr | fail_cross)
    cltv_save = grey & ~cltv_fail

    rows = [
        {"stage": "auto_approve", "n": int(auto.sum()), "plain_english": "Safe band — approved on PD alone"},
        {"stage": "grey_zone", "n": int(grey.sum()), "plain_english": "Mid-band — needs CLTV checks"},
        {"stage": "cltv_save", "n": int(cltv_save.sum()), "plain_english": "Grey zone kept (PR + Cross PD pass)"},
        {"stage": "cltv_decline", "n": int(cltv_fail.sum()), "plain_english": "Grey zone declined (CLTV fail)"},
        {"stage": "outright_decline", "n": int(high.sum()), "plain_english": "Above upper PD limit — declined"},
    ]
    return pd.DataFrame(rows)


def policy_tradeoff_summary(
    scored_pnl: pd.DataFrame,
    cutoffs: dict,
    *,
    window_start: str,
    window_end: str,
    burn_in_before: str = "197501",
    economics: dict | None = None,
    bad_customer: dict | None = None,
    reference: float = 731882.0,
    sensitivity_params: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    """Bundle sensitivity + benchmark compare + mid-band funnel for README / UI."""
    base_ev = evaluate_cutoffs(
        scored_pnl,
        cutoffs,
        window_start=window_start,
        window_end=window_end,
        burn_in_before=burn_in_before,
        economics=economics,
        bad_customer=bad_customer,
    )
    sens_cfg = sensitivity_params or {
        "pd_ins_high": [-0.01, 0.0],
        "pd_css": [0.0, 0.05],
    }
    sensitivity = {
        param: policy_sensitivity(
            scored_pnl,
            cutoffs,
            param,
            deltas,
            window_start=window_start,
            window_end=window_end,
            burn_in_before=burn_in_before,
            economics=economics,
            bad_customer=bad_customer,
        ).to_dict(orient="records")
        for param, deltas in sens_cfg.items()
    }
    return {
        "benchmark": compare_to_benchmark(base_ev, reference),
        "sensitivity": sensitivity,
        "midband_funnel": midband_funnel(
            scored_pnl,
            cutoffs,
            window_start=window_start,
            window_end=window_end,
            burn_in_before=burn_in_before,
            bad_customer=bad_customer,
        ).to_dict(orient="records"),
        "cutoffs": dict(cutoffs),
    }
