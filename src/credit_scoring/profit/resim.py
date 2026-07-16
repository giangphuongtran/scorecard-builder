"""Closed-loop re-simulation: monthly assemble → score → strategy → grow pool."""

from __future__ import annotations

import pandas as pd

from credit_scoring.behavioral.build import build_behavioral_for_month
from credit_scoring.behavioral.periods import derive_proc_period1, get_cust_unique
from credit_scoring.simulation.aggregates import (
    build_cust_active_flag,
    build_cust_all_agg,
    build_cust_product_agg,
)
from credit_scoring.simulation.assembly import assemble_abt_month
from credit_scoring.simulation.pool import _resolve_seed_period, append_approved_to_pool

from .rules import apply_strategy, rules_from_params
from .scoring import score_abt_application


def _seed_approved_tx(
    transactions: pd.DataFrame,
    sim_params: dict,
    profit_params: dict,
    start: str,
) -> pd.DataFrame:
    """Build the opening approved transaction pool.

    ``opening_book`` (default): all loans with ``fin_period < start`` — the
    pre-window book. Keeps ``act_cins`` / ``act_call`` histories available when
    CSS approvals are selective. Set ``seed_mode: single`` to use only
    ``sim_params.seed_period``.
    """
    mode = str(
        profit_params.get("seed_mode")
        or sim_params.get("seed_mode")
        or "opening_book"
    ).lower()
    if mode in {"opening_book", "pre_window", "before_start"}:
        return transactions.loc[
            transactions["fin_period"].astype(str) < str(start)
        ].copy()
    seed_period = _resolve_seed_period(transactions, sim_params)
    return transactions.loc[transactions["fin_period"] == seed_period].copy()


def run_closed_loop_resim(
    production: pd.DataFrame,
    transactions: pd.DataFrame,
    default_df: pd.DataFrame,
    behavioral_params: dict,
    sim_params: dict,
    profit_params: dict,
    packages: dict,
    points_tables: dict,
    calibrations: dict,
    *,
    start_period: str | None = None,
    end_period: str | None = None,
    verbose: bool = True,
    secondary: dict | None = None,
    rules_override: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Month-by-month sim with Gate B scoring + params:profit strategy.

    Features each month are built from the growing ``approved_tx`` pool
    (reject-inference feedback). Returns (abt_app with defaults, decisions).

    Optional ``secondary`` (``pr`` / ``cross`` bundles) attaches response and
    Cross PD scores so mid-band rules can fire inside ``apply_strategy``.
    """
    rules = rules_override if rules_override is not None else rules_from_params(profit_params)
    start = start_period or profit_params.get("window_start") or sim_params["start_period"]
    end = end_period or profit_params.get("window_end") or sim_params["end_period"]
    approved_tx = _seed_approved_tx(transactions, sim_params, profit_params, start)

    all_periods = sorted(production["period"].unique())
    target_periods = [p for p in all_periods if start <= p <= end]

    all_abt = []
    all_decisions = []

    # Split application PD vs secondary artifacts
    app_packages = {k: packages[k] for k in ("ins", "css") if k in packages}
    app_points = {k: points_tables[k] for k in ("ins", "css") if k in points_tables}
    app_cals = {k: calibrations[k] for k in ("ins", "css") if k in calibrations}

    if secondary is None:
        secondary = {}
        for key, col_map in (("pr", None), ("cross", None)):
            if key in packages:
                secondary[key] = {
                    "package": packages[key],
                    "points": points_tables[key],
                    "calib": calibrations[key],
                }

    for proc_period in target_periods:
        proc_period1 = derive_proc_period1(proc_period)
        month_prod = production[production["period"] == proc_period]
        cust_unique = get_cust_unique(production, proc_period)

        behavioral = build_behavioral_for_month(
            approved_tx, production, proc_period, proc_period1, behavioral_params
        )
        agg_all = build_cust_all_agg(approved_tx, month_prod, cust_unique, proc_period)
        agg_ins = build_cust_product_agg(approved_tx, cust_unique, proc_period1, "ins")
        agg_css = build_cust_product_agg(approved_tx, cust_unique, proc_period1, "css")
        active_flag = build_cust_active_flag(approved_tx, proc_period1)

        abt_month = assemble_abt_month(
            month_prod,
            behavioral,
            agg_all,
            agg_ins,
            agg_css,
            active_flag,
            proc_period,
        )

        scored = score_abt_application(
            abt_month,
            app_packages,
            app_points,
            app_cals,
            secondary=secondary or None,
        )
        merge_cols = ["aid", "score", "pd"]
        for c in ("pr", "cross_pd", "pr_score", "cross_score"):
            if c in scored.columns:
                merge_cols.append(c)
        strategy_in = abt_month.merge(scored[merge_cols], on="aid", how="left")
        decisions_month = apply_strategy(strategy_in, rules)

        approved_tx = append_approved_to_pool(
            approved_tx, transactions, decisions_month, proc_period
        )

        all_abt.append(abt_month)
        all_decisions.append(decisions_month)

        if verbose:
            n_apps = len(abt_month)
            n_appr = (decisions_month["decision"] == "A").sum()
            print(
                f"{proc_period} -> {proc_period1}: {n_apps:>5} apps, "
                f"{n_appr:>5} approved, pool={len(approved_tx):>7,}"
            )

    abt = pd.concat(all_abt, ignore_index=True)
    decisions = pd.concat(all_decisions, ignore_index=True)

    abt_app = abt.merge(
        default_df[["aid", "default3", "default6", "default12"]],
        on="aid",
        how="left",
    )
    return abt_app, decisions
