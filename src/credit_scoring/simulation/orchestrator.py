import pandas as pd
from .pool import _resolve_seed_period
from .aggregates import build_cust_all_agg, build_cust_product_agg, build_cust_active_flag
from .assembly import assemble_abt_month
from .decisions import apply_decision_engine_v1
from .pool import append_approved_to_pool
from ..behavioral.periods import derive_proc_period1, get_cust_unique
from ..behavioral.build import build_behavioral_for_month

def run_simulation(production: pd.DataFrame,
                   transactions: pd.DataFrame,
                   default_df: pd.DataFrame,
                   params: dict,
                   sim_params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Month-by-month ABT assembly with approved-pool feedback loop.

    Actions:
    1. Seed approved_tx from loans originated at seed_period (burn-in history).
    2. For each proc_period in [start_period, end_period]:
       a. Build behavioral features from approved_tx (not raw transactions).
       b. Build customer aggregates (product, cross-product, active flag).
       c. Assemble abt_month (one row per application).
       d. Apply decision engine v1.
       e. Append approved loans to pool for next month.
    3. Concatenate all months, attach default3/6/12 labels → abt_app.
    """
    seed_period = _resolve_seed_period(transactions, sim_params)
    approved_tx = transactions[transactions["fin_period"] == seed_period].copy()

    all_periods = sorted(production["period"].unique())
    target_periods = [
        p for p in all_periods
        if sim_params["start_period"] <= p <= sim_params["end_period"]
    ]

    all_abt = []
    all_decisions = []

    for proc_period in target_periods:
        proc_period1 = derive_proc_period1(proc_period)
        month_prod = production[production["period"] == proc_period]
        cust_unique = get_cust_unique(production, proc_period)

        behavioral = build_behavioral_for_month(
            transactions, production, proc_period, proc_period1, params
        )
        agg_all = build_cust_all_agg(transactions, month_prod, cust_unique, proc_period)
        agg_ins = build_cust_product_agg(transactions, cust_unique, proc_period1, "ins")
        agg_css = build_cust_product_agg(transactions, cust_unique, proc_period1, "css")
        active_flag = build_cust_active_flag(transactions, proc_period1)

        abt_month = assemble_abt_month(
            month_prod, behavioral, agg_all, agg_ins, agg_css, active_flag, proc_period
        )
        decisions_month = apply_decision_engine_v1(abt_month, proc_period, sim_params)

        approved_tx = append_approved_to_pool(
            approved_tx, transactions, decisions_month, proc_period
        )

        all_abt.append(abt_month)
        all_decisions.append(decisions_month)

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