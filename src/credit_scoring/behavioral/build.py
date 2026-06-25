import pandas as pd
from .periods import derive_proc_periodf, get_cust_unique, derive_proc_period1
from .filters import filter_transactions
from .pivot import pivot_customer_monthly
from .rolling import make_abt

def build_behavioral_for_month(transaction: pd.DataFrame,
                                 production: pd.DataFrame,
                                 proc_period: str,
                                 proc_period1: str,
                                 params: dict) -> pd.DataFrame:
    """End-to-end behavioral feature build for one application month.

    Actions:
    1. Derive proc_periodf (start of 12-month lookback).
    2. Identify applicants (cust_unique) from production.
    3. Filter transaction rows to [proc_periodf, proc_period1] for those cids.
       In Phase 3, transaction is the approved_tx pool (not raw transactions).
    4. Pivot → roll → counter → return 157-column table keyed by cid.
    """
    proc_periodf = derive_proc_periodf(proc_period1, params["max_length"])
    cust_unique = get_cust_unique(production, proc_period)

    transaction_filtered = filter_transactions(
        transaction, cust_unique, proc_periodf, proc_period1, params["grace_days"]
    )

    abt_beh = pivot_customer_monthly(transaction_filtered, cust_unique, params["product_streams"])
    return make_abt(abt_beh, proc_period1, params)

def build_behavioral_all_months(transaction: pd.DataFrame,
                                  production: pd.DataFrame,
                                  params: dict,
                                  output_dir: str = ".",
                                  start_period: str = "197502",
                                  end_period: str = "198712") -> list:
    """Optional batch export: one behavioral parquet per application month.

    Actions:
    1. Loop proc_period from start_period to end_period.
    2. For each month, call build_behavioral_for_month and write
       behavioral_{proc_period1}.parquet to output_dir.
    3. Return summary list of (proc_period, proc_period1, path, n_rows).

    Note: uses the transaction argument as-is. For exploration this is raw
    transactions; in the Phase 3 simulation loop the same function receives
    approved_tx instead (reject-inference-aware history).
    """
    start_period = start_period or params["start_period"]
    end_period = end_period or params["end_period"]
    all_periods = sorted(production["period"].unique())
    target_periods = [p for p in all_periods if start_period <= p <= end_period]
    monthly_frames = []
    for proc_period in target_periods:
        proc_period1 = derive_proc_period1(proc_period)
        abt = build_behavioral_for_month(
            transaction, production, proc_period, proc_period1, params
        ).copy()
        abt["proc_period"] = proc_period
        abt["proc_period1"] = proc_period1
        monthly_frames.append(abt)
        print(f"{proc_period} -> {proc_period1}: {len(abt):>6} rows")
    if not monthly_frames:
        return pd.DataFrame()
    return pd.concat(monthly_frames, ignore_index=True)