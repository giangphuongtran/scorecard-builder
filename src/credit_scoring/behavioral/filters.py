import pandas as pd

def filter_transactions(transaction: pd.DataFrame,
                        cust_unique: pd.Series,
                        proc_periodf: str,
                        proc_period1: str,
                        grace_days: int = 15) -> pd.DataFrame:
    """Slice transactions to the behavioral window and derive the days indicator.

    Actions:
    1. Keep rows whose cid is in cust_unique (only applicants this month).
    2. Keep rows whose snapshot period is in [proc_periodf, proc_period1].
    3. Derive days = pay_days + grace_days (SAS: pay_days+15; no lower clip).
       Negative values mean paid before the grace window — kept intentionally.
    4. Return a copy; due_installments stays as the due counter.
    """
    mask_cid = transaction["cid"].isin(cust_unique)
    mask_period = (transaction["period"] >= proc_periodf) & (transaction["period"] <= proc_period1)
    filtered_df = transaction.loc[mask_cid & mask_period].copy()

    filtered_df["days"] = filtered_df["pay_days"] + grace_days

    return filtered_df