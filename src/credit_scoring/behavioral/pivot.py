import pandas as pd

def pivot_one_stream(transaction_filtered: pd.DataFrame, stream: dict) -> pd.DataFrame:
    """Pivot one product stream to wide customer-month columns.

    Actions:
    1. Filter to product stream (ins, css, or all if product is None).
    2. Aggregate to (cid, period) taking max(days) and max(due_installments)
       — worst loan dominates, per credit-policy convention.
    3. Pivot wide: one row per cid, columns {prefix}_Days_{YYYYMM} / {prefix}_Due_{YYYYMM}.
    4. Missing months stay NaN (filled later by rolling logic, not with 0).
    """
    prefix = stream["prefix"]
    product = stream["product"]

    if product is None:
        stream_df = transaction_filtered
    else:
        stream_df = transaction_filtered[transaction_filtered["product"] == product]

    grouped = (
        stream_df.groupby(["cid", "period"])
        .agg(days=("days", "max"), due_installments=("due_installments", "max"))
        .reset_index()
    )

    pivot = grouped.pivot(index="cid", columns="period", values=["days", "due_installments"])
    pivot.columns = [
        f"{prefix}_{'Days' if metric == 'days' else 'Due'}_{period}"
        for metric, period in pivot.columns
    ]

    return pivot.reset_index()


def pivot_customer_monthly(transaction_filtered: pd.DataFrame,
                            cust_unique: pd.Series,
                            product_streams: list) -> pd.DataFrame:
    """Build the wide abt_beh scaffold: cid + up to 72 monthly columns.

    Actions:
    1. Start with one row per applicant cid (left spine from cust_unique).
    2. For each product stream (Ins / Css / All), pivot independently.
    3. Left-merge each stream on cid — customers with no css history get NaN css cols.
    """
    abt_beh = pd.DataFrame({"cid": cust_unique.values})

    for stream in product_streams:
        stream_pivot = pivot_one_stream(transaction_filtered, stream)
        abt_beh = abt_beh.merge(stream_pivot, on="cid", how="left")

    return abt_beh


def build_period_list(proc_period1: str, window: int) -> list:
    """Return chronological list of YYYYMM months in a rolling window.

    Actions:
    1. Anchor on proc_period1 (last month in window).
    2. Walk back (window - 1) months.
    3. Return oldest-first (e.g. build_period_list('197501', 3) → ['197411','197412','197501']).
    """
    anchor = pd.Period(proc_period1, freq="M")
    periods = [(anchor - i).strftime("%Y%m") for i in range(window)]
    return list(reversed(periods))