import pandas as pd

def derive_proc_period1(proc_period: str) -> str:
    """Convert application month → last month of usable behavioral history.

    Actions:
    1. Parse proc_period as a monthly Period (YYYYMM).
    2. Subtract one month — features must not include the application month itself
       (that would be look-ahead leakage).
    3. Return as YYYYMM string (e.g. 197502 → 197501).
    """
    return (pd.Period(proc_period, freq="M") - 1).strftime("%Y%m")


def derive_proc_periodf(proc_period1: str, max_length: int) -> str:
    """First snapshot month in the 12-month behavioral lookback window.

    Actions:
    1. Anchor on proc_period1 (end of history).
    2. Step back (max_length - 2) months — matches the SAS window that feeds
       agr12_* / ags12_* (12 months ending at proc_period1).
    3. Return YYYYMM string (e.g. 197501 → 197403).
    """
    anchor = pd.Period(proc_period1, freq="M")
    return (anchor - max_length + 2).strftime("%Y%m")


def get_cust_unique(production: pd.DataFrame, proc_period: str) -> pd.Series:
    """Customer IDs with at least one application in proc_period.

    Actions:
    1. Filter production to rows where period == proc_period.
    2. Return deduplicated cid values (one row per customer, not per loan).
    """
    mask = production["period"] == proc_period
    return production.loc[mask, "cid"].drop_duplicates().reset_index(drop=True)