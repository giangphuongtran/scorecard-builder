import pandas as pd

def assemble_abt_month(month_prod: pd.DataFrame,
                       behavioral: pd.DataFrame,
                       agg_all: pd.DataFrame,
                       agg_ins: pd.DataFrame,
                       agg_css: pd.DataFrame,
                       active_flag: pd.DataFrame,
                       proc_period: str) -> pd.DataFrame:
    """Join application + behavioral + customer aggregates into one modeling row per aid.

    Actions:
    1. Start from month_prod (every application this month).
    2. Left-merge agg_all on aid (loan-level cumulative debt ratios).
    3. Left-merge behavioral on cid (157 customer-level payment-history features).
    4. Left-merge agg_ins / agg_css / active_flag on cid (portfolio snapshots).
    5. Stamp period = proc_period. Assert aid uniqueness.
    """
    abt = (
        month_prod
        .merge(agg_all, on="aid", how="left")
        .merge(behavioral, on="cid", how="left")
        .merge(agg_ins, on="cid", how="left")
        .merge(agg_css, on="cid", how="left")
        .merge(active_flag, on="cid", how="left")
    )
    abt["period"] = proc_period
    assert abt["aid"].is_unique, "duplicate aid after assembly — check merge keys"
    return abt