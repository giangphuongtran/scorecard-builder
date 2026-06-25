import pandas as pd

def append_approved_to_pool(approved_tx: pd.DataFrame,
                            transactions: pd.DataFrame,
                            decision_month: pd.DataFrame,
                            proc_period: str) -> pd.DataFrame:
    """Grow the approved transaction pool after each simulation month.

    Actions:
    1. Select loans originated this month: fin_period == proc_period (NOT snapshot period).
    2. Keep only aids with decision == 'A'.
    3. Concatenate their full transaction histories into approved_tx.

    This is the reject-inference mechanism: rejected applicants' future payment
    history never enters the pool that feeds next month's behavioral features.
    """
    month_trans = transactions[transactions["fin_period"] == proc_period]
    approved_aids = decision_month.loc[decision_month["decision"] == "A", "aid"]
    month_trans_approved = month_trans[month_trans["aid"].isin(approved_aids)]
    return pd.concat([approved_tx, month_trans_approved], ignore_index=True)

def _resolve_seed_period(transactions: pd.DataFrame, sim_params: dict) -> str:
    """Return seed_period from sim_params, or earliest fin_period if None."""
    seed = sim_params.get("seed_period")
    if seed is None:
        return sorted(transactions["fin_period"].unique())[0]
    return seed