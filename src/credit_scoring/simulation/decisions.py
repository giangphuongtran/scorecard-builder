import pandas as pd

def apply_decision_engine_v1(abt_month: pd.DataFrame,
                              proc_period: str,
                              sim_params: dict) -> pd.DataFrame:
    """Benchmark v1 decision engine — no PD scorecard yet (Phase 5 adds that).

    Actions (per application row):
    1. Default decision='A', decline_reason='999ok'.
    2. If proc_period < burn_in_before → keep approve (burn-in months).
    3. Else if product=='css' and act_cus_active != 1 → decline with reason
       '998 not active customer' (mandatory rule for entire project).
    4. Return slim decision table (aid-level, not full abt).
    """
    decision = pd.Series("A", index=abt_month.index)
    decline_reason = pd.Series("999ok", index=abt_month.index)

    burn_in = proc_period < sim_params["burn_in_before"]
    not_active_css = (
        (abt_month["product"] == "css") & (abt_month["act_cus_active"] != 1)
    )

    decline_mask = not_active_css & (not burn_in)
    decision.loc[decline_mask] = "N"
    decline_reason.loc[decline_mask] = "998 not active customer"

    out = abt_month[["cid", "aid", "product", "period",
                     "app_loan_amount", "app_n_installments"]].copy()
    out["decision"] = decision.values
    out["decline_reason"] = decline_reason.values
    return out