import pandas as pd

def build_cust_product_agg(approved_tx: pd.DataFrame,
                            cust_unique: pd.Series,
                            proc_period1: str,
                            version: str) -> pd.DataFrame:
    """Per-customer product aggregates for ins or css (act_cins_* / act_ccss_*).

    Actions:
    Part A — Snapshot at period == proc_period1 (current portfolio state):
      1. Filter approved_tx to product and applicant cids.
      2. Keep rows at proc_period1 only.
      3. Group by cid: count active loans, max due, min paid/left installments.
      4. Compute utilization ratios across the customer's loans:
         utl = sum(paid)/sum(n_installments), dueutl = sum(due)/sum(n_installments),
         cc = (sum(installment)+sum(spendings))/sum(income).

    Part B — Lifetime history with period <= proc_period1:
      5. Compute seniority = months from fin_period to proc_period1 (+1).
      6. Aggregate max/min seniority, distinct aid count, count status B/C closures.

    Customers with no loans of this product are absent (NaN after left-join in assembly).
    """
    v = version
    base = approved_tx[
        (approved_tx["product"] == version) & (approved_tx["cid"].isin(cust_unique))
    ]

    snap = base[base["period"] == proc_period1]

    snap_agg = snap.groupby("cid").agg(
        **{
            f"act_c{v}_n_loans_act": ("aid", "count"),
            f"act_c{v}_maxdue": ("due_installments", "max"),
            f"act_c{v}_min_pninst": ("paid_installments", "min"),
            f"act_c{v}_min_lninst": ("leftn_installments", "min"),
        }
    )

    snap_sums = snap.groupby("cid").agg(
        paid_sum=("paid_installments", "sum"),
        due_sum=("due_installments", "sum"),
        n_sum=("n_installments", "sum"),
        installment_sum=("installment", "sum"),
        spendings_sum=("spendings", "sum"),
        income_sum=("income", "sum"),
    )

    snap_agg[f"act_c{v}_utl"] = snap_sums["paid_sum"] / snap_sums["n_sum"]
    snap_agg[f"act_c{v}_dueutl"] = snap_sums["due_sum"] / snap_sums["n_sum"]
    snap_agg[f"act_c{v}_cc"] = (
        snap_sums["installment_sum"] + snap_sums["spendings_sum"]
    ) / snap_sums["income_sum"]

    hist = base[base["period"] <= proc_period1].copy()
    anchor = pd.Period(proc_period1, freq="M")
    hist["seniority"] = hist["fin_period"].apply(
        lambda fp: (anchor - pd.Period(fp, freq="M")).n + 1
    )

    hist_agg = hist.groupby("cid").agg(
        **{
            f"act_c{v}_seniority": ("seniority", "max"),
            f"act_c{v}_min_seniority": ("seniority", "min"),
            f"act_c{v}_n_loans_hist": ("aid", "nunique"),
        }
    )

    n_statC = (
        hist[hist["status"] == "C"]
        .groupby("cid")["aid"].nunique()
        .rename(f"act_c{v}_n_statC")
    )
    n_statB = (
        hist[hist["status"] == "B"]
        .groupby("cid")["aid"].nunique()
        .rename(f"act_c{v}_n_statB")
    )

    out = (
        snap_agg
        .join(hist_agg, how="outer")
        .join(n_statC, how="outer")
        .join(n_statB, how="outer")
        .reset_index()
    )
    return out

def build_cust_active_flag(approved_tx: pd.DataFrame,
                            proc_period1: str) -> pd.DataFrame:
    """Flag customers with at least one active loan at proc_period1.

    Actions:
    1. Filter approved_tx to period == proc_period1 and status == 'A'.
    2. Deduplicate cid → act_cus_active = 1.
    3. Customers not in this table get NaN after left-join (= not active).

    Used by decision engine: css applicants with act_cus_active != 1 are declined.
    """
    snap = approved_tx[
        (approved_tx["period"] == proc_period1) & (approved_tx["status"] == "A")
    ]
    cids = snap["cid"].drop_duplicates()
    return pd.DataFrame({"cid": cids.values, "act_cus_active": 1}).reset_index(drop=True)

def build_cust_all_agg(approved_tx: pd.DataFrame,
                        month_prod: pd.DataFrame,
                        cust_unique: pd.Series,
                        proc_period: str) -> pd.DataFrame:
    """Cross-product cumulative aggregates — one row per new application (aid).

    Actions:
    1. Collect currently active loans: approved_tx, status='A', period=proc_period.
    2. Append this month's new applications from month_prod (rename app_* → contract cols).
    3. Sort each customer's rows by origination time (aid[3:11] as YYYYMMDD proxy).
    4. Within cid, compute running sums of installment/spendings and loan counts by product.
    5. Derive act_call_cc = (installment_cum + spendings_cum) / income on each row.
    6. Keep only the new-application rows — merge key is aid, not cid.
    """
    active = approved_tx[
        (approved_tx["status"] == "A")
        & (approved_tx["period"] == proc_period)
        & (approved_tx["cid"].isin(cust_unique))
    ][["cid", "aid", "product", "installment", "spendings", "income"]].copy()
    active["is_new"] = False

    new_apps = month_prod[month_prod["cid"].isin(cust_unique)][
        ["cid", "aid", "product", "app_n_installments", "app_installment", "app_spendings", "app_income"]
    ].rename(columns={
        "app_n_installments": "n_installments",
        "app_installment": "installment",
        "app_spendings": "spendings",
        "app_income": "income",
    }).copy()
    new_apps["is_new"] = True

    combined = pd.concat([active, new_apps], ignore_index=True)
    combined["time"] = combined["aid"].str[3:11]
    combined = combined.sort_values(["cid", "time"]).reset_index(drop=True)

    combined["is_ins"] = (combined["product"] == "ins").astype(int)
    combined["is_css"] = (combined["product"] == "css").astype(int)

    grp = combined.groupby("cid")
    combined["installment_cum"] = grp["installment"].cumsum()
    combined["spendings_cum"] = grp["spendings"].cumsum()
    combined["act_cins_n_loan"] = grp["is_ins"].cumsum()
    combined["act_ccss_n_loan"] = grp["is_css"].cumsum()
    combined["act_call_n_loan"] = combined["act_cins_n_loan"] + combined["act_ccss_n_loan"]
    combined["act_call_cc"] = (
        combined["installment_cum"] + combined["spendings_cum"]
    ) / combined["income"]

    return combined.loc[
        combined["is_new"],
        ["aid", "act_call_cc", "act_cins_n_loan", "act_ccss_n_loan", "act_call_n_loan"],
    ].reset_index(drop=True)