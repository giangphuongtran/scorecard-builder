import pandas as pd
import numpy as np
from .pivot import build_period_list

def add_counter_features(abt_beh: pd.DataFrame,
                        proc_period1: str,
                        params: dict) -> pd.DataFrame:
    """Hand-crafted policy counters on the All-product (CMaxA) stream.

    Actions (for each window in {3,6,9,12}):
    1. Pull CMaxA_Due_{p} and CMaxA_Days_{p} for months in the window.
    2. act{n}_n_arrears      = count months where due >= 1 (any delinquency).
    3. act{n}_n_arrears_days  = count months where days > 15 (past grace period).
    4. act{n}_n_good_days     = count months where 0 < days < 15 (paid inside grace).
    These three consistently rank among top-IV features in the reference scorecards.
    """
    columns = {"cid": abt_beh["cid"]}

    for window in params["window_lengths"]:
        window_periods = build_period_list(proc_period1, window)

        due_cols = [f"CMaxA_Due_{p}" for p in window_periods]
        days_cols = [f"CMaxA_Days_{p}" for p in window_periods]

        due_existing = [c for c in due_cols if c in abt_beh.columns]
        days_existing = [c for c in days_cols if c in abt_beh.columns]

        due_data = abt_beh[due_existing] if due_existing else pd.DataFrame(np.nan, index=abt_beh.index, columns=due_cols)
        days_data = abt_beh[days_existing] if days_existing else pd.DataFrame(np.nan, index=abt_beh.index, columns=days_cols)

        columns[f"act{window}_n_arrears"] = (due_data >= 1).sum(axis=1)
        columns[f"act{window}_n_arrears_days"] = (days_data > 15).sum(axis=1)
        columns[f"act{window}_n_good_days"] = ((days_data > 0) & (days_data < 15)).sum(axis=1)

    return pd.DataFrame(columns)


def make_abt(abt_beh: pd.DataFrame,
              proc_period1: str,
              params: dict) -> pd.DataFrame:
    """Combine rolling + counter features into the 157-column behavioral table.

    Actions:
    1. Call make_rolling_features → 145 cols (cid + 72 ags + 72 agr).
    2. Call add_counter_features → 13 cols (cid + 12 act* counters).
    3. Merge on cid (one-to-one). Raw CMax*_YYYYMM pivot columns are dropped.
    4. Assert shape: 1 + 72 + 72 + 12 = 157 columns.
    """
    abt_rolling = make_rolling_features(abt_beh, proc_period1, params)
    abt_counters = add_counter_features(abt_beh, proc_period1, params)

    abt = abt_rolling.merge(abt_counters, on="cid", how="left", validate="one_to_one")

    expected_cols = 1 + 72 + 72 + 12
    assert abt.shape[1] == expected_cols, f"Expected {expected_cols} columns, got {abt.shape[1]}"
    assert abt["cid"].is_unique, "cid is not unique in final ABT"

    return abt


def make_rolling_features(abt_beh: pd.DataFrame,
                            proc_period1: str,
                            params: dict) -> pd.DataFrame:
    """Compress monthly columns into 144 rolling-window features (72 ags + 72 agr).

    Actions (for each base_var × window × stat):
    1. Collect source columns {base_var}_{YYYYMM} for the window ending at proc_period1.
    2. Count nmiss = NaN months in window + months with no column at all.
    3. Compute ags* (lenient): Mean/Max/Min with skipna=True over available months.
    4. Compute agr* (strict): copy ags*, but set to NaN when nmiss > nmiss_threshold
       — signals "not enough history" to the WOE binner's Missing bucket.
    """
    window_lengths = params["window_lengths"]
    stats = params["stats"]
    base_vars = params["base_vars"]
    nmiss_threshold = params["nmiss_threshold"]

    columns = {"cid": abt_beh["cid"]}

    for base_var in base_vars:
        for window in window_lengths:
            window_periods = build_period_list(proc_period1, window)
            window_cols = [f"{base_var}_{period}" for period in window_periods]

            existing_cols = [c for c in window_cols if c in abt_beh.columns]
            missing_cols = [c for c in window_cols if c not in abt_beh.columns]

            if existing_cols:
                window_data = abt_beh[existing_cols]
            else:
                window_data = pd.DataFrame(np.nan, index=abt_beh.index, columns=window_cols)

            nmiss = window_data.isna().sum(axis=1) + len(missing_cols)

            for stat in stats:
                ags_name = f"ags{window}_{stat}_{base_var}"
                agr_name = f"agr{window}_{stat}_{base_var}"

                if stat == "Mean":
                    ags_values = window_data.mean(axis=1, skipna=True)
                elif stat == "Max":
                    ags_values = window_data.max(axis=1, skipna=True)
                elif stat == "Min":
                    ags_values = window_data.min(axis=1, skipna=True)
                else:
                    raise ValueError(f"Unknown stat: {stat}")

                agr_values = ags_values.where(nmiss <= nmiss_threshold, np.nan)

                columns[ags_name] = ags_values
                columns[agr_name] = agr_values

    return pd.DataFrame(columns)