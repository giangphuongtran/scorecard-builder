"""Cross-sell response labels (SAS all_contents.sas port).

Builds ``cross_response``, ``cross_aid``, ``cross_after_months``, and
``default_cross*`` / ``cross_app_*`` on an application ABT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _period_to_int(period: str) -> int:
    """YYYYMM -> months since year 0 (ordinal for differences)."""
    p = int(str(period))
    y, m = divmod(p, 100)
    return y * 12 + m


def _int_to_period(ord_m: int) -> str:
    y, m = divmod(ord_m, 12)
    if m == 0:
        y -= 1
        m = 12
    return f"{y:04d}{m:02d}"


def _month_index(period: str, first_period: str) -> int:
    """1-based index of period in the calendar starting at first_period."""
    return _period_to_int(period) - _period_to_int(first_period) + 1


def attach_cross_labels(
    abt: pd.DataFrame,
    decisions: pd.DataFrame | None = None,
    *,
    response_n_months: int = 6,
    response_product: str = "css",
) -> pd.DataFrame:
    """Attach SAS-style cross-sell labels to each application row.

    Logic (matches ``all_contents.sas``):
    1. Build accepted ``css`` apps per ``(cid, period)``.
    2. For each application in period ``P``, look for an accepted CSS in
       months ``P+1 .. P+(response_n_months-1)`` (latest wins).
    3. Join that CSS aid's ``default12`` (and loan fields) as ``default_cross*``.
    """
    df = abt.copy()
    if decisions is not None and "decision" not in df.columns:
        cols = [c for c in ("aid", "decision", "decline_reason") if c in decisions.columns]
        df = df.merge(decisions[cols], on="aid", how="left")

    if "decision" not in df.columns:
        raise KeyError("abt/decisions must include decision")

    need = {"cid", "aid", "period", "product", "decision"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"missing columns for cross labels: {sorted(missing)}")

    work = df.copy()
    work["period"] = work["period"].astype(str)
    work["aid"] = work["aid"].astype(str)
    work["cid"] = work["cid"].astype(str)

    # Accepted CSS response matrix: one aid per cid-period
    res = (
        work.loc[
            work["decision"].eq("A") & work["product"].eq(response_product),
            ["cid", "aid", "period"],
        ]
        .sort_values(["cid", "period", "aid"])
        .drop_duplicates(["cid", "period"], keep="first")
    )
    if res.empty:
        out = work.copy()
        out["cross_response"] = 0
        out["cross_aid"] = ""
        out["cross_after_months"] = np.nan
        out["default_cross12"] = np.nan
        return out

    wide = res.pivot(index="cid", columns="period", values="aid")
    periods = sorted(wide.columns.astype(str).tolist())
    first_period = periods[0]
    n_periods = len(periods)
    # Map period -> 1-based array index aligned with SAS (column order)
    period_to_idx = {p: i + 1 for i, p in enumerate(periods)}

    # Lookup default / loan attrs by aid
    aid_attrs = work.set_index("aid")
    default_cols = [c for c in ("default3", "default6", "default9", "default12") if c in work.columns]
    loan_cols = [c for c in ("app_loan_amount", "app_n_installments") if c in work.columns]

    cross_response = np.zeros(len(work), dtype=int)
    cross_aid = np.array([""] * len(work), dtype=object)
    cross_after = np.full(len(work), np.nan)

    # Precompute cid -> array of aids aligned to period indices (NaN if none)
    cid_arrays: dict[str, list] = {}
    for cid, row in wide.iterrows():
        arr = [None] * (n_periods + 1)  # 1-based
        for p, aid in row.items():
            idx = period_to_idx[str(p)]
            arr[idx] = None if pd.isna(aid) else str(aid)
        cid_arrays[str(cid)] = arr

    for i, (_, row) in enumerate(work.iterrows()):
        cid = str(row["cid"])
        period = str(row["period"])
        arr = cid_arrays.get(cid)
        if arr is None:
            continue
        # SAS: index = months_from_first + 2  (== period array index + 1 = next month)
        # max_index = index + response_n_months - 2
        if period not in period_to_idx:
            # Period outside production grid
            p_ord = _period_to_int(period)
            f_ord = _period_to_int(first_period)
            index = (p_ord - f_ord) + 2
        else:
            index = period_to_idx[period] + 1
        max_index = index + response_n_months - 2
        if not (1 <= index <= n_periods and 1 <= max_index <= n_periods):
            # Clamp searchable window to available range
            lo = max(1, min(index, n_periods))
            hi = max(1, min(max_index, n_periods))
            if hi < lo:
                continue
        else:
            lo, hi = index, max_index

        found_aid = None
        found_after = None
        for j in range(hi, lo - 1, -1):
            if 1 <= j <= n_periods and arr[j] is not None:
                found_aid = arr[j]
                found_after = j - index + 1
                break
        if found_aid is not None:
            cross_response[i] = 1
            cross_aid[i] = found_aid
            cross_after[i] = found_after

    out = work.copy()
    out["cross_response"] = cross_response
    out["cross_aid"] = cross_aid
    out["cross_after_months"] = cross_after

    # Join defaults / loan amount from cross_aid
    for src in default_cols:
        dst = src.replace("default", "default_cross")
        out[dst] = np.nan
    for src in loan_cols:
        out[f"cross_{src}"] = np.nan

    responders = out["cross_response"] == 1
    if responders.any():
        ca = out.loc[responders, "cross_aid"].astype(str)
        for src in default_cols:
            dst = src.replace("default", "default_cross")
            mapped = ca.map(
                lambda a, c=src: (
                    aid_attrs.at[a, c]
                    if a in aid_attrs.index and c in aid_attrs.columns
                    else np.nan
                )
            )
            out.loc[responders, dst] = mapped.values
        for src in loan_cols:
            mapped = ca.map(
                lambda a, c=src: (
                    aid_attrs.at[a, c]
                    if a in aid_attrs.index and c in aid_attrs.columns
                    else np.nan
                )
            )
            out.loc[responders, f"cross_{src}"] = mapped.values

    # Normalize default_cross12 like prepare_target
    if "default_cross12" in out.columns:
        out["default_cross12"] = out["default_cross12"].map(
            {".i": 0, ".d": 0, 0: 0, 1: 1, 0.0: 0, 1.0: 1}
        )

    return out


def cross_label_summary(abt_with_labels: pd.DataFrame) -> dict:
    """QA summary for notebook §8."""
    ins = abt_with_labels.loc[
        (abt_with_labels.get("product") == "ins")
        & (abt_with_labels.get("decision") == "A")
    ]
    n_ins = len(ins)
    n_resp = int(ins["cross_response"].sum()) if n_ins and "cross_response" in ins else 0
    responders = ins.loc[ins["cross_response"] == 1] if n_ins else ins
    n_cross_def = (
        int(responders["default_cross12"].notna().sum())
        if "default_cross12" in responders.columns
        else 0
    )
    br = (
        float(responders["default_cross12"].mean())
        if n_cross_def
        else float("nan")
    )
    return {
        "n_ins_accepted": n_ins,
        "n_responders": n_resp,
        "response_rate": n_resp / n_ins if n_ins else float("nan"),
        "n_default_cross12_nonnull": n_cross_def,
        "default_cross12_rate": br,
        "has_cross_response": "cross_response" in abt_with_labels.columns,
        "has_default_cross12": "default_cross12" in abt_with_labels.columns,
    }
