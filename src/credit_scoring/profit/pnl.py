"""Loan P&L accounting (course formula)."""

from __future__ import annotations

import pandas as pd


def installment_amount(loan: float, n: int, apr_monthly: float) -> float:
    """Annuity installment: loan * apr * (1+apr)^n / ((1+apr)^n - 1)."""
    loan = float(loan)
    n = int(n)
    apr_monthly = float(apr_monthly)
    if n <= 0:
        raise ValueError("n_installments must be positive")
    if apr_monthly == 0:
        return loan / n
    factor = (1.0 + apr_monthly) ** n
    return loan * apr_monthly * factor / (factor - 1.0)


def loan_pnl(
    loan: float,
    n_installments: int,
    product: str,
    default12,
    economics: dict,
) -> dict:
    """Per-loan Income / EL / Profit matching SAS calibration accounting."""
    if product not in economics:
        raise KeyError(f"Unknown product {product}")
    eco = economics[product]
    lgd = float(eco["lgd"])
    apr_monthly = float(eco["apr_annual"]) / 12.0
    provision = float(eco.get("provision", 0.0))

    if pd.isna(default12) or default12 in (".i", ".d"):
        default12 = 0
    default12 = int(default12)

    inst = installment_amount(loan, n_installments, apr_monthly)
    el = float(loan) * lgd if default12 == 1 else 0.0
    income = 0.0
    if default12 == 0:
        income = n_installments * inst + float(loan) * (provision - 1.0)
    profit = income - el
    return {
        "income": income,
        "el": el,
        "profit": profit,
        "installment": inst,
        "lgd": lgd,
        "apr_monthly": apr_monthly,
    }


def compute_pnl_table(scored: pd.DataFrame, economics: dict) -> pd.DataFrame:
    """Add income / el / profit / installment columns to a scored frame."""
    out = scored.copy()
    rows = [
        loan_pnl(
            loan=r.app_loan_amount,
            n_installments=int(r.app_n_installments),
            product=r.product,
            default12=r.default12,
            economics=economics,
        )
        for r in out.itertuples(index=False)
    ]
    pnl = pd.DataFrame(rows)
    for col in ("income", "el", "profit", "installment"):
        out[col] = pnl[col].to_numpy()
    return out


def filter_profit_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Keep rows with period in [start, end] inclusive (YYYYMM strings)."""
    return df.loc[df["period"].astype(str).between(start, end)].copy()
