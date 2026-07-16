"""Human-friendly labels and short interpretations for scorecard feature names."""

from __future__ import annotations

import re

import pandas as pd

_STAT_TOKENS: dict[str, str] = {
    "n_loan": "loan count",
    "n_loans_act": "active loans",
    "n_loans_hist": "historical loans",
    "maxdue": "max due installments",
    "min_pninst": "min paid installments",
    "min_lninst": "min remaining installments",
    "utl": "utilization (paid / total installments)",
    "dueutl": "due utilization",
    "cc": "credit capacity ratio",
    "seniority": "max loan seniority (months)",
    "min_seniority": "min loan seniority",
    "n_statc": "closed loans (status C)",
    "n_statb": "closed loans (status B)",
}

_PRODUCT_TOKENS: dict[str, str] = {
    "cins": "installment (INS)",
    "ccss": "cash/card (CSS)",
    "call": "all products",
}

_APP_TOKENS: dict[str, str] = {
    "loan_amount": "requested loan amount",
    "n_installments": "requested term (months)",
    "installment": "requested installment",
    "spendings": "requested spendings",
    "income": "declared income",
    "age": "applicant age",
    "gender": "applicant gender",
    "marital": "marital status",
    "education": "education level",
    "housing": "housing type",
    "seniority": "employment seniority",
}


def strip_woe(name: str) -> str:
    """Remove ``_WOE`` suffix from encoded feature names."""
    if isinstance(name, str) and name.endswith("_WOE"):
        return name[: -len("_WOE")]
    return str(name)


def _title_words(text: str) -> str:
    return " ".join(w.capitalize() for w in text.replace("_", " ").split())


def display_label(raw: str) -> str:
    """Short display label for forms and tables."""
    name = strip_woe(raw)
    low = name.lower()

    if low.startswith("app_"):
        tail = low[4:]
        for key, label in _APP_TOKENS.items():
            if tail == key or tail.endswith(f"_{key}"):
                return label.capitalize() if label == label.lower() else label
        return _title_words(tail)

    if low.startswith("act_"):
        body = low[4:]
        for prod_key, prod_short in (("cins", "INS"), ("ccss", "CSS"), ("call", "All products")):
            if body.startswith(prod_key + "_"):
                stat = body[len(prod_key) + 1 :]
                stat_label = _STAT_TOKENS.get(stat, _title_words(stat))
                return f"{prod_short} {stat_label}"
        if body == "cus_active":
            return "Customer active flag"
        return _title_words(body)

    if low.startswith("agr"):
        m = re.match(r"agr(\d+)_(.+)", low)
        if m:
            window, tail = m.group(1), m.group(2)
            return f"{window}m arrears aggregate ({_title_words(tail)})"
        return _title_words(low)

    if low.startswith("ags"):
        return _title_words(low[3:])

    return _title_words(name)


def interpret_feature(raw: str) -> str:
    """One-line business meaning for tooltips and report tables."""
    name = strip_woe(raw)
    low = name.lower()

    if low.startswith("app_"):
        return "Application-time attribute captured at origination."

    if low.startswith("act_cins_"):
        stat = low.replace("act_cins_", "")
        detail = _STAT_TOKENS.get(stat, stat.replace("_", " "))
        return f"Installment-lane behavioral snapshot or history: {detail}."

    if low.startswith("act_ccss_"):
        stat = low.replace("act_ccss_", "")
        detail = _STAT_TOKENS.get(stat, stat.replace("_", " "))
        return f"Cash/card-lane behavioral snapshot or history: {detail}."

    if low.startswith("act_call_"):
        stat = low.replace("act_call_", "")
        detail = _STAT_TOKENS.get(stat, stat.replace("_", " "))
        return f"Cross-product cumulative behavior at application: {detail}."

    if low == "act_cus_active":
        return "Whether the customer has an active loan in the decision month (CSS gate)."

    if low.startswith("agr"):
        return "A rolling total built from the customer's recent months across all products; used in risk rules and policy checks."

    if low.startswith("ags"):
        return "A summary statistic built from the customer's payment behavior history."

    return "An input used to build the risk score for this loan type."


def variables_table_frame(raw_features: list[str]) -> pd.DataFrame:
    """Build feature / label / interpretation table for the workbench."""
    if not raw_features:
        return pd.DataFrame(columns=["Feature", "Label", "What it measures"])
    rows = [
        {
            "Feature": feat,
            "Label": display_label(feat),
            "What it measures": interpret_feature(feat),
        }
        for feat in raw_features
    ]
    return pd.DataFrame(rows)