"""Offline profit strategy and closed-loop re-sim helpers."""

from .cutoff import find_optimal_cutoff, profit_curve_by_pd
from .pnl import compute_pnl_table, filter_profit_window, installment_amount, loan_pnl
from .rules import apply_strategy, compare_strategies, evaluate_strategy, rules_from_params
from .scoring import (
    apply_bins_notebook,
    normalize_calib_params,
    score_abt_application,
    score_product_slice,
    score_to_pd,
)

__all__ = [
    "apply_bins_notebook",
    "apply_strategy",
    "compare_strategies",
    "compute_pnl_table",
    "evaluate_strategy",
    "filter_profit_window",
    "find_optimal_cutoff",
    "installment_amount",
    "loan_pnl",
    "normalize_calib_params",
    "profit_curve_by_pd",
    "rules_from_params",
    "score_abt_application",
    "score_product_slice",
    "score_to_pd",
]
