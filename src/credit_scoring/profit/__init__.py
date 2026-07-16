"""Offline profit strategy and closed-loop re-sim helpers."""

from .cutoff import find_cutoff_at_ar, find_optimal_cutoff, profit_curve_by_pd
from .cutoff_explore import (
    build_asif_scored_frame,
    cutoffs_yaml_snippet,
    evaluate_cutoffs,
    export_asif_scored,
    load_asif_scored,
    load_workbench_product_bundle,
    save_workbench_product_bundle,
)
from .pnl import compute_pnl_table, filter_profit_window, installment_amount, loan_pnl
from .rules import apply_strategy, compare_strategies, evaluate_strategy, rules_from_params
from .scoring import (
    apply_bins_notebook,
    normalize_calib_params,
    score_abt_application,
    score_product_slice,
    score_secondary_model,
    score_to_pd,
)

__all__ = [
    "apply_bins_notebook",
    "apply_strategy",
    "build_asif_scored_frame",
    "compare_strategies",
    "compute_pnl_table",
    "cutoffs_yaml_snippet",
    "evaluate_cutoffs",
    "evaluate_strategy",
    "export_asif_scored",
    "filter_profit_window",
    "find_cutoff_at_ar",
    "find_optimal_cutoff",
    "installment_amount",
    "load_asif_scored",
    "load_workbench_product_bundle",
    "loan_pnl",
    "normalize_calib_params",
    "profit_curve_by_pd",
    "rules_from_params",
    "save_workbench_product_bundle",
    "score_abt_application",
    "score_product_slice",
    "score_secondary_model",
    "score_to_pd",
]
