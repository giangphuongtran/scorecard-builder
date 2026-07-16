"""Thin Kedro nodes for the offline profit pipeline."""

from __future__ import annotations

import html
import tempfile
from pathlib import Path

import numpy as np

from credit_scoring.profit.pnl import compute_pnl_table, filter_profit_window
from credit_scoring.profit.rules import apply_strategy, evaluate_strategy, rules_from_params
from credit_scoring.profit.scoring import score_abt_application
from credit_scoring.mlflow_utils import maybe_log_run


def score_application_abt_node(
    abt_app,
    pd_ins_v2,
    pd_css_v5,
    points_table_ins_v2,
    points_table_css_v5,
    calibration_params_ins_v2,
    calibration_params_css_v5,
):
    """Score full ABT with frozen Gate B packages."""
    packages = {"ins": pd_ins_v2, "css": pd_css_v5}
    points_tables = {"ins": points_table_ins_v2, "css": points_table_css_v5}
    calibrations = {
        "ins": calibration_params_ins_v2,
        "css": calibration_params_css_v5,
    }
    return score_abt_application(abt_app, packages, points_tables, calibrations)


def apply_strategy_node(scored_abt, profit_params):
    """Apply St_yours (or params-driven) strategy rules."""
    rules = rules_from_params(profit_params)
    return apply_strategy(scored_abt, rules)


def report_profit_node(scored_abt, decisions_strategy, profit_params):
    """Compute loan P&L, evaluate strategy, write summary + HTML."""
    rules = rules_from_params(profit_params)
    economics = profit_params["economics"]
    scored_pnl = compute_pnl_table(scored_abt, economics)
    window = filter_profit_window(
        scored_pnl,
        profit_params["window_start"],
        profit_params["window_end"],
    )
    evaluation = evaluate_strategy(
        window,
        decisions_strategy,
        profit_params["window_start"],
        profit_params["window_end"],
    )

    merged = window.drop(columns=["decision", "decline_reason"], errors="ignore").merge(
        decisions_strategy[["aid", "decision", "decline_reason"]],
        on="aid",
        how="left",
    )

    summary = {
        "best_strategy": "St_yours (bad + PD)",
        "source": "offline",
        "cutoffs": dict(rules["cutoffs"]),
        "bad_customer": dict(rules.get("bad_customer") or {}),
        "offline_total_profit": evaluation["total_profit"],
        "total_income": evaluation["total_income"],
        "total_el": evaluation["total_el"],
        "ar_ins": evaluation["ar_ins"],
        "ar_css": evaluation["ar_css"],
        "bad_rate_ins": evaluation["bad_rate_ins"],
        "bad_rate_css": evaluation["bad_rate_css"],
        "n_apps": evaluation["n_apps"],
        "n_accept": evaluation["n_accept"],
        "n_N": evaluation["n_N"],
        "reference": float(profit_params.get("reference", 0)),
        "window": [profit_params["window_start"], profit_params["window_end"]],
        "note": (
            "Offline as-if cut-off on fixed historical ABT. "
            "Course-style closed-loop needs 04b re-sim; reference is published course benchmark."
        ),
    }

    report_html = _render_profit_html(summary)

    # Optional MLflow logging for profit evaluation + selected policy cut-offs.
    try:
        report_summary = {
            "total_profit": float(evaluation.get("total_profit", 0.0)),
            "total_income": float(evaluation.get("total_income", 0.0)),
            "total_el": float(evaluation.get("total_el", 0.0)),
            "ar_ins": float(evaluation.get("ar_ins", np.nan))
            if evaluation.get("ar_ins") == evaluation.get("ar_ins")
            else np.nan,
            "ar_css": float(evaluation.get("ar_css", np.nan))
            if evaluation.get("ar_css") == evaluation.get("ar_css")
            else np.nan,
            "bad_rate_ins": float(evaluation.get("bad_rate_ins", np.nan))
            if evaluation.get("bad_rate_ins") == evaluation.get("bad_rate_ins")
            else np.nan,
            "bad_rate_css": float(evaluation.get("bad_rate_css", np.nan))
            if evaluation.get("bad_rate_css") == evaluation.get("bad_rate_css")
            else np.nan,
        }
        with tempfile.TemporaryDirectory() as td:
            report_path = Path(td) / "profit_report.html"
            report_path.write_text(report_html or "", encoding="utf-8")
            maybe_log_run(
                run_name="profit_strategy",
                params={
                    "window_start": profit_params.get("window_start"),
                    "window_end": profit_params.get("window_end"),
                    "burn_in_before": profit_params.get("burn_in_before"),
                    **{f"cutoff_{k}": v for k, v in dict(rules.get("cutoffs") or {}).items()},
                },
                metrics={k: v for k, v in report_summary.items() if v == v},
                artifacts={"profit_report.html": report_path},
                tags={"pipeline": "profit", "node": "report_profit_node"},
            )
    except Exception:
        pass

    return merged, summary, report_html


def _render_profit_html(summary: dict) -> str:
    rows = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in summary.items()
        if k not in ("note",)
    )
    note = html.escape(str(summary.get("note", "")))
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Profit strategy (offline)</title></head><body>"
        "<h1>Profit strategy — offline</h1>"
        f"<p>{note}</p>"
        f"<table border='1' cellpadding='4'>{rows}</table>"
        "</body></html>"
    )
