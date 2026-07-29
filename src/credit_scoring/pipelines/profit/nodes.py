"""Thin Kedro nodes for the offline profit pipeline (CLTV production policy)."""

from __future__ import annotations

import html
import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from credit_scoring.mlflow_utils import log_policy_run, maybe_log_run
from credit_scoring.profit.cutoff_explore import export_asif_scored
from credit_scoring.profit.pnl import compute_pnl_table, filter_profit_window
from credit_scoring.profit.rules import apply_strategy, evaluate_strategy, rules_from_params
from credit_scoring.profit.scoring import score_abt_application

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return _REPO_ROOT / p


def load_profit_artifacts(artifacts_cfg: dict) -> tuple[dict, dict, dict, dict | None]:
    """Load Gate B packages / points / calib from ``params:profit.artifacts`` paths."""
    packages: dict = {}
    points_tables: dict = {}
    calibrations: dict = {}
    secondary: dict = {}

    for key in ("ins", "css", "pr", "cross"):
        if key not in artifacts_cfg:
            continue
        spec = artifacts_cfg[key]
        pkg = pickle.loads(_resolve_path(spec["package"]).read_bytes())
        pts = pd.read_parquet(_resolve_path(spec["points"]))
        cal = json.loads(_resolve_path(spec["calib"]).read_text(encoding="utf-8"))
        if key in ("ins", "css"):
            packages[key] = pkg
            points_tables[key] = pts
            calibrations[key] = cal
        else:
            secondary[key] = {"package": pkg, "points": pts, "calib": cal}

    return packages, points_tables, calibrations, (secondary or None)


def score_application_abt_node(abt_app, profit_params):
    """Score ABT with frozen v6 PD + secondary PR / Cross models from params."""
    artifacts_cfg = profit_params.get("artifacts") or {}
    packages, points_tables, calibrations, secondary = load_profit_artifacts(artifacts_cfg)
    return score_abt_application(
        abt_app,
        packages,
        points_tables,
        calibrations,
        secondary=secondary,
    )


def apply_strategy_node(scored_abt, profit_params):
    """Apply CLTV mid-band production strategy rules."""
    rules = rules_from_params(profit_params)
    return apply_strategy(scored_abt, rules)


def report_profit_node(scored_abt, decisions_strategy, profit_params):
    """Compute loan P&L, evaluate CLTV policy, write summary + HTML + as-if frame."""
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
        "best_strategy": "CLTV mid-band (production)",
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
            "Offline as-if cut-off on fixed historical ABT (CLTV production policy). "
            "Closed-loop re-sim is a separate check; reference is the published benchmark."
        ),
    }

    report_html = _render_profit_html(summary)

    # Keep Streamlit tuner frame in sync with the scored CLTV columns.
    asif_meta = {
        "window_start": profit_params["window_start"],
        "window_end": profit_params["window_end"],
        "burn_in_before": profit_params.get("burn_in_before", "197501"),
        "reference": float(profit_params.get("reference", 0)),
        "artifacts": profit_params.get("artifacts") or {},
        "secondary_keys": ["pr", "cross"],
        "n_rows": int(len(scored_pnl)),
        "save_version": "v6",
        "cutoffs": dict(rules["cutoffs"]),
    }
    try:
        export_asif_scored(scored_pnl, asif_meta)
    except Exception:
        pass

    try:
        with tempfile.TemporaryDirectory() as td:
            report_path = Path(td) / "profit_report.html"
            report_path.write_text(report_html or "", encoding="utf-8")
            log_policy_run(
                variant="cltv_production",
                cutoffs=dict(rules.get("cutoffs") or {}),
                metrics={
                    "total_profit": float(evaluation.get("total_profit", 0.0)),
                    "total_income": float(evaluation.get("total_income", 0.0)),
                    "total_el": float(evaluation.get("total_el", 0.0)),
                    "ar_ins": float(evaluation["ar_ins"])
                    if evaluation.get("ar_ins") == evaluation.get("ar_ins")
                    else np.nan,
                    "ar_css": float(evaluation["ar_css"])
                    if evaluation.get("ar_css") == evaluation.get("ar_css")
                    else np.nan,
                    "bad_rate_ins": float(evaluation["bad_rate_ins"])
                    if evaluation.get("bad_rate_ins") == evaluation.get("bad_rate_ins")
                    else np.nan,
                    "bad_rate_css": float(evaluation["bad_rate_css"])
                    if evaluation.get("bad_rate_css") == evaluation.get("bad_rate_css")
                    else np.nan,
                    "n_accept": float(evaluation.get("n_accept") or 0),
                },
                artifacts={"profit_report.html": report_path},
                tags={"pipeline": "profit", "node": "report_profit_node"},
                run_name="profit_cltv",
            )
    except Exception:
        maybe_log_run(run_name="profit_cltv", params={}, metrics={}, tags={"pipeline": "profit"})

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
        "<title>Profit strategy (offline CLTV)</title></head><body>"
        "<h1>CLTV mid-band (production) — offline</h1>"
        f"<p>{note}</p>"
        f"<table border='1' cellpadding='4'>{rows}</table>"
        "</body></html>"
    )
