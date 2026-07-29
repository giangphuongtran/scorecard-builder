"""Offline champion vs challenger A/B policy comparison."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd

from credit_scoring.mlflow_utils import log_policy_run
from credit_scoring.profit.cutoff_explore import DEFAULT_ASIF_PATH, DEFAULT_META_PATH, evaluate_cutoffs, load_asif_scored


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj:
            return None
        return obj
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    return obj


def _metrics_from_ev(ev: dict) -> dict[str, float]:
    out = {}
    for key in (
        "total_profit",
        "total_income",
        "total_el",
        "ar_ins",
        "ar_css",
        "bad_rate_ins",
        "bad_rate_css",
        "n_accept",
        "n_apps",
    ):
        val = ev.get(key)
        try:
            fv = float(val)
        except (TypeError, ValueError):
            continue
        if fv == fv:
            out[key] = fv
    return out


def _decline_mix(ev: dict) -> list[dict[str, Any]]:
    reasons = ev.get("decline_reasons")
    if not isinstance(reasons, pd.DataFrame) or not len(reasons):
        return []
    return reasons.to_dict(orient="records")


def compare_policies_node(ab_test_params, profit_params):
    """Compare champion vs challenger cutoffs on the frozen as-if scored frame."""
    asif_path = Path(ab_test_params.get("asif_path") or DEFAULT_ASIF_PATH)
    meta_path = Path(ab_test_params.get("meta_path") or DEFAULT_META_PATH)
    scored, meta = load_asif_scored(asif_path, meta_path)

    window_start = profit_params.get("window_start", meta.get("window_start", "197501"))
    window_end = profit_params.get("window_end", meta.get("window_end", "198712"))
    burn_in = profit_params.get("burn_in_before", meta.get("burn_in_before", "197501"))
    economics = profit_params.get("economics") or {}
    bad_customer = profit_params.get("bad_customer") or {}
    reference = float(ab_test_params.get("reference", profit_params.get("reference", 731882)))

    champion_cfg = ab_test_params.get("champion") or {}
    challenger_cfg = ab_test_params.get("challenger") or {}
    champ_cuts = dict(champion_cfg.get("cutoffs") or profit_params.get("cutoffs") or {})
    chall_cuts = dict(challenger_cfg.get("cutoffs") or {})
    champ_name = str(champion_cfg.get("name") or "cltv_production")
    chall_name = str(challenger_cfg.get("name") or "challenger")

    champ_ev = evaluate_cutoffs(
        scored,
        champ_cuts,
        window_start=window_start,
        window_end=window_end,
        burn_in_before=burn_in,
        economics=economics,
        bad_customer=bad_customer,
    )
    chall_ev = evaluate_cutoffs(
        scored,
        chall_cuts,
        window_start=window_start,
        window_end=window_end,
        burn_in_before=burn_in,
        economics=economics,
        bad_customer=bad_customer,
    )

    champ_profit = float(champ_ev.get("total_profit") or 0.0)
    chall_profit = float(chall_ev.get("total_profit") or 0.0)
    champ_bad = float(champ_ev.get("bad_rate_ins") or 0.0) + float(champ_ev.get("bad_rate_css") or 0.0)
    chall_bad = float(chall_ev.get("bad_rate_ins") or 0.0) + float(chall_ev.get("bad_rate_css") or 0.0)

    if chall_profit > champ_profit:
        winner = chall_name
    elif chall_profit < champ_profit:
        winner = champ_name
    else:
        # Tie-break on lower combined bad rate.
        winner = chall_name if chall_bad < champ_bad else champ_name

    summary = {
        "champion": {
            "name": champ_name,
            "cutoffs": champ_cuts,
            "metrics": _metrics_from_ev(champ_ev),
            "decline_reasons": _decline_mix(champ_ev),
        },
        "challenger": {
            "name": chall_name,
            "cutoffs": chall_cuts,
            "metrics": _metrics_from_ev(chall_ev),
            "decline_reasons": _decline_mix(chall_ev),
        },
        "delta": {
            "delta_profit": chall_profit - champ_profit,
            "delta_n_accept": int(chall_ev.get("n_accept") or 0) - int(champ_ev.get("n_accept") or 0),
            "delta_ar_ins": float(chall_ev.get("ar_ins") or 0) - float(champ_ev.get("ar_ins") or 0),
            "delta_ar_css": float(chall_ev.get("ar_css") or 0) - float(champ_ev.get("ar_css") or 0),
            "delta_bad_rate_ins": float(chall_ev.get("bad_rate_ins") or 0)
            - float(champ_ev.get("bad_rate_ins") or 0),
            "delta_bad_rate_css": float(chall_ev.get("bad_rate_css") or 0)
            - float(champ_ev.get("bad_rate_css") or 0),
        },
        "winner": winner,
        "reference": reference,
        "note": (
            "Offline historical replay on frozen as-if scored frame. "
            "Logged to MLflow experiment credit_scoring. Not a live closed-loop A/B."
        ),
        "source_frame": str(asif_path),
    }

    report_html = _render_ab_html(summary)

    # Log both variants for experiment tracking.
    for variant, ev, cuts in (
        ("champion", champ_ev, champ_cuts),
        ("challenger", chall_ev, chall_cuts),
    ):
        name = champ_name if variant == "champion" else chall_name
        log_policy_run(
            variant=variant,
            cutoffs=cuts,
            metrics=_metrics_from_ev(ev),
            artifacts={"ab_test_summary.json": summary} if variant == "challenger" else None,
            tags={
                "pipeline": "ab_test",
                "policy_name": name,
                "winner": winner,
            },
            run_name=f"ab_test_{variant}_{name}",
        )

    return _json_safe(summary), report_html


def _render_ab_html(summary: dict) -> str:
    champ = summary["champion"]
    chall = summary["challenger"]
    delta = summary["delta"]

    def _metric_rows(block: dict) -> str:
        m = block.get("metrics") or {}
        return "".join(
            f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
            for k, v in m.items()
        )

    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Policy A/B (offline)</title></head><body>"
        "<h1>Champion vs challenger (offline)</h1>"
        f"<p>{html.escape(str(summary.get('note', '')))}</p>"
        f"<p><b>Winner:</b> {html.escape(str(summary.get('winner')))}</p>"
        f"<h2>Champion — {html.escape(champ['name'])}</h2>"
        f"<table border='1' cellpadding='4'>{_metric_rows(champ)}</table>"
        f"<h2>Challenger — {html.escape(chall['name'])}</h2>"
        f"<table border='1' cellpadding='4'>{_metric_rows(chall)}</table>"
        "<h2>Challenger − champion</h2>"
        "<table border='1' cellpadding='4'>"
        + "".join(
            f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
            for k, v in delta.items()
        )
        + "</table></body></html>"
    )
