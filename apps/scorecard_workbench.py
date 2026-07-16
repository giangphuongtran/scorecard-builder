"""Streamlit scorecard report: model quality, stability, profit policy, and officer tool."""

from __future__ import annotations

import json
import pickle
import sys
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
APPS_DIR = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(APPS_DIR) not in sys.path:
    sys.path.insert(0, str(APPS_DIR))

import streamlit as st

from workbench_copy import (
    DATA_DISCLAIMER,
    DECISION_PLAIN,
    GLOSSARY,
    MEASURED_OUTCOMES,
    PIPELINE_PHASES,
    PRODUCT_NAMES,
    TAGLINE,
    TITLE,
    github_badge,
)
from credit_scoring.profit.cutoff_explore import (
    DEFAULT_ASIF_PATH,
    DEFAULT_BUNDLE_DIR,
    DEFAULT_META_PATH,
    evaluate_cutoffs,
    load_asif_scored,
    load_workbench_product_bundle,
    u_curves_by_product,
)
from credit_scoring.profit.pnl import installment_amount
from credit_scoring.profit.rules import apply_strategy, rules_from_params
from credit_scoring.profit.scoring import score_abt_application
from credit_scoring.scorecard.big_scorecard import _bin_condition
from credit_scoring.scorecard.feature_labels import (
    display_label,
    interpret_feature,
    strip_woe,
    variables_table_frame,
)
from credit_scoring.scorecard.reports import (
    CUTOFF_DISPLAY,
    POLICY_FLOW_STEPS,
    format_report_frame,
    fig_bin_metric_over_time,
    fig_gini_time,
    fig_roc_before_after,
    fig_score_hist,
    fig_score_vs_bad_rate,
    fig_variable_importance,
    section_blurb,
    splitting_points_display,
)

st.set_page_config(
    page_title="Scorecard workbench",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_CUTOFF_ORDER = ("pd_css", "pd_ins_high", "pd_ins_low", "pr_min", "cross_pd_max")


def _fmt_pct(value, decimals: int = 2) -> str:
    """Format a 0-1 fraction as a percentage string, tolerant of NaN/None."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if v != v:  # NaN check
        return "n/a"
    return f"{v:.{decimals}%}"


def _fmt_money(value, currency: str = "PLN") -> str:
    """Format a number as thousands-separated currency, tolerant of NaN/None."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if v != v:
        return "n/a"
    return f"{v:,.0f} {currency}"


@st.cache_data(show_spinner=False)
def _load_yml() -> dict:
    return yaml.safe_load((ROOT / "conf" / "base" / "parameters.yml").read_text())


@st.cache_data(show_spinner=False)
def _load_asif():
    path = ROOT / DEFAULT_ASIF_PATH
    meta_path = ROOT / DEFAULT_META_PATH
    if not path.exists():
        return None
    return load_asif_scored(path, meta_path)


@st.cache_data(show_spinner=False)
def _load_bundle(product: str) -> dict:
    return load_workbench_product_bundle(product, bundle_dir=ROOT / DEFAULT_BUNDLE_DIR)


@st.cache_data(show_spinner=False)
def _evaluate_frozen_policy(_yml_text: str, _asif_mtime: float) -> dict | None:
    yml = yaml.safe_load(_yml_text)
    asif_path = ROOT / DEFAULT_ASIF_PATH
    if not asif_path.exists():
        return None
    asif, _ = load_asif_scored(asif_path, ROOT / DEFAULT_META_PATH)
    if asif is None or not len(asif):
        return None
    profit = yml.get("profit") or {}
    cuts = dict(profit.get("cutoffs") or {})
    return evaluate_cutoffs(
        asif,
        cuts,
        window_start=profit.get("window_start", "197501"),
        window_end=profit.get("window_end", "198712"),
        burn_in_before=profit.get("burn_in_before", "197501"),
        economics=profit.get("economics") or {},
        bad_customer=profit.get("bad_customer") or {},
    )


def _raw_features(bundle: dict) -> list[str]:
    pkg = bundle.get("model_package") or {}
    feats = pkg.get("features") or []
    return [strip_woe(f) for f in feats]


def _show_df(df: pd.DataFrame | None) -> None:
    if df is not None and len(df):
        st.dataframe(format_report_frame(df), use_container_width=True)


def _show_fig(fig) -> None:
    if fig is not None:
        st.pyplot(fig, clear_figure=True)


def _section(title: str) -> None:
    st.subheader(title)
    blurb = section_blurb(title)
    if blurb:
        st.caption(blurb)


def _frozen_cutoffs(yml: dict) -> dict:
    return dict((yml.get("profit") or {}).get("cutoffs") or {})


def _profit_eval(yml: dict) -> dict | None:
    yml_text = (ROOT / "conf" / "base" / "parameters.yml").read_text()
    asif_path = ROOT / DEFAULT_ASIF_PATH
    mtime = asif_path.stat().st_mtime if asif_path.exists() else 0.0
    return _evaluate_frozen_policy(yml_text, mtime)


def _render_landing() -> None:
    st.title(TITLE)
    st.markdown(TAGLINE)
    st.markdown(github_badge(), unsafe_allow_html=True)

    with st.expander("About this project — how it works, step by step", expanded=False):
        st.markdown("**The pipeline, in plain English**")
        for step_title, step_detail in PIPELINE_PHASES:
            st.markdown(f"**{step_title}** — {step_detail}")

        st.markdown("")
        st.markdown("**What we measured**")
        cols = st.columns(len(MEASURED_OUTCOMES))
        for col, (label, value, note) in zip(cols, MEASURED_OUTCOMES):
            col.metric(label, value, help=note)

        st.markdown("")
        st.info(DATA_DISCLAIMER)

    with st.expander("New to credit-risk terms? Quick glossary"):
        for term, definition in GLOSSARY.items():
            st.markdown(f"**{term}** — {definition}")


def _render_toolbar(available: list[str]) -> tuple[str, dict]:
    st.markdown("**Scorecard dossier**")
    
    # Map raw keys to full names for display, fallback to key if name is missing
    display_options = {p: PRODUCT_NAMES.get(p, p.upper()) for p in available}
    
    selected_display = st.selectbox(
        "Scorecard dossier", 
        options=list(display_options.keys()), 
        format_func=lambda x: display_options[x],
        index=0, 
        label_visibility="collapsed"
    )
    
    bundle = _load_bundle(selected_display)
    if not bundle:
        st.warning(
            f"No bundle at `{DEFAULT_BUNDLE_DIR / selected_display}`. "
            "Save via `save_workbench_product_bundle` after fit in notebook 05."
        )
    return selected_display, bundle


def _render_policy_flow() -> None:
    st.markdown("**How decisions are made**")
    for i, step in enumerate(POLICY_FLOW_STEPS, 1):
        st.markdown(f"{i}. {step}")


def _render_scenario_examples(cuts: dict) -> None:
    st.markdown("**What happens to a typical applicant?**")
    pd_css = float(cuts.get("pd_css", 0))
    pd_ins_high = float(cuts.get("pd_ins_high", 0))
    pd_ins_low = float(cuts.get("pd_ins_low", 0))
    pr_min = float(cuts.get("pr_min", 0))
    cross_max = float(cuts.get("cross_pd_max", 0))
    rows = [
        {
            "Type of applicant": "Low-risk installment applicant",
            "Loan type": "INS",
            "Their risk score": f"Estimated default risk ≤ {_fmt_pct(pd_ins_low)} — a strong profile",
            "What happens": "Automatically approved",
        },
        {
            "Type of applicant": "Borderline installment applicant",
            "Loan type": "INS",
            "Their risk score": (
                f"Estimated default risk between {_fmt_pct(pd_ins_low)} and {_fmt_pct(pd_ins_high)} — grey zone"
            ),
            "What happens": (
                f"Approved only if they're likely to take the loan (≥ {_fmt_pct(pr_min)} chance) "
                f"AND their risk on the other loan type is low (≤ {_fmt_pct(cross_max)})"
            ),
        },
        {
            "Type of applicant": "High-risk cash/card applicant",
            "Loan type": "CSS",
            "Their risk score": f"Estimated default risk above {_fmt_pct(pd_css)}",
            "What happens": "Declined",
        },
    ]
    _show_df(pd.DataFrame(rows))
    with st.expander("Try a live example in the Officer tool"):
        st.caption("Open the Officer tool tab to score one application yourself against these same thresholds.")


def _render_cutoff_cards(cuts: dict) -> None:
    cols = st.columns(len(_CUTOFF_ORDER))
    for col, key in zip(cols, _CUTOFF_ORDER):
        label, hint = CUTOFF_DISPLAY.get(key, (key, ""))
        val = cuts.get(key)
        col.metric(label, _fmt_pct(val) if val is not None else "n/a", help=hint)


def _render_portfolio_metrics(ev: dict) -> None:
    st.caption(
        "Estimated profit if this policy had been applied to historical applications "
        "(not a live, currently-running result)."
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Estimated profit",
        _fmt_money(ev.get("total_profit", 0.0)),
        help="Total profit across both loan types, replayed on historical data.",
    )
    c2.metric(
        "Approval rate — INS",
        _fmt_pct(ev.get("ar_ins", np.nan)),
        help="Share of installment applications approved under this policy.",
    )
    c3.metric(
        "Approval rate — CSS",
        _fmt_pct(ev.get("ar_css", np.nan)),
        help="Share of cash/card applications approved under this policy.",
    )
    c4.metric(
        "Default rate (approved) — INS / CSS",
        f"{_fmt_pct(ev.get('bad_rate_ins', np.nan))} / {_fmt_pct(ev.get('bad_rate_css', np.nan))}",
        help="Among approved loans, the share that actually defaulted.",
    )
    c5.metric("Loans approved", f"{int(ev.get('n_accept') or 0):,}")


def tab_model_variables(bundle: dict) -> None:
    _section("Model variables")
    raw = _raw_features(bundle)
    if not raw:
        st.info("No model features in bundle.")
        return
    _show_df(variables_table_frame(raw))


def tab_splitting(bundle: dict) -> None:
    _section("Splitting points")
    pts = bundle.get("points_table")
    big = bundle.get("big_scorecard")
    raw = _raw_features(bundle)
    if pts is None:
        st.info("No points_table in bundle.")
        return
    splitting = splitting_points_display(big, pts, raw)
    if len(splitting):
        st.dataframe(splitting, use_container_width=True)
    else:
        st.info("No splitting rows for selected features.")


def tab_model(bundle: dict) -> None:
    report = bundle.get("model_report") or {}
    pkg = bundle.get("model_package") or {}

    _section("Discriminant power of model")
    _show_df(report.get("Main_measures", pd.DataFrame()))

    _section("Effects (β, p, VIF)")
    effects = report.get("Effects")
    if effects is None or (isinstance(effects, pd.DataFrame) and effects.empty):
        effects = bundle.get("effects")
    if effects is not None and len(effects):
        effects = effects.copy()
        if "feature" in effects.columns:
            effects["feature"] = effects["feature"].map(strip_woe)
        _show_df(effects)

    _section("Scale of variable's scorecard points")
    importance = report.get("Variable importance", pd.DataFrame())
    _show_df(importance)
    _show_fig(fig_variable_importance(importance))

    _section("Gini over time")
    gini = bundle.get("gini_time")
    if gini is None:
        gini = report.get("Gini_over_time")
    _show_fig(fig_gini_time(gini if gini is not None else pd.DataFrame()))

    _section("Stability of scorecard points")
    tr = bundle.get("scored_train")
    va = bundle.get("scored_valid")
    _show_fig(
        fig_score_hist(
            tr["score"] if isinstance(tr, pd.DataFrame) and "score" in tr.columns else None,
            va["score"] if isinstance(va, pd.DataFrame) and "score" in va.columns else None,
        )
    )


def tab_calibration(bundle: dict) -> None:
    report = bundle.get("model_report") or {}
    calib = bundle.get("calibration") or {}

    _section("Platt parameters")
    params = report.get("Calibration_params")
    if params is None or (isinstance(params, pd.DataFrame) and params.empty):
        raw = calib.get("params", {})
        if isinstance(raw, dict) and ("a" in raw or "b" in raw):
            params = pd.DataFrame([{"a": raw.get("a"), "b": raw.get("b")}])
    _show_df(params if params is not None else pd.DataFrame())

    _section("Diagnostics")
    diag = report.get("Calibration_diagnostics")
    if diag is None or (isinstance(diag, pd.DataFrame) and diag.empty):
        d = calib.get("diagnostics")
        diag = pd.DataFrame([d]) if d else pd.DataFrame()
    _show_df(diag)

    _section("ROC before / after")
    scored = bundle.get("scored_valid")
    pkg = bundle.get("model_package") or {}
    target = str(pkg.get("target") or (calib.get("params") or {}).get("target") or "default12")
    if isinstance(scored, pd.DataFrame) and "score" in scored.columns and target in scored.columns:
        y = scored[target]
        score = scored["score"]
        pd_s = scored["pd"] if "pd" in scored.columns else None
        if pd_s is None and calib.get("params"):
            from credit_scoring.profit.scoring import normalize_calib_params, score_to_pd

            try:
                a, b = normalize_calib_params(calib.get("params", calib))
                pd_s = pd.Series(score_to_pd(score.to_numpy(), a, b), index=score.index)
            except Exception:
                pd_s = None
        _show_fig(fig_roc_before_after(y, score, pd_s))
    else:
        st.info("Need scored_valid with score + target for ROC.")

    _section("Score deciles")
    dec = report.get("Calibration_deciles")
    if dec is None or (isinstance(dec, pd.DataFrame) and dec.empty):
        dec = bundle.get("cal_table")
    if dec is not None and len(dec):
        _show_df(dec)
        _show_fig(fig_score_vs_bad_rate(dec))


def tab_gini_vars(bundle: dict) -> None:
    _section("Gini vars (final)")
    report = bundle.get("model_report") or {}
    gini = report.get("Gini_vars_final")
    if gini is not None and len(gini):
        _show_df(gini)
    else:
        st.info("No Gini_vars_final in bundle.")

    _section("Variable stability over time")
    vr = bundle.get("variable_report") or {}
    for feat, blocks in vr.items():
        period = blocks.get("period") if isinstance(blocks, dict) else None
        if not isinstance(period, pd.DataFrame) or period.empty:
            continue
        st.markdown(f"### {display_label(feat)}")
        st.caption(interpret_feature(feat))
        _show_fig(
            fig_bin_metric_over_time(
                period, metric="share", title=f"{display_label(feat)}: share by bin"
            )
        )


def tab_profit_policy(yml: dict) -> None:
    _section("Profit & policy")
    cuts = _frozen_cutoffs(yml)

    _render_policy_flow()
    st.divider()
    _render_scenario_examples(cuts)
    st.divider()
    st.markdown("**Frozen thresholds**")
    _render_cutoff_cards(cuts)

    threshold_rows = [
        {
            "What it controls": CUTOFF_DISPLAY.get(key, (key, ""))[0],
            "Threshold": _fmt_pct(cuts[key]),
            "In plain English": CUTOFF_DISPLAY.get(key, (key, ""))[1],
        }
        for key in _CUTOFF_ORDER
        if key in cuts
    ]
    if threshold_rows:
        _show_df(pd.DataFrame(threshold_rows))

    ev = _profit_eval(yml)
    if ev is None:
        st.warning(
            f"Missing as-if scored frame at `{DEFAULT_ASIF_PATH}`. "
            "Export once from notebook 05 (`export_asif_scored`) after scoring."
        )
        return

    if ev.get("midband_empty"):
        st.error("Mid-band empty: pd_ins_high must be greater than pd_ins_low in parameters.yml.")

    st.divider()
    st.markdown("**Portfolio impact**")
    _render_portfolio_metrics(ev)

    _section("By product")
    bp = ev.get("by_product")
    if isinstance(bp, pd.DataFrame) and len(bp):
        _show_df(bp)
    else:
        st.info("No accepted loans under frozen cutoffs.")

    st.subheader("Decline reasons")
    st.caption("Why applications were declined under the frozen policy.")
    reasons = ev.get("decline_reasons")
    if isinstance(reasons, pd.DataFrame) and len(reasons):
        _show_df(reasons)

    st.subheader("PD U-curves (exploration)")
    st.caption(
        "Single-product PD-only curves — cumulative profit if you ranked by PD alone. "
        "Does not include mid-band, bad-customer, or inactive-CSS rules."
    )
    asif, _ = _load_asif()
    curves = u_curves_by_product(asif)
    pd_ins_high = float(cuts.get("pd_ins_high", 0.0))
    pd_css = float(cuts.get("pd_css", 0.0))
    try:
        import plotly.graph_objects as go

        for prod, curve in curves.items():
            if curve is None or curve.empty:
                continue
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=curve["pd"],
                    y=curve["profit_cum"],
                    mode="lines",
                    name="profit_cum",
                )
            )
            mark = pd_ins_high if prod == "ins" else pd_css
            fig.add_vline(x=mark, line_dash="dash", annotation_text=f"cutoff={mark:.4f}")
            fig.update_layout(
                title=f"{prod}: cumulative profit vs PD (PD-only)",
                xaxis_title="PD threshold",
                yaxis_title="profit_cum",
                height=360,
            )
            st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        for prod, curve in curves.items():
            if curve is not None and len(curve):
                st.line_chart(curve.set_index("pd")["profit_cum"], height=280)


@lru_cache(maxsize=32)
def _load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=32)
def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=8)
def _load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def _load_profit_artifacts(profit_yml: dict) -> dict:
    """Load frozen scorecard + points + calibration artifacts for scoring."""
    arts = profit_yml.get("artifacts") or {}
    out = {}
    for key in ("ins", "css", "pr", "cross"):
        if key not in arts:
            continue
        a = arts[key]
        pkg_path = str((ROOT / a["package"]).resolve())
        out[key] = {
            "package": _load_pickle(pkg_path),
            "points": _load_parquet(str((ROOT / a["points"]).resolve())),
            "calib": _load_json(str((ROOT / a["calib"]).resolve())),
        }
    return out


def _nominal_options(spec: dict) -> list[str]:
    cat_map = spec.get("category_map") or {}
    missing = str(spec.get("missing_label", "Missing"))
    other = str(spec.get("other_label", "<OTHERS>"))
    raw_levels = sorted({str(k) for k in cat_map})
    return [""] + raw_levels + [missing, other]


def _numeric_bin_hint(spec: dict, feature: str, binning_maps: dict) -> str:
    intervals = spec.get("intervals") or []
    if not intervals:
        edges = spec.get("edges")
        if edges:
            return f"Numeric; edges: {edges}"
        return interpret_feature(feature)
    sample = ", ".join(_bin_condition(binning_maps, feature, iv) for iv in intervals[:3])
    if len(intervals) > 3:
        sample += ", …"
    return f"Numeric bins: {sample}"


def _render_decision_card(out: dict) -> None:
    decision = str(out.get("decision", ""))
    reason = str(out.get("decline_reason", ""))
    label, meaning = DECISION_PLAIN.get(decision, (decision, ""))

    if decision == "A":
        st.success(f"**{label}** — {meaning}")
    elif decision == "D":
        st.error(f"**{label}** — {meaning}")
    else:
        st.warning(f"**{label}** — {meaning}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk of default (PD)", _fmt_pct(out.get("pd")), help="Estimated probability this applicant defaults.")
    c2.metric("Raw model score", f"{float(out.get('score', 0.0)):,.1f}", help="Internal points score before conversion to a probability.")
    c3.metric("Expected loss", _fmt_money(out.get("expected_el")), help="Expected money lost to default if this loan is issued.")
    c4.metric("Expected profit (estimate)", _fmt_money(out.get("expected_profit_proxy")), help="Rough expected profit, before overhead costs.")

    if decision == "D" and reason:
        st.caption(f"Decline reason: {reason}")


def _score_officer_local(
    *,
    product: str,
    period: str,
    app_loan_amount: float,
    app_n_installments: int,
    act_cus_active: int,
    agr12_max_cmaxa_due: float,
    raw_features: dict,
    cuts: dict,
    profit_yml: dict,
) -> dict:
    """Score one application locally (fallback when FastAPI is unavailable)."""
    abt = {
        "aid": ["client_1"],
        "cid": ["client_1"],
        "product": [product],
        "period": [period],
        "app_loan_amount": [float(app_loan_amount)],
        "app_n_installments": [int(app_n_installments)],
        "act_cus_active": [int(act_cus_active)],
        "agr12_Max_CMaxA_Due": [float(agr12_max_cmaxa_due)],
        **{k: [v] for k, v in raw_features.items()},
    }
    abt_df = pd.DataFrame(abt)

    artifacts = _load_profit_artifacts(profit_yml)
    packages = {product: artifacts[product]["package"]}
    points_tables = {product: artifacts[product]["points"]}
    calibrations = {product: artifacts[product]["calib"]}

    secondary = None
    if product == "ins":
        secondary = {}
        if "pr" in artifacts:
            secondary["pr"] = {
                "package": artifacts["pr"]["package"],
                "points": artifacts["pr"]["points"],
                "calib": artifacts["pr"]["calib"],
            }
        if "cross" in artifacts:
            secondary["cross"] = {
                "package": artifacts["cross"]["package"],
                "points": artifacts["cross"]["points"],
                "calib": artifacts["cross"]["calib"],
            }
        if not secondary:
            secondary = None

    scored = score_abt_application(
        abt_df,
        packages,
        points_tables,
        calibrations,
        secondary=secondary,
    )

    rules = rules_from_params(
        {
            "window_start": profit_yml["window_start"],
            "window_end": profit_yml["window_end"],
            "burn_in_before": profit_yml.get("burn_in_before", "197501"),
            "economics": profit_yml.get("economics", {}),
            "cutoffs": dict(cuts),
            "bad_customer": dict(profit_yml.get("bad_customer") or {}),
        }
    )
    decision = apply_strategy(scored, rules).iloc[0]

    pd_val = float(scored.iloc[0]["pd"])
    score_val = float(scored.iloc[0]["score"])

    eco = profit_yml.get("economics") or {}
    eco_p = eco[product]
    lgd = float(eco_p["lgd"])
    apr_monthly = float(eco_p["apr_annual"]) / 12.0
    provision = float(eco_p.get("provision", 0.0))
    inst = installment_amount(app_loan_amount, app_n_installments, apr_monthly)
    income_good = app_n_installments * inst + app_loan_amount * (provision - 1.0)
    expected_el = pd_val * app_loan_amount * lgd
    expected_profit = (1.0 - pd_val) * income_good - expected_el

    return {
        "product": product,
        "period": period,
        "score": round(score_val, 4),
        "pd": round(pd_val, 4),
        "decision": str(decision["decision"]),
        "decline_reason": str(decision["decline_reason"]),
        "expected_el": round(float(expected_el), 4),
        "expected_profit_proxy": round(float(expected_profit), 4),
    }


def tab_officer_tool(yml: dict) -> None:
    _section("Officer tool")
    profit = yml.get("profit") or {}
    if not profit:
        st.info("No `profit` config found in parameters.yml.")
        return

    cuts = _frozen_cutoffs(yml)

    if not profit.get("artifacts"):
        st.warning("Missing `profit.artifacts` in parameters.yml; officer tool needs frozen packages.")
        return

    period = str(profit.get("window_start", "197501"))

    use_api = st.toggle("Use FastAPI scoring", value=True)
    api_url = st.text_input(
        "FastAPI /score endpoint",
        value="http://localhost:8000/score",
        disabled=not use_api,
    )

    product = st.selectbox("Product", options=["ins", "css"], index=0)
    app_loan_amount = st.number_input(
        display_label("app_loan_amount"), min_value=0.0, value=5000.0, step=100.0
    )
    app_n_installments = st.number_input(
        display_label("app_n_installments"), min_value=1, value=12, step=1
    )

    act_cus_active = 1 if product == "ins" else st.number_input(
        display_label("act_cus_active"), min_value=0, max_value=1, value=1, step=1
    )
    bad_customer_cfg = profit.get("bad_customer") or {}
    default_agr12 = float(bad_customer_cfg.get("threshold", 0.0)) if bad_customer_cfg.get("enabled") else 0.0
    agr12 = st.number_input(
        display_label("agr12_Max_CMaxA_Due"),
        min_value=0.0,
        value=default_agr12,
        step=1.0,
        help=interpret_feature("agr12_Max_CMaxA_Due"),
    )

    artifacts = _load_profit_artifacts(profit)
    if product not in artifacts:
        st.error(f"No frozen artifacts loaded for product={product}.")
        return

    main_pkg = artifacts[product]["package"]
    binning_maps = dict(main_pkg.get("binning_maps") or {})
    raw_needed = list(binning_maps.keys())
    if product == "ins":
        for sec_key in ("pr", "cross"):
            if sec_key in artifacts:
                sec_maps = artifacts[sec_key]["package"].get("binning_maps") or {}
                binning_maps.update(sec_maps)
                raw_needed.extend(list(sec_maps.keys()))
    raw_needed = sorted(set(raw_needed))

    raw_specs: dict[str, dict] = {}
    for feat in raw_needed:
        raw_specs[feat] = binning_maps.get(feat) or {}

    with st.form("officer_form"):
        raw_inputs: dict = {}
        for feat in raw_needed:
            spec = raw_specs.get(feat) or {}
            label = display_label(feat)
            help_text = interpret_feature(feat)
            ftype = spec.get("type", "numeric")
            if ftype == "nominal":
                options = _nominal_options(spec)
                choice = st.selectbox(label, options=options, index=0, help=help_text, key=f"feat_{feat}")
                raw_inputs[feat] = choice
            else:
                hint = _numeric_bin_hint(spec, feat, binning_maps)
                raw_inputs[feat] = st.text_input(
                    label, value="", help=f"{help_text} {hint}", key=f"feat_{feat}"
                )

        submitted = st.form_submit_button("Score client")

    if not submitted:
        return

    def _parse_raw(spec: dict, txt: str):
        if txt is None:
            return np.nan
        s = str(txt).strip()
        if s == "":
            return np.nan
        if spec.get("type") == "numeric":
            return float(s)
        return s

    parsed_features = {
        feat: _parse_raw(raw_specs.get(feat) or {}, txt) for feat, txt in raw_inputs.items()
    }

    with st.spinner("Scoring and applying decision strategy..."):
        if use_api:
            payload = {
                "product": product,
                "period": period,
                "app_loan_amount": float(app_loan_amount),
                "app_n_installments": int(app_n_installments),
                "act_cus_active": int(act_cus_active),
                "agr12_Max_CMaxA_Due": float(agr12),
                "features": parsed_features,
                "cutoffs": {k: float(v) for k, v in dict(cuts).items()},
            }
            try:
                req = urllib.request.Request(
                    api_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    raw = resp.read().decode("utf-8")
                    out = json.loads(raw)
            except Exception as e:
                st.warning(f"FastAPI scoring failed ({e}). Falling back to local scoring.")
                out = _score_officer_local(
                    product=product,
                    period=period,
                    app_loan_amount=float(app_loan_amount),
                    app_n_installments=int(app_n_installments),
                    act_cus_active=int(act_cus_active),
                    agr12_max_cmaxa_due=float(agr12),
                    raw_features=parsed_features,
                    cuts=cuts,
                    profit_yml=profit,
                )
        else:
            out = _score_officer_local(
                product=product,
                period=period,
                app_loan_amount=float(app_loan_amount),
                app_n_installments=int(app_n_installments),
                act_cus_active=int(act_cus_active),
                agr12_max_cmaxa_due=float(agr12),
                raw_features=parsed_features,
                cuts=cuts,
                profit_yml=profit,
            )

    st.success("Decision computed.")
    _render_decision_card(out)


def main() -> None:
    _render_landing()
    yml = _load_yml()

    products = ["ins", "css", "pr", "cross"]
    available = [
        p
        for p in products
        if (ROOT / DEFAULT_BUNDLE_DIR / p / "model_package_light.json").exists()
    ]
    if not available:
        available = products

    _product, bundle = _render_toolbar(available)

    tabs = st.tabs(
        [
            "Model quality",
            "Stability over time",
            "Profit & policy",
            "Officer tool",
        ]
    )
    with tabs[0]:
        tab_model(bundle)
        tab_calibration(bundle)
    with tabs[1]:
        tab_model_variables(bundle)
        tab_gini_vars(bundle)
        tab_splitting(bundle)
    with tabs[2]:
        tab_profit_policy(yml)
    with tabs[3]:
        tab_officer_tool(yml)


if __name__ == "__main__":
    main()