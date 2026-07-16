"""Variable / model report tables and HTML scorecard export."""

from __future__ import annotations

import base64
import html as html_lib
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credit_scoring.scorecard.selection import check_vif
from credit_scoring.scorecard.stability import (
    _woe_display_bins,
    bin_stability_report,
    check_woe_monotonicity,
    compute_period_br_gaps,
    compute_static_br_gaps,
)

MIDBAND_STRATEGY_BLURB = """\
How the approval policy decides (installment loans use a grey zone; cash/card loans don't):
- Serious payment history problem? → Decline.
- Estimated default risk above the upper limit? → Decline outright.
- Estimated default risk at or below the safe limit? → Approve automatically.
- Risk in between (the grey zone)? → Approve only if the applicant is likely enough to take the \
loan AND their risk on the other loan type is low enough; otherwise decline.
Also: cash/card applicants with no active relationship get no automatic decision; cash/card risk \
above its limit is declined; the warm-up period is always auto-approved.
The exact thresholds live in conf/base/parameters.yml under profit.cutoffs.
"""

POLICY_FLOW_STEPS: list[str] = [
    "Warm-up period → every application is approved while the portfolio history builds up.",
    "Serious arrears check → decline if the applicant's payment history is too poor.",
    "CSS (cash/card): if the customer has no active relationship, no automatic decision is made.",
    "CSS: decline if the estimated default risk is above the CSS risk limit.",
    "INS (installment): decline outright if the estimated default risk is above the upper risk limit.",
    "INS: approve automatically if the estimated default risk is at or below the safe (lower) limit.",
    "INS grey zone: approve only if the applicant is likely enough to take the loan AND their risk on other loans is low enough.",
]

CUTOFF_DISPLAY: dict[str, tuple[str, str]] = {
    "pd_css": (
        "CSS risk limit",
        "Cash/card applications riskier than this are declined.",
    ),
    "pd_ins_high": (
        "INS risk limit (upper)",
        "Installment applications riskier than this are declined outright.",
    ),
    "pd_ins_low": (
        "INS safe limit (lower)",
        "Installment applications safer than this are approved automatically.",
    ),
    "pr_min": (
        "Minimum take-up likelihood",
        "For grey-zone installment applicants: the minimum estimated chance they'll actually take the loan if offered.",
    ),
    "cross_pd_max": (
        "Cross-product risk limit",
        "For grey-zone installment applicants: their maximum allowed risk on the other loan type.",
    ),
}

SECTION_HINTS: dict[str, str] = {
    "Variables in the model": (
        "The inputs this scorecard actually uses. Each loan type (INS, CSS, PR, Cross) has its own set."
    ),
    "Model variables": (
        "The inputs selected for this loan type's model, with plain-language labels."
    ),
    "Splitting points for variables": (
        "How each input is grouped into risk bands. \"Condition\" shows the raw values or ranges in "
        "each band, and \"Points\" shows how much each band adds to the final score."
    ),
    "Scale of scorecard": "The minimum and maximum possible score, and how raw numbers convert into points.",
    "Scale of variable's scorecard points": (
        "How much each input contributes to the total score, as a share. A larger share means that "
        "input has more influence on the final decision."
    ),
    "Discriminant power of model": (
        "How well the model separates risky from safe applicants, on training vs. validation data, "
        "plus a couple of technical health checks (Gini, AR_diff, VIF, p-value)."
    ),
    "Effects in the model": (
        "Each input's weight in the model, whether that weight is statistically meaningful, and whether "
        "it overlaps too much with other inputs (technical terms: coefficient/β, p-value, VIF — lower VIF, "
        "under 3, is healthier)."
    ),
    "Effects (β, p, VIF)": (
        "Each input's weight in the model, whether that weight is statistically meaningful, and whether "
        "it overlaps too much with other inputs (technical terms: coefficient/β, p-value, VIF — lower VIF, "
        "under 3, is healthier)."
    ),
    "Gini over time": (
        "How well the model separates risky from safe applicants, tracked month by month. A sharp drop "
        "would signal the model losing accuracy."
    ),
    "Stability of scorecard points": (
        "Are applicants in the test period scored similarly to applicants in the training period? A big "
        "shift can mean the population has changed."
    ),
    "Scorecard points and bad rates": (
        "Sanity check: applicants with a higher score should default less often. This should trend "
        "smoothly, not zig-zag."
    ),
    "Calibration": (
        "The step that converts the raw score into an actual probability of default. Shows the "
        "conversion formula, an accuracy score for how close the predicted probabilities are to what "
        "actually happened (technical term: Brier score — lower is better), and how well the model "
        "ranks risk before vs. after this conversion."
    ),
    "Gini vars (final)": (
        "How much each individual input contributes to separating risky from safe applicants, plus how "
        "each input's risk bands behave over time."
    ),
    "Platt parameters": "The two numbers used to convert a raw score into a probability of default.",
    "Diagnostics": (
        "How accurate the converted probabilities are on validation data (technical term: Brier score — "
        "lower is better)."
    ),
    "ROC before / after": "How well the model ranks risk before vs. after converting the score to a probability.",
    "Score deciles": (
        "Applicants split into ten equal groups by score. Actual default rates should climb steadily "
        "from the safest group to the riskiest."
    ),
    "Splitting points": (
        "How each input is grouped into risk bands. \"Condition\" shows the raw values or ranges in "
        "each band."
    ),
    "Profit & policy": (
        "Estimated profit if this approval policy had been applied to historical applications "
        "(not a live, currently-running result)."
    ),
    "Profit strategy": (
        "Estimated profit if this approval policy had been applied to historical applications."
    ),
    "Decision policy card": "A quick summary of the current approval thresholds and their estimated impact.",
    "Variable stability over time": (
        "How the mix of applicants in each risk band shifts across the validation period, for each input."
    ),
    "Officer tool": "Score one application by hand using the same models and policy as the rest of this report.",
    "Profit": MIDBAND_STRATEGY_BLURB,
}


def section_blurb(title: str) -> str:
    """Return a short section description for Streamlit captions."""
    return SECTION_HINTS.get(title, "")


COLUMN_LABELS: dict[str, str] = {
    "feature": "Variable",
    "variable": "Variable",
    "bin": "Risk band",
    "condition": "Condition",
    "points_round": "Points",
    "beta": "Weight (β)",
    "pvalue": "p-value (significance)",
    "vif": "VIF (overlap check)",
    "importance_pct": "Importance share",
    "min_points": "Min points",
    "max_points": "Max points",
    "range": "Point range",
    "gini": "Gini (separation score)",
    "period": "Period",
    "ar_diff": "Approval rate difference",
    "decline_reason": "Decline reason",
    "n": "Count",
    "n_accept": "Approved loans",
    "total_profit": "Total profit",
    "total_income": "Total income",
    "total_el": "Expected loss",
    "bad_rate": "Default rate",
    "product": "Loan type",
    "Parameter": "Parameter",
    "Label": "Label",
    "Value": "Value",
    "mean_score": "Average score",
    "brier_before": "Accuracy before calibration (Brier)",
    "brier_after": "Accuracy after calibration (Brier)",
    "auc_before": "Ranking power before (AUC)",
    "auc_after": "Ranking power after (AUC)",
    "n_train": "Training count",
    "bads_train": "Training defaults",
    "goods_train": "Training non-defaults",
    "share_train": "Training share",
    "woe": "Weight of Evidence (WOE)",
    "iv_component": "Predictive strength (IV)",
    "bad_rate_train": "Training default rate",
    "decile": "Risk group (1=safest, 10=riskiest)",
    "pd": "Default risk (PD)",
    "score": "Score",
}


def format_report_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    """Round floats, rename columns, and map feature names for workbench display."""
    if df is None or not len(df):
        return pd.DataFrame() if df is None else df.copy()
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(4)
    if "feature" in out.columns:
        from credit_scoring.scorecard.feature_labels import display_label, strip_woe

        out["feature"] = out["feature"].map(
            lambda f: display_label(strip_woe(str(f))) if pd.notna(f) else f
        )
    if "variable" in out.columns:
        from credit_scoring.scorecard.feature_labels import display_label, strip_woe

        out["variable"] = out["variable"].map(
            lambda f: display_label(strip_woe(str(f))) if pd.notna(f) else f
        )
    rename = {c: COLUMN_LABELS[c] for c in out.columns if c in COLUMN_LABELS}
    if rename:
        out = out.rename(columns=rename)
    return out


def variable_importance_from_points(points_table: pd.DataFrame) -> pd.DataFrame:
    """Per-feature score range as importance share."""
    scale = (
        points_table.groupby("feature")["points"]
        .agg(min_points="min", max_points="max")
        .reset_index()
    )
    scale["range"] = scale["max_points"] - scale["min_points"]
    total_range = float(scale["range"].sum())
    scale["importance_pct"] = scale["range"] / total_range if total_range else 0.0
    return scale.sort_values("importance_pct", ascending=False)


def fig_variable_importance(importance: pd.DataFrame | None):
    """Horizontal bar chart of per-feature importance share."""
    if importance is None or importance.empty or "importance_pct" not in importance.columns:
        return None
    from credit_scoring.scorecard.feature_labels import display_label, strip_woe

    df = importance.sort_values("importance_pct", ascending=True).copy()
    feat_col = "feature" if "feature" in df.columns else None
    if not feat_col:
        return None
    labels = [display_label(strip_woe(str(f))) for f in df[feat_col]]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(df))))
    ax.barh(labels, df["importance_pct"])
    ax.set_xlabel("Importance share")
    ax.set_title("Variable importance (point range share)")
    fig.tight_layout()
    return fig


def splitting_points_display(
    big_scorecard: pd.DataFrame | None,
    points_table: pd.DataFrame,
    raw_features: list[str],
) -> pd.DataFrame:
    """Workbench-friendly splitting table: friendly labels, rounded points only."""
    splitting = splitting_points_table(big_scorecard, points_table, raw_features)
    if splitting.empty:
        return splitting
    from credit_scoring.scorecard.feature_labels import display_label, strip_woe

    out = splitting.copy()
    out["variable"] = out["variable"].map(lambda v: display_label(strip_woe(str(v))))
    keep = [c for c in ["variable", "bin", "condition", "points_round"] if c in out.columns]
    return out[keep].rename(columns={"variable": "Variable", "bin": "Bin", "condition": "Condition", "points_round": "Points"})


def profit_summary_frame(summary: dict | None, *, kind: str = "one") -> pd.DataFrame:
    """Turn a profit eval / conclusion dict into a report table.

    ``kind``: ``one`` (single product), ``all`` (portfolio scalars),
    ``by_product`` (multi-row from ``summary['by_product']``).
    """
    if not summary:
        return pd.DataFrame()
    if kind == "by_product":
        bp = summary.get("by_product")
        if isinstance(bp, pd.DataFrame) and len(bp):
            return bp.copy()
        if isinstance(bp, list) and bp:
            return pd.DataFrame(bp)
        return pd.DataFrame()
    if kind == "all":
        keys = [
            "total_profit",
            "n_accept",
            "ar_ins",
            "ar_css",
            "bad_rate_ins",
            "bad_rate_css",
            "reference",
            "beats_reference",
            "best_strategy",
        ]
    else:
        keys = [
            "product",
            "total_profit",
            "n_accept",
            "ar",
            "bad_rate",
            "cutoff",
            "reference",
            "beats_reference",
        ]
    row = {k: summary[k] for k in keys if k in summary}
    # Allow nested pd_only / midband payloads.
    if not row and "pd_only" in summary and isinstance(summary["pd_only"], dict):
        return profit_summary_frame(
            {**summary["pd_only"], "reference": summary.get("reference")},
            kind="all",
        )
    return pd.DataFrame([row]) if row else pd.DataFrame()


def splitting_points_table(
    big_scorecard: pd.DataFrame | None,
    points_table: pd.DataFrame,
    raw_features: list[str],
) -> pd.DataFrame:
    """Final-feature splitting table with points attached."""
    if big_scorecard is None or not len(big_scorecard):
        return pd.DataFrame()
    splitting = big_scorecard.loc[
        big_scorecard["variable"].isin(raw_features),
        [
            c
            for c in [
                "variable",
                "bin",
                "condition",
                "n_train",
                "bads_train",
                "goods_train",
                "share_train",
                "woe",
                "iv_component",
                "bad_rate_train",
            ]
            if c in big_scorecard.columns
        ],
    ].copy()
    splitting = splitting.sort_values(["variable", "bin"]).reset_index(drop=True)
    return _enrich_splitting_with_points(splitting, points_table)


def qc_table_for_report(qc_table: pd.DataFrame | None, raw_features: list[str]) -> pd.DataFrame:
    """QC table with selected features sorted first."""
    if qc_table is None or not len(qc_table):
        return pd.DataFrame()
    qc_view = qc_table.copy()
    if raw_features:
        qc_view["_sel"] = qc_view["variable"].isin(raw_features).astype(int)
        qc_view = qc_view.sort_values(
            ["_sel", "safe_to_include", "variable"], ascending=[False, False, True]
        ).drop(columns=["_sel"])
    return qc_view


def build_variable_report(
    train_binned: pd.DataFrame,
    big_scorecard: pd.DataFrame,
    kept: list[str],
    target: str,
    time_col: str,
    params: dict,
    period_tables: dict | None = None,
    binning_maps: dict | None = None,
) -> dict:
    """Per-variable static / period / WOE QC blocks."""
    report: dict = {}
    if period_tables is None:
        period_tables, _ = bin_stability_report(
            train_binned, kept, target, time_col, params
        )
    static_cols = [
        "bin",
        "condition",
        "bad_rate_train",
        "share_train",
        "n_train",
        "bads_train",
        "goods_train",
        "woe",
    ]
    for feat in kept:
        static = big_scorecard.loc[
            big_scorecard["variable"] == feat, [c for c in static_cols if c in big_scorecard.columns]
        ].copy()
        period = period_tables.get(feat, pd.DataFrame())
        static_gaps = (
            compute_static_br_gaps(big_scorecard, feat, binning_maps)
            if binning_maps is not None
            else pd.DataFrame()
        )
        period_gaps = (
            compute_period_br_gaps(period, feat, binning_maps)
            if binning_maps is not None
            else pd.DataFrame()
        )
        if binning_maps is not None:
            woe_ordered, woe_inc, woe_dec, woe_direction = check_woe_monotonicity(
                big_scorecard, feat, binning_maps
            )
            woe_display = _woe_display_bins(big_scorecard, feat, binning_maps)
        else:
            woe_ordered, woe_inc, woe_dec, woe_direction = (
                pd.DataFrame(),
                False,
                False,
                "unknown",
            )
            woe_display = pd.DataFrame()

        report[feat] = {
            "static": static,
            "period": period,
            "static_gaps": static_gaps,
            "period_gaps": period_gaps,
            "woe_ordered": woe_ordered,
            "woe_display": woe_display,
            "woe_monotonic": {
                "increasing": woe_inc,
                "decreasing": woe_dec,
                "direction": woe_direction,
            },
        }
    return report


def build_model_report(
    model_package: dict,
    points_table: pd.DataFrame,
    gini_time: pd.DataFrame,
    calibration: dict,
    cal_table: pd.DataFrame | None,
    gini_vars: pd.DataFrame | None,
    stability_gate: dict | None = None,
) -> dict[str, pd.DataFrame]:
    """ASB Model_report sheets as a dict of DataFrames."""
    m = model_package["metrics"]
    main = pd.DataFrame(
        {
            "Measure": [
                "gini_train",
                "gini_valid",
                "ar_diff",
                "max_vif",
                "max_pvalue",
                "n_features",
                "nnegative_betas",
            ],
            "Value": [
                m.get("gini_train"),
                m.get("gini_valid"),
                m.get("ar_diff"),
                m.get("max_vif"),
                m.get("max_pvalue"),
                m.get("n_features"),
                m.get("n_negative_betas", m.get("nnegative_betas")),
            ],
        }
    )
    if stability_gate:
        main = pd.concat(
            [
                main,
                pd.DataFrame(
                    {
                        "Measure": [
                            "stability_gate_pass",
                            "n_stable_vars",
                            "n_unstable_vars",
                            "min_safe_ratio",
                            "max_bad_rate_swing_unsafe",
                        ],
                        "Value": [
                            stability_gate.get("gate_pass"),
                            stability_gate.get("n_safe"),
                            stability_gate.get("n_unsafe"),
                            stability_gate.get("min_safe_ratio"),
                            stability_gate.get("max_bad_rate_swing_unsafe"),
                        ],
                    }
                ),
            ],
            ignore_index=True,
        )
    effects = model_package.get("effects")
    if effects is None:
        feature_cols = model_package["features"]
        train_subset = model_package.get("train_subset")
        if train_subset is not None and all(f in train_subset.columns for f in feature_cols):
            vif_series = check_vif(train_subset[feature_cols])
        else:
            vif_series = pd.Series(dtype=float)
        model = model_package["model"]
        effects = pd.DataFrame(
            [
                {
                    "feature": feat,
                    "beta": float(model.params[feat]),
                    "pvalue": float(model.pvalues[feat]),
                    "vif": float(vif_series.get(feat, np.nan)),
                }
                for feat in feature_cols
            ]
        )
    if effects is not None and len(effects) and "feature" in effects.columns:
        effects = effects.copy()
        effects["feature"] = effects["feature"].map(
            lambda f: f[: -len("_WOE")] if isinstance(f, str) and f.endswith("_WOE") else f
        )
    importance = variable_importance_from_points(points_table)
    scorecard = points_table.copy()
    scorecard["points_round"] = scorecard["points"].round(0)

    raw = calibration.get("params", calibration) if calibration else {}
    if not isinstance(raw, dict):
        raw = {}
    cal_params = (
        pd.DataFrame([{"a": raw.get("a"), "b": raw.get("b")}])
        if ("a" in raw or "b" in raw)
        else pd.DataFrame(columns=["a", "b"])
    )
    diag_raw = (calibration or {}).get("diagnostics") or {}
    cal_diag = pd.DataFrame([diag_raw]) if diag_raw else pd.DataFrame()
    cal_deciles = cal_table.copy() if cal_table is not None and len(cal_table) else pd.DataFrame()
    # Keep Calibration key for backward compat = params only (no target concat).
    cal_legacy = cal_params.copy()

    final_vars = [f[: -len("_WOE")] if f.endswith("_WOE") else f for f in model_package["features"]]
    if gini_vars is not None and len(gini_vars):
        gini_final = gini_vars[gini_vars["variable"].isin(final_vars)]
    else:
        gini_final = pd.DataFrame()
    return {
        "Main_measures": main,
        "Effects": effects,
        "Gini_over_time": gini_time if gini_time is not None else pd.DataFrame(),
        "Scorecard": scorecard,
        "Variable importance": importance,
        "Calibration": cal_legacy,
        "Calibration_params": cal_params,
        "Calibration_diagnostics": cal_diag,
        "Calibration_deciles": cal_deciles,
        "Gini_vars_final": gini_final,
    }


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _df_html(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df is None or df.empty:
        return "<p><em>empty</em></p>"
    return df.head(max_rows).to_html(index=False, float_format=lambda x: f"{x:.4f}")


def _hint(title: str, extra: str = "") -> str:
    text = SECTION_HINTS.get(title, "")
    if extra:
        text = f"{text} {extra}".strip()
    if not text:
        return ""
    return f'<p class="hint">{html_lib.escape(text)}</p>'


def _section(sid: int, title: str, content: str, *, extra_hint: str = "") -> str:
    return (
        f'<section id="s{sid}"><h2>{title}</h2>'
        f"{_hint(title, extra_hint)}{content}</section>"
    )


def _split_points_html(splitting: pd.DataFrame) -> str:
    """Render splitting table with variable name shown only on first row of each group."""
    if splitting is None or splitting.empty:
        return "<p><em>empty</em></p>"

    cols = list(splitting.columns)
    if "variable" not in cols:
        return _df_html(splitting)

    header = "".join(f"<th>{html_lib.escape(str(c))}</th>" for c in cols)
    rows_html: list[str] = []
    prev_var = object()
    for _, row in splitting.iterrows():
        var = row["variable"]
        first = var != prev_var
        prev_var = var
        cells: list[str] = []
        for c in cols:
            val = row[c]
            if c == "variable":
                if first:
                    cells.append(
                        f'<td class="var-name">{html_lib.escape("" if pd.isna(val) else str(val))}</td>'
                    )
                else:
                    cells.append("<td></td>")
                continue
            if pd.isna(val):
                cells.append("<td></td>")
            elif isinstance(val, (float, np.floating)):
                cells.append(f"<td>{float(val):.4f}</td>")
            else:
                cells.append(f"<td>{html_lib.escape(str(val))}</td>")
        rows_html.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<table class="dataframe splitting">'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table>"
    )


def _scale_table(
    *,
    train_min: float,
    train_max: float,
    points_min: float,
    points_max: float,
    base_points: float,
    factor: float | None,
    offset: float | None,
) -> str:
    rows = [
        ("train_score_min", train_min),
        ("train_score_max", train_max),
        ("points_min", points_min),
        ("points_max", points_max),
        ("base_points", base_points),
        ("factor", factor if factor is not None else np.nan),
        ("offset", offset if offset is not None else np.nan),
    ]
    df = pd.DataFrame(rows, columns=["Measure", "Value"])
    return _df_html(df)


def _calibration_html(
    *,
    model_report: dict[str, pd.DataFrame],
    calibration: dict | None,
    cal_table: pd.DataFrame | None,
    target_label: str,
    roc_html: str = "",
) -> str:
    params = model_report.get("Calibration_params")
    if params is None or params.empty:
        raw = (calibration or {}).get("params", calibration or {})
        if isinstance(raw, dict) and ("a" in raw or "b" in raw):
            params = pd.DataFrame([{"a": raw.get("a"), "b": raw.get("b")}])
        else:
            params = model_report.get("Calibration", pd.DataFrame())

    diag = model_report.get("Calibration_diagnostics", pd.DataFrame())
    if (diag is None or diag.empty) and calibration and calibration.get("diagnostics"):
        diag = pd.DataFrame([calibration["diagnostics"]])

    deciles = model_report.get("Calibration_deciles", pd.DataFrame())
    if (deciles is None or deciles.empty) and cal_table is not None and len(cal_table):
        deciles = cal_table

    # Drop provenance columns if callers still pass them.
    for frame_name, frame in (("params", params), ("deciles", deciles)):
        if frame is not None and not frame.empty:
            drop = [c for c in ("target", "score_col", "default12") if c in frame.columns]
            if drop:
                if frame_name == "params":
                    params = frame.drop(columns=drop)
                else:
                    deciles = frame.drop(columns=drop)

    decile_plot = _plot_calibration(deciles)
    parts = [
        "<h3>Platt parameters</h3>",
        _df_html(params),
        "<h3>Diagnostics</h3>",
        _df_html(diag) if diag is not None and len(diag) else "<p><em>empty</em></p>",
        "<h3>ROC before / after</h3>",
        roc_html or "<p><em>No scored sample passed for ROC.</em></p>",
        "<h3>Score deciles (validation)</h3>",
        _df_html(deciles) if deciles is not None and len(deciles) else "<p><em>empty</em></p>",
        decile_plot,
    ]
    return "".join(parts)



def _profit_html(
    product: str,
    profit_one: dict | None,
    profit_all: dict | None,
) -> str:
    one_df = profit_summary_frame(profit_one, kind="one")
    by_prod = profit_summary_frame(profit_all, kind="by_product")
    all_df = profit_summary_frame(profit_all, kind="all")
    one_block = (
        _df_html(one_df)
        if len(one_df)
        else "<p><em>Not run yet — run notebook §6 (one-product PD as-if smoke) or Streamlit workbench.</em></p>"
    )
    if len(by_prod):
        prod_block = _df_html(by_prod)
    elif len(all_df):
        prod_block = _df_html(all_df)
    else:
        prod_block = (
            "<p><em>Not run yet — Streamlit Profit tab or notebook §12 "
            "(closed-loop / as-if portfolio).</em></p>"
        )
    totals = _df_html(all_df) if len(all_df) else ""
    return (
        f"<h3>One product ({html_lib.escape(product)})</h3>{one_block}"
        f"<h3>All products (by product)</h3>{prod_block}"
        + (f"<h3>Portfolio totals</h3>{totals}" if len(all_df) and len(by_prod) else "")
    )


def _period_xticks(periods: list[str], tick_every: int = 12) -> list[str]:
    """Subsample period labels (yearly ticks on monthly series); always keep last."""
    if tick_every <= 0 or len(periods) <= tick_every:
        return list(periods)
    xticks = list(periods[::tick_every])
    if periods[-1] not in xticks:
        xticks.append(periods[-1])
    return xticks


def fig_gini_time(gini_time: pd.DataFrame | None, *, tick_every: int = 12):
    """Matplotlib figure: period Gini."""
    if gini_time is None or gini_time.empty or "gini" not in gini_time.columns:
        return None
    fig, ax = plt.subplots(figsize=(8, 3))
    ordered = gini_time.sort_values("period")
    periods = ordered["period"].astype(str).tolist()
    ax.plot(periods, ordered["gini"], marker="o", ms=3)
    ax.set_title("Gini over time")
    ax.set_xlabel("period")
    ax.set_ylabel("gini")
    ax.set_xticks(_period_xticks(sorted(set(periods)), tick_every))
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    return fig


def fig_score_hist(train_scores: pd.Series | None, valid_scores: pd.Series | None):
    """Matplotlib figure: train/valid score histograms."""
    if train_scores is None and valid_scores is None:
        return None
    fig, ax = plt.subplots(figsize=(8, 3))
    if train_scores is not None and len(train_scores):
        ax.hist(train_scores.dropna(), bins=30, alpha=0.5, label="train", density=True)
    if valid_scores is not None and len(valid_scores):
        ax.hist(valid_scores.dropna(), bins=30, alpha=0.5, label="valid", density=True)
    ax.set_title("Score distribution")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_score_vs_bad_rate(cal_table: pd.DataFrame | None):
    """Matplotlib figure: mean score vs bad rate (decile line)."""
    if cal_table is None or cal_table.empty:
        return None
    xcol = "mean_score" if "mean_score" in cal_table.columns else None
    ycol = "bad_rate" if "bad_rate" in cal_table.columns else None
    if not xcol or not ycol:
        return None
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(cal_table[xcol], cal_table[ycol], marker="o")
    ax.set_title("Score vs bad rate")
    ax.set_xlabel("mean score")
    ax.set_ylabel("bad rate")
    fig.tight_layout()
    return fig


def fig_roc_before_after(
    y: np.ndarray | pd.Series | None,
    score: np.ndarray | pd.Series | None,
    pd_calibrated: np.ndarray | pd.Series | None = None,
):
    """ROC curves for raw score (oriented) and calibrated PD."""
    if y is None or score is None:
        return None
    from sklearn.metrics import roc_curve

    y_arr = np.asarray(y, dtype=float)
    score_arr = np.asarray(score, dtype=float)
    if y_arr.size < 2 or np.unique(y_arr).size < 2:
        return None

    # Orient score so higher = higher PD risk when AUC < 0.5
    from sklearn.metrics import roc_auc_score

    auc_s = float(roc_auc_score(y_arr, score_arr))
    score_plot = -score_arr if auc_s < 0.5 else score_arr

    fig, ax = plt.subplots(figsize=(5, 4))
    fpr_s, tpr_s, _ = roc_curve(y_arr, score_plot)
    ax.plot(fpr_s, tpr_s, label=f"score (AUC={max(auc_s, 1 - auc_s):.3f})")
    if pd_calibrated is not None:
        pd_arr = np.asarray(pd_calibrated, dtype=float)
        auc_p = float(roc_auc_score(y_arr, pd_arr))
        fpr_p, tpr_p, _ = roc_curve(y_arr, pd_arr)
        ax.plot(fpr_p, tpr_p, label=f"calibrated PD (AUC={auc_p:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC before / after calibration")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def fig_bin_metric_over_time(
    period_table: pd.DataFrame | None,
    *,
    metric: str = "bad_rate",
    title: str | None = None,
    tick_every: int = 12,
):
    """Line plot of bad_rate or share by bin over period."""
    if period_table is None or period_table.empty:
        return None
    if metric not in period_table.columns or "period" not in period_table.columns:
        return None
    if "bin" not in period_table.columns:
        return None
    fig, ax = plt.subplots(figsize=(8, 3))
    for bin_label, sub in period_table.groupby("bin", observed=False):
        sub = sub.sort_values("period")
        ax.plot(
            sub["period"].astype(str),
            sub[metric],
            marker="o",
            ms=3,
            label=str(bin_label),
        )
    periods = sorted(period_table["period"].astype(str).unique())
    ax.set_xticks(_period_xticks(periods, tick_every))
    ax.set_title(title or f"{metric} by bin over time")
    ax.set_xlabel("period")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def _plot_gini_time(gini_time: pd.DataFrame) -> str:
    fig = fig_gini_time(gini_time)
    if fig is None:
        return ""
    return f'<img alt="gini" src="data:image/png;base64,{_fig_to_base64(fig)}"/>'


def _plot_score_hist(train_scores: pd.Series | None, valid_scores: pd.Series | None) -> str:
    fig = fig_score_hist(train_scores, valid_scores)
    if fig is None:
        return ""
    return f'<img alt="scores" src="data:image/png;base64,{_fig_to_base64(fig)}"/>'


def _plot_calibration(cal_table: pd.DataFrame | None) -> str:
    fig = fig_score_vs_bad_rate(cal_table)
    if fig is None:
        return ""
    return f'<img alt="cal" src="data:image/png;base64,{_fig_to_base64(fig)}"/>'


def _plot_roc_html(
    y: np.ndarray | pd.Series | None,
    score: np.ndarray | pd.Series | None,
    pd_calibrated: np.ndarray | pd.Series | None,
) -> str:
    fig = fig_roc_before_after(y, score, pd_calibrated)
    if fig is None:
        return ""
    return f'<img alt="roc" src="data:image/png;base64,{_fig_to_base64(fig)}"/>'


def _gini_vars_html(
    gini_final: pd.DataFrame | None,
    variable_report: dict | None,
) -> str:
    parts = [_df_html(gini_final) if gini_final is not None and len(gini_final) else "<p><em>empty</em></p>"]
    if variable_report:
        for feat, blocks in variable_report.items():
            period = blocks.get("period") if isinstance(blocks, dict) else None
            if period is None or not isinstance(period, pd.DataFrame) or period.empty:
                continue
            parts.append(f"<h3>{html_lib.escape(str(feat))}</h3>")
            for metric, title in (
                ("bad_rate", f"{feat}: bad rate by bin over time"),
                ("share", f"{feat}: share by bin over time"),
            ):
                fig = fig_bin_metric_over_time(period, metric=metric, title=title)
                if fig is not None:
                    parts.append(
                        f'<img alt="{html_lib.escape(metric)}" '
                        f'src="data:image/png;base64,{_fig_to_base64(fig)}"/>'
                    )
    return "".join(parts)


def _enrich_splitting_with_points(
    splitting: pd.DataFrame,
    points_table: pd.DataFrame,
) -> pd.DataFrame:
    if splitting.empty or points_table is None or points_table.empty:
        return splitting
    pts = points_table.copy()
    pts["variable"] = pts["feature"].map(
        lambda f: f[: -len("_WOE")] if isinstance(f, str) and f.endswith("_WOE") else f
    )
    pts = pts[["variable", "bin", "points"]].drop_duplicates(subset=["variable", "bin"])
    pts["points_round"] = pts["points"].round(0)
    out = splitting.merge(pts, on=["variable", "bin"], how="left")
    return out


def render_scorecard_html(
    *,
    product: str,
    model_package: dict,
    points_table: pd.DataFrame,
    big_scorecard: pd.DataFrame | None = None,
    model_report: dict[str, pd.DataFrame] | None = None,
    gini_time: pd.DataFrame | None = None,
    calibration: dict | None = None,
    cal_table: pd.DataFrame | None = None,
    train_scores: pd.Series | None = None,
    valid_scores: pd.Series | None = None,
    qc_table: pd.DataFrame | None = None,
    profit_one: dict | None = None,
    profit_all: dict | None = None,
    variable_report: dict | None = None,
    scored_for_roc: pd.DataFrame | None = None,
    output_path: str | Path | None = None,
) -> str:
    """Render an optional HTML dossier snapshot (Streamlit is the primary UI)."""
    if model_report is None:
        model_report = build_model_report(
            model_package,
            points_table,
            gini_time if gini_time is not None else pd.DataFrame(),
            calibration or {},
            cal_table,
            None,
        )

    features = model_package.get("features", [])
    raw_feats = [f[: -len("_WOE")] if f.endswith("_WOE") else f for f in features]
    importance = model_report.get("Variable importance", pd.DataFrame())
    score_scale = {
        "min": float(points_table["points"].min()) if len(points_table) else np.nan,
        "max": float(points_table["points"].max()) if len(points_table) else np.nan,
    }
    global_min = float(train_scores.min()) if train_scores is not None and len(train_scores) else np.nan
    global_max = float(train_scores.max()) if train_scores is not None and len(train_scores) else np.nan
    base_points = float(points_table.attrs.get("base_points", np.nan)) if hasattr(points_table, "attrs") else np.nan
    factor = points_table.attrs.get("factor") if hasattr(points_table, "attrs") else None
    offset = points_table.attrs.get("offset") if hasattr(points_table, "attrs") else None
    target_label = str(model_package.get("target", "") or (calibration or {}).get("params", {}).get("target", ""))

    splitting = splitting_points_table(big_scorecard, points_table, raw_feats)

    gini_plot_src = gini_time if gini_time is not None else model_report.get("Gini_over_time")
    cal_deciles_for_plot = model_report.get("Calibration_deciles")
    if cal_deciles_for_plot is None or cal_deciles_for_plot.empty:
        cal_deciles_for_plot = cal_table

    roc_html = ""
    if scored_for_roc is not None and len(scored_for_roc) and target_label in scored_for_roc.columns:
        y = scored_for_roc[target_label]
        score = scored_for_roc["score"] if "score" in scored_for_roc.columns else None
        pd_col = scored_for_roc["pd"] if "pd" in scored_for_roc.columns else None
        if score is not None:
            # Reconstruct Platt PD if needed
            if pd_col is None and calibration:
                from credit_scoring.profit.scoring import normalize_calib_params, score_to_pd

                try:
                    a, b = normalize_calib_params(calibration.get("params", calibration))
                    pd_col = pd.Series(score_to_pd(score.to_numpy(), a, b), index=score.index)
                except Exception:
                    pd_col = None
            roc_html = _plot_roc_html(y, score, pd_col)

    sections: list[tuple[str, str, str]] = [
        (
            "Variables in the model",
            f"<ul>{''.join(f'<li>{html_lib.escape(f)}</li>' for f in raw_feats)}</ul>",
            "",
        ),
        ("Splitting points for variables", _split_points_html(splitting), ""),
        (
            "Scale of scorecard",
            _scale_table(
                train_min=global_min,
                train_max=global_max,
                points_min=score_scale["min"],
                points_max=score_scale["max"],
                base_points=base_points,
                factor=float(factor) if factor is not None else None,
                offset=float(offset) if offset is not None else None,
            ),
            "",
        ),
        ("Scale of variable's scorecard points", _df_html(importance), ""),
        ("Discriminant power of model", _df_html(model_report.get("Main_measures")), ""),
        ("Effects in the model", _df_html(model_report.get("Effects")), ""),
        ("Gini over time", _plot_gini_time(gini_plot_src if gini_plot_src is not None else pd.DataFrame()), ""),
        ("Stability of scorecard points", _plot_score_hist(train_scores, valid_scores), ""),
        (
            "Calibration",
            _calibration_html(
                model_report=model_report,
                calibration=calibration,
                cal_table=cal_table,
                target_label=target_label,
                roc_html=roc_html,
            ),
            f"Fitted on label `{target_label}`." if target_label else "",
        ),
        (
            "Gini vars (final)",
            _gini_vars_html(model_report.get("Gini_vars_final"), variable_report),
            "",
        ),
        ("Profit", _profit_html(product, profit_one, profit_all), ""),
    ]

    body = "\n".join(
        _section(i, title, content, extra_hint=extra)
        for i, (title, content, extra) in enumerate(sections, 1)
    )
    toc = "\n".join(
        f'<li><a href="#s{i}">{title}</a></li>'
        for i, (title, _, _) in enumerate(sections, 1)
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Scorecard report — {html_lib.escape(product)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
       margin: 2rem; color: #222; }}
h1 {{ border-bottom: 2px solid #333; padding-bottom: .3rem; }}
h2 {{ margin-top: 2rem; color: #1a1a1a; }}
h3 {{ margin-top: 1rem; color: #333; font-size: 1rem; }}
.hint {{ color: #555; font-size: 0.92rem; margin: 0.25rem 0 0.75rem; max-width: 52rem;
         white-space: pre-wrap; }}
table {{ border-collapse: collapse; font-size: 12px; margin: .5rem 0 1rem; }}
th, td {{ border: 1px solid #ccc; padding: 4px 8px; }}
th {{ background: #f0f0f0; }}
td.var-name {{ background: #ffe8cc; font-weight: 600; }}
table.splitting td:first-child {{ min-width: 8rem; }}
nav ul {{ columns: 2; }}
img {{ max-width: 100%; height: auto; }}
</style>
</head>
<body>
<h1>Scorecard report — {html_lib.escape(product)}</h1>
<p>Features: {len(raw_feats)} | Target: {html_lib.escape(target_label)} | Snapshot (prefer Streamlit workbench)</p>
<nav><h2>Contents</h2><ul>{toc}</ul></nav>
{body}
</body>
</html>
"""
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
    return html