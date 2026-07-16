"""As-if scored frame export and fast cutoff evaluation for the workbench."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from credit_scoring.profit.cutoff import profit_curve_by_pd
from credit_scoring.profit.pnl import compute_pnl_table
from credit_scoring.profit.rules import apply_strategy, evaluate_strategy
from credit_scoring.profit.scoring import score_abt_application
from credit_scoring.mlflow_utils import maybe_log_run


# Paths are resolved from repo root (not process cwd).
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ASIF_PATH = _REPO_ROOT / "data/08_reporting/asif_scored_for_tuner.parquet"
DEFAULT_META_PATH = _REPO_ROOT / "data/08_reporting/asif_scored_for_tuner_meta.json"
DEFAULT_BUNDLE_DIR = _REPO_ROOT / "data/08_reporting/workbench_bundle"


def _dataframe_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Cast parquet-hostile dtypes (Interval / ordered categorical Interval) to string.

    ``pd.qcut`` decile labels are Interval categories; pyarrow cannot cast those
    Arrow dictionary Interval values to a parquet-friendly struct.
    """
    out = df.copy()
    for col in out.columns:
        series = out[col]
        dtype = series.dtype
        # IntervalArray / Categorical of Intervals / object holding Intervals
        if isinstance(dtype, pd.IntervalDtype):
            out[col] = series.astype(str)
            continue
        if isinstance(dtype, pd.CategoricalDtype):
            cats = dtype.categories
            if len(cats) and isinstance(cats[0], pd.Interval):
                out[col] = series.astype(str)
                continue
            # Plain categoricals are fine; keep them. If still fails downstream,
            # string cast is safer for report dumps.
            out[col] = series.astype(str)
            continue
        if dtype is object:
            sample = series.dropna().head(1)
            if len(sample) and isinstance(sample.iloc[0], pd.Interval):
                out[col] = series.astype(str)
    return out


def evaluate_cutoffs(
    scored_pnl: pd.DataFrame,
    cutoffs: dict,
    *,
    window_start: str,
    window_end: str,
    burn_in_before: str = "197501",
    economics: dict | None = None,
    bad_customer: dict | None = None,
) -> dict:
    """Apply strategy cutoffs on a frozen as-if scored+P&L frame and evaluate.

    Returns the same shape as :func:`evaluate_strategy` plus ``cuts`` and a
    ``midband_empty`` flag when ``pd_ins_high <= pd_ins_low``.
    """
    rules = {
        "window_start": window_start,
        "window_end": window_end,
        "burn_in_before": burn_in_before,
        "economics": economics or {},
        "cutoffs": dict(cutoffs),
        "bad_customer": dict(bad_customer or {}),
    }
    decisions = apply_strategy(scored_pnl, rules)
    ev = evaluate_strategy(scored_pnl, decisions, window_start, window_end)
    hi = cutoffs.get("pd_ins_high")
    lo = cutoffs.get("pd_ins_low")
    midband_empty = (
        hi is not None and lo is not None and float(hi) <= float(lo)
    )
    reasons = (
        decisions.loc[decisions["decision"].eq("D"), "decline_reason"]
        .value_counts()
        .rename_axis("decline_reason")
        .reset_index(name="n")
        if len(decisions)
        else pd.DataFrame(columns=["decline_reason", "n"])
    )
    ev["cuts"] = dict(cutoffs)
    ev["midband_empty"] = bool(midband_empty)
    ev["decline_reasons"] = reasons
    ev["decisions"] = decisions
    return ev


def cutoffs_yaml_snippet(cutoffs: dict) -> str:
    """YAML fragment for paste-into ``profit.cutoffs``."""
    body = yaml.safe_dump({"cutoffs": dict(cutoffs)}, sort_keys=False, default_flow_style=False)
    return body


def build_asif_scored_frame(
    abt: pd.DataFrame,
    packages: dict,
    points_tables: dict,
    calibrations: dict,
    economics: dict,
    *,
    secondary: dict | None = None,
) -> pd.DataFrame:
    """Score ABT and attach course P&L columns (as-if, no closed-loop)."""
    scored = score_abt_application(
        abt, packages, points_tables, calibrations, secondary=secondary
    )
    return compute_pnl_table(scored, economics)


def export_asif_scored(
    scored_pnl: pd.DataFrame,
    meta: dict[str, Any],
    *,
    parquet_path: Path | str = DEFAULT_ASIF_PATH,
    meta_path: Path | str = DEFAULT_META_PATH,
) -> Path:
    """Write as-if scored parquet + JSON provenance for the Streamlit workbench."""
    path = Path(parquet_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scored_pnl.to_parquet(path, index=False)
    mpath = Path(meta_path)
    mpath.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return path


def load_asif_scored(
    parquet_path: Path | str = DEFAULT_ASIF_PATH,
    meta_path: Path | str = DEFAULT_META_PATH,
) -> tuple[pd.DataFrame, dict]:
    """Load as-if scored frame and meta JSON."""
    df = pd.read_parquet(parquet_path)
    meta: dict = {}
    mpath = Path(meta_path)
    if mpath.exists():
        meta = json.loads(mpath.read_text(encoding="utf-8"))
    return df, meta


def pr_cross_heatmap(
    scored_pnl: pd.DataFrame,
    *,
    window_start: str,
    window_end: str,
    burn_in_before: str,
    economics: dict,
    bad_customer: dict | None,
    base_cutoffs: dict,
    pr_grid: np.ndarray | list[float],
    cross_grid: np.ndarray | list[float],
) -> pd.DataFrame:
    """Grid total_profit over pr_min × cross_pd_max at fixed PD band cutoffs."""
    rows = []
    for pr_min in pr_grid:
        for cross_max in cross_grid:
            cuts = {
                **base_cutoffs,
                "pr_min": float(pr_min),
                "cross_pd_max": float(cross_max),
            }
            ev = evaluate_cutoffs(
                scored_pnl,
                cuts,
                window_start=window_start,
                window_end=window_end,
                burn_in_before=burn_in_before,
                economics=economics,
                bad_customer=bad_customer,
            )
            rows.append(
                {
                    "pr_min": float(pr_min),
                    "cross_pd_max": float(cross_max),
                    "total_profit": ev["total_profit"],
                    "ar_ins": ev["ar_ins"],
                    "ar_css": ev["ar_css"],
                    "n_accept": ev["n_accept"],
                }
            )
    return pd.DataFrame(rows)


def _quantile_candidates(
    values: pd.Series,
    n: int,
    *,
    min_val: float = 0.0,
    max_val: float = 1.0,
    include: list[float] | None = None,
    quantiles: np.ndarray | None = None,
) -> list[float]:
    """Build a compact threshold candidate list from data quantiles."""
    s = pd.Series(values).dropna()
    if s.empty:
        return []

    if quantiles is None:
        # Use inner quantiles first; extremes often create degenerate strategies.
        quantiles = np.linspace(0.05, 0.95, max(3, int(n)))

    cand = list(np.quantile(s.to_numpy(dtype=float), quantiles))
    cand = [float(x) for x in cand if min_val <= float(x) <= max_val]

    if include:
        for x in include:
            xf = float(x)
            if min_val <= xf <= max_val:
                cand.append(xf)

    cand = sorted(set(cand))
    return [float(x) for x in cand]


def optimize_cutoffs(
    scored_pnl: pd.DataFrame,
    *,
    window_start: str,
    window_end: str,
    burn_in_before: str,
    economics: dict,
    bad_customer: dict | None,
    pd_ins_low: float,
    constraints: dict | None = None,
    grid_sizes: dict | None = None,
    near_opt_rel_tol: float = 0.01,
    top_n: int = 10,
) -> dict:
    """Constrained optimization over the 4-cutoff policy.

    The decision policy is defined in :func:`credit_scoring.profit.rules.apply_strategy`.
    This optimizer searches over:
    - ``pd_css``
    - ``pd_ins_high``
    - ``pr_min``
    - ``cross_pd_max``

    ``pd_ins_low`` is treated as fixed input so the search stays 4D and fast.

    Constraints are applied on the resulting evaluation metrics:
    - ``min_ar_ins`` / ``min_ar_css``: lower bounds on acceptance rates.
    - ``max_bad_rate_ins`` / ``max_bad_rate_css``: upper bounds on accepted bad rates.
    - ``min_n_accept``: minimum accepted count across portfolio.
    """
    constraints = dict(constraints or {})
    grid_sizes = dict(
        {
            "pd_css": 8,
            "pd_ins_high": 8,
            "pr_min": 8,
            "cross_pd_max": 8,
        },
        **(grid_sizes or {}),
    )

    # Candidate grids from data distributions (quantiles keep runtime stable).
    css_pd = scored_pnl.loc[scored_pnl["product"].eq("css"), "pd"]
    ins_pd = scored_pnl.loc[scored_pnl["product"].eq("ins"), "pd"]
    ins_pr = scored_pnl.loc[scored_pnl["product"].eq("ins"), "pr"]
    ins_cross = scored_pnl.loc[scored_pnl["product"].eq("ins"), "cross_pd"]

    # Include current pd_ins_low in candidates to reduce mid-band emptiness risk.
    pd_ins_low_f = float(pd_ins_low)
    pd_ins_high_cand = _quantile_candidates(
        ins_pd,
        grid_sizes["pd_ins_high"],
        min_val=0.0,
        max_val=1.0,
        include=[pd_ins_low_f + 1e-6],
    )
    pd_css_cand = _quantile_candidates(
        css_pd,
        grid_sizes["pd_css"],
        min_val=0.0,
        max_val=1.0,
    )
    pr_cand = _quantile_candidates(
        ins_pr,
        grid_sizes["pr_min"],
        min_val=0.0,
        max_val=1.0,
    )
    cross_cand = _quantile_candidates(
        ins_cross,
        grid_sizes["cross_pd_max"],
        min_val=0.0,
        max_val=1.0,
    )

    # Guard: keep search finite even if quantiles collapse.
    if not (pd_css_cand and pd_ins_high_cand and pr_cand and cross_cand):
        raise ValueError("insufficient scored_pnl data to build cutoff candidates")

    # Avoid degenerate mid-band (pd_ins_low < pd <= pd_ins_high).
    pd_ins_high_cand = [x for x in pd_ins_high_cand if x > pd_ins_low_f]
    if not pd_ins_high_cand:
        raise ValueError("no pd_ins_high candidates satisfy pd_ins_low < pd_ins_high")

    def _constraints_ok(ev: dict) -> bool:
        def _get_float(key: str) -> float | None:
            v = ev.get(key)
            if v is None:
                return None
            try:
                return float(v)
            except Exception:
                return None

        # Acceptance constraints.
        min_ar_ins = constraints.get("min_ar_ins")
        if min_ar_ins is not None:
            ar_ins = _get_float("ar_ins")
            if ar_ins is None or np.isnan(ar_ins) or ar_ins < float(min_ar_ins):
                return False

        min_ar_css = constraints.get("min_ar_css")
        if min_ar_css is not None:
            ar_css = _get_float("ar_css")
            if ar_css is None or np.isnan(ar_css) or ar_css < float(min_ar_css):
                return False

        # Bad-rate constraints.
        max_br_ins = constraints.get("max_bad_rate_ins")
        if max_br_ins is not None:
            br_ins = _get_float("bad_rate_ins")
            if br_ins is None or np.isnan(br_ins) or br_ins > float(max_br_ins):
                return False

        max_br_css = constraints.get("max_bad_rate_css")
        if max_br_css is not None:
            br_css = _get_float("bad_rate_css")
            if br_css is None or np.isnan(br_css) or br_css > float(max_br_css):
                return False

        min_n_accept = constraints.get("min_n_accept")
        if min_n_accept is not None:
            n_accept = _get_float("n_accept")
            if n_accept is None or np.isnan(n_accept) or n_accept < float(min_n_accept):
                return False

        return True

    rows: list[dict] = []
    for pd_css in pd_css_cand:
        for pd_ins_high in pd_ins_high_cand:
            for pr_min in pr_cand:
                for cross_pd_max in cross_cand:
                    cuts = {
                        "pd_css": float(pd_css),
                        "pd_ins_high": float(pd_ins_high),
                        "pd_ins_low": pd_ins_low_f,
                        "pr_min": float(pr_min),
                        "cross_pd_max": float(cross_pd_max),
                    }
                    ev = evaluate_cutoffs(
                        scored_pnl,
                        cuts,
                        window_start=window_start,
                        window_end=window_end,
                        burn_in_before=burn_in_before,
                        economics=economics,
                        bad_customer=bad_customer,
                    )

                    row = {
                        **cuts,
                        "total_profit": float(ev.get("total_profit", 0.0)),
                        "ar_ins": ev.get("ar_ins"),
                        "ar_css": ev.get("ar_css"),
                        "bad_rate_ins": ev.get("bad_rate_ins"),
                        "bad_rate_css": ev.get("bad_rate_css"),
                        "n_accept": int(ev.get("n_accept") or 0),
                        "midband_empty": bool(ev.get("midband_empty", False)),
                    }
                    rows.append(row)

    all_df = pd.DataFrame(rows)
    feasible_rows = []
    for r in rows:
        if r["midband_empty"]:
            continue
        ev_like = {
            "total_profit": r["total_profit"],
            "ar_ins": r["ar_ins"],
            "ar_css": r["ar_css"],
            "bad_rate_ins": r["bad_rate_ins"],
            "bad_rate_css": r["bad_rate_css"],
            "n_accept": r["n_accept"],
        }
        if _constraints_ok(ev_like):
            feasible_rows.append(r)
    feasible_df = pd.DataFrame(feasible_rows)

    if feasible_df.empty:
        # Keep it robust for UI: pick the best overall and flag infeasibility.
        best_row = all_df.sort_values("total_profit", ascending=False).iloc[0]
        best_cuts = {
            "pd_css": float(best_row["pd_css"]),
            "pd_ins_high": float(best_row["pd_ins_high"]),
            "pd_ins_low": float(best_row["pd_ins_low"]),
            "pr_min": float(best_row["pr_min"]),
            "cross_pd_max": float(best_row["cross_pd_max"]),
        }
        maybe_log_run(
            run_name="profit_cutoff_optimization__fallback",
            params={"pd_ins_low": pd_ins_low_f, **{f"constraint_{k}": v for k, v in constraints.items()}},
            metrics={"best_total_profit": float(best_row["total_profit"])},
            artifacts={"best_cutoffs.json": json.dumps(best_cuts, default=str, indent=2)},
            tags={"pipeline": "profit", "node": "optimize_cutoffs"},
        )
        return {
            "best_cutoffs": best_cuts,
            "feasible_count": 0,
            "best_is_feasible": False,
            "best_metrics": {
                "total_profit": float(best_row["total_profit"]),
                "ar_ins": best_row["ar_ins"],
                "ar_css": best_row["ar_css"],
                "bad_rate_ins": best_row["bad_rate_ins"],
                "bad_rate_css": best_row["bad_rate_css"],
                "n_accept": int(best_row["n_accept"]),
            },
            "near_optimal_bands": {},
            "top_policies": [],
            "search_size": int(len(all_df)),
        }

    best_row = feasible_df.sort_values("total_profit", ascending=False).iloc[0]
    best_profit = float(best_row["total_profit"])

    near_df = feasible_df.loc[
        feasible_df["total_profit"] >= best_profit * (1.0 - float(near_opt_rel_tol))
    ]

    def _band(col: str) -> dict:
        vals = near_df[col].astype(float)
        return {
            "min": float(vals.min()),
            "max": float(vals.max()),
            "median": float(vals.median()),
            "count": int(len(vals)),
        }

    top_df = feasible_df.sort_values("total_profit", ascending=False).head(int(top_n))
    top_policies = []
    for _, r in top_df.iterrows():
        top_policies.append(
            {
                "pd_css": float(r["pd_css"]),
                "pd_ins_high": float(r["pd_ins_high"]),
                "pd_ins_low": float(r["pd_ins_low"]),
                "pr_min": float(r["pr_min"]),
                "cross_pd_max": float(r["cross_pd_max"]),
                "total_profit": float(r["total_profit"]),
                "ar_ins": float(r["ar_ins"]) if r["ar_ins"] == r["ar_ins"] else None,
                "ar_css": float(r["ar_css"]) if r["ar_css"] == r["ar_css"] else None,
                "bad_rate_ins": float(r["bad_rate_ins"])
                if r["bad_rate_ins"] == r["bad_rate_ins"]
                else None,
                "bad_rate_css": float(r["bad_rate_css"])
                if r["bad_rate_css"] == r["bad_rate_css"]
                else None,
                "n_accept": int(r["n_accept"]),
            }
        )

    maybe_log_run(
        run_name="profit_cutoff_optimization",
        params={"pd_ins_low": pd_ins_low_f, **{f"constraint_{k}": v for k, v in constraints.items()}},
        metrics={"best_total_profit": best_profit, "feasible_count": len(feasible_df)},
        artifacts={
            "best_cutoffs.json": json.dumps(
                {
                    "pd_css": float(best_row["pd_css"]),
                    "pd_ins_high": float(best_row["pd_ins_high"]),
                    "pd_ins_low": float(best_row["pd_ins_low"]),
                    "pr_min": float(best_row["pr_min"]),
                    "cross_pd_max": float(best_row["cross_pd_max"]),
                },
                default=str,
                indent=2,
            ),
            "top_policies.json": json.dumps(top_policies, default=str, indent=2),
        },
        tags={"pipeline": "profit", "node": "optimize_cutoffs"},
    )

    return {
        "best_cutoffs": {
            "pd_css": float(best_row["pd_css"]),
            "pd_ins_high": float(best_row["pd_ins_high"]),
            "pd_ins_low": float(best_row["pd_ins_low"]),
            "pr_min": float(best_row["pr_min"]),
            "cross_pd_max": float(best_row["cross_pd_max"]),
        },
        "best_metrics": {
            "total_profit": float(best_profit),
            "ar_ins": best_row["ar_ins"],
            "ar_css": best_row["ar_css"],
            "bad_rate_ins": best_row["bad_rate_ins"],
            "bad_rate_css": best_row["bad_rate_css"],
            "n_accept": int(best_row["n_accept"]),
        },
        "near_optimal_bands": {
            "pd_css": _band("pd_css"),
            "pd_ins_high": _band("pd_ins_high"),
            "pr_min": _band("pr_min"),
            "cross_pd_max": _band("cross_pd_max"),
            "pd_ins_low": {
                "min": float(pd_ins_low_f),
                "max": float(pd_ins_low_f),
                "median": float(pd_ins_low_f),
                "count": int(len(near_df)),
            },
        },
        "top_policies": top_policies,
        "feasible_count": int(len(feasible_df)),
        "best_is_feasible": True,
        "search_size": int(len(all_df)),
    }


def u_curves_by_product(scored_pnl: pd.DataFrame, products: tuple[str, ...] = ("ins", "css")) -> dict[str, pd.DataFrame]:
    """1-D cumulative profit vs PD per product (as-if)."""
    return {p: profit_curve_by_pd(scored_pnl, p, pd_col="pd") for p in products}


def save_workbench_product_bundle(
    product: str,
    payload: dict[str, Any],
    *,
    bundle_dir: Path | str = DEFAULT_BUNDLE_DIR,
) -> Path:
    """Persist one product's fit artifacts for the Streamlit workbench.

    Expected keys (any may be omitted): model_package, points_table, calibration,
    cal_table, gini_time, qc_table, big_scorecard, model_report, variable_report,
    scored_train, scored_valid, uni.
    """
    out = Path(bundle_dir) / product
    out.mkdir(parents=True, exist_ok=True)

    if "points_table" in payload and payload["points_table"] is not None:
        pts = payload["points_table"]
        _dataframe_for_parquet(pts).to_parquet(out / "points_table.parquet", index=False)
        attrs = {
            "base_points": pts.attrs.get("base_points"),
            "factor": pts.attrs.get("factor"),
            "offset": pts.attrs.get("offset"),
        }
        (out / "points_attrs.json").write_text(json.dumps(attrs, default=str), encoding="utf-8")

    for key in ("cal_table", "gini_time", "qc_table", "big_scorecard"):
        df = payload.get(key)
        if isinstance(df, pd.DataFrame) and len(df):
            _dataframe_for_parquet(df).to_parquet(out / f"{key}.parquet", index=False)

    if "model_report" in payload and isinstance(payload["model_report"], dict):
        for name, df in payload["model_report"].items():
            if isinstance(df, pd.DataFrame) and len(df):
                safe = str(name).replace(" ", "_")
                _dataframe_for_parquet(df).to_parquet(out / f"report_{safe}.parquet", index=False)

    if "variable_report" in payload and isinstance(payload["variable_report"], dict):
        vr_dir = out / "variable_report"
        vr_dir.mkdir(exist_ok=True)
        summary = []
        for feat, blocks in payload["variable_report"].items():
            summary.append(feat)
            for bname, val in blocks.items():
                if isinstance(val, pd.DataFrame) and len(val):
                    _dataframe_for_parquet(val).to_parquet(
                        vr_dir / f"{feat}__{bname}.parquet", index=False
                    )
                elif isinstance(val, dict):
                    (vr_dir / f"{feat}__{bname}.json").write_text(
                        json.dumps(val, default=str), encoding="utf-8"
                    )
        (vr_dir / "features.json").write_text(json.dumps(summary), encoding="utf-8")

    for key in ("scored_train", "scored_valid"):
        df = payload.get(key)
        if isinstance(df, pd.DataFrame) and len(df):
            keep = [c for c in df.columns if c in ("aid", "score", "pd") or c.endswith("12") or c in ("period", "product")]
            # Prefer target columns present
            cols = list(dict.fromkeys(keep + [c for c in df.columns if c in ("default12", "cross_response", "default_cross12")]))
            slim = df[[c for c in cols if c in df.columns]]
            _dataframe_for_parquet(slim).to_parquet(out / f"{key}.parquet", index=False)

    # Pickle-free serializable package subset
    pkg = payload.get("model_package") or {}
    pkg_light = {
        "product": pkg.get("product"),
        "target": pkg.get("target"),
        "features": pkg.get("features"),
        "metrics": pkg.get("metrics"),
    }
    if "effects" in pkg and isinstance(pkg["effects"], pd.DataFrame):
        _dataframe_for_parquet(pkg["effects"]).to_parquet(out / "effects.parquet", index=False)
    (out / "model_package_light.json").write_text(
        json.dumps(pkg_light, indent=2, default=str), encoding="utf-8"
    )

    calib = payload.get("calibration")
    if isinstance(calib, dict):
        (out / "calibration.json").write_text(json.dumps(calib, indent=2, default=str), encoding="utf-8")

    return out


def load_workbench_product_bundle(
    product: str,
    *,
    bundle_dir: Path | str = DEFAULT_BUNDLE_DIR,
) -> dict[str, Any]:
    """Load a product bundle written by :func:`save_workbench_product_bundle`."""
    root = Path(bundle_dir) / product
    if not root.exists():
        return {}
    out: dict[str, Any] = {"product": product}

    pts_path = root / "points_table.parquet"
    if pts_path.exists():
        pts = pd.read_parquet(pts_path)
        attrs_path = root / "points_attrs.json"
        if attrs_path.exists():
            attrs = json.loads(attrs_path.read_text(encoding="utf-8"))
            for k, v in attrs.items():
                if v is not None:
                    pts.attrs[k] = v
        out["points_table"] = pts

    for key in ("cal_table", "gini_time", "qc_table", "big_scorecard", "effects"):
        p = root / f"{key}.parquet"
        if p.exists():
            out[key] = pd.read_parquet(p)

    report: dict[str, pd.DataFrame] = {}
    key_alias = {
        "Main_measures": "Main_measures",
        "Effects": "Effects",
        "Gini_over_time": "Gini_over_time",
        "Scorecard": "Scorecard",
        "Variable_importance": "Variable importance",
        "Calibration": "Calibration",
        "Calibration_params": "Calibration_params",
        "Calibration_diagnostics": "Calibration_diagnostics",
        "Calibration_deciles": "Calibration_deciles",
        "Gini_vars_final": "Gini_vars_final",
    }
    for p in root.glob("report_*.parquet"):
        raw = p.stem[len("report_") :]
        key = key_alias.get(raw, raw.replace("_", " "))
        report[key] = pd.read_parquet(p)
    if report:
        out["model_report"] = report

    vr_dir = root / "variable_report"
    if vr_dir.exists():
        feats_path = vr_dir / "features.json"
        feats = json.loads(feats_path.read_text(encoding="utf-8")) if feats_path.exists() else []
        vr: dict = {}
        for feat in feats:
            vr[feat] = {}
            for p in vr_dir.glob(f"{feat}__*.parquet"):
                bname = p.stem.split("__", 1)[1]
                vr[feat][bname] = pd.read_parquet(p)
            for p in vr_dir.glob(f"{feat}__*.json"):
                bname = p.stem.split("__", 1)[1]
                vr[feat][bname] = json.loads(p.read_text(encoding="utf-8"))
        out["variable_report"] = vr

    for key in ("scored_train", "scored_valid"):
        p = root / f"{key}.parquet"
        if p.exists():
            out[key] = pd.read_parquet(p)

    pkg_path = root / "model_package_light.json"
    if pkg_path.exists():
        pkg = json.loads(pkg_path.read_text(encoding="utf-8"))
        if "effects" in out:
            pkg["effects"] = out["effects"]
        out["model_package"] = pkg

    cal_path = root / "calibration.json"
    if cal_path.exists():
        out["calibration"] = json.loads(cal_path.read_text(encoding="utf-8"))

    return out
