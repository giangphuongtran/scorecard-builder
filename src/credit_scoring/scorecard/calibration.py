"""Score-to-PD calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, roc_auc_score


def calibrate_pd(scores_df: pd.DataFrame, target: str = "default12") -> dict:
    """Fit Platt-style score-to-PD calibration and return diagnostics.

    Actions:
    1. Fit ``logit(PD) = a + b * score`` on the scored sample.
    2. Compute before/after AUC and Brier diagnostics.
    3. Return serializable calibration parameters.
    """
    y = scores_df[target].values
    score = scores_df["score"].values

    x = sm.add_constant(score)
    calib_model = sm.Logit(y, x).fit(disp=0)

    if hasattr(calib_model.params, "index"):
        param_index = calib_model.params.index.tolist()
        a = float(calib_model.params["const"]) if "const" in param_index else float(calib_model.params.iloc[0])
        slope_cols = [c for c in param_index if c != "const"]
        b = float(calib_model.params[slope_cols[0]]) if slope_cols else 0.0
    else:
        a = float(calib_model.params[0])
        b = float(calib_model.params[1]) if len(calib_model.params) > 1 else 0.0
    pd_calibrated = calib_model.predict(x)

    auc_before = roc_auc_score(y, score)
    auc_after = roc_auc_score(y, pd_calibrated)

    score_norm = (score - score.min()) / (score.max() - score.min() + 1e-12)
    brier_before = brier_score_loss(y, score_norm)
    brier_after = brier_score_loss(y, pd_calibrated)

    diagnostics = {
        "auc_before": float(auc_before),
        "auc_after": float(auc_after),
        "brier_before": float(brier_before),
        "brier_after": float(brier_after),
        "mean_pd_predicted": float(pd_calibrated.mean()),
        "mean_pd_actual": float(y.mean()),
    }

    return {
        "params": {"a": a, "b": b, "target": target, "score_col": "score"},
        "diagnostics": diagnostics,
    }
