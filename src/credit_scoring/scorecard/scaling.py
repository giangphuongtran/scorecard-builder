"""Score scaling and applicant scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd


def scale_scorecard(model_package: dict, factor: float, offset: float) -> pd.DataFrame:
    """Convert logit coefficients into bin-level scorecard points.

    Actions:
    1. Read model betas and attached WOE tables.
    2. Compute per-bin points as ``-beta * WOE * factor``.
    3. Store base points in ``points_table.attrs`` for traceable scoring.
    """
    model = model_package["model"]
    features = model_package["features"]
    intercept = model.params.get("const", 0.0)
    base_points = offset - factor * intercept
    rows = []

    for feat in features:
        beta = model.params[feat]
        woe_table = model_package.get("woe_tables", {}).get(feat)

        if woe_table is None:
            rows.append(
                {"feature": feat, "bin": "<ALL>", "woe": np.nan, "points": np.nan}
            )
            continue

        for _, row in woe_table.iterrows():
            rows.append(
                {
                    "feature": feat,
                    "bin": row["bin"],
                    "woe": row["woe"],
                    "points": -beta * row["woe"] * factor,
                }
            )

    points_table = pd.DataFrame(rows)
    points_table.attrs["base_points"] = base_points
    points_table.attrs["intercept"] = intercept
    points_table.attrs["factor"] = factor
    points_table.attrs["offset"] = offset
    return points_table


def score_applicants(
    df_woe: pd.DataFrame,
    model_package: dict,
    points_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute applicant-level scores from points or the linear predictor.

    Actions:
    1. Start from applicant ids.
    2. If ``points_table`` is supplied, sum bin points plus base points.
    3. Otherwise use the fitted model linear predictor directly.
    """
    features = model_package["features"]
    id_col = model_package.get("id_col", "aid")

    out = pd.DataFrame({id_col: df_woe[id_col].values})

    if points_table is not None:
        base_points = points_table.attrs.get("base_points", 0.0)
        total = pd.Series(base_points, index=df_woe.index, dtype=float)

        for feat in features:
            raw_feat = feat[: -len("_WOE")]
            grp_col = f"{raw_feat}_GRP"
            feat_points_map = (
                points_table[points_table["feature"] == feat]
                .set_index("bin")["points"]
                .to_dict()
            )
            contrib = df_woe[grp_col].map(feat_points_map).fillna(0.0)
            out[f"{feat}_points"] = contrib.values
            total = total + contrib

        out["score"] = total.values
    else:
        model = model_package["model"]
        x = df_woe[features].copy()
        x.insert(0, "const", 1.0)
        out["score"] = x.values @ model.params[["const"] + features].values

    return out.rename(columns={id_col: "aid"}) if id_col != "aid" else out
