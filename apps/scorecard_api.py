"""FastAPI scoring + decision endpoint for officer-style approvals.

This service loads frozen Gate B artifacts from `conf/base/parameters.yml`
(profit.artifacts) and applies the same scorecard->PD->policy-decision logic
used in the Streamlit workbench.
"""

from __future__ import annotations

import json
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from credit_scoring.profit.pnl import installment_amount
from credit_scoring.profit.rules import apply_strategy, rules_from_params
from credit_scoring.profit.scoring import score_abt_application

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _load_params() -> dict:
    return yaml.safe_load((ROOT / "conf" / "base" / "parameters.yml").read_text())


@lru_cache(maxsize=32)
def _load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=16)
def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=8)
def _load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@lru_cache(maxsize=1)
def _load_profit_artifacts() -> dict[str, dict[str, Any]]:
    profit = _load_params().get("profit") or {}
    arts = profit.get("artifacts") or {}
    out: dict[str, dict[str, Any]] = {}
    for key in ("ins", "css", "pr", "cross"):
        if key not in arts:
            continue
        a = arts[key]
        out[key] = {
            "package": _load_pickle(str((ROOT / a["package"]).resolve())),
            "points": _load_parquet(str((ROOT / a["points"]).resolve())),
            "calib": _load_json(str((ROOT / a["calib"]).resolve())),
        }
    return out


class ScoreRequest(BaseModel):
    product: Literal["ins", "css"] = Field(..., description="Applicant product lane.")
    period: str = Field(..., description="Observation/decision month as YYYYMM.")
    app_loan_amount: float = Field(..., ge=0)
    app_n_installments: int = Field(..., ge=1)
    act_cus_active: int = Field(1, ge=0, le=1, description="Used for CSS inactive -> N.")
    agr12_Max_CMaxA_Due: float = Field(0.0, description="Bad-customer rule input.")
    features: dict[str, Any] = Field(
        ..., description="Raw feature values required by the frozen scorecard."
    )
    cutoffs: dict[str, float] | None = Field(
        None, description="Decision cutoffs; if omitted, uses parameters.yml defaults."
    )


class ScoreResponse(BaseModel):
    product: str
    period: str
    score: float
    pd: float
    decision: str
    decline_reason: str
    expected_el: float
    expected_profit_proxy: float
    used_cutoffs: dict[str, float]


def _required_raw_features(product: str, artifacts: dict[str, dict[str, Any]]) -> set[str]:
    primary = set((artifacts[product]["package"].get("binning_maps") or {}).keys())
    if product == "ins":
        for sec in ("pr", "cross"):
            if sec in artifacts:
                primary |= set((artifacts[sec]["package"].get("binning_maps") or {}).keys())
    return primary


def _score_one(req: ScoreRequest) -> ScoreResponse:
    params = _load_params()
    profit = params.get("profit") or {}
    economics = profit.get("economics") or {}
    bad_customer = profit.get("bad_customer") or {}

    default_cuts = dict((profit.get("cutoffs") or {}))
    used_cuts = dict(req.cutoffs or default_cuts)
    if used_cuts.get("pd_ins_high") is not None and used_cuts.get("pd_ins_low") is not None:
        if float(used_cuts["pd_ins_high"]) <= float(used_cuts["pd_ins_low"]):
            raise HTTPException(
                status_code=400,
                detail="Invalid cutoffs: require pd_ins_high > pd_ins_low so INS mid-band is non-empty.",
            )

    artifacts = _load_profit_artifacts()
    if req.product not in artifacts:
        raise HTTPException(status_code=400, detail=f"Missing artifacts for product={req.product}")

    needed = _required_raw_features(req.product, artifacts)
    missing = sorted(list(needed - set(req.features.keys())))
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required features for scoring: {missing[:10]}{'...' if len(missing)>10 else ''}",
        )

    # Build ABT-like single-row frame. default12 is not needed for scoring.
    abt = {
        "aid": ["client_1"],
        "cid": ["client_1"],
        "product": [req.product],
        "period": [req.period],
        "app_loan_amount": [float(req.app_loan_amount)],
        "app_n_installments": [int(req.app_n_installments)],
        "act_cus_active": [int(req.act_cus_active)],
        "agr12_Max_CMaxA_Due": [float(req.agr12_Max_CMaxA_Due)],
        **{k: [v] for k, v in req.features.items()},
    }
    abt_df = pd.DataFrame(abt)

    packages = {req.product: artifacts[req.product]["package"]}
    points_tables = {req.product: artifacts[req.product]["points"]}
    calibrations = {req.product: artifacts[req.product]["calib"]}

    secondary = None
    if req.product == "ins":
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
            "window_start": profit.get("window_start", "197501"),
            "window_end": profit.get("window_end", "198712"),
            "burn_in_before": profit.get("burn_in_before", "197501"),
            "economics": economics,
            "cutoffs": used_cuts,
            "bad_customer": bad_customer,
        }
    )
    decision_row = apply_strategy(scored, rules).iloc[0]

    pd_val = float(scored.iloc[0]["pd"])
    score_val = float(scored.iloc[0]["score"])

    eco_p = economics[req.product]
    lgd = float(eco_p["lgd"])
    apr_monthly = float(eco_p["apr_annual"]) / 12.0
    provision = float(eco_p.get("provision", 0.0))
    inst = installment_amount(req.app_loan_amount, req.app_n_installments, apr_monthly)
    income_good = req.app_n_installments * inst + req.app_loan_amount * (provision - 1.0)
    expected_el = pd_val * req.app_loan_amount * lgd
    expected_profit = (1.0 - pd_val) * income_good - expected_el

    return ScoreResponse(
        product=req.product,
        period=req.period,
        score=score_val,
        pd=pd_val,
        decision=str(decision_row["decision"]),
        decline_reason=str(decision_row["decline_reason"]),
        expected_el=float(expected_el),
        expected_profit_proxy=float(expected_profit),
        used_cutoffs={k: float(v) for k, v in used_cuts.items()},
    )


app = FastAPI(title="Credit scoring API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    return _score_one(req)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("apps.scorecard_api:app", host="127.0.0.1", port=8000, reload=False)

