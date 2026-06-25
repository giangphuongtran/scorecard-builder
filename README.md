# Credit Scoring (Python + Kedro)

End-to-end credit scorecard build on the SGH Advanced Scorecard Builder dataset: EDA → behavioral features → monthly simulation → application PD models (Ins + Css) → profit strategy → behavioral PD + monitoring.

Full roadmap and glossary: `[documents/plan.md](documents/plan.md)`. Kedro setup and commands: `[documents/kedro_guide.md](documents/kedro_guide.md)`.

## Stack

- **Python:** pandas, numpy, scikit-learn, statsmodels, optbinning
- **Pipeline:** Kedro (`kedro run`)
- **Tracking (optional):** MLflow
- **Serving (Phase 5b):** FastAPI

## Quick start

```bash
uv sync
uv run kedro run                    # full pipeline
uv run kedro run --pipeline=load_raw
```

Raw SAS inputs live under `data/raw/` (gitignored, licensed course data). Intermediate outputs use Kedro layers under `data/`.

## Project layout

```
credit-scoring/
├── conf/                  # Kedro catalog + parameters
├── documents/             # plan.md, kedro_guide.md
├── notebooks/             # Phase work (01_eda_raw, 02_behavioral_features, …)
├── src/credit_scoring/    # behavioral/, simulation/, pipelines/
└── data/                  # raw (gitignored) + Kedro outputs
```

## Models (hunt-mode scope)


| Model                 | Status   | Notes                                  |
| --------------------- | -------- | -------------------------------------- |
| PD Ins / PD Css       | Phase 4  | Application scorecards, WOE + logistic |
| Behavioral PD         | Phase 5b | Monthly re-score on active loans       |
| Cross PD Css / PR Css | Optional | Post-hunt                              |


## Data note

Trained on SGH course data (not redistributable). Keep the repo private until license terms are confirmed; share view-only access for interviews.