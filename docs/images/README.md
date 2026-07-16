# README image assets

Place exported screenshots and Kedro-viz PNGs here. Paths are referenced from the root [`README.md`](../../README.md).

## Kedro pipeline diagrams

Export after local data is available under `data/01_raw/`:

```bash
uv run kedro viz run --save-file docs/images/kedro-pipeline-default.png
uv run kedro viz run --pipeline=behavioral --save-file docs/images/kedro-pipeline-behavioral.png
uv run kedro viz run --pipeline=scorecard --save-file docs/images/kedro-pipeline-scorecard.png
uv run kedro viz run --pipeline=profit --save-file docs/images/kedro-pipeline-profit.png
```

If `--save-file` is unavailable in your Kedro Viz version, open `kedro viz run`, select the pipeline in the UI, and screenshot the graph at ~1400px width.

| File | Pipeline | Status |
|------|----------|--------|
| `kedro-pipeline-default.png` | `__default__` | Present — `load_raw` → `simulation` |
| `kedro-pipeline-simulation.png` | `simulation` only | Present — optional detail view |
| `kedro-pipeline-behavioral.png` | `behavioral` | **TODO** — standalone behavioral export |
| `kedro-pipeline-scorecard.png` | `scorecard` | Present |
| `kedro-pipeline-profit.png` | `profit` | Present |

## Streamlit workbench screenshots

Capture from `uv run python -m streamlit run apps/scorecard_workbench.py` after notebook 05 has written bundles to `data/08_reporting/workbench_bundle/`.

| `streamlit-landing.png` | Landing | **TODO** |
| `streamlit-model-quality.png` | Model quality | **TODO** |
| `streamlit-profit-policy.png` | Profit | **TODO** |

Recommended width: 1200–1600px. PNG format.

## Optional results chart

| File | Source |
|------|--------|
| `results-gini-comparison.png` | Notebook or workbench — INS vs CSS validation Gini |
