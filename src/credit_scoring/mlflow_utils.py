"""Optional MLflow logging helpers.

This module is safe to import even when MLflow is not installed; all
functions no-op by default unless explicitly enabled via environment vars.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


def _mlflow_import():
    try:
        import mlflow  # type: ignore

        return mlflow
    except Exception:
        return None


def mlflow_enabled() -> bool:
    """Enable MLflow logging when `MLFLOW_ENABLE` is set to a truthy value."""
    v = os.getenv("MLFLOW_ENABLE", "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def get_experiment_name(default: str = "credit_scoring") -> str:
    return os.getenv("MLFLOW_EXPERIMENT_NAME", default)


def maybe_log_run(
    *,
    run_name: str,
    params: Mapping[str, Any] | None = None,
    metrics: Mapping[str, float] | None = None,
    artifacts: Mapping[str, Any] | None = None,
    tags: Mapping[str, str] | None = None,
) -> None:
    """Log an MLflow run if enabled; otherwise do nothing."""
    if not mlflow_enabled():
        return

    mlflow = _mlflow_import()
    if mlflow is None:
        return

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(get_experiment_name())

    # Keep logging tolerant: MLflow can fail if a tracking server is misconfigured.
    try:
        with mlflow.start_run(run_name=run_name):
            if tags:
                for k, v in tags.items():
                    mlflow.set_tag(str(k), str(v))

            if params:
                for k, v in params.items():
                    if v is None:
                        continue
                    mlflow.log_param(str(k), str(v))

            if metrics:
                for k, v in metrics.items():
                    if v is None:
                        continue
                    mlflow.log_metric(str(k), float(v))

            if artifacts:
                with tempfile.TemporaryDirectory() as td:
                    tmpdir = Path(td)
                    for name, payload in artifacts.items():
                        # Payload can be bytes/str or a file path.
                        if payload is None:
                            continue
                        if isinstance(payload, (str, Path)) and Path(payload).exists():
                            mlflow.log_artifact(str(payload))
                            continue
                        out_path = tmpdir / name
                        if isinstance(payload, bytes):
                            out_path.write_bytes(payload)
                        elif isinstance(payload, str):
                            out_path.write_text(payload, encoding="utf-8")
                        else:
                            out_path.write_text(
                                json.dumps(payload, default=str, indent=2),
                                encoding="utf-8",
                            )
                        mlflow.log_artifact(str(out_path))
    except Exception:
        # Logging must never break training.
        return

