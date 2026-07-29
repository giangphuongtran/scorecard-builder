"""Optional MLflow logging helpers.

Logging is on by default through ``params:mlflow`` / ``conf/base/mlflow.yml``.
The default tracking backend is ``sqlite:///mlruns.db``. Disable with
``MLFLOW_ENABLE=0`` or ``params:mlflow.enabled: false`` via the project hook.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

_CONFIG: dict[str, Any] = {
    "enabled": True,
    "tracking_uri": "sqlite:///mlruns.db",
    "experiment_name": "credit_scoring",
}


def configure_mlflow_from_params(params: Mapping[str, Any] | None) -> None:
    """Apply Kedro ``params:mlflow`` (called from project hooks)."""
    if not params:
        return
    block = params.get("mlflow") if "mlflow" in params else params
    if not isinstance(block, Mapping):
        return
    if "enabled" in block:
        _CONFIG["enabled"] = bool(block["enabled"])
    if block.get("tracking_uri"):
        _CONFIG["tracking_uri"] = str(block["tracking_uri"])
    if block.get("experiment_name"):
        _CONFIG["experiment_name"] = str(block["experiment_name"])

    # Mirror into env so nested workers / notebooks inherit the same defaults.
    os.environ.setdefault("MLFLOW_TRACKING_URI", str(_CONFIG["tracking_uri"]))
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", str(_CONFIG["experiment_name"]))


def _mlflow_import():
    try:
        import mlflow  # type: ignore

        return mlflow
    except Exception:
        return None


def mlflow_enabled() -> bool:
    """Env ``MLFLOW_ENABLE`` overrides config; otherwise use configured default (on)."""
    env = os.getenv("MLFLOW_ENABLE")
    if env is not None and str(env).strip() != "":
        return str(env).strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(_CONFIG.get("enabled", True))


def get_experiment_name(default: str | None = None) -> str:
    return os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        str(_CONFIG.get("experiment_name") or default or "credit_scoring"),
    )


def get_tracking_uri() -> str | None:
    return os.getenv("MLFLOW_TRACKING_URI", _CONFIG.get("tracking_uri"))


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

    # Keep logging tolerant: MLflow can fail if a tracking server is misconfigured.
    try:
        tracking_uri = get_tracking_uri()
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            # MLflow 3.x file store needs an explicit opt-in when using a path URI.
            if "://" not in str(tracking_uri) or str(tracking_uri).startswith("file:"):
                os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

        mlflow.set_experiment(get_experiment_name())

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
                    try:
                        fv = float(v)
                    except (TypeError, ValueError):
                        continue
                    if fv != fv:  # NaN
                        continue
                    mlflow.log_metric(str(k), fv)

            if artifacts:
                with tempfile.TemporaryDirectory() as td:
                    tmpdir = Path(td)
                    for name, payload in artifacts.items():
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


def log_policy_run(
    *,
    variant: str,
    cutoffs: Mapping[str, Any],
    metrics: Mapping[str, float],
    artifacts: Mapping[str, Any] | None = None,
    tags: Mapping[str, str] | None = None,
    run_name: str | None = None,
) -> None:
    """Standard wrapper for champion / challenger / production policy runs."""
    merged_tags = {"variant": str(variant), **dict(tags or {})}
    maybe_log_run(
        run_name=run_name or f"policy_{variant}",
        params={f"cutoff_{k}": v for k, v in dict(cutoffs).items()},
        metrics=metrics,
        artifacts=artifacts,
        tags=merged_tags,
    )
