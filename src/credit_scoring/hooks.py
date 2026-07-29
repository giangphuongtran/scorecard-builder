"""Kedro project hooks — configure MLflow defaults at session start."""

from __future__ import annotations

from typing import Any

from kedro.framework.hooks import hook_impl

from credit_scoring.mlflow_utils import configure_mlflow_from_params


class ProjectHooks:
    """Apply ``params:mlflow`` so tracking is on by default for Kedro runs."""

    @hook_impl
    def after_context_created(self, context) -> None:  # type: ignore[no-untyped-def]
        try:
            params: dict[str, Any] = context.params or {}
        except Exception:
            params = {}
        mlflow_params = params.get("mlflow") if isinstance(params, dict) else None
        if mlflow_params is None and isinstance(params, dict):
            # Also accept a top-level block loaded via conf/base/mlflow.yml if present.
            try:
                mlflow_params = context.config_loader["mlflow"]
            except Exception:
                mlflow_params = None
        configure_mlflow_from_params({"mlflow": mlflow_params} if mlflow_params else params)
