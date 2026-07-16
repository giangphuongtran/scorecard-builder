"""Helpers to merge Kedro / notebook parameter groups into a flat scorecard config."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

PRODUCT_KEYS = ("ins", "css", "pr", "cross")


def merge_scorecard_params(binning: dict, model: dict) -> dict:
    """Flatten ``params:binning`` and ``params:model`` into one dict.

    Later keys win on collision; ``model`` overrides ``binning``.
    """
    return {**binning, **model}


def load_product_scorecard_params(yml: dict, product: str) -> dict[str, Any]:
    """Merge shared ``scorecard`` keys with ``scorecard.<product>`` (+ model scale).

    Raises KeyError if ``product`` is missing from ``yml['scorecard']``.
    """
    if product not in PRODUCT_KEYS:
        raise ValueError(f"unknown product {product!r}; expected one of {PRODUCT_KEYS}")
    sc = yml.get("scorecard") or {}
    if product not in sc:
        raise KeyError(f"scorecard.{product} missing from parameters.yml")
    shared = {k: deepcopy(v) for k, v in sc.items() if k not in PRODUCT_KEYS}
    out = {**shared, **deepcopy(sc[product])}
    model = yml.get("model") or {}
    out.setdefault("factor", model.get("factor"))
    out.setdefault("offset", model.get("offset"))
    if "prefixes" in out and not isinstance(out["prefixes"], tuple):
        out["prefixes"] = tuple(out["prefixes"])
    if "blocked" in out and not isinstance(out["blocked"], tuple):
        out["blocked"] = tuple(out["blocked"])
    if "number_features" in out and not isinstance(out["number_features"], list):
        out["number_features"] = list(out["number_features"])
    return out


def artifact_registry(
    profit_params: dict,
    *,
    models_dir: Path | str,
    root: Path | str | None = None,
) -> list[dict[str, Any]]:
    """List YAML-wired and on-disk packages for ins/css/pr/cross."""
    root_p = Path(root) if root is not None else None
    models_p = Path(models_dir)
    arts = profit_params.get("artifacts") or {}
    glob_pat = {
        "ins": "pd_ins_v*.pkl",
        "css": "pd_css_v*.pkl",
        "pr": "pr_css_v*.pkl",
        "cross": "cross_pd_css_v*.pkl",
    }
    rows = []
    for product in PRODUCT_KEYS:
        yaml_paths = arts.get(product) or {}
        pkg_rel = yaml_paths.get("package")
        pkg_abs = (root_p / pkg_rel) if (root_p is not None and pkg_rel) else (
            Path(pkg_rel) if pkg_rel else None
        )
        rows.append(
            {
                "product": product,
                "yaml_package": pkg_rel,
                "yaml_exists": bool(pkg_abs and pkg_abs.exists()) if pkg_abs else False,
                "on_disk": sorted(p.name for p in models_p.glob(glob_pat[product])),
            }
        )
    return rows
