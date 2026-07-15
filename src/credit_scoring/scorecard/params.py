"""Helpers to merge Kedro parameter groups into a flat scorecard config."""

from __future__ import annotations


def merge_scorecard_params(binning: dict, model: dict) -> dict:
    """Flatten ``params:binning`` and ``params:model`` into one dict.

    Later keys win on collision; ``model`` overrides ``binning``.
    """
    return {**binning, **model}
