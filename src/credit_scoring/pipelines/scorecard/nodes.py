"""Thin Kedro nodes for the scorecard pipeline."""

from __future__ import annotations

from credit_scoring.scorecard.orchestration import (
    calibrate_both,
    partition_and_bin,
    train_both_products,
    woe_and_screen,
)
from credit_scoring.scorecard.fit import prepare_abt
from credit_scoring.scorecard.params import merge_scorecard_params


def prepare_abt_node(abt_app, decisions, model_params):
    """Merge ABT + decisions and keep accepted applicants."""
    return prepare_abt(abt_app, decisions, model_params, accepted_only=True)


def partition_and_bin_node(abt_model, binning_params, model_params):
    """Partition train/valid and fit/apply bins for both products."""
    params = merge_scorecard_params(binning_params, model_params)
    return partition_and_bin(abt_model, params)


def woe_and_screen_node(train_binned, valid_binned, binning_maps, binning_params, model_params):
    """Encode WOE and prescreen features for both products."""
    params = merge_scorecard_params(binning_params, model_params)
    return woe_and_screen(train_binned, valid_binned, binning_maps, params)


def train_scorecard_node(
    train_woe,
    valid_woe,
    woe_maps,
    feature_screen_report,
    binning_params,
    model_params,
):
    """Train PD models, build points tables, and score validation."""
    params = merge_scorecard_params(binning_params, model_params)
    model_packages, points_tables, model_diagnostics, scores_valid = train_both_products(
        train_woe, valid_woe, woe_maps, feature_screen_report, params
    )
    return (
        model_packages.get("ins"),
        model_packages.get("css"),
        points_tables,
        model_diagnostics,
        scores_valid,
    )


def calibrate_node(scores_valid, model_params):
    """Calibrate PD per product from validation scores."""
    return calibrate_both(scores_valid, model_params)
