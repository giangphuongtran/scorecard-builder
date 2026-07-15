"""Kedro scorecard pipeline (PD Ins + PD Css)."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    calibrate_node,
    partition_and_bin_node,
    prepare_abt_node,
    train_scorecard_node,
    woe_and_screen_node,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=prepare_abt_node,
                inputs=["abt_app", "decisions", "params:model"],
                outputs="abt_model",
                name="prepare_abt_node",
            ),
            node(
                func=partition_and_bin_node,
                inputs=["abt_model", "params:binning", "params:model"],
                outputs=["binning_maps", "train_binned", "valid_binned"],
                name="partition_and_bin_node",
            ),
            node(
                func=woe_and_screen_node,
                inputs=[
                    "train_binned",
                    "valid_binned",
                    "binning_maps",
                    "params:binning",
                    "params:model",
                ],
                outputs=[
                    "woe_maps",
                    "train_woe",
                    "valid_woe",
                    "iv_table",
                    "feature_screen_report",
                ],
                name="woe_and_screen_node",
            ),
            node(
                func=train_scorecard_node,
                inputs=[
                    "train_woe",
                    "valid_woe",
                    "woe_maps",
                    "feature_screen_report",
                    "params:binning",
                    "params:model",
                ],
                outputs=[
                    "pd_ins",
                    "pd_css",
                    "points_tables",
                    "model_diagnostics",
                    "scores_valid",
                ],
                name="train_scorecard_node",
            ),
            node(
                func=calibrate_node,
                inputs=["scores_valid", "params:model"],
                outputs=["calibration_params", "calibration_diagnostics"],
                name="calibrate_node",
            ),
        ]
    )
