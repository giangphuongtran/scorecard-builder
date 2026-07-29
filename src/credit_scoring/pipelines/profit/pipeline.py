"""Kedro profit pipeline (offline CLTV production policy evaluation)."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import apply_strategy_node, report_profit_node, score_application_abt_node


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=score_application_abt_node,
                inputs=["abt_app_cross", "params:profit"],
                outputs="scored_abt_profit",
                name="score_application_abt_node",
            ),
            node(
                func=apply_strategy_node,
                inputs=["scored_abt_profit", "params:profit"],
                outputs="decisions_strategy",
                name="apply_strategy_node",
            ),
            node(
                func=report_profit_node,
                inputs=["scored_abt_profit", "decisions_strategy", "params:profit"],
                outputs=["profit_by_loan", "profit_summary", "profit_report"],
                name="report_profit_node",
            ),
        ]
    )
