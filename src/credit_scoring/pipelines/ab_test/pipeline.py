"""Kedro A/B pipeline — offline champion vs challenger policy compare."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import compare_policies_node


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=compare_policies_node,
                inputs=["params:ab_test", "params:profit"],
                outputs=["ab_test_summary", "ab_test_report"],
                name="compare_policies_node",
            ),
        ]
    )
