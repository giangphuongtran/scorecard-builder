from kedro.pipeline import Pipeline, node, pipeline
from .nodes import build_behavioral, build_behavioral_abt, train_behavioral, monitor_behavioral, score_api

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=build_behavioral,
                inputs=["production_raw", "transactions_raw"],
                outputs="behavioral_stub",
                name="build_behavioral_node"),
            node(
                func=build_behavioral_abt,
                inputs="behavioral_stub",
                outputs="behavioral_abt_stub",
                name="build_behavioral_abt_node"),
            node(
                func=train_behavioral,
                inputs="behavioral_abt_stub",
                outputs="pd_behavioral_stub",
                name="train_behavioral_node"),
            node(
                func=monitor_behavioral,
                inputs="behavioral_abt_stub",
                outputs="monitoring_stub",
                name="monitor_behavioral_node"),
            node(
                func=score_api,
                inputs="pd_behavioral_stub",
                outputs="api_readme_stub",
                name="score_api_node"),
        ]
    )