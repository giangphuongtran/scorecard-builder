from kedro.pipeline import Pipeline, node, pipeline
from .nodes import build_behavioral_node

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=build_behavioral_node,
                inputs=["production_parquet", "transactions_parquet", "params:behavioral"],
                outputs="behavioral_features",
                name="build_behavioral_node"),
        ]
    )