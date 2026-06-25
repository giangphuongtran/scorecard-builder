# src/credit_scoring/pipelines/simulation/pipeline.py
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import run_simulation_node

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=run_simulation_node,
                inputs=["production_parquet", "transactions_parquet", "default_parquet", "params:behavioral", "params:simulation"],
                outputs=["abt_app", "decisions"],
                name="run_simulation_node",
            )
        ]
    )