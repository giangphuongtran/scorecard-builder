from kedro.pipeline import Pipeline, node, pipeline
from .nodes import standardize_columns

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=standardize_columns,
            inputs="production_raw",
            outputs="production_parquet",
            name="standardize_production_node",
        ),
        node(
            func=standardize_columns,
            inputs="transactions_raw",
            outputs="transactions_parquet",
            name="standardize_transactions_node",
        ),
        node(
            func=standardize_columns,
            inputs="default_raw",
            outputs="default_parquet",
            name="standardize_default_node",
        ),
    ])