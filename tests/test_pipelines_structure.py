from credit_scoring.pipelines.behavioral.pipeline import create_pipeline as create_behavioral_pipeline
from credit_scoring.pipelines.simulation.pipeline import create_pipeline as create_simulation_pipeline


def test_behavioral_pipeline_exposes_single_export_node():
    pipeline = create_behavioral_pipeline()

    assert [node.name for node in pipeline.nodes] == ["build_behavioral_node"]
    node = next(iter(pipeline.nodes))
    assert list(node.inputs) == [
        "production_parquet",
        "transactions_parquet",
        "params:behavioral",
    ]
    assert list(node.outputs) == ["behavioral_features"]


def test_simulation_pipeline_exposes_single_run_node():
    pipeline = create_simulation_pipeline()

    assert [node.name for node in pipeline.nodes] == ["run_simulation_node"]
    node = next(iter(pipeline.nodes))
    assert list(node.inputs) == [
        "production_parquet",
        "transactions_parquet",
        "default_parquet",
        "params:behavioral",
        "params:simulation",
    ]
    assert list(node.outputs) == ["abt_app", "decisions"]
