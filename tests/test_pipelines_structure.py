from credit_scoring.pipelines.behavioral.pipeline import create_pipeline as create_behavioral_pipeline
from credit_scoring.pipelines.profit.pipeline import create_pipeline as create_profit_pipeline
from credit_scoring.pipelines.scorecard.pipeline import create_pipeline as create_scorecard_pipeline
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


def test_scorecard_pipeline_exposes_five_nodes_with_expected_io():
    pipeline = create_scorecard_pipeline()
    names = [node.name for node in pipeline.nodes]
    assert names == [
        "prepare_abt_node",
        "partition_and_bin_node",
        "woe_and_screen_node",
        "train_scorecard_node",
        "calibrate_node",
    ]

    by_name = {node.name: node for node in pipeline.nodes}

    assert list(by_name["prepare_abt_node"].inputs) == [
        "abt_app",
        "decisions",
        "params:model",
    ]
    assert list(by_name["prepare_abt_node"].outputs) == ["abt_model"]

    assert list(by_name["partition_and_bin_node"].outputs) == [
        "binning_maps",
        "train_binned",
        "valid_binned",
    ]
    assert list(by_name["woe_and_screen_node"].outputs) == [
        "woe_maps",
        "train_woe",
        "valid_woe",
        "iv_table",
        "feature_screen_report",
    ]
    assert list(by_name["train_scorecard_node"].outputs) == [
        "pd_ins",
        "pd_css",
        "points_tables",
        "model_diagnostics",
        "scores_valid",
    ]
    assert list(by_name["calibrate_node"].outputs) == [
        "calibration_params",
        "calibration_diagnostics",
    ]


def test_profit_pipeline_exposes_three_nodes():
    pipeline = create_profit_pipeline()
    names = [node.name for node in pipeline.nodes]
    assert names == [
        "score_application_abt_node",
        "apply_strategy_node",
        "report_profit_node",
    ]
    by_name = {node.name: node for node in pipeline.nodes}
    assert list(by_name["score_application_abt_node"].inputs) == [
        "abt_app_cross",
        "params:profit",
    ]
    assert list(by_name["apply_strategy_node"].outputs) == ["decisions_strategy"]
    assert list(by_name["report_profit_node"].outputs) == [
        "profit_by_loan",
        "profit_summary",
        "profit_report",
    ]
