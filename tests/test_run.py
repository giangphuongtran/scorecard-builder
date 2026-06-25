from credit_scoring.pipeline_registry import register_pipelines


def test_pipeline_registry_exposes_expected_pipelines():
    pipelines = register_pipelines()

    assert set(pipelines) == {"__default__", "load_raw", "behavioral", "simulation"}


def test_default_pipeline_matches_load_raw_plus_simulation():
    pipelines = register_pipelines()

    default_nodes = {node.name for node in pipelines["__default__"].nodes}
    load_raw_nodes = {node.name for node in pipelines["load_raw"].nodes}
    simulation_nodes = {node.name for node in pipelines["simulation"].nodes}
    behavioral_nodes = {node.name for node in pipelines["behavioral"].nodes}

    assert default_nodes == load_raw_nodes | simulation_nodes
    assert default_nodes.isdisjoint(behavioral_nodes)
