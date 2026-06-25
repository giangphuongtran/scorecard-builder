import pandas as pd

from credit_scoring.pipelines.simulation.nodes import run_simulation_node


def test_run_simulation_node_passes_through_expected_arguments(monkeypatch):
    production = pd.DataFrame({"aid": [1], "cid": [10], "period": ["197501"]})
    transactions = pd.DataFrame({"cid": [10], "fin_period": ["197501"]})
    default_df = pd.DataFrame({"aid": [1], "default12": [0]})
    behavioral_params = {"max_length": 12}
    simulation_params = {"start_period": "197501", "end_period": "197501"}
    captured = {}

    def fake_run_simulation(
        production_arg,
        transactions_arg,
        default_arg,
        params=None,
        sim_params=None,
    ):
        captured["production"] = production_arg
        captured["transactions"] = transactions_arg
        captured["default_df"] = default_arg
        captured["behavioral_params"] = params
        captured["simulation_params"] = sim_params
        return (
            pd.DataFrame({"aid": [1], "period": ["197501"]}),
            pd.DataFrame({"aid": [1], "decision": ["A"]}),
        )

    monkeypatch.setattr(
        "credit_scoring.pipelines.simulation.nodes.run_simulation",
        fake_run_simulation,
    )

    abt_app, decisions = run_simulation_node(
        production,
        transactions,
        default_df,
        behavioral_params,
        simulation_params,
    )

    assert captured["production"] is production
    assert captured["transactions"] is transactions
    assert captured["default_df"] is default_df
    assert captured["behavioral_params"] is behavioral_params
    assert captured["simulation_params"] is simulation_params
    assert list(abt_app.columns) == ["aid", "period"]
    assert list(decisions.columns) == ["aid", "decision"]
