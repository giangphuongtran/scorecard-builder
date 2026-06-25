import pandas as pd

from credit_scoring.pipelines.behavioral.nodes import build_behavioral_node


def test_build_behavioral_node_passes_arguments_in_expected_order(monkeypatch):
    production = pd.DataFrame({"cid": [1], "period": ["197502"]})
    transactions = pd.DataFrame({"cid": [1], "period": ["197501"]})
    params = {"start_period": "197502", "end_period": "197502"}
    captured = {}

    def fake_build_behavioral_all_months(transaction_arg, production_arg, params_arg):
        captured["transaction"] = transaction_arg
        captured["production"] = production_arg
        captured["params"] = params_arg
        return pd.DataFrame({"cid": [1], "proc_period": ["197502"]})

    monkeypatch.setattr(
        "credit_scoring.pipelines.behavioral.nodes.build_behavioral_all_months",
        fake_build_behavioral_all_months,
    )

    out = build_behavioral_node(production, transactions, params)

    assert captured["transaction"] is transactions
    assert captured["production"] is production
    assert captured["params"] is params
    assert list(out.columns) == ["cid", "proc_period"]
