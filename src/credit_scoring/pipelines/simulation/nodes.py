# pipelines/simulation/nodes.py  (thin wrapper)
from credit_scoring.simulation.orchestrator import run_simulation

def run_simulation_node(production, transactions, default_df, behavioral_params, simulation_params):
    return run_simulation(
        production, transactions, default_df,
        params=behavioral_params,
        sim_params=simulation_params,
    )