# src/credit_scoring/pipeline_registry.py

from typing import Dict
from kedro.pipeline import Pipeline

# 1. Import the pipeline creation functions from your subfolders
from .pipelines.load_raw.pipeline import create_pipeline as create_load_raw_pl
from .pipelines.behavioral.pipeline import create_pipeline as create_behavioral_pl
# Import your other pipelines here once created...
# from .pipeline.simulation.pipeline import create_pipeline as create_simulation_pl
# from .pipeline.scorecard.pipeline import create_pipeline as create_scorecard_pl
# from .pipeline.strategy.pipeline import create_pipeline as create_strategy_pl

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines."""

    # 2. Instantiate the individual pipelines
    load_raw_pipeline = create_load_raw_pl()
    behavioral_pipeline = create_behavioral_pl()
    
    # simulation_pipeline = create_simulation_pl()
    # scorecard_pipeline = create_scorecard_pl()
    # strategy_pipeline = create_strategy_pl()

    # 3. Stitch them together in dependency order
    # Phase 6 goal: A single continuous chain.
    # Note: In Kedro, simply adding pipelines merges their nodes. The DAG 
    # automatically determines execution order based on inputs and outputs.
    full_end_to_end_pipeline = (
        load_raw_pipeline + 
        behavioral_pipeline 
        # + simulation_pipeline 
        # + scorecard_pipeline 
        # + strategy_pipeline
    )

    return {
        "__default__": full_end_to_end_pipeline,
        
        # You can also register them individually so you can run just one piece 
        # via the CLI: `kedro run --pipeline=behavioral`
        "load_raw": load_raw_pipeline,
        "behavioral": behavioral_pipeline,
    }