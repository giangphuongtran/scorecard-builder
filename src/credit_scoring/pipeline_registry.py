# src/credit_scoring/pipeline_registry.py

from typing import Dict
from kedro.pipeline import Pipeline

# 1. Import the pipeline creation functions from your subfolders
from .pipelines.load_raw.pipeline import create_pipeline as create_load_raw_pl
from .pipelines.behavioral.pipeline import create_pipeline as create_behavioral_pl
from .pipelines.simulation.pipeline import create_pipeline as create_simulation_pl

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines."""

    # 2. Instantiate the individual pipelines
    load_raw_pipeline = create_load_raw_pl()
    behavioral_pipeline = create_behavioral_pl()
    simulation_pipeline = create_simulation_pl()

    # The simulation pipeline rebuilds behavioral features from the approved pool.
    # Keep the standalone behavioral export runnable, but exclude it from the
    # default chain to avoid redundant work and mixed semantics.
    full_end_to_end_pipeline = load_raw_pipeline + simulation_pipeline

    return {
        "__default__": full_end_to_end_pipeline,
        "load_raw": load_raw_pipeline,
        "behavioral": behavioral_pipeline,
        "simulation": simulation_pipeline,
    }