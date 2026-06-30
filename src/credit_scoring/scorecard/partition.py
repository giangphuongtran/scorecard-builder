"""Train/validation time partitioning."""

from __future__ import annotations

import pandas as pd


def partition_abt(
    df: pd.DataFrame,
    train_end: str,
    valid_start: str,
    time_col: str = "period",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ABT into non-overlapping train and validation windows.

    Actions:
    1. Keep rows with ``time_col <= train_end`` for train.
    2. Keep rows with ``time_col >= valid_start`` for validation.
    3. Return both partitions as a tuple.
    """
    work = df.copy()
    df_train = work[work[time_col] <= train_end]
    df_valid = work[work[time_col] >= valid_start]
    return df_train, df_valid
