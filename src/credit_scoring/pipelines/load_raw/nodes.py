import pandas as pd

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Receives a loaded DataFrame from Kedro's catalog, standardizes 
    its structural schema, and returns it for automated writing.
    """
    df.columns = df.columns.str.lower()
    return df