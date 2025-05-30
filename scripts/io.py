import pandas as pd
import numpy as np
import glob
import os

from typing import List


def load_msna(filepath: str) -> pd.DataFrame:
    df = _load_msna(filepath)
    df = filter_cols(df)
    df = format_dtype(df)
    return df


def _load_msna(filepath: str) -> pd.DataFrame:
    """Loads the MSNA data into a Pandas DataFrame.
    
    Args:
        filepath (str): The filepath to one of the MSNA files (ex:
            'MSNA091_rest_burstcomments_*.txt')

    Returns:
        pd.DataFrame

    """
    with open(filepath, "r", encoding = "utf-8", errors = "replace") as file:
        lines = file.readlines()
        
    cols = lines[4].strip().split('\t')[1:]
    cols = ["Timestamp"] + cols + ["Comments"]
    
    df = pd.read_csv(
        filepath, 
        encoding = "unicode_escape", 
        sep = '\t', 
        skiprows = 9, 
        names = cols
    )
    df[cols[:-1]] = df[cols[:-1]].apply(pd.to_numeric)

    # Add a binary column for the ground truth label. 
    df["Burst"] = (df["Comments"] == "#11 BURST ").astype(np.float32)
    df["Beat"] = (df["Comments"] == "#1 BEAT ").astype(np.float32)

    return df 


def get_all_paths(basedir: str) -> List[str]:
    all_paths = []
    for path in glob.glob(basedir):
        all_paths.extend(
            glob.glob(os.path.join(path, "*_Emma.txt"))
        ) 

    return all_paths


def filter_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Timestamp",
        "ECG",
        "NIBP",
        "Respiratory Waveform",
        "Raw MSNA",
        "Integrated MSNA",
        "ECG Peaks",
        "Burst",
        "Beat"
    ]
    return df[cols]


def format_dtype(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)

    df["Burst"] = np.array(df["Burst"]).astype(bool)
    df["Beat"] = np.array(df["Beat"]).astype(bool)
    
    return df


