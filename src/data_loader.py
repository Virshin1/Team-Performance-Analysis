"""
Data Loader Module for Team Performance Analysis.
Loads the raw UCI Garment Worker Productivity dataset and applies foundational data hygiene.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "garments_worker_productivity.csv"

def load_raw_data(filepath: str | Path = None) -> pd.DataFrame:
    """
    Loads raw CSV data from the specified path or default data directory.
    """
    if filepath is None:
        filepath = DATA_PATH
    
    df = pd.read_csv(filepath)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies empirical data cleaning based on dataset exploration:
    1. Trims whitespace from string columns (fixing 'finishing ' vs 'finishing').
    2. Normalizes department names ('sweing' -> 'sewing').
    3. Fixes data entry column swap for finishing rows on 2015-03-09 where overtime
       was entered as 0 and the overtime value was entered into incentive.
    4. Parses dates into datetime format and extracts calendar features.
    """
    df_clean = df.copy()

    # 1. Clean string columns
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    # 2. Normalize department typos
    df_clean['department'] = df_clean['department'].replace({'sweing': 'sewing'})

    # 3. Parse date
    df_clean['date'] = pd.to_datetime(df_clean['date'], format='%m/%d/%Y')

    # 4. Resolve data entry swap anomaly on 2015-03-09 for finishing department
    # On 3/9/2015, finishing records recorded 0 overtime and values like 960, 1080, 2880, 3600 in incentive
    # These match worker count * standard overtime minutes (120 mins/worker).
    mask_swap = (df_clean['date'] == '2015-03-09') & (df_clean['department'] == 'finishing') & (df_clean['over_time'] == 0) & (df_clean['incentive'] >= 960)
    for idx in df_clean[mask_swap].index:
        val = df_clean.loc[idx, 'incentive']
        df_clean.loc[idx, 'over_time'] = val
        df_clean.loc[idx, 'incentive'] = 0

    return df_clean

def get_cleaned_data(filepath: str | Path = None) -> pd.DataFrame:
    """
    Convenience wrapper to load and clean the dataset.
    """
    raw_df = load_raw_data(filepath)
    return clean_data(raw_df)

if __name__ == "__main__":
    df = get_cleaned_data()
    print(f"Data loaded successfully! Shape: {df.shape}")
    print(f"Departments: {df['department'].value_counts().to_dict()}")
    print(f"Incentive max after correction: {df['incentive'].max()}")
