from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BATCH_DIR = DATA_DIR / "batches"
INCOMING_DIR = DATA_DIR / "incoming"
HOLDOUT_PATH = DATA_DIR / "holdout.csv"

TARGET = "Revenue"
# OperatingSystems, Browser, Region and TrafficType are stored as numbers but are
# category codes (browser 2 is not "twice" browser 1), so they are categorical here
CATEGORICAL = ["Month", "VisitorType", "OperatingSystems", "Browser", "Region", "TrafficType"]
NUMERIC = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
           "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates",
           "SpecialDay", "Weekend"]
FEATURES = CATEGORICAL + NUMERIC


def clean(df):
    """Turn the TRUE/FALSE columns into 0/1, make the code columns text so they are
    treated as categories, and remove exact duplicate sessions."""
    df = df.drop_duplicates().copy()
    for col in [TARGET, "Weekend"]:
        df[col] = df[col].astype(str).str.upper().eq("TRUE").astype(int)
    for col in CATEGORICAL:
        df[col] = df[col].astype(str)
    return df[FEATURES + [TARGET]].reset_index(drop=True)


def read(path):
    # Keep the code columns as text when reading the CSV files back
    return pd.read_csv(path, dtype={col: str for col in CATEGORICAL})


def batch_files(folder=None):
    return sorted(Path(folder or BATCH_DIR).glob("batch_*.csv"))


def load_training_data(folder=None):
    """All batches that have arrived so far, as one table."""
    files = batch_files(folder)
    if not files:
        raise FileNotFoundError(f"no batch_*.csv files in {folder or BATCH_DIR}")
    df = pd.concat([read(f) for f in files], ignore_index=True)
    return df[FEATURES], df[TARGET], [f.name for f in files]


def load_holdout(path=None):
    # Every column is returned, not only FEATURES: a champion trained with an older
    # feature list picks the columns it was fitted on
    df = read(path or HOLDOUT_PATH)
    return df.drop(columns=[TARGET]), df[TARGET]
