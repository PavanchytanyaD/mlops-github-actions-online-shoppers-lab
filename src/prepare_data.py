"""Download the UCI Online Shoppers Purchasing Intention data once and split it into a fixed holdout set
and four training batches. Batch 1 goes to data/batches/ (data we already have),
batches 2-4 go to data/incoming/ and arrive later with ingest_batch.py."""
import io
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data import BATCH_DIR, HOLDOUT_PATH, INCOMING_DIR, TARGET, clean

URL = "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"
N_BATCHES = 4
SEED = 42


def download():
    archive = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL).read()))
    with archive.open("online_shoppers_intention.csv") as f:
        return pd.read_csv(f)


if __name__ == "__main__":
    df = clean(download())
    pool, holdout = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=SEED)
    pool = pool.sample(frac=1, random_state=SEED)

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    holdout.to_csv(HOLDOUT_PATH, index=False)
    for i, part in enumerate(np.array_split(np.arange(len(pool)), N_BATCHES), start=1):
        folder = BATCH_DIR if i == 1 else INCOMING_DIR
        pool.iloc[part].to_csv(folder / f"batch_{i:02d}.csv", index=False)

    print(f"{len(df)} rows -> holdout {len(holdout)}, {N_BATCHES} batches of about {len(pool) // N_BATCHES}")
