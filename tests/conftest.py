import pandas as pd
import pytest
import yaml

import data
import registry


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A copy of the repo layout in a temp folder: two small batches, a small holdout,
    params.yaml and an empty models/ folder. Nothing in the real repo is touched."""
    (tmp_path / "data" / "batches").mkdir(parents=True)
    (tmp_path / "models").mkdir()
    holdout = data.read(data.HOLDOUT_PATH).sample(1500, random_state=0)
    holdout.to_csv(tmp_path / "data" / "holdout.csv", index=False)
    batch = data.read(data.BATCH_DIR / "batch_01.csv")
    batch.iloc[:1200].to_csv(tmp_path / "data" / "batches" / "batch_01.csv", index=False)
    batch.iloc[1200:].to_csv(tmp_path / "data" / "batches" / "batch_02.csv", index=False)
    # Same model settings, but minimums of 0: these tests check that train -> evaluate ->
    # record works, not how good the model is (that is the gate's job in the workflow)
    params = yaml.safe_load((data.ROOT / "params.yaml").read_text())
    params["gate"].update(min_pr_auc=0, min_roc_auc=0, max_calibration_error=1)
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(params))

    for module in [data, registry]:
        monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(data, "BATCH_DIR", tmp_path / "data" / "batches")
    monkeypatch.setattr(data, "HOLDOUT_PATH", tmp_path / "data" / "holdout.csv")
    monkeypatch.setattr(registry, "REGISTRY_PATH", tmp_path / "models" / "registry.json")
    return tmp_path
