"""Runs the same steps as the retraining workflow in a temp copy of the repo:
train -> evaluate -> record, twice, then a worse model that must be rejected."""
import json

import data
import evaluate_model
import registry
import train_model


def run(ws, timestamp):
    train_model.train(timestamp, out_dir=ws, batch_dir=data.BATCH_DIR)
    report = evaluate_model.evaluate(timestamp, model_dir=ws, registry_path=registry.REGISTRY_PATH)
    registry.record(timestamp, source_dir=ws, path=registry.REGISTRY_PATH)
    return report


def test_first_model_becomes_champion(workspace):
    report = run(workspace, "001")
    assert report["decision"] == "promote"
    assert report["training"]["training_rows"] == 2441
    assert report["holdout_rows"] == 1500   # the temp holdout, not the real one
    reg = json.loads((workspace / "models" / "registry.json").read_text())
    assert reg["champion"]["version"] == "v1"
    assert (workspace / "models" / "model_001.joblib").exists()
    assert (workspace / "metrics" / "001_metrics.json").exists()


def test_worse_model_is_rejected_and_champion_kept(workspace, monkeypatch):
    run(workspace, "001")
    # A much weaker model (two shallow trees) must not replace the champion
    params = train_model.load_params()
    params["model"].update(max_iter=2, max_leaf_nodes=2)
    monkeypatch.setattr(train_model, "load_params", lambda path=None: params)
    report = run(workspace, "002")

    assert report["decision"] == "reject"
    failed = [c["check"] for c in report["checks"] if not c["passed"]]
    assert failed == ["Not worse than champion"]   # the relaxed minimums all pass
    reg = json.loads((workspace / "models" / "registry.json").read_text())
    assert reg["champion"]["version"] == "v1"
    assert [h["decision"] for h in reg["history"]] == ["promote", "reject"]
    assert not (workspace / "models" / "model_002.joblib").exists()


def test_report_is_complete(workspace):
    report = run(workspace, "001")
    for key in ["pr_auc", "roc_auc", "brier", "calibration_error", "buyers_in_top_10pct"]:
        assert 0 <= report["metrics"][key] <= 1
    text = evaluate_model.summary_markdown(report)
    assert "PROMOTE" in text and "| Not worse than champion | pass |" in text
