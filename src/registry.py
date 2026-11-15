"""models/registry.json keeps the current champion and every model that was evaluated."""
import argparse
import json
import shutil

import data
from data import ROOT

REGISTRY_PATH = ROOT / "models" / "registry.json"


def load_registry(path=None):
    path = path or REGISTRY_PATH
    if not path.exists():
        return {"champion": None, "history": []}
    return json.loads(path.read_text())


def record(timestamp, source_dir=None, path=None):
    """Add the evaluated model to the history. If it passed the gate, copy it into
    models/ as the next version and make it the champion."""
    source_dir = source_dir or data.ROOT
    path = path or REGISTRY_PATH
    report = json.loads((source_dir / f"{timestamp}_metrics.json").read_text())
    registry = load_registry(path)
    entry = {
        "timestamp": timestamp,
        "decision": report["decision"],
        "pr_auc": report["metrics"]["pr_auc"],
        "roc_auc": report["metrics"]["roc_auc"],
        "training_rows": report["training"]["training_rows"],
    }
    if report["decision"] == "promote":
        version = sum(1 for h in registry["history"] if h["decision"] == "promote") + 1
        model_file = f"model_{timestamp}.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_dir / model_file, path.parent / model_file)
        entry["version"] = f"v{version}"
        registry["champion"] = {"version": entry["version"], "timestamp": timestamp,
                                "model_file": model_file, **{k: entry[k] for k in ["pr_auc", "roc_auc"]}}
    registry["history"].append(entry)
    path.write_text(json.dumps(registry, indent=2) + "\n")

    metrics_dir = path.parent.parent / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    shutil.copy(source_dir / f"{timestamp}_metrics.json", metrics_dir / f"{timestamp}_metrics.json")
    return entry


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    args = parser.parse_args()
    entry = record(args.timestamp)
    print(f"Recorded {args.timestamp}: {entry['decision']} {entry.get('version', '')}")
