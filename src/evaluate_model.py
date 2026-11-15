import argparse
import json
import os

import joblib
import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)

import data
from data import load_holdout
import registry as model_registry
from train_model import load_params


def calibration_error(y, proba, bins=10):
    """Expected calibration error: average gap between predicted and actual
    purchase rate, over 10 equal-width probability bins."""
    y, proba = np.asarray(y), np.asarray(proba)
    which = np.minimum((proba * bins).astype(int), bins - 1)
    gap = 0.0
    for b in range(bins):
        in_bin = which == b
        if in_bin.any():
            gap += in_bin.mean() * abs(y[in_bin].mean() - proba[in_bin].mean())
    return float(gap)


def score(model, X, y):
    proba = model.predict_proba(X)[:, 1]
    # Take the 10% of sessions most likely to buy: what share of all buyers is that?
    top = np.argsort(-proba)[: int(len(y) * 0.1)]
    pred = (proba >= 0.5).astype(int)
    return {
        "pr_auc": round(float(average_precision_score(y, proba)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "brier": round(float(brier_score_loss(y, proba)), 4),
        "calibration_error": round(calibration_error(y, proba), 4),
        "buyers_in_top_10pct": round(float(np.asarray(y)[top].sum() / np.sum(y)), 4),
        "precision_at_0.5": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall_at_0.5": round(float(recall_score(y, pred)), 4),
        "f1_at_0.5": round(float(f1_score(y, pred)), 4),
    }


def quality_gate(new, champion, gate):
    """Returns (promote, list of checks). champion is None for the first model."""
    checks = [
        ("PR-AUC above minimum", new["pr_auc"] >= gate["min_pr_auc"],
         f"{new['pr_auc']} >= {gate['min_pr_auc']}"),
        ("ROC-AUC above minimum", new["roc_auc"] >= gate["min_roc_auc"],
         f"{new['roc_auc']} >= {gate['min_roc_auc']}"),
        ("Calibration error below maximum", new["calibration_error"] <= gate["max_calibration_error"],
         f"{new['calibration_error']} <= {gate['max_calibration_error']}"),
    ]
    if champion is None:
        checks.append(("Not worse than champion", True, "no champion yet"))
    else:
        floor = round(champion["pr_auc"] - gate["max_pr_auc_drop"], 4)
        checks.append(("Not worse than champion", new["pr_auc"] >= floor,
                       f"{new['pr_auc']} >= {champion['pr_auc']} - {gate['max_pr_auc_drop']} = {floor}"))
    return all(ok for _, ok, _ in checks), checks


def evaluate(timestamp, model_dir=None, registry_path=None):
    model_dir = model_dir or data.ROOT
    registry_path = registry_path or model_registry.REGISTRY_PATH
    gate = load_params()["gate"]
    X, y = load_holdout()
    new = score(joblib.load(model_dir / f"model_{timestamp}.joblib"), X, y)

    # The champion is scored again on the same holdout rows, so both numbers are comparable
    registry = model_registry.load_registry(registry_path)
    champion = None
    if registry["champion"]:
        champion_model = joblib.load(registry_path.parent / registry["champion"]["model_file"])
        champion = score(champion_model, X, y)

    promote, checks = quality_gate(new, champion, gate)
    train_info = json.loads((model_dir / f"model_{timestamp}_train.json").read_text())
    report = {
        "timestamp": timestamp,
        "decision": "promote" if promote else "reject",
        "training": train_info,
        "holdout_rows": int(len(y)),
        "metrics": new,
        "champion_version": registry["champion"]["version"] if registry["champion"] else None,
        "champion_metrics": champion,
        "checks": [{"check": name, "passed": ok, "detail": detail} for name, ok, detail in checks],
    }
    (model_dir / f"{timestamp}_metrics.json").write_text(json.dumps(report, indent=2))
    return report


def summary_markdown(report):
    new, old = report["metrics"], report["champion_metrics"] or {}
    lines = [
        f"## Model {report['timestamp']}: {report['decision'].upper()}",
        "",
        f"Trained on {report['training']['training_rows']} rows "
        f"({', '.join(report['training']['batches'])}), scored on {report['holdout_rows']} holdout rows.",
        "",
        f"| Metric | New model | Champion ({report['champion_version'] or 'none'}) |",
        "|---|---|---|",
    ]
    for key in ["pr_auc", "roc_auc", "brier", "calibration_error", "buyers_in_top_10pct"]:
        lines.append(f"| {key} | {new[key]} | {old.get(key, '-')} |")
    lines += ["", "| Check | Result | Detail |", "|---|---|---|"]
    for c in report["checks"]:
        lines.append(f"| {c['check']} | {'pass' if c['passed'] else 'FAIL'} | {c['detail']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()

    report = evaluate(args.timestamp)
    text = summary_markdown(report)
    print(text)

    # Hand the decision to the next jobs in the workflow and show the table on the run page
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"promote={'true' if report['decision'] == 'promote' else 'false'}\n")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(text)
