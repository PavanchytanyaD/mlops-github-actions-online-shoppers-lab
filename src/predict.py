"""Score website sessions with the champion model (or another model file).

    python src/predict.py data/sample_visitors.csv
"""
import argparse

import joblib

import registry
from data import read


def load_champion():
    champion = registry.load_registry()["champion"]
    if champion is None:
        raise SystemExit("No champion model yet. Run the retraining workflow first.")
    return joblib.load(registry.REGISTRY_PATH.parent / champion["model_file"]), champion["version"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="CSV file with the session columns")
    parser.add_argument("--model", help="path to a .joblib model instead of the champion")
    args = parser.parse_args()

    model, name = (joblib.load(args.model), args.model) if args.model else load_champion()
    sessions = read(args.csv)
    sessions["purchase_probability"] = model.predict_proba(sessions)[:, 1].round(3)
    print(f"Model: {name}")
    print(sessions[["Month", "VisitorType", "ProductRelated", "PageValues", "BounceRates",
                    "purchase_probability"]].to_string(index=False))
