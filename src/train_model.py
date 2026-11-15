import argparse
import json

import joblib
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

import data
from data import CATEGORICAL, NUMERIC, load_training_data


def load_params(path=None):
    with open(path or data.ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def build_model(model_params):
    # Categories become integer codes; the boosting model treats them as categorical
    encode = ColumnTransformer(
        [("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
          CATEGORICAL)],
        remainder="passthrough",
    )
    boosting = HistGradientBoostingClassifier(
        categorical_features=list(range(len(CATEGORICAL))),
        early_stopping=True,
        **model_params,
    )
    return make_pipeline(encode, boosting)


def train(timestamp, out_dir=None, batch_dir=None):
    out_dir = out_dir or data.ROOT
    params = load_params()
    X, y, batches = load_training_data(batch_dir)
    model = build_model(params["model"]).fit(X[CATEGORICAL + NUMERIC], y)

    model_path = out_dir / f"model_{timestamp}.joblib"
    joblib.dump(model, model_path)
    info = {
        "timestamp": timestamp,
        "training_rows": int(len(y)),
        "batches": batches,
        "positive_rate": round(float(y.mean()), 4),
        "boosting_iterations": int(model[-1].n_iter_),
    }
    (out_dir / f"model_{timestamp}_train.json").write_text(json.dumps(info, indent=2))
    return model_path, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()

    print(f"Timestamp received from GitHub Actions: {args.timestamp}")
    path, info = train(args.timestamp)
    print(f"Trained on {info['training_rows']} rows from {len(info['batches'])} batches, "
          f"{info['boosting_iterations']} boosting iterations -> {path.name}")
