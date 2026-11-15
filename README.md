# Continuous Training with a Quality Gate: MLOps GitHub Actions Lab

This lab uses GitHub Actions to retrain a machine learning model every time new
data or code is pushed, check the new model against the current one, and only
keep it if it passes a quality gate. Every model is versioned with a timestamp,
its metrics are committed to the repository, and `models/registry.json` records
which version is the current champion. A second workflow runs the tests and builds
a Docker image on every push.

The model predicts whether a visit to an online store will end in a purchase, from
how the visitor browsed the site (pages viewed, time spent, bounce and exit rates,
month, visitor type).

> **Lab Assignment 4 submission.** Adapted from
> [Github_Labs/Lab2](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Github_Labs/Lab2)
> in the [raminmohammadi/MLOps](https://github.com/raminmohammadi/MLOps) course
> repository. The structure is the same (a workflow that retrains on push to main,
> `train_model.py` and `evaluate_model.py` that take a `--timestamp`, models and
> metrics versioned by timestamp and committed back by the workflow), and the
> dataset, model, evaluation and the decision of which model to keep are new. See
> [My Modifications](#my-modifications).

---

## Dataset

The [Online Shoppers Purchasing Intention dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
from the UCI Machine Learning Repository. It is not used in any of the course labs.

- 12,330 visits (sessions) to an online store over one year. Each session belongs to
  a different user.
- 17 inputs:
  - **pages and time:** how many administrative, informational and product pages the
    visitor opened and how long they spent on each type
  - **Google Analytics measures:** `BounceRates` (left after one page), `ExitRates`
    (how often the pages were the last one viewed) and `PageValues` (the average value
    of the pages the visitor saw before a purchase)
  - **context:** `Month`, `SpecialDay` (closeness to a holiday such as Valentine's
    Day), `Weekend`, `VisitorType` (new or returning), and codes for operating system,
    browser, region and traffic source
- Target `Revenue`: did the session end in a purchase? 15.5% did (15.6% after
  removing duplicates).
- `OperatingSystems`, `Browser`, `Region` and `TrafficType` are stored as numbers but
  are category codes (browser 2 is not "twice" browser 1), so they are read as text
  and treated as categories.
- 125 exact duplicate sessions are removed, which leaves 12,205.

`src/prepare_data.py` downloads the data and splits it once:

| File | Rows | Used for |
|---|---|---|
| `data/holdout.csv` | 2,441 (20%, stratified) | Fixed test set. No model is ever trained on it, so every model is scored on the same rows. |
| `data/batches/batch_01.csv` ... `batch_04.csv` | about 2,441 each | Training data. The model trains on every batch in `data/batches/`. |
| `data/incoming/` | | Batches that have not "arrived" yet. `src/ingest_batch.py` moves the next one into `data/batches/`. |
| `data/sample_visitors.csv` | 6 | Sessions from the holdout set (without the answer) for trying out predictions. |

License: CC BY 4.0. Sakar, C. and Kastro, Y. (2018). Online Shoppers Purchasing
Intention Dataset. UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q

---

## My Modifications

The original lab trains a Random Forest on random synthetic data
(`make_classification` with a random number of rows), scores it on newly generated
data, saves every model and commits it. In this version:

- **Real data that arrives over time:** the Online Shoppers data, split into a fixed
  holdout set and four training batches. Moving a batch into `data/batches/` and
  pushing it starts a retraining run, the way new labelled data would arrive in a
  real project.
- **Model:** `HistGradientBoostingClassifier` with the six category columns handled
  as native categorical features, early stopping, and settings in `params.yaml`.
- **Evaluation on a fixed holdout set:** PR-AUC (the main metric, since only about 16% of
  sessions end in a purchase), ROC-AUC, Brier score, calibration error, the share of
  all buyers found in the 10% of sessions the model ranks highest, and precision,
  recall and F1 at 0.5.
- **Calibration is measured, not just named:** the original README describes a
  calibration workflow, but its workflow file only retrains. Here the calibration
  error is computed for every model and is one of the gate checks.
- **Quality gate:** a new model replaces the champion only if it passes every check
  in `params.yaml` (see [The Quality Gate](#the-quality-gate)). The champion is
  scored again on the same holdout rows in the same run, so the two numbers can be
  compared directly.
- **The workflow branches on the decision:** after training and evaluation, either
  the `promote` job (store the model, make it the champion) or the
  `record-rejection` job (log it, keep the champion) runs. On a pull request
  neither runs, so a PR shows the gate result without changing anything.
- **Model registry:** `models/registry.json` holds the current champion and the
  history of every model that was evaluated, with its decision and scores.
- **CI with tests and a Docker build:** a separate `CI` workflow runs pytest, then
  builds the Docker image and trains, evaluates and predicts inside it.
- **Prediction script:** `src/predict.py` scores sessions with the champion.
- **Workflow fixes:** current action versions (`checkout@v4`, `setup-python@v5`),
  the timestamp passed between jobs as a job output, `concurrency` so two runs never
  update the registry at once, the commit made as `github-actions[bot]`, and model
  files passed between jobs as an artifact.

---

## How It Works

### Workflows

| Workflow | File | Runs on | Jobs |
|---|---|---|---|
| CI | `.github/workflows/ci.yml` | every push and pull request | `tests` (pytest), then `docker` (build the image, run the pipeline inside it) |
| Model Retraining with Quality Gate | `.github/workflows/model_retraining_on_push.yml` | push to `main` or a pull request that changes `data/batches/`, `src/`, `params.yaml` or `requirements.txt`, or **Run workflow** in the Actions tab | `train-and-evaluate`, then `promote` **or** `record-rejection` |

### Retraining workflow, step by step

```
train-and-evaluate ──► promote             (gate passed, push to main)
                  └──► record-rejection    (gate failed, push to main)
                       (neither on a pull request)
```

**Job `train-and-evaluate`**

| Step | What it does |
|---|---|
| Checkout code, Set up Python, Install dependencies | Python 3.11 and the pinned packages in `requirements.txt` |
| Generate and Store Timestamp | `date -u +%Y%m%d%H%M%S`, saved as a step output and used as the model's version tag |
| Train Model | `python src/train_model.py --timestamp <ts>`: trains on every file in `data/batches/`, writes `model_<ts>.joblib` and `model_<ts>_train.json` (rows, batches, boosting iterations) |
| Evaluate Model and Apply Quality Gate | `python src/evaluate_model.py --timestamp <ts>`: scores the new model and the champion on `data/holdout.csv`, applies the gate, writes `<ts>_metrics.json`, sets the job output `promote=true/false` and writes a summary table to the run page |
| Upload model and metrics | Passes the three files to the next job as an artifact |

**Job `promote`** (only if `promote=true` and not a pull request): downloads the
artifact, runs `python src/registry.py --timestamp <ts>` to copy the model into
`models/`, make it the champion and add it to the history, copies the metrics into
`metrics/`, then commits and pushes `promote model <ts>`.

**Job `record-rejection`** (only if `promote=false` and not a pull request): the same
script adds the model to the history as rejected and keeps the champion. Only the
registry and metrics are committed; the rejected model file is not stored.

Commits made by `GITHUB_TOKEN` do not start new workflow runs, so the bot's commit
does not trigger another retraining.

### The Quality Gate

The thresholds are in `params.yaml`:

| Check | Rule | Why |
|---|---|---|
| PR-AUC above minimum | `pr_auc >= 0.60` | A weak model is never shipped, even as the first one |
| ROC-AUC above minimum | `roc_auc >= 0.85` | Same, with a metric that does not depend on the share of buyers |
| Calibration error below maximum | `calibration_error <= 0.05` | The predicted purchase probability should be close to the real purchase rate |
| Not worse than champion | `pr_auc >= champion_pr_auc - 0.005` | A new model may be very slightly worse (noise), but not clearly worse |

The first model has no champion to compare with, so only the first three checks
apply. The calibration error is the average gap between the predicted and actual
purchase rate over ten probability bins. With 2,441 holdout sessions it moves by
about 0.01 between models, so the limit is set at 0.05.

### Model registry

`models/registry.json`:

```json
{
  "champion": {"version": "v4", "timestamp": "20261115170036",
               "model_file": "model_20261115170036.joblib", "pr_auc": 0.7542, "roc_auc": 0.9337},
  "history": [
    {"timestamp": "20261115165722", "decision": "promote", "pr_auc": 0.7247,
     "roc_auc": 0.9258, "training_rows": 2441, "version": "v1"},
    ...
  ]
}
```

`metrics/<timestamp>_metrics.json` has the full report for every model: training
info, all metrics, the champion's metrics on the same rows, and each gate check with
its numbers.

---

## Results

Every row below is a real GitHub Actions run in this repository.

| Version | Trigger | Training rows | PR-AUC | ROC-AUC | Brier | Calibration error | Buyers in top 10% | Decision |
|---|---|---|---|---|---|---|---|---|
| v1 | Run workflow (first model) | 2,441 (batch 1) | 0.7247 | 0.9258 | 0.0735 | 0.0264 | 50.3% | promote |
| v2 | push `add batch 02` | 4,882 | 0.7299 | 0.9278 | 0.0730 | 0.0292 | 49.5% | promote |
| v3 | push `add batch 03` | 7,323 | 0.7400 | 0.9315 | 0.0702 | 0.0177 | 50.5% | promote |
| v4 | push `add batch 04` | 9,764 | 0.7542 | 0.9337 | 0.0698 | 0.0176 | 50.5% | promote |
| (none) | [pull request #1](https://github.com/PavanchytanyaD/mlops-github-actions-online-shoppers-lab/pull/1): drop `PageValues` | 9,764 | 0.3663 | 0.7926 | | | | **reject** |

- **Every new batch made the model better.** PR-AUC goes from 0.725 with one batch to
  0.754 with four, and ROC-AUC from 0.926 to 0.934, so each batch passed the gate.
- **The 10% of sessions the model ranks highest contain about half of all buyers**,
  against 10% for picking sessions at random. A store could show offers or a chat
  prompt to those visitors first.
- **At a 0.5 cut-off the champion (v4) finds 65% of buyers, and 71% of the sessions it
  flags really buy.**
- **The gate rejected a real change.** Pull request #1 removed `PageValues`, which only
  stores that use Google Analytics page values would have. The workflow trained and
  scored it on the pull request: PR-AUC fell from 0.7542 to 0.3663, three of the four
  checks failed, and the run page showed **REJECT**. The PR was closed without merging
  and the champion stayed v4. `PageValues` carries most of the signal: 22% of sessions
  have a page value above 0, and 56% of those end in a purchase, against 4% of the
  others.

The summary the workflow writes on the run page for pull request #1:

```
## Model 20261115170208: REJECT
| pr_auc  | 0.3663 | 0.7542 |
| roc_auc | 0.7926 | 0.9337 |
| PR-AUC above minimum    | FAIL | 0.3663 >= 0.6 |
| ROC-AUC above minimum   | FAIL | 0.7926 >= 0.85 |
| Not worse than champion | FAIL | 0.3663 >= 0.7542 - 0.005 = 0.7492 |
```

Scoring the sample sessions with the champion (the first three bought, the last three
did not):

```
$ python src/predict.py data/sample_visitors.csv
Model: v4
Month       VisitorType  ProductRelated  PageValues  BounceRates  purchase_probability
  Nov Returning_Visitor             338    1.114150     0.007520                 0.682
  May Returning_Visitor               9   17.270400     0.000000                 0.560
  Jul Returning_Visitor             310    0.098621     0.000039                 0.138
  May Returning_Visitor              42    0.000000     0.005000                 0.002
  Nov       New_Visitor              11    0.000000     0.027273                 0.039
  Oct Returning_Visitor              28    0.000000     0.000000                 0.118
```

---

## Project Structure

```
.
├── .github/workflows/
│   ├── ci.yml                          # tests + Docker build on every push and PR
│   └── model_retraining_on_push.yml    # train -> evaluate -> gate -> promote / record
├── params.yaml               # model settings and gate thresholds
├── requirements.txt          # pinned packages for training and prediction
├── requirements-dev.txt      # requirements + pytest
├── pytest.ini
├── Dockerfile
├── data/
│   ├── holdout.csv           # fixed test set
│   ├── batches/              # training batches that have arrived
│   ├── incoming/             # batches still to arrive
│   └── sample_visitors.csv
├── src/
│   ├── data.py               # columns, cleaning, loading batches and holdout
│   ├── prepare_data.py       # download and split the dataset (run once)
│   ├── ingest_batch.py       # move the next batch from incoming/ to batches/
│   ├── train_model.py        # train on all batches
│   ├── evaluate_model.py     # metrics, champion comparison, quality gate
│   ├── registry.py           # store the model and update models/registry.json
│   └── predict.py            # score sessions with the champion
├── models/                   # versioned models + registry.json (written by the workflow)
├── metrics/                  # one metrics report per model (written by the workflow)
└── tests/
    ├── conftest.py           # temp copy of the repo layout for the tests
    ├── test_data.py          # cleaning, columns, holdout never in training data
    ├── test_gate.py          # each gate rule, calibration error
    └── test_pipeline.py      # train -> evaluate -> record, promotion and rejection
```

---

## How to Run

### Prerequisites

- A GitHub account (to run the workflows)
- Python 3.11 and Git
- Docker (optional, to build the image locally)

### Step 1: Get your own copy

Fork this repository on GitHub (the workflows need to push to your copy), then clone it:

```sh
git clone https://github.com/<your-username>/mlops-github-actions-online-shoppers-lab.git
cd mlops-github-actions-online-shoppers-lab
```

### Step 2: Allow the workflow to push

The `promote` and `record-rejection` jobs commit to `main`. The workflow asks for
`contents: write` itself, but if your account or organization limits workflow
permissions, go to **Settings > Actions > General > Workflow permissions** and choose
**Read and write permissions**. On a fork, also enable workflows in the **Actions** tab.

### Step 3: Run the tests and the pipeline locally

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

The same steps the workflow runs, with your own timestamp:

```sh
python src/train_model.py --timestamp local1
python src/evaluate_model.py --timestamp local1
python src/predict.py data/sample_visitors.csv --model model_local1.joblib
```

`evaluate_model.py` prints the metrics and gate table. These files stay in the repo
root and are ignored by git; only the workflow writes to `models/` and `metrics/`.

### Step 4: Start over with a fresh registry (optional)

This repository already has models v1 to v4 and all four batches in `data/batches/`.
To replay the whole story in your fork, delete the registry and models and put
batches 2 to 4 back in `data/incoming/`:

```sh
git rm -q models/*.joblib models/registry.json metrics/*_metrics.json
mkdir -p data/incoming
git mv data/batches/batch_0[234].csv data/incoming/
git commit -m "start with one batch"
git push
```

The push changes `data/batches/`, so it starts the retraining workflow. With no
champion in the registry, the model trained on batch 1 only has to pass the
minimums and becomes v1. When the run finishes, `promote model <timestamp>` appears
in the commit history; get it with `git pull`. (You can also start a run at any time
with **Actions > Model Retraining with Quality Gate > Run workflow**.)

### Step 5: Add a new batch of data

```sh
python src/ingest_batch.py
git add data/
git commit -m "add batch 02"
git push
```

The push starts the retraining workflow. Open the **Actions** tab, click the run and
look at:

- the graph: `train-and-evaluate`, then `promote` or `record-rejection`
- the summary below the graph: the metrics of the new model and the champion, and
  each gate check
- the artifact `model-<timestamp>` with the model and metrics files

When it finishes, `git pull` to get the bot's commit with the new model and registry.

### Step 6: Try a change in a pull request

```sh
git checkout -b my-experiment
# change params.yaml or src/, for example the learning rate
git commit -am "try a higher learning rate"
git push -u origin my-experiment
```

Open a pull request. Both workflows run on it: the tests and Docker build, and a
retraining run that shows whether the change would be promoted. Nothing is written to
`main` until the PR is merged.

### Step 7: Use the champion

```sh
python src/predict.py data/sample_visitors.csv
```

### Step 8: Docker

```sh
docker build -t online-shoppers-model .
docker run --rm online-shoppers-model                   # score the sample sessions with the champion
docker run --rm online-shoppers-model sh -c "python src/train_model.py --timestamp docker && python src/evaluate_model.py --timestamp docker"
```

### Rebuilding the data (optional)

`data/` is already in the repository. To create it again from the UCI download:

```sh
cd src && python prepare_data.py
```

---

## Troubleshooting

- **The retraining workflow did not start after my push:** it only runs when
  `data/batches/`, `src/`, `params.yaml` or `requirements.txt` change. The very first
  push of a new repository does not start path-filtered workflows; use **Run
  workflow** in the Actions tab.
- **`promote` fails with "Permission denied" or 403:** see Step 2.
- **`git push` is rejected after a retraining run:** the bot pushed a commit. Run
  `git pull --rebase`, then push again.
- **`No champion model yet`:** no model has been promoted. Run the retraining
  workflow first, or pass `--model` with a local model file.
