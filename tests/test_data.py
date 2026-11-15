import pandas as pd

import data


def test_clean_converts_flags_and_drops_duplicates():
    row = {**{c: 1 for c in data.NUMERIC}, **{c: 2 for c in data.CATEGORICAL},
           "Weekend": "FALSE", "Revenue": "TRUE"}
    clean = data.clean(pd.DataFrame([row, row]))
    assert len(clean) == 1
    assert clean["Revenue"].tolist() == [1]
    assert clean["Weekend"].tolist() == [0]
    assert clean["Browser"].tolist() == ["2"]      # code columns become text


def test_files_have_the_model_columns():
    for f in [data.HOLDOUT_PATH, *data.batch_files(), *data.batch_files(data.INCOMING_DIR)]:
        cols = pd.read_csv(f, nrows=1).columns.tolist()
        assert set(data.FEATURES + [data.TARGET]) <= set(cols), f.name


def test_code_columns_are_read_as_text():
    X, _ = data.load_holdout()
    for col in data.CATEGORICAL:
        assert X[col].map(type).eq(str).all(), col


def test_holdout_is_never_in_training_batches():
    holdout = data.read(data.HOLDOUT_PATH)
    batches = pd.concat(data.read(f) for f in [*data.batch_files(), *data.batch_files(data.INCOMING_DIR)])
    overlap = holdout.merge(batches, on=holdout.columns.tolist(), how="inner")
    assert overlap.empty


def test_holdout_has_both_classes():
    _, y = data.load_holdout()
    assert set(y.unique()) == {0, 1}
    assert 0.1 < y.mean() < 0.2
