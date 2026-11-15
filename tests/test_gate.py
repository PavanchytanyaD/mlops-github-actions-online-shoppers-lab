import pytest

from evaluate_model import calibration_error, quality_gate

GATE = {"min_pr_auc": 0.60, "min_roc_auc": 0.85, "max_calibration_error": 0.05, "max_pr_auc_drop": 0.005}
GOOD = {"pr_auc": 0.75, "roc_auc": 0.93, "calibration_error": 0.02}


def test_first_model_only_needs_the_minimums():
    promote, checks = quality_gate(GOOD, None, GATE)
    assert promote
    assert checks[-1][2] == "no champion yet"


@pytest.mark.parametrize("change", [
    {"pr_auc": 0.59},
    {"roc_auc": 0.80},
    {"calibration_error": 0.06},
])
def test_each_minimum_can_reject(change):
    promote, checks = quality_gate({**GOOD, **change}, None, GATE)
    assert not promote
    assert sum(not ok for _, ok, _ in checks) == 1


def test_small_drop_is_accepted_big_drop_is_rejected():
    champion = {"pr_auc": 0.75}
    assert quality_gate({**GOOD, "pr_auc": 0.746}, champion, GATE)[0]
    assert not quality_gate({**GOOD, "pr_auc": 0.70}, champion, GATE)[0]


def test_calibration_error():
    assert calibration_error([0, 1, 0, 1], [0.0, 1.0, 0.0, 1.0]) == 0
    assert calibration_error([0, 0, 0, 0], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)
