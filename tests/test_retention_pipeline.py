import numpy as np
import pytest
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from employee_retention.retention_pipeline import find_best_threshold

def test_find_best_threshold_perfect_separation():
    # True labels: 0=leaver, 1=stayer
    y_true = np.array([0, 0, 1, 1])

    # Probabilities of staying
    y_probs = np.array([0.9, 0.85, 0.2, 0.1])
    # leave_probs = [0.1, 0.15, 0.8, 0.9]
    # Prediction is 1 if leave_probs >= t, 0 otherwise
    # If t is between 0.15 and 0.80, prediction is [0, 0, 1, 1], which perfectly matches y_true.
    # We expect F1=1.0.
    # find_best_threshold will return the first threshold that gives the max F1.
    # The first threshold > 0.15 is 0.16.

    thresh, f1 = find_best_threshold(y_true, y_probs)

    assert f1 == 1.0
    assert np.isclose(thresh, 0.16, atol=1e-2)

def test_find_best_threshold_all_zeros():
    # What if all true labels are 0?
    y_true = np.array([0, 0, 0, 0])
    y_probs = np.array([0.8, 0.8, 0.8, 0.8])
    # leave_probs = [0.2, 0.2, 0.2, 0.2]
    # For t <= 0.2, pred = [1, 1, 1, 1], f1 = 0.0 for pos_label=0
    # For t > 0.2, pred = [0, 0, 0, 0], f1 = 1.0 for pos_label=0
    # The first threshold > 0.2 is 0.21.

    thresh, f1 = find_best_threshold(y_true, y_probs)

    assert f1 == 1.0
    assert np.isclose(thresh, 0.21, atol=1e-2)

def test_find_best_threshold_no_leaver():
    # If there are no leavers (0), f1 score for pos_label=0 might raise UndefinedMetricWarning
    y_true = np.array([1, 1, 1, 1])
    y_probs = np.array([0.8, 0.8, 0.8, 0.8])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        thresh, f1 = find_best_threshold(y_true, y_probs)

    # since no true 0s, F1 will be 0.0 everywhere
    assert f1 == 0.0
    # threshold should just be the first one (0.05)
    assert np.isclose(thresh, 0.05, atol=1e-2)

def test_find_best_threshold_mixed():
    # A mixed scenario
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_probs = np.array([0.7, 0.6, 0.4, 0.9, 0.2, 0.3])

    thresh, f1 = find_best_threshold(y_true, y_probs)

    # The best threshold > 0.7 gives F1 of ~0.666
    assert np.isclose(f1, 0.666, atol=1e-2)
    assert np.isclose(thresh, 0.80, atol=1e-2)
