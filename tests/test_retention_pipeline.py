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
import pytest
import pandas as pd
import numpy as np

from employee_retention.retention_pipeline import engineer_features

def test_engineer_features_empty_df():
    # An empty dataframe or one without the expected columns
    df = pd.DataFrame({"OtherColumn": [1, 2]})
    result = engineer_features(df)

    # Should create base features using default values (0)
    assert "Engagement_Index" in result.columns
    assert "Promotion_Rate" in result.columns
    assert (result["Engagement_Index"] == 0).all()
    assert (result["Promotion_Rate"] == 0).all()

    # Conditional features should not be present
    assert "Overtime_SensitiveRole" not in result.columns

def test_engineer_features_base_calculations():
    # Valid data for basic features
    df = pd.DataFrame({
        "JobSatisfaction": [3, 4, 1],
        "WorkLifeBalance": [2, 3, 4],
        "YearsSinceLastPromotion": [2, 5, 0],
        "YearsAtCompany": [3, 4, 0]
    })

    result = engineer_features(df)

    # Engagement_Index = JobSatisfaction * WorkLifeBalance
    expected_engagement = pd.Series([6, 12, 4], name="Engagement_Index")
    pd.testing.assert_series_equal(result["Engagement_Index"], expected_engagement)

    # Promotion_Rate = YearsSinceLastPromotion / (YearsAtCompany + 1)
    expected_promotion = pd.Series([2/4, 5/5, 0/1], name="Promotion_Rate", dtype=float)
    pd.testing.assert_series_equal(result["Promotion_Rate"], expected_promotion)

def test_engineer_features_conditional_overtime():
    # Has necessary columns for Overtime_SensitiveRole
    df = pd.DataFrame({
        "Doing_Overtime": [1, 0, 1],
        "Laboratory_Technician": [1, 1, 0]
    })

    result = engineer_features(df)

    assert "Overtime_SensitiveRole" in result.columns
    expected_overtime = pd.Series([1, 0, 0], name="Overtime_SensitiveRole")
    pd.testing.assert_series_equal(result["Overtime_SensitiveRole"], expected_overtime)

def test_engineer_features_polynomials():
    # At least two key_cols to trigger polynomial features
    # key_cols = ['Age', 'JobSatisfaction', 'MonthlyIncome', 'YearsAtCompany']
    df = pd.DataFrame({
        "Age": [30, 40],
        "JobSatisfaction": [3, 4]
    })

    result = engineer_features(df)

    # PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    # So we should get 'Age JobSatisfaction' in the output
    assert "Age JobSatisfaction" in result.columns
    expected_interaction = pd.Series([90.0, 160.0], name="Age JobSatisfaction", dtype=float)
    pd.testing.assert_series_equal(result["Age JobSatisfaction"], expected_interaction)

def test_engineer_features_polynomials_duplicate_column():
    # If a polynomial feature already exists in df, it shouldn't be duplicated
    df = pd.DataFrame({
        "Age": [30, 40],
        "JobSatisfaction": [3, 4],
        "Age JobSatisfaction": [0.0, 0.0] # Already exists
    })

    result = engineer_features(df)

    # The duplicate shouldn't overwrite the original column
    assert "Age JobSatisfaction" in result.columns
    expected_interaction = pd.Series([0.0, 0.0], name="Age JobSatisfaction", dtype=float)
    pd.testing.assert_series_equal(result["Age JobSatisfaction"], expected_interaction)
