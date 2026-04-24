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
