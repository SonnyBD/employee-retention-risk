import numpy as np
import pandas as pd
from unittest.mock import Mock
from employee_retention.retention_pipeline import assign_risk

def test_assign_risk_standard():
    mock_model = Mock()

    # predict_proba returns array where column 1 is stay_prob
    # So leave_prob will be 1 - stay_prob
    # Let's set stay_prob from 0.9 to 0.0, so leave_prob from 0.1 to 1.0
    stay_probs = np.linspace(0.9, 0.0, 10) # 0.9, 0.8, 0.7, ..., 0.0
    probs = np.zeros((10, 2))
    probs[:, 1] = stay_probs
    mock_model.predict_proba.return_value = probs

    X = pd.DataFrame(np.zeros((10, 2)))

    # Act
    # leave_probs will be 0.1, 0.2, ..., 1.0
    # 90th percentile of [0.1...1.0] is 0.91
    # bins: [0, 0.5, 0.91, 1]
    # <= 0.5: Low
    # 0.5 < x <= 0.91: Moderate
    # 0.91 < x <= 1: High
    result = assign_risk(mock_model, X, threshold=0.5)

    assert len(result) == 10

    # Check leave_probs
    expected_leave_probs = 1 - stay_probs
    np.testing.assert_allclose(result['Leave_Probability'].values, expected_leave_probs)

    # Check risk tiers
    expected_tiers = [
        'Low', 'Low', 'Low', 'Low', 'Low', # 0.1, 0.2, 0.3, 0.4, 0.5
        'Moderate', 'Moderate', 'Moderate', 'Moderate', # 0.6, 0.7, 0.8, 0.9
        'High' # 1.0
    ]
    assert list(result['Risk_Tier']) == expected_tiers

def test_assign_risk_high_threshold():
    # If threshold is very high, e.g. 0.95
    # threshold + 0.01 = 0.96
    # 90th percentile = 0.91
    # max(high_boundary, threshold + 0.01) = 0.96
    # bins: [0, 0.95, 0.96, 1]
    mock_model = Mock()
    stay_probs = np.linspace(0.9, 0.0, 10)
    probs = np.zeros((10, 2))
    probs[:, 1] = stay_probs
    mock_model.predict_proba.return_value = probs

    X = pd.DataFrame(np.zeros((10, 2)))

    result = assign_risk(mock_model, X, threshold=0.95)

    expected_tiers = [
        'Low', 'Low', 'Low', 'Low', 'Low', 'Low', 'Low', 'Low', 'Low', 'High'
    ]
    # Actually wait: 1.0 is > 0.96, so it's High.
    # What about between 0.95 and 0.96? we don't have any in this array.
    assert list(result['Risk_Tier']) == expected_tiers

def test_assign_risk_all_low_leave_prob():
    # What if everyone is super happy and likely to stay?
    # leave_prob = [0.01, 0.02, 0.03, 0.04, 0.05]
    mock_model = Mock()
    stay_probs = np.array([0.99, 0.98, 0.97, 0.96, 0.95])
    probs = np.zeros((5, 2))
    probs[:, 1] = stay_probs
    mock_model.predict_proba.return_value = probs

    X = pd.DataFrame(np.zeros((5, 2)))

    result = assign_risk(mock_model, X, threshold=0.5)

    # 90th percentile of [0.01, 0.02, 0.03, 0.04, 0.05] is 0.046
    # max(0.046, 0.51) = 0.51
    # bins: [0, 0.5, 0.51, 1]
    # leave_probs are all <= 0.05, so all are 'Low'
    assert all(result['Risk_Tier'] == 'Low')
