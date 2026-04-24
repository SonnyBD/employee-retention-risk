"""Streamlit dashboard for the employee retention risk model.

The pipeline (employee_retention.retention_pipeline) produces these
artifacts in outputs/, all of which are required here:
  - final_calibrated_model.joblib
  - final_scaler.joblib                (fit on all engineered columns)
  - all_feature_columns.joblib         (column order scaler expects)
  - selected_feature_names.joblib      (columns the model actually uses)
  - decision_threshold.joblib          (tuned on validation set)

Inference order matches training: engineer -> reindex to full column
set -> scale -> subset to selected columns -> predict.
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st

from employee_retention.retention_pipeline import engineer_features


OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')


@st.cache_resource
def load_artifacts():
    """Load all pipeline artifacts and initialize explainer. Raises a clear error if missing."""
    required = {
        'model': 'final_calibrated_model.joblib',
        'scaler': 'final_scaler.joblib',
        'all_cols': 'all_feature_columns.joblib',
        'selected': 'selected_feature_names.joblib',
        'threshold': 'decision_threshold.joblib',
    }
    missing = [f for f in required.values() if not os.path.exists(os.path.join(OUTPUTS_DIR, f))]
    if missing:
        raise FileNotFoundError(
            "Missing pipeline artifacts: " + ", ".join(missing) +
            "\nRun `python -m employee_retention.retention_pipeline` first."
        )

    artifacts = {k: joblib.load(os.path.join(OUTPUTS_DIR, v)) for k, v in required.items()}

    # Pre-initialize SHAP explainer to avoid blocking on every request
    model = artifacts['model']
    base_pipeline = model.calibrated_classifiers_[0].estimator
    base_xgb = base_pipeline.named_steps['xgb']

    artifacts['explainer'] = shap.TreeExplainer(base_xgb)

    return artifacts


st.set_page_config(page_title="Employee Retention Risk Dashboard", layout="centered")
st.title("📊 Employee Attrition Risk Predictor")
st.markdown(
    "Estimate the likelihood of an employee leaving and understand the key "
    "drivers behind the prediction."
)

try:
    artifacts = load_artifacts()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

model = artifacts['model']
scaler = artifacts['scaler']
all_feature_columns = artifacts['all_cols']
selected_features = artifacts['selected']
threshold = artifacts['threshold']
explainer = artifacts['explainer']


with st.form("input_form"):
    st.subheader("Input Employee Information")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 60, 35)
        monthly_income = st.slider("Monthly Income", 1000, 20000, 5000)
        years_at_company = st.slider("Years at Company", 0, 40, 5)
        years_since_promo = st.slider("Years Since Last Promotion", 0, 15, 2)
        years_in_role = st.slider("Years in Current Role", 0, 18, 3)
        years_with_mgr = st.slider("Years with Current Manager", 0, 17, 3)
        num_companies = st.slider("Number of Companies Worked", 0, 10, 2)
        total_working = st.slider("Total Working Years", 0, 40, 10)
    with col2:
        job_sat = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
        env_sat = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
        job_involve = st.slider("Job Involvement (1-4)", 1, 4, 3)
        wlb = st.slider("Work-Life Balance (1-4)", 1, 4, 3)
        distance = st.slider("Distance From Home (km)", 1, 30, 5)
        ot = st.selectbox("Doing Overtime?", [0, 1])
        travel = st.selectbox(
            "Business Travel", ["Non_Travel", "Travel_Rarely", "Travel_Frequently"]
        )
        marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        role = st.selectbox(
            "Job Role",
            ["Laboratory_Technician", "Sales_Executive", "Research_Scientist",
             "Sales_Representative", "Human_Resources", "Other"],
        )

    submitted = st.form_submit_button("Predict Risk")


if submitted:
    # Build the single-row input. Any engineered column not supplied
    # below will be filled with 0 via reindex.
    raw = {
        "Age": age,
        "JobSatisfaction": job_sat,
        "EnvironmentSatisfaction": env_sat,
        "JobInvolvement": job_involve,
        "WorkLifeBalance": wlb,
        "DistanceFromHome": distance,
        "YearsSinceLastPromotion": years_since_promo,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_role,
        "YearsWithCurrManager": years_with_mgr,
        "NumCompaniesWorked": num_companies,
        "TotalWorkingYears": total_working,
        "Doing_Overtime": int(ot),
        "MonthlyIncome": monthly_income,
        # One-hot categoricals (all zero by default, flip the selected one)
        "Non_Travel": 1 if travel == "Non_Travel" else 0,
        "Travel_Rarely": 1 if travel == "Travel_Rarely" else 0,
        "Single": 1 if marital == "Single" else 0,
        "Laboratory_Technician": 1 if role == "Laboratory_Technician" else 0,
        "Research_Scientist": 1 if role == "Research_Scientist" else 0,
        "Sales_Representative": 1 if role == "Sales_Representative" else 0,
        "Human_Resources": 1 if role == "Human_Resources" else 0,
    }
    input_df = pd.DataFrame([raw])

    # Preprocess in the same order as training:
    # engineer -> reindex to full column set -> scale -> subset
    engineered = engineer_features(input_df)
    engineered_full = engineered.reindex(columns=all_feature_columns, fill_value=0)
    scaled_full = scaler.transform(engineered_full)

    scaled_full_df = pd.DataFrame(scaled_full, columns=all_feature_columns)
    scaled_selected = scaled_full_df[selected_features].values

    # Predict. Model's positive class (y=1) is "Retained", so leave
    # probability is 1 - P(retained).
    stay_prob = model.predict_proba(scaled_selected)[0][1]
    leave_prob = 1 - stay_prob
    flagged_high_risk = leave_prob >= threshold

    st.markdown(f"### ⚠️ Predicted chance of leaving: **{leave_prob * 100:.2f}%**")
    st.caption(
        f"Model threshold (tuned on validation set): {threshold:.2f} · "
        f"This employee is {'FLAGGED' if flagged_high_risk else 'not flagged'} as high risk."
    )

    # SHAP explanation using cached explainer
    shap_values = explainer(scaled_selected)

    st.subheader("Top Contributing Features")
    top_features = pd.DataFrame({
        "Feature": selected_features,
        "SHAP Value": shap_values.values[0],
    }).sort_values(by="SHAP Value", key=np.abs, ascending=False).head(5)
    st.table(top_features)

    st.subheader("Feature Impact Visual")
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.bar(shap_values[0], show=False)
    st.pyplot(fig)
