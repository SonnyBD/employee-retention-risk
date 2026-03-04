import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib

from employee_retention.retention_pipeline import engineer_features
from sklearn.preprocessing import StandardScaler

# Load trained model and scaler
model = joblib.load("outputs/final_calibrated_model.joblib")
scaler = joblib.load("outputs/final_scaler.joblib")
selected_features = pd.read_csv("outputs/selected_features.csv")['Selected_Features'].tolist()

st.set_page_config(page_title="Employee Retention Risk Dashboard", layout="centered")
st.title("\U0001F4CA Employee Attrition Risk Predictor")
st.markdown("Estimate the likelihood of an employee leaving and understand the key drivers behind the prediction.")

with st.form("input_form"):
    st.subheader("Input Employee Information")
    age = st.slider("Age", 18, 60, 35)
    job_sat = st.slider("Job Satisfaction (1–4)", 1, 4, 3)
    wlb = st.slider("Work-Life Balance (1–4)", 1, 4, 3)
    ot = st.selectbox("Doing Overtime?", [0, 1])
    yslp = st.slider("Years Since Last Promotion", 0, 10, 2)
    yatc = st.slider("Years at Company", 0, 40, 5)
    lt = st.selectbox("Laboratory Technician?", [0, 1])
    mi = st.slider("Monthly Income", 1000, 20000, 5000)

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    input_df = pd.DataFrame([{
        "Age": age,
        "JobSatisfaction": job_sat,
        "WorkLifeBalance": wlb,
        "YearsSinceLastPromotion": yslp,
        "YearsAtCompany": yatc,
        "Doing_Overtime": ot,
        "Laboratory_Technician": lt,
        "MonthlyIncome": mi
    }])

    # Feature engineering and selection
    engineered = engineer_features(input_df)
    engineered = engineered.reindex(columns=selected_features, fill_value=0)
    scaled_input = scaler.transform(engineered)

    # Prediction
    risk_prob = model.predict_proba(scaled_input)[0][1]
    risk_percent = round((1 - risk_prob) * 100, 2)

    st.markdown(f"### ⬆️ Predicted Retention Risk: **{risk_percent}%** chance of leaving")

    # SHAP Explanation
    base_model = getattr(model, 'estimator', getattr(model, 'base_estimator', model))
    explainer = shap.Explainer(base_model)
    shap_values = explainer(scaled_input)

    st.subheader("Top Contributing Features")
    top_features = pd.DataFrame({
        "Feature": selected_features,
        "SHAP Value": shap_values.values[0]
    }).sort_values(by="SHAP Value", key=np.abs, ascending=False).head(5)
    st.table(top_features)

    st.subheader("Feature Impact Visual")
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.bar(shap_values[0], show=False)
    st.pyplot(fig)
