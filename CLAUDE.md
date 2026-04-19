# CLAUDE.md — Employee Retention Risk Project

## Project Overview

A machine learning pipeline for predicting employee attrition risk using XGBoost, SHAP explainability, and probability calibration. The goal is to help HR teams identify at-risk employees and understand the key drivers behind potential departures.

## Repository Structure

```
employee-retention-risk/
├── data/                          # Input datasets (Excel + CSV)
├── employee_retention/
│   ├── __init__.py
│   └── retention_pipeline.py      # Core ML pipeline (load, engineer, train, evaluate)
├── notebooks/                     # Jupyter walkthrough notebook
├── outputs/                       # Generated charts, CSVs, and saved model artifacts
├── .github/workflows/ci.yml       # GitHub Actions CI
├── app.py                         # Streamlit dashboard
├── setup.py                       # Package config
├── requirements.txt               # Python dependencies
└── README.md
```

## Key Commands

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the ML pipeline (generates all artifacts the app needs)
```bash
python -m employee_retention.retention_pipeline
```

### Launch the Streamlit app
```bash
streamlit run app.py
```

### Run a Jupyter notebook
```bash
jupyter notebook notebooks/
```

## Tech Stack

- **Python 3.12**
- **scikit-learn** — preprocessing, RFE, calibration
- **XGBoost** — gradient boosted classifier
- **imbalanced-learn** — SMOTE (applied inside the CV loop via `imblearn.pipeline.Pipeline`)
- **SHAP** — model explainability
- **Streamlit** — interactive dashboard (`app.py`)
- **pandas / numpy / matplotlib / joblib**

## ML Pipeline Summary (`retention_pipeline.py`)

1. `load_data()` — reads preprocessed Excel data
2. `engineer_features()` — creates derived features (Engagement Index, Promotion Rate, Overtime × Role, polynomial interactions); drops duplicate column names
3. Three-way split: train / validation (for threshold tuning) / test
4. Scaler fit on training data only
5. `select_features()` — variance threshold → correlation filter → RFE (top 20); returns column *names*
6. `train_model()` — `GridSearchCV` over an `imblearn` pipeline `[SMOTE → XGBClassifier]`, so SMOTE runs only inside each training fold (no leakage). The tuned pipeline is then wrapped in `CalibratedClassifierCV(cv=5)` using the original (non-resampled) training distribution.
7. `find_best_threshold()` — tunes the F1-optimal decision threshold on the **validation set**, not the test set; optimises for the leaver class (`pos_label=0`)
8. `evaluate_model()` — classification report, confusion matrix, AUC-ROC, precision-recall curve on the test set
9. `assign_risk()` — buckets employees into Low / Moderate / High risk tiers
10. `explain_employee()` — per-employee SHAP explanation via the base XGB estimator

## Artifacts Persisted to `outputs/`

Running the pipeline writes every file the Streamlit app depends on:

- `final_calibrated_model.joblib` — the calibrated model
- `final_scaler.joblib` — `StandardScaler` fit on **all** engineered columns
- `all_feature_columns.joblib` — ordered list of engineered columns (for reindex at inference)
- `selected_feature_names.joblib` — 20 column names the model actually consumes
- `decision_threshold.joblib` — the threshold tuned on the validation set
- `selected_features.csv`, `retention_risk_predictions.csv`, `retention_risk_tiers.csv`
- `SHAP_Global_Importance_HR_Friendly.png`, `precision_recall_curve.png`

The app loads everything via `@st.cache_resource` and errors out cleanly if any artifact is missing.

## Known Limitations

- The input form in `app.py` captures the most important engineered features (satisfaction scores, overtime, tenure, travel, role, marital status); remaining one-hot columns default to 0 via reindex. Predictions are directionally useful but not identical to what full IBM HR records would produce.
- Feature selection is performed once on the training set rather than inside the CV loop. This is a known small source of optimism in reported CV scores, but keeps the pipeline simple. SMOTE leakage (the much larger concern) is fixed.

## Development Notes

- The `np.int = int` compatibility shim in `retention_pipeline.py` is placed **before** SHAP/XGBoost imports to handle a numpy deprecation in older versions of those libraries.
- Risk tiering uses the 90th percentile of predicted leave probability as the High/Moderate boundary, making it threshold-adaptive.
- Positive class (`y=1`) is **Retained**; leave probability is therefore `1 - model.predict_proba(X)[:, 1]`.

## Git Workflow

- Main development branch: `main`
- Push with: `git push origin main`
