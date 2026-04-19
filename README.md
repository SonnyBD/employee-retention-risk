# 📊 Employee Retention Risk Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green?logo=opensourceinitiative&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Made with](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=jupyter)
![CI](https://github.com/SonnyBD/employee-retention-risk/actions/workflows/ci.yml/badge.svg)

[![View in nbviewer](https://img.shields.io/badge/View%20Notebook-nbviewer-orange?logo=jupyter)](https://nbviewer.org/github/SonnyBD/employee-retention-risk/blob/main/notebooks/Full_Retention_Model_Walkthrough.ipynb)

A machine learning–driven People Analytics project that identifies employees at risk of leaving and explains the drivers behind attrition. Built to support HR teams in making proactive, data-informed retention decisions.

![SHAP Feature Impact](outputs/SHAP_Global_Importance_HR_Friendly.png)

---

## 🧠 Objective

Build a calibrated, interpretable predictive model on HR data that estimates employee attrition risk, uncovers key retention factors, and segments employees into actionable risk tiers for HR intervention.

---

## 🛠️ Tools & Techniques

- Python (pandas, scikit-learn, imbalanced-learn, SHAP)
- XGBoost Classifier + Recursive Feature Elimination (RFE)
- SMOTE for class balancing (applied inside CV folds via `imblearn.pipeline.Pipeline` to prevent leakage)
- Probability calibration (Platt scaling via `CalibratedClassifierCV`)
- SHAP for model explainability
- Matplotlib for visualizations

---

## 🔄 Workflow Summary

1. Data cleaning and feature engineering
2. Three-way split: train / validation / test
3. Feature selection with RFE on training data
4. XGBoost tuning with SMOTE applied per fold, then probability calibration
5. Threshold tuning on the **validation set** (not the test set)
6. Risk scoring and percentile-based tiering (Low, Moderate, High)
7. SHAP-based interpretation of feature importance
8. Final outputs: CSV reports, risk segmentation, SHAP visualizations, and serialized model artifacts

---

## 📈 Example Pipeline Output

```
Output directory: ./outputs
Engineering features...
After correlation filtering: 44 features
Best parameters: {'xgb__learning_rate': 0.1, ...}
Best threshold on validation set: 0.20 (F1: 0.45)
Employee 1 top risk factors:
  JobSatisfaction     -1.24
  Doing_Overtime      -0.89
  Single               1.03
```

Actual metrics vary slightly between runs depending on the random split, but F1 for the minority (leaver) class typically lands in the 0.40–0.50 range with AUC-ROC around 0.80. Accuracy alone is misleading on this imbalanced dataset — focus on recall and F1 for the leaver class.

---

## 📆 Why This Matters

- 📉 Replacing a lost employee can cost ~33% of their annual salary
- ⏱️ Manual retention tracking is inefficient and reactive
- 🧠 This model helps HR prioritize who to retain and why — with explainability

> Empower HR teams with data-backed decisions instead of gut-feeling.

---

## 📂 Repository Structure

```
employee-retention-risk/
│
├── data/                           # Raw and preprocessed input data
│   ├── IBM_Test_Project_Preprocessed_Data.xlsx
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── notebooks/                      # Jupyter notebook walkthrough
│   └── Full_Retention_Model_Walkthrough.ipynb
│
├── outputs/                        # Generated artifacts (created by the pipeline)
│   ├── final_calibrated_model.joblib
│   ├── final_scaler.joblib
│   ├── all_feature_columns.joblib
│   ├── selected_feature_names.joblib
│   ├── decision_threshold.joblib
│   ├── selected_features.csv
│   ├── SHAP_Global_Importance_HR_Friendly.png
│   └── precision_recall_curve.png
│
├── employee_retention/             # Modular ML pipeline
│   ├── __init__.py
│   └── retention_pipeline.py
│
├── app.py                          # Streamlit dashboard
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

---

## 🚀 How to Run This Project

```bash
# 1. Clone the repository
git clone https://github.com/SonnyBD/employee-retention-risk.git
cd employee-retention-risk

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline script (generates model + all artifacts the app needs)
python -m employee_retention.retention_pipeline

# 4. Launch the Streamlit dashboard
streamlit run app.py

# 5. (Optional) Open the notebook walkthrough
jupyter notebook notebooks/Full_Retention_Model_Walkthrough.ipynb
```

---

## 📁 Dataset

This project uses the [IBM HR Analytics Employee Attrition dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) from Kaggle, with additional preprocessing and feature engineering.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📜 License

This project is licensed under the MIT License — open for use with attribution.

---

## 👤 Author

Built by [Sonny Bigras-Dewan](https://www.linkedin.com/in/sonny-bigras-dewan/) — let's connect!
