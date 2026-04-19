"""
Employee Retention Risk Prediction Pipeline
------------------------------------------------
Builds a calibrated XGBoost classifier for predicting employee attrition.

Key design choices
- SMOTE runs *inside* each CV fold via ``imblearn.pipeline.Pipeline``
  to prevent leakage into validation folds.
- A three-way split (train / validation / test) keeps threshold tuning
  independent of the final evaluation set.
- The tuned ``CalibratedClassifierCV`` wraps an imblearn pipeline, so
  ``app.py`` can drill down to the base XGB estimator for SHAP.
- All artifacts the Streamlit app needs are persisted to ``outputs/``.
"""

import os
import joblib
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import shap

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, f1_score,
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier

# numpy compat shim — must precede SHAP / XGBoost imports on older builds
if not hasattr(np, 'int'):
    np.int = int

# ------------------------------------------------------------------
# Data Loading & Feature Engineering
# ------------------------------------------------------------------

def load_data(file_path):
    """Load preprocessed HR data from an Excel file."""
    return pd.read_excel(file_path)


def engineer_features(df):
    """Add derived features with limited polynomial interactions.

    Duplicate column names produced by ``PolynomialFeatures`` (when
    input columns already exist in *df*) are dropped automatically.
    """
    df = df.copy()

    df['Engagement_Index'] = df.get('JobSatisfaction', 0) * df.get('WorkLifeBalance', 0)
    df['Promotion_Rate'] = df.get('YearsSinceLastPromotion', 0) / (df.get('YearsAtCompany', 0) + 1)

    if 'Doing_Overtime' in df.columns and 'Laboratory_Technician' in df.columns:
        df['Overtime_SensitiveRole'] = df['Doing_Overtime'] * df['Laboratory_Technician']

    key_cols = [c for c in ['Age', 'JobSatisfaction', 'MonthlyIncome', 'YearsAtCompany'] if c in df.columns]
    if len(key_cols) >= 2:
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        poly_features = poly.fit_transform(df[key_cols])
        poly_names = poly.get_feature_names_out(key_cols)
        poly_df = pd.DataFrame(poly_features, columns=poly_names, index=df.index)
        # Drop columns that already exist to avoid duplicates
        new_cols = [c for c in poly_df.columns if c not in df.columns]
        df = pd.concat([df, poly_df[new_cols]], axis=1)

    return df

# ------------------------------------------------------------------
# Feature Selection
# ------------------------------------------------------------------

def select_features(X, y, n_features=20):
    """Variance → correlation filter → RFE.  Returns a boolean mask
    and the list of selected column *names* (requires X to be a DataFrame
    or the caller to supply column names separately).
    """
    print(f"Starting feature selection with {X.shape[1]} features")

    # 1. Variance threshold
    var_sel = VarianceThreshold(threshold=0.05)
    X_var = var_sel.fit_transform(X)
    var_mask = var_sel.get_support()
    print(f"After variance filtering: {X_var.shape[1]} features")

    # 2. Correlation filter (drop one of each pair > 0.8)
    corr = pd.DataFrame(X_var).corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_idx = {int(c) for c in upper.columns if any(upper[c] > 0.8)}
    keep_idx = [i for i in range(X_var.shape[1]) if i not in drop_idx]
    X_corr = X_var[:, keep_idx]
    print(f"After correlation filtering: {X_corr.shape[1]} features")

    # Map surviving indices back to original column space
    var_indices = np.where(var_mask)[0]
    corr_indices = var_indices[keep_idx]

    # 3. RFE
    estimator = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
    rfe = RFE(estimator, n_features_to_select=min(n_features, X_corr.shape[1]), step=10)
    rfe.fit(X_corr, y)
    print("RFE completed")

    final_mask = np.zeros(X.shape[1], dtype=bool)
    final_mask[corr_indices[rfe.support_]] = True
    return final_mask

# ------------------------------------------------------------------
# Model Training & Evaluation
# ------------------------------------------------------------------

def train_model(X_train, y_train, scale_pos_weight):
    """Train XGBoost inside an imblearn Pipeline (SMOTE per fold),
    then wrap in CalibratedClassifierCV for reliable probabilities.

    Returns ``(calibrated_model,)`` — the calibrated model whose
    ``calibrated_classifiers_[i].estimator`` is the imblearn pipeline.
    """
    imb_pipe = ImbPipeline([
        ('smote', SMOTE(random_state=42, k_neighbors=3, sampling_strategy=0.5)),
        ('xgb', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')),
    ])

    param_grid = {
        'xgb__n_estimators': [100, 200],
        'xgb__max_depth': [3, 5],
        'xgb__learning_rate': [0.01, 0.1],
        'xgb__scale_pos_weight': [1, scale_pos_weight, scale_pos_weight * 1.5],
    }

    grid = GridSearchCV(imb_pipe, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Best parameters:", grid.best_params_)

    best_pipeline = grid.best_estimator_

    # Calibrate on the *original* (non-resampled) training distribution
    calibrated = CalibratedClassifierCV(best_pipeline, method='sigmoid', cv=5)
    calibrated.fit(X_train, y_train)
    return calibrated


def evaluate_model(model, X_test, y_test, output_dir):
    """Print classification metrics and save precision-recall curve."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("AUC-ROC:", roc_auc_score(y_test, y_prob))

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    prc_path = os.path.join(output_dir, 'precision_recall_curve.png')
    plt.savefig(prc_path, bbox_inches='tight')
    plt.close()
    print(f"Precision-recall curve saved to {prc_path}")

# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def find_best_threshold(y_true, y_probs):
    """Sweep thresholds and pick the one that maximises F1 for the
    *leaver* class (``pos_label=0``, i.e. 1 − P(retained)).
    """
    leave_probs = 1 - y_probs                       # leave probability
    thresholds = np.arange(0.05, 0.90, 0.01)
    f1s = [f1_score(y_true, (leave_probs >= t).astype(int), pos_label=0)
           for t in thresholds]
    best_idx = int(np.argmax(f1s))
    return thresholds[best_idx], f1s[best_idx]


def assign_risk(model, X, threshold=0.5):
    stay_prob = model.predict_proba(X)[:, 1]
    leave_prob = 1 - stay_prob
    high_boundary = np.percentile(leave_prob, 90)
    risk_tiers = pd.cut(
        leave_prob,
        bins=[0, threshold, max(high_boundary, threshold + 0.01), 1],
        labels=['Low', 'Moderate', 'High'],
        include_lowest=True,
    )
    return pd.DataFrame({'Leave_Probability': leave_prob, 'Risk_Tier': risk_tiers})


def explain_employee(explainer, X_row, feature_names, top_n=3):
    shap_vals = explainer.shap_values(X_row)
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'contribution': shap_vals[0],
    }).sort_values(by='contribution', key=np.abs, ascending=False)
    return shap_df.head(top_n)

# ------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # --- Load & engineer ---
    data = load_data(os.path.join(base_dir, 'data', 'IBM_Test_Project_Preprocessed_Data.xlsx'))
    X = data.drop(columns=['EmployeeNumber', 'Retained'])
    y = data['Retained']

    print("Engineering features...")
    X_engineered = engineer_features(X)
    all_feature_columns = list(X_engineered.columns)

    # --- Three-way split: train / validation / test ---
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_engineered, y, test_size=0.20, random_state=42,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.15, random_state=42,
    )

    # --- Scale (fit on training data only) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # --- Feature selection (on training data) ---
    feat_mask = select_features(X_train_scaled, y_train, n_features=20)
    selected_feature_names = [c for c, keep in zip(all_feature_columns, feat_mask) if keep]

    X_train_sel = X_train_scaled[:, feat_mask]
    X_val_sel = X_val_scaled[:, feat_mask]
    X_test_sel = X_test_scaled[:, feat_mask]

    # --- Train (SMOTE inside CV via imblearn pipeline) ---
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = train_model(X_train_sel, y_train, scale_pos_weight)

    # --- Threshold tuning on the **validation** set ---
    val_probs = model.predict_proba(X_val_sel)[:, 1]
    best_thresh, best_f1 = find_best_threshold(y_val, val_probs)
    print(f"Best threshold on validation set: {best_thresh:.2f} (F1: {best_f1:.3f})")

    # --- Evaluate on the held-out test set ---
    evaluate_model(model, X_test_sel, y_test, output_dir)

    # --- Risk tiering ---
    risk_df = assign_risk(model, X_test_sel, threshold=best_thresh)

    # --- SHAP global importance ---
    try:
        base_pipeline = model.calibrated_classifiers_[0].estimator
    except AttributeError:
        base_pipeline = model.calibrated_classifiers_[0].base_estimator
    base_xgb = base_pipeline.named_steps['xgb']

    explainer = shap.TreeExplainer(base_xgb)
    shap_vals = explainer.shap_values(X_test_sel)
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    shap_series = pd.Series(mean_abs_shap, index=selected_feature_names).sort_values()
    shap_series.plot(kind='barh', figsize=(10, 6))
    plt.title("Feature Importance (Average SHAP Impact)")
    plt.xlabel("Mean Absolute SHAP Value")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'SHAP_Global_Importance_HR_Friendly.png'))
    plt.close()

    # --- Per-employee explanations ---
    for i, row in enumerate(X_test_sel[:5]):
        explanation = explain_employee(explainer, row.reshape(1, -1), selected_feature_names)
        print(f"Employee {i + 1} top risk factors:")
        print(explanation.to_string(index=False))

    # --- Persist all artifacts the app needs ---
    joblib.dump(model, os.path.join(output_dir, 'final_calibrated_model.joblib'))
    joblib.dump(scaler, os.path.join(output_dir, 'final_scaler.joblib'))
    joblib.dump(all_feature_columns, os.path.join(output_dir, 'all_feature_columns.joblib'))
    joblib.dump(selected_feature_names, os.path.join(output_dir, 'selected_feature_names.joblib'))
    joblib.dump(best_thresh, os.path.join(output_dir, 'decision_threshold.joblib'))

    # --- CSV reports ---
    test_leave_prob = 1 - model.predict_proba(X_test_sel)[:, 1]
    preds_df = pd.DataFrame({
        'Predictions': (test_leave_prob >= best_thresh).astype(int),
    })
    preds_df.to_csv(os.path.join(output_dir, 'retention_risk_predictions.csv'), index=False)
    risk_df.to_csv(os.path.join(output_dir, 'retention_risk_tiers.csv'), index=False)
    pd.DataFrame({'Selected_Features': selected_feature_names}).to_csv(
        os.path.join(output_dir, 'selected_features.csv'), index=False,
    )

    print("\n✅ Pipeline complete — all artifacts saved to", output_dir)


if __name__ == '__main__':
    main()
