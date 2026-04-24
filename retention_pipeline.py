"""
Employee Retention Risk Prediction Pipeline
------------------------------------------------
Builds a calibrated XGBoost model for predicting employee attrition, with
SHAP interpretability and threshold tuning for improved recall on leavers.

Pipeline design notes:
- SMOTE is applied INSIDE the CV loop via an imblearn Pipeline, so
  synthetic samples never leak into validation folds.
- Probability calibration uses cross-validation over the original
  (non-resampled) training distribution.
- The decision threshold is tuned on a held-out validation split, not
  on the test set.
- Feature selection is performed on training data only; the resulting
  column list and the fitted scaler are persisted so the Streamlit app
  can reproduce identical preprocessing at inference time.
"""

import numpy as np
import os
import joblib
import pandas as pd
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
    precision_recall_curve, f1_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier


# ---------------------------
# Data Loading and Features
# ---------------------------

def load_data(file_path):
    """Load preprocessed HR data from an Excel file."""
    return pd.read_excel(file_path)


def engineer_features(df):
    """Add derived features and limited polynomial interactions.

    Safe to call on a single-row dataframe at inference time as long as
    the input contains the columns referenced below.
    """
    df = df.copy()
    df['Engagement_Index'] = df['JobSatisfaction'] * df['WorkLifeBalance']
    df['Promotion_Rate'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
    df['Overtime_SensitiveRole'] = df['Doing_Overtime'] * df['Laboratory_Technician']

    key_cols = ['Age', 'JobSatisfaction', 'MonthlyIncome', 'YearsAtCompany']
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_features = poly.fit_transform(df[key_cols])
    poly_feature_names = poly.get_feature_names_out(key_cols)
    # Keep only the interaction terms; PolynomialFeatures also re-emits the
    # original features, which would duplicate column names in the output.
    interaction_mask = [' ' in n for n in poly_feature_names]
    interaction_names = [n for n, m in zip(poly_feature_names, interaction_mask) if m]
    interaction_values = poly_features[:, interaction_mask]
    poly_df = pd.DataFrame(interaction_values, columns=interaction_names, index=df.index)
    return pd.concat([df, poly_df], axis=1)


# ---------------------------
# Feature Selection
# ---------------------------

def select_features(X_scaled, y, feature_names, n_features=20):
    """Variance threshold -> correlation filter -> RFE.

    Returns the list of selected feature *names* (not indices), which
    is what the Streamlit app needs to reindex inference inputs.
    """
    print(f"Starting feature selection with {X_scaled.shape[1]} features")

    # Variance filter
    vt = VarianceThreshold(threshold=0.05)
    X_vt = vt.fit_transform(X_scaled)
    vt_mask = vt.get_support()
    vt_names = [n for n, keep in zip(feature_names, vt_mask) if keep]
    print(f"After variance filtering: {X_vt.shape[1]} features")

    # Correlation filter
    X_vt_df = pd.DataFrame(X_vt, columns=vt_names)
    corr_matrix = X_vt_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > 0.8)]
    kept_names = [c for c in vt_names if c not in to_drop]
    X_corr = X_vt_df[kept_names].values
    print(f"After correlation filtering: {X_corr.shape[1]} features")

    # RFE
    if X_corr.shape[1] <= n_features:
        print(f"Only {X_corr.shape[1]} features remain after filtering; skipping RFE")
        return kept_names

    estimator = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
    rfe = RFE(estimator, n_features_to_select=n_features, step=10)
    rfe.fit(X_corr, y)
    selected_names = [n for n, keep in zip(kept_names, rfe.support_) if keep]
    print(f"RFE selected {len(selected_names)} features")
    return selected_names


# ---------------------------
# Model Training & Evaluation
# ---------------------------

def train_model(X_train, y_train):
    """Tune XGBoost with SMOTE inside each CV fold, then calibrate.

    Optimization target is F1 for the LEAVER class (label=0), because
    that is the class HR actually cares about detecting. Returns
    (base_xgb, calibrated_model).
    """
    from sklearn.metrics import make_scorer
    leaver_f1 = make_scorer(f1_score, pos_label=0)

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    pipe = ImbPipeline([
        ('smote', SMOTE(random_state=42, k_neighbors=3, sampling_strategy=0.5)),
        ('xgb', XGBClassifier(random_state=42, eval_metric='logloss')),
    ])

    param_grid = {
        'xgb__n_estimators': [100, 200],
        'xgb__max_depth': [3, 5],
        'xgb__learning_rate': [0.01, 0.1],
        'xgb__scale_pos_weight': [1, scale_pos_weight, scale_pos_weight * 1.5],
    }

    grid = GridSearchCV(pipe, param_grid, cv=5, scoring=leaver_f1, n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Best parameters:", grid.best_params_)
    print(f"Best CV leaver-F1: {grid.best_score_:.3f}")

    best_pipeline = grid.best_estimator_
    base_xgb = best_pipeline.named_steps['xgb']

    # Calibrate the full pipeline (SMOTE + XGB) on the ORIGINAL distribution.
    # CalibratedClassifierCV with cv=5 ensures calibration uses held-out folds.
    calibrated_model = CalibratedClassifierCV(best_pipeline, method='sigmoid', cv=5)
    calibrated_model.fit(X_train, y_train)
    return base_xgb, calibrated_model


def evaluate_model(model, X_test, y_test, output_dir, leave_threshold=0.5):
    """Report metrics and save a PR curve.

    leave_threshold is applied to P(leave): if leave_prob >= threshold,
    the employee is flagged as a leaver (predicted class 0).
    """
    stay_prob = model.predict_proba(X_test)[:, 1]
    leave_prob = 1 - stay_prob
    y_pred = np.where(leave_prob >= leave_threshold, 0, 1)  # 0 = leaver, 1 = stayer

    print(f"\n--- Evaluation at leave_threshold={leave_threshold:.2f} ---")
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix (rows=true, cols=pred; 0=leaver, 1=stayer):\n",
          confusion_matrix(y_test, y_pred))
    print("AUC-ROC:", roc_auc_score(y_test, stay_prob))

    precision, recall, _ = precision_recall_curve(y_test, leave_prob, pos_label=0)
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall (leavers)')
    plt.ylabel('Precision (leavers)')
    plt.title('Precision-Recall Curve for Leaver Class')
    prc_path = os.path.join(output_dir, 'precision_recall_curve.png')
    plt.savefig(prc_path, bbox_inches='tight')
    plt.close()
    print(f"Precision-recall curve saved to {prc_path}")


# ---------------------------
# Utility Functions
# ---------------------------

def find_best_threshold(y_true, leave_probs):
    """Grid search leave-probability threshold to maximize F1 on leavers.

    y_true uses the pipeline's convention: 1 = retained, 0 = leaver.
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1s = []
    for t in thresholds:
        y_pred = np.where(leave_probs >= t, 0, 1)
        f1s.append(f1_score(y_true, y_pred, pos_label=0, zero_division=0))
    best_idx = int(np.argmax(f1s))
    return thresholds[best_idx], f1s[best_idx]


def assign_risk(model, X, leave_threshold=0.5):
    """Bucket employees into Low / Moderate / High risk based on leave_prob.

    The leave_threshold (tuned to maximize leaver-F1) separates Low from
    Moderate. The 90th percentile of leave_prob separates Moderate from High.
    """
    leave_prob = 1 - model.predict_proba(X)[:, 1]
    high_cut = np.percentile(leave_prob, 90)
    low_cut = min(leave_threshold, high_cut - 1e-6)
    risk_tiers = pd.cut(
        leave_prob,
        bins=[-0.001, low_cut, high_cut, 1.001],
        labels=['Low', 'Moderate', 'High'],
    )
    return pd.DataFrame({'Leave_Probability': leave_prob, 'Risk_Tier': risk_tiers})


def explain_employee(explainer, X_row, feature_names, top_n=3):
    shap_vals = explainer.shap_values(X_row)
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'contribution': shap_vals[0],
    }).sort_values(by='contribution', key=np.abs, ascending=False)
    return shap_df.head(top_n)


# ---------------------------
# Entry Point
# ---------------------------

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # 1. Load and engineer
    data = load_data(os.path.join(base_dir, 'data', 'IBM_Test_Project_Preprocessed_Data.xlsx'))
    X = data.drop(columns=['EmployeeNumber', 'Retained'])
    y = data['Retained']

    print("Engineering features...")
    X_engineered = engineer_features(X)
    all_feature_columns = X_engineered.columns.tolist()

    # 2. Three-way split: train / validation (for threshold) / test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_engineered, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
    )

    # 3. Fit scaler on training data only (all engineered columns)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 4. Feature selection (training data only)
    selected_names = select_features(
        X_train_scaled, y_train.values, all_feature_columns, n_features=20
    )
    selected_idx = [all_feature_columns.index(n) for n in selected_names]
    X_train_sel = X_train_scaled[:, selected_idx]
    X_val_sel = X_val_scaled[:, selected_idx]
    X_test_sel = X_test_scaled[:, selected_idx]

    # 5. Train (SMOTE inside CV) and calibrate
    base_xgb, model = train_model(X_train_sel, y_train.values)

    # 6. Threshold tuning on validation set (NOT test)
    y_val_stay = model.predict_proba(X_val_sel)[:, 1]
    y_val_leave = 1 - y_val_stay
    best_thresh, best_f1 = find_best_threshold(y_val.values, y_val_leave)
    print(f"\nBest leave-probability threshold on validation set: "
          f"{best_thresh:.2f} (leaver F1: {best_f1:.3f})")

    # 7. Final evaluation on test set using the tuned threshold
    evaluate_model(model, X_test_sel, y_test.values, output_dir, leave_threshold=best_thresh)

    # 8. Risk tiers on test set
    risk_df = assign_risk(model, X_test_sel, leave_threshold=best_thresh)

    # 9. SHAP global importance (on base XGB, selected feature space)
    explainer = shap.TreeExplainer(base_xgb)
    shap_vals = explainer.shap_values(X_test_sel)
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    shap_imp = pd.Series(mean_abs_shap, index=selected_names).sort_values()
    plt.figure(figsize=(10, 6))
    shap_imp.plot(kind='barh')
    plt.title("Feature Importance (Mean |SHAP|)")
    plt.xlabel("Mean Absolute SHAP Value")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'SHAP_Global_Importance_HR_Friendly.png'))
    plt.close()

    for i, row in enumerate(X_test_sel[:5]):
        explanation = explain_employee(explainer, row.reshape(1, -1), selected_names)
        print(f"\nEmployee {i + 1} top risk factors:")
        print(explanation.to_string(index=False))

    # 10. Persist artifacts required by app.py
    leave_prob_test = 1 - model.predict_proba(X_test_sel)[:, 1]
    final_preds = (leave_prob_test >= best_thresh).astype(int)  # 1 = flagged leaver
    pd.DataFrame({'Flagged_Leaver': final_preds}).to_csv(
        os.path.join(output_dir, 'retention_risk_predictions.csv'), index=False
    )
    risk_df.to_csv(os.path.join(output_dir, 'retention_risk_tiers.csv'), index=False)
    pd.DataFrame({'Selected_Features': selected_names}).to_csv(
        os.path.join(output_dir, 'selected_features.csv'), index=False
    )

    joblib.dump(model, os.path.join(output_dir, 'final_calibrated_model.joblib'))
    joblib.dump(scaler, os.path.join(output_dir, 'final_scaler.joblib'))
    joblib.dump(all_feature_columns, os.path.join(output_dir, 'all_feature_columns.joblib'))
    joblib.dump(selected_names, os.path.join(output_dir, 'selected_feature_names.joblib'))
    joblib.dump(float(best_thresh), os.path.join(output_dir, 'decision_threshold.joblib'))
    print("\nSaved model, scaler, feature lists, and threshold to outputs/")


if __name__ == '__main__':
    main()
