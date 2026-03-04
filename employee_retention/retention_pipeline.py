"""
Employee Retention Risk Prediction Pipeline
------------------------------------------------
This script builds a machine learning pipeline for predicting employee attrition using XGBoost, SHAP interpretability, and threshold optimization for better recall on minority classes.

Structure follows GitHub script conventions for clarity, reproducibility, and modularity.
"""

import os
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
    precision_recall_curve, f1_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

if not hasattr(np, 'int'):
    np.int = int

# ---------------------------
# Data Loading and Features
# ---------------------------

def load_data(file_path):
    """Load preprocessed HR data from an Excel file."""
    return pd.read_excel(file_path)

def engineer_features(df):
    """Add derived features with limited polynomial features."""
    df = df.copy()
    df['Engagement_Index'] = df['JobSatisfaction'] * df['WorkLifeBalance']
    df['Promotion_Rate'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
    df['Overtime_SensitiveRole'] = df['Doing_Overtime'] * df['Laboratory_Technician']
    key_cols = ['Age', 'JobSatisfaction', 'MonthlyIncome', 'YearsAtCompany']
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_features = poly.fit_transform(df[key_cols])
    poly_feature_names = poly.get_feature_names_out(key_cols)
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
    return pd.concat([df, poly_df], axis=1)

# ---------------------------
# Feature Selection
# ---------------------------

def select_features(X, y, n_features=20):
    """Select top features using RFE with aggressive pre-filtering."""
    print(f"Starting feature selection with {X.shape[1]} features")
    selector = VarianceThreshold(threshold=0.05)
    X_filtered = selector.fit_transform(X)
    variance_mask = selector.get_support()
    print(f"After variance filtering: {X_filtered.shape[1]} features")

    X_filtered_df = pd.DataFrame(X_filtered)
    corr_matrix = X_filtered_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
    keep_cols = [i for i in range(X_filtered_df.shape[1]) if i not in to_drop]
    X_filtered = X_filtered_df.iloc[:, keep_cols].values
    print(f"After correlation filtering: {X_filtered.shape[1]} features")

    corr_mask = np.zeros(X_filtered_df.shape[1], dtype=bool)
    corr_mask[keep_cols] = True
    combined_mask = np.zeros(X.shape[1], dtype=bool)
    variance_indices = np.where(variance_mask)[0]
    corr_indices = variance_indices[keep_cols]
    combined_mask[corr_indices] = True

    estimator = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
    selector = RFE(estimator, n_features_to_select=n_features, step=10)
    selector = selector.fit(X_filtered, y)
    print("RFE completed")

    final_support = np.zeros(X.shape[1], dtype=bool)
    rfe_support = selector.support_
    final_support[corr_indices[rfe_support]] = True
    return final_support

# ---------------------------
# Model Training & Evaluation
# ---------------------------

def train_model(X, y, scale_pos_weight):
    """Train an XGBoost model with GridSearchCV and calibrate probabilities."""
    xgb = XGBClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'scale_pos_weight': [1, scale_pos_weight, scale_pos_weight * 1.5]
    }
    grid = GridSearchCV(xgb, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X, y)
    print("Best parameters:", grid.best_params_)

    base_model = grid.best_estimator_
    calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
    calibrated_model.fit(X, y)
    return base_model, calibrated_model


def evaluate_model(model, X_test, y_test, output_dir):
    """Evaluate model performance and plot precision-recall curve."""
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("AUC-ROC:", roc_auc_score(y_test, y_pred_prob))

    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    prc_path = os.path.join(output_dir, 'precision_recall_curve.png')
    plt.savefig(prc_path, bbox_inches='tight')
    plt.close()
    print(f"Precision-recall curve saved to {prc_path}")

# ---------------------------
# Utility Functions
# ---------------------------

def find_best_threshold(y_true, y_probs):
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1s = [f1_score(y_true, y_probs > t) for t in thresholds]
    best_thresh = thresholds[np.argmax(f1s)]
    return best_thresh, max(f1s)

def assign_risk(model, X, threshold=0.5):
    y_pred_prob = model.predict_proba(X)[:, 1]
    retention_risk = 1 - y_pred_prob
    low_threshold = threshold
    high_threshold = np.percentile(retention_risk, 90)
    risk_tiers = pd.cut(retention_risk, bins=[0, low_threshold, high_threshold, 1],
                        labels=['Low', 'Moderate', 'High'], include_lowest=True)
    return pd.DataFrame({'Retention_Risk': retention_risk, 'Risk_Tier': risk_tiers})

def explain_employee(explainer, X_test_row, feature_names, top_n=3):
    shap_vals = explainer.shap_values(X_test_row)
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'contribution': shap_vals[0]
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

    data = load_data(os.path.join(base_dir, 'data', 'IBM_Test_Project_Preprocessed_Data.xlsx'))
    X = data.drop(columns=['EmployeeNumber', 'Retained'])
    y = data['Retained']

    print("Engineering features...")
    X_engineered = engineer_features(X)
    X_train, X_test, y_train, y_test = train_test_split(X_engineered, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    selected_features = select_features(X_train_scaled, y_train, n_features=20)
    X_train_sel = X_train_scaled[:, selected_features]
    X_test_sel = X_test_scaled[:, selected_features]
    feature_names = X_train.columns[selected_features].tolist()

    smote = SMOTE(random_state=42, k_neighbors=3, sampling_strategy=0.5)
    X_resampled, y_resampled = smote.fit_resample(X_train_sel, y_train)
    scale_pos_weight = (y_resampled == 0).sum() / (y_resampled == 1).sum()

    base_model, model = train_model(X_resampled, y_resampled, scale_pos_weight)
    evaluate_model(model, X_test_sel, y_test, output_dir)

    y_pred_prob = model.predict_proba(X_test_sel)[:, 1]
    best_thresh, best_f1 = find_best_threshold(y_test, y_pred_prob)
    final_preds = (y_pred_prob > best_thresh).astype(int)
    print(f"Best threshold: {best_thresh:.2f}, F1: {best_f1:.3f}")

    risk_df = assign_risk(model, X_test_sel, threshold=best_thresh)

    explainer = shap.TreeExplainer(base_model)
    shap_vals = explainer.shap_values(X_test_sel)
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    shap_df = pd.Series(mean_abs_shap, index=feature_names).sort_values()
    shap_df.plot(kind='barh', figsize=(10, 6))
    plt.title("Feature Importance (Average SHAP Impact)")
    plt.xlabel("Mean Absolute SHAP Value")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'SHAP_Global_Importance_HR_Friendly.png'))
    plt.close()

    for i, row in enumerate(X_test_sel[:5]):
        explanation = explain_employee(explainer, row.reshape(1, -1), feature_names)
        print(f"Employee {i + 1} top risk factors:")
        print(explanation.to_string(index=False))

    pd.DataFrame({'Predictions': final_preds}).to_csv(os.path.join(output_dir, 'retention_risk_predictions.csv'), index=False)
    risk_df.to_csv(os.path.join(output_dir, 'retention_risk_tiers.csv'), index=False)
    pd.DataFrame({'Selected_Features': feature_names}).to_csv(os.path.join(output_dir, 'selected_features.csv'), index=False)


if __name__ == '__main__':
    main()
