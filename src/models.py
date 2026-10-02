"""
Model Development, Cross-Validation, and Comparison Module.
Implements the exact model architectures and benchmarks taught in /Users/virshin/VScode/ML:
Linear Regression, Logistic Regression, KNN, Decision Trees, Bagging, AdaBoost, and Gradient Boosting.
Zero emojis.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    BaggingRegressor,
    AdaBoostRegressor,
    AdaBoostClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    median_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

def get_regression_models() -> Dict[str, Any]:
    """
    Returns candidate regression models from /Users/virshin/VScode/ML coursework.
    """
    return {
        "Multiple Linear Regression": LinearRegression(),
        "K-Nearest Neighbors Regressor": KNeighborsRegressor(n_neighbors=5),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=5, random_state=42),
        "AdaBoost Regressor": AdaBoostRegressor(n_estimators=100, learning_rate=0.05, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    }

def get_classification_models() -> Dict[str, Any]:
    """
    Returns candidate classification models from /Users/virshin/VScode/ML coursework.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors Classifier": KNeighborsClassifier(n_neighbors=5),
        "Base Decision Tree": DecisionTreeClassifier(max_depth=3, criterion="entropy", random_state=42),
        "Bagging Classifier": BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=3, criterion="entropy"),
            n_estimators=100,
            random_state=42
        ),
        "AdaBoost Classifier": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        ),
        "Gradient Boosting Classifier": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
    }

def evaluate_regression_experiments(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_folds: int = 10
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Performs 10-fold cross-validation and test set evaluation for regression models.
    """
    models = get_regression_models()
    records = []
    fitted_models = {}

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for name, model in models.items():
        cv_r2 = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
        cv_neg_rmse = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_mae = mean_absolute_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)
        test_medae = median_absolute_error(y_test, y_pred)
        safe_actual = np.where(y_test == 0, 1e-5, y_test)
        test_mape = np.mean(np.abs((y_test - y_pred) / safe_actual)) * 100

        records.append({
            'Model': name,
            '10-Fold CV R2': round(float(cv_r2.mean()), 4),
            'Test R2': round(float(test_r2), 4),
            'Test RMSE': round(float(test_rmse), 4),
            'Test MAE': round(float(test_mae), 4)
        })
        fitted_models[name] = model

    results_df = pd.DataFrame(records).sort_values(by='Test R2', ascending=False).reset_index(drop=True)
    return results_df, fitted_models

def evaluate_classification_experiments(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_folds: int = 10
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Performs 10-fold stratified cross-validation and test set evaluation matching Assignment_10.ipynb.
    """
    models = get_classification_models()
    records = []
    fitted_models = {}

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for name, model in models.items():
        cv_acc = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
        cv_f1 = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1')
        cv_auc = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc')

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            test_auc = roc_auc_score(y_test, y_proba)
        else:
            test_auc = 0.5

        test_acc = accuracy_score(y_test, y_pred)
        test_prec = precision_score(y_test, y_pred, zero_division=0)
        test_rec = recall_score(y_test, y_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)

        records.append({
            'Model': name,
            '10-Fold CV Accuracy (%)': round(float(cv_acc.mean() * 100), 2),
            'Test Accuracy (%)': round(float(test_acc * 100), 2),
            'Test ROC-AUC': round(float(test_auc), 4),
            'Test F1': round(float(test_f1), 4),
            'Test Precision': round(float(test_prec), 4),
            'Test Recall': round(float(test_rec), 4)
        })
        fitted_models[name] = model

    results_df = pd.DataFrame(records).sort_values(by='Test F1', ascending=False).reset_index(drop=True)
    return results_df, fitted_models
