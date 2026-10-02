"""
Comprehensive Training, Evaluation, and Visualization Pipeline.
Executes end-to-end model development, hyperparameter tuning, metric calculation,
statistical significance extraction, SHAP analysis, and plot artifact generation.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Scikit-Learn & Models
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, median_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    ExtraTreesRegressor,
    ExtraTreesClassifier
)
import shap

# Local modules
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data_loader import get_cleaned_data
from src.preprocessing import engineer_features, prepare_modeling_data
from src.models import evaluate_regression_experiments, evaluate_classification_experiments
from src.explainability import compute_ols_statistical_justification, compute_shap_explanations

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "eda_plots"
EVAL_DIR = ARTIFACTS_DIR / "model_eval"

for d in [MODELS_DIR, ARTIFACTS_DIR, PLOTS_DIR, EVAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Styling for plots
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def generate_eda_visualizations(df_eng: pd.DataFrame):
    """
    Generates high-resolution EDA figures saved to artifacts/eda_plots/.
    """
    print("Generating EDA visualizations...")

    # 1. Target Distribution
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.histplot(df_eng['actual_productivity'], kde=True, color='#1f77b4', bins=30, label='Actual Productivity', ax=ax, alpha=0.6)
    sns.histplot(df_eng['targeted_productivity'], kde=True, color='#ff7f0e', bins=30, label='Targeted Productivity', ax=ax, alpha=0.4)
    ax.axvline(df_eng['actual_productivity'].mean(), color='#1f77b4', linestyle='--', linewidth=1.5, label=f"Actual Mean ({df_eng['actual_productivity'].mean():.2f})")
    ax.axvline(df_eng['targeted_productivity'].mean(), color='#ff7f0e', linestyle='--', linewidth=1.5, label=f"Target Mean ({df_eng['targeted_productivity'].mean():.2f})")
    ax.set_title("Distribution of Actual vs. Targeted Team Productivity", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Productivity Score (Output / Standard Target)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_productivity_distribution.png")
    plt.close()

    # 2. Correlation Matrix
    corr_cols = [
        'actual_productivity', 'targeted_productivity', 'overtime_per_worker',
        'incentive_per_worker', 'smv', 'no_of_workers', 'log_wip',
        'no_of_style_change', 'idle_time'
    ]
    corr = df_eng[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-0.4, vmax=0.6,
                linewidths=0.5, cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_title("Correlation Heatmap of Key Operational Team Factors", fontsize=14, fontweight='bold', pad=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_correlation_matrix.png")
    plt.close()

    # 3. Overtime Fatigue Curve
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.regplot(
        data=df_eng, x='overtime_per_worker', y='actual_productivity',
        scatter_kws={'alpha': 0.35, 'color': '#2ca02c', 's': 25},
        line_kws={'color': '#d62728', 'linewidth': 2.5},
        order=2, ax=ax
    )
    ax.axvline(120, color='black', linestyle=':', linewidth=1.5, label='Fatigue Threshold (120 min/worker)')
    ax.set_title("Team Overtime vs. Productivity: Evidence of Diminishing Returns", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Overtime per Worker (Minutes / Day)", fontsize=11)
    ax.set_ylabel("Actual Productivity Score", fontsize=11)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_overtime_fatigue.png")
    plt.close()

    # 4. Performance by Team
    team_perf = df_eng.groupby('team')[['actual_productivity', 'targeted_productivity']].mean().reset_index()
    team_perf = team_perf.sort_values(by='actual_productivity', ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    x = np.arange(len(team_perf))
    width = 0.35
    ax.bar(x - width/2, team_perf['actual_productivity'], width, label='Actual Productivity', color='#3b82f6')
    ax.bar(x + width/2, team_perf['targeted_productivity'], width, label='Targeted Productivity', color='#94a3b8')
    ax.set_title("Mean Team Performance vs Target across Factory Teams 1 - 12", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Team ID", fontsize=11)
    ax.set_ylabel("Mean Productivity", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Team {t}" for t in team_perf['team']])
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_team_performance_comparison.png")
    plt.close()

    # 5. Impact of Style Changes
    style_perf = df_eng.groupby('no_of_style_change')['actual_productivity'].agg(['mean', 'std', 'count']).reset_index()
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    bars = ax.bar(style_perf['no_of_style_change'].astype(str), style_perf['mean'],
           yerr=style_perf['std']/np.sqrt(style_perf['count']), capsize=5, color=['#10b981', '#f59e0b', '#ef4444'])
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),  # 4 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    ax.set_title("Impact of Style Changes on Team Output (Process Disruption Penalty)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Number of Garment Style Changes in Shift", fontsize=11)
    ax.set_ylabel("Mean Actual Productivity", fontsize=11)
    ax.set_ylim(0, 0.9)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_style_changes_impact.png")
    plt.close()

    print("EDA plots generated successfully.")

def train_and_evaluate_everything():
    """
    Main driver executing the end-to-end model development and evaluation.
    """
    print("--- 1. Loading and Cleaning Data ---")
    df_clean = get_cleaned_data()
    df_eng = engineer_features(df_clean)

    # Save processed dataset
    processed_csv = DATA_DIR / "processed_team_performance.csv"
    df_eng.to_csv(processed_csv, index=False)
    print(f"Processed dataset saved to {processed_csv}")

    # Generate EDA plots
    generate_eda_visualizations(df_eng)

    print("\n--- 2. Preparing Modeling Splits ---")
    bundle = prepare_modeling_data(df_eng, test_size=0.2, random_state=42)
    X_train_trans = bundle['X_train_trans']
    X_test_trans = bundle['X_test_trans']
    y_reg_train = bundle['y_reg_train']
    y_reg_test = bundle['y_reg_test']
    y_clf_train = bundle['y_clf_train']
    y_clf_test = bundle['y_clf_test']

    # Save fitted preprocessor
    joblib.dump(bundle['preprocessor'], MODELS_DIR / "preprocessor.pkl")
    print("Saved preprocessor to models/preprocessor.pkl")

    print("\n--- 3. Running Regression Experiments ---")
    reg_df, reg_models = evaluate_regression_experiments(X_train_trans, y_reg_train, X_test_trans, y_reg_test)
    print(reg_df.to_string())

    print("\n--- 4. Running Classification Experiments ---")
    clf_df, clf_models = evaluate_classification_experiments(X_train_trans, y_clf_train, X_test_trans, y_clf_test)
    print(clf_df.to_string())

    # Save benchmark tables
    reg_df.to_csv(EVAL_DIR / "regression_benchmark.csv", index=False)
    clf_df.to_csv(EVAL_DIR / "classification_benchmark.csv", index=False)

    print("\n--- 5. Hyperparameter Tuning on Best Models ---")
    # Fine-tuning Gradient Boosting Regressor
    print("Tuning Gradient Boosting Regressor...")
    gbr = GradientBoostingRegressor(random_state=42)
    param_grid_reg = {
        'n_estimators': [100, 160, 220],
        'learning_rate': [0.03, 0.06, 0.1],
        'max_depth': [3, 4, 5],
        'subsample': [0.8, 0.9, 1.0]
    }
    grid_reg = GridSearchCV(gbr, param_grid_reg, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid_reg.fit(X_train_trans, y_reg_train)
    best_regressor = grid_reg.best_estimator_
    print(f"Best Regressor Params: {grid_reg.best_params_}")

    # Evaluate tuned regressor
    y_reg_pred = best_regressor.predict(X_test_trans)
    tuned_reg_rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
    tuned_reg_mae = mean_absolute_error(y_reg_test, y_reg_pred)
    tuned_reg_r2 = r2_score(y_reg_test, y_reg_pred)
    tuned_reg_medae = median_absolute_error(y_reg_test, y_reg_pred)
    tuned_reg_mape = np.mean(np.abs((y_reg_test - y_reg_pred) / np.maximum(y_reg_test, 1e-5))) * 100
    print(f"Tuned Regressor Test RMSE: {tuned_reg_rmse:.4f}, R2: {tuned_reg_r2:.4f}, MAE: {tuned_reg_mae:.4f}")

    # Fine-tuning Gradient Boosting Classifier
    print("Tuning Gradient Boosting Classifier...")
    gb_clf = GradientBoostingClassifier(random_state=42)
    param_grid_clf = {
        'n_estimators': [100, 150, 200],
        'learning_rate': [0.03, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'subsample': [0.8, 0.9, 1.0]
    }
    grid_clf = GridSearchCV(gb_clf, param_grid_clf, cv=5, scoring='f1', n_jobs=-1)
    grid_clf.fit(X_train_trans, y_clf_train)
    best_classifier = grid_clf.best_estimator_
    print(f"Best Classifier Params: {grid_clf.best_params_}")

    # Evaluate tuned classifier
    y_clf_pred = best_classifier.predict(X_test_trans)
    y_clf_proba = best_classifier.predict_proba(X_test_trans)[:, 1]
    tuned_clf_acc = accuracy_score(y_clf_test, y_clf_pred)
    tuned_clf_prec = precision_score(y_clf_test, y_clf_pred)
    tuned_clf_rec = recall_score(y_clf_test, y_clf_pred)
    tuned_clf_f1 = f1_score(y_clf_test, y_clf_pred)
    tuned_clf_auc = roc_auc_score(y_clf_test, y_clf_proba)
    print(f"Tuned Classifier Test Acc: {tuned_clf_acc:.4f}, F1: {tuned_clf_f1:.4f}, ROC-AUC: {tuned_clf_auc:.4f}")

    # Save best models
    joblib.dump(best_regressor, MODELS_DIR / "best_productivity_regressor.pkl")
    joblib.dump(best_classifier, MODELS_DIR / "best_target_classifier.pkl")
    print("Saved best models to models/")

    print("\n--- 6. Computing Explainability & Statistical Justification ---")
    ols_df, ols_model = compute_ols_statistical_justification(X_train_trans, y_reg_train)
    ols_df.to_csv(ARTIFACTS_DIR / "statistical_justification.csv", index=False)
    print("Saved statistical justifications to artifacts/statistical_justification.csv")

    # SHAP Explanations
    shap_values, explainer, importance_df = compute_shap_explanations(best_regressor, X_train_trans)
    importance_df.to_csv(ARTIFACTS_DIR / "shap_feature_importance.csv", index=False)
    joblib.dump(explainer, MODELS_DIR / "shap_explainer.pkl")
    print("Saved SHAP explainer and feature importance table.")

    # Generate SHAP Beeswarm Summary Plot
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    shap.summary_plot(shap_values, X_train_trans, max_display=12, show=False)
    plt.title("SHAP Feature Attribution: Impact on Team Productivity", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(PLOTS_DIR / "eval_shap_summary.png")
    plt.close('all')

    print("\n--- 7. Generating Evaluation Diagnostic Visualizations ---")
    # Regression Benchmark Plot
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    sns.barplot(data=reg_df, x='Test R2', y='Model', palette='Blues_r', ax=ax)
    ax.set_title("Model Comparison: Out-of-Sample Regression R² Score", fontsize=13, fontweight='bold')
    ax.set_xlabel("R² Score (Higher is Better)", fontsize=11)
    ax.set_ylabel("")
    for i, p in enumerate(ax.patches):
        val = reg_df.loc[i, 'Test R2']
        ax.annotate(f"{val:.3f}", (p.get_width() + 0.01, p.get_y() + p.get_height() / 2),
                    va='center', fontsize=9, fontweight='bold')
    ax.set_xlim(-0.05, 0.55)
    plt.tight_layout()
    fig.savefig(EVAL_DIR / "eval_model_comparison_reg.png")
    plt.close()

    # Classification Benchmark Plot
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    sns.barplot(data=clf_df, x='Test F1', y='Model', palette='Greens_r', ax=ax)
    ax.set_title("Model Comparison: Classification F1 Score (Target Achievement)", fontsize=13, fontweight='bold')
    ax.set_xlabel("F1 Score (Higher is Better)", fontsize=11)
    ax.set_ylabel("")
    for i, p in enumerate(ax.patches):
        val = clf_df.loc[i, 'Test F1']
        ax.annotate(f"{val:.3f}", (p.get_width() + 0.01, p.get_y() + p.get_height() / 2),
                    va='center', fontsize=9, fontweight='bold')
    ax.set_xlim(0, 1.0)
    plt.tight_layout()
    fig.savefig(EVAL_DIR / "eval_model_comparison_clf.png")
    plt.close()

    # Residuals Diagnostic Plot for Regressor
    residuals = y_reg_test - y_reg_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    # Residual vs Fitted
    ax1.scatter(y_reg_pred, residuals, alpha=0.5, color='#3b82f6', edgecolors='none', s=35)
    ax1.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax1.set_title("Residuals vs. Fitted Values", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Fitted (Predicted) Productivity", fontsize=10)
    ax1.set_ylabel("Residuals (Actual - Predicted)", fontsize=10)

    # Residual Distribution
    sns.histplot(residuals, kde=True, color='#8b5cf6', bins=25, ax=ax2)
    ax2.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax2.set_title("Distribution of Residual Errors", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Residual Error", fontsize=10)
    ax2.set_ylabel("Density", fontsize=10)
    plt.tight_layout()
    fig.savefig(EVAL_DIR / "eval_residuals_diagnostic.png")
    plt.close()

    # ROC & Precision-Recall Curves
    fpr, tpr, _ = roc_curve(y_clf_test, y_clf_proba)
    prec, rec, _ = precision_recall_curve(y_clf_test, y_clf_proba)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    ax1.plot(fpr, tpr, color='#2563eb', lw=2, label=f"Tuned GB (AUC = {tuned_clf_auc:.3f})")
    ax1.plot([0, 1], [0, 1], color='#94a3b8', linestyle='--', lw=1.5, label='Random Baseline')
    ax1.set_title("ROC Curve (Target Met Classification)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("False Positive Rate", fontsize=10)
    ax1.set_ylabel("True Positive Rate", fontsize=10)
    ax1.legend(loc="lower right")

    ax2.plot(rec, prec, color='#059669', lw=2, label=f"PR Curve (AP = {np.trapezoid(prec, rec):.3f})")
    ax2.set_title("Precision-Recall Curve", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Recall", fontsize=10)
    ax2.set_ylabel("Precision", fontsize=10)
    ax2.legend(loc="lower left")
    plt.tight_layout()
    fig.savefig(EVAL_DIR / "eval_roc_pr_curves.png")
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_clf_test, y_clf_pred)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=['Target Missed (0)', 'Target Met (1)'],
                yticklabels=['Target Missed (0)', 'Target Met (1)'])
    ax.set_title("Test Confusion Matrix (Unseen 240 Shifts)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11)
    ax.set_ylabel("True Class", fontsize=11)
    plt.tight_layout()
    fig.savefig(EVAL_DIR / "eval_confusion_matrix.png")
    plt.close()

    # Baselines calculation
    baseline_reg_pred = np.full_like(y_reg_test, y_reg_train.mean())
    baseline_rmse = float(np.sqrt(mean_squared_error(y_reg_test, baseline_reg_pred)))
    baseline_r2 = float(r2_score(y_reg_test, baseline_reg_pred))
    rmse_improv_pct = round(float((1 - tuned_reg_rmse / baseline_rmse) * 100), 2)

    majority_class = y_clf_train.mode()[0]
    baseline_clf_acc = float(accuracy_score(y_clf_test, np.full_like(y_clf_test, majority_class)))

    # Save comprehensive metadata and metrics JSON
    meta = {
        'dataset': {
            'total_rows': len(df_eng),
            'train_rows': len(X_train_trans),
            'test_rows': len(X_test_trans),
            'total_features': len(bundle['feature_names']),
            'departments': df_clean['department'].value_counts().to_dict(),
            'teams_count': int(df_clean['team'].nunique()),
            'target_achievement_ratio': float(df_eng['target_met'].mean())
        },
        'regression_metrics': {
            'best_model': 'Tuned Gradient Boosting Regressor',
            'test_rmse': round(float(tuned_reg_rmse), 4),
            'test_mae': round(float(tuned_reg_mae), 4),
            'test_r2': round(float(tuned_reg_r2), 4),
            'test_medae': round(float(tuned_reg_medae), 4),
            'test_mape': round(float(tuned_reg_mape), 2),
            'baseline_rmse': round(baseline_rmse, 4),
            'baseline_r2': round(baseline_r2, 4),
            'rmse_improvement_pct': rmse_improv_pct
        },
        'classification_metrics': {
            'best_model': 'Tuned Gradient Boosting Classifier',
            'test_accuracy': round(float(tuned_clf_acc), 4),
            'test_precision': round(float(tuned_clf_prec), 4),
            'test_recall': round(float(tuned_clf_rec), 4),
            'test_f1': round(float(tuned_clf_f1), 4),
            'test_roc_auc': round(float(tuned_clf_auc), 4),
            'baseline_accuracy': round(baseline_clf_acc, 4),
            'confusion_matrix': cm.tolist()
        },
        'feature_names': bundle['feature_names'],
        'numeric_cols': bundle['numeric_cols'],
        'categorical_cols': bundle['categorical_cols'],
        'raw_feature_names': bundle['raw_feature_names']
    }

    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=4)

    with open(ARTIFACTS_DIR / "model_evaluation_metrics.json", "w") as f:
        json.dump(meta, f, indent=4)

    print("\n--- Pipeline Completed Successfully! All models, plots, and metrics are saved. ---")

if __name__ == "__main__":
    train_and_evaluate_everything()
