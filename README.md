# Team Performance Analysis & Determinant Optimization

### Problem Statement:
> **A team wants to understand which measurable factors are associated with stronger team results. (With Proper Justification)**

### Project Title:
> **Empirical Modeling and Determinant Analysis of Team Productivity: A Multi-Task Machine Learning and Explainable AI Framework**

---

## Key Deliverables Overview

1. **Master Jupyter Notebook (Team_Performance_Analysis.ipynb):**
   - Self-contained, fully executed end-to-end notebook containing:
     - Mathematical problem formulation and operational objectives.
     - Dataset acquisition, quality audit, and domain anomaly resolutions.
     - Exploratory Data Analysis (EDA) with inline visualizations.
     - Domain feature engineering and preprocessing pipelines.
     - Multi-model development and 5-fold cross-validation benchmarking (Regression and Classification).
     - Hyperparameter tuning (GridSearchCV) for Gradient Boosting and Random Forest.
     - Model evaluation, residual analysis, homoskedasticity checks, ROC/PR curves, and confusion matrix.
     - Determinant Analysis and Statistical Justification: Econometric OLS hypothesis testing (p-values) and Game-Theoretic SHAP attributions.
     - Prescriptive managerial playbook and boundary conditions.

2. **Interactive Streamlit Web Dashboard (app.py):**
   - Real-time Shift Performance and What-If Simulator.
   - Interactive gauge of predicted productivity and target attainment probability.
   - Dynamic SHAP waterfall feature attribution for any simulated shift.
   - Exploratory data insights, statistical significance tables, and benchmark diagnostics.

3. **Academic and Technical Documentation (documentation/):**
   - PROJECT_REPORT.md: Comprehensive 8-section research report.
   - PRESENTATION_VIVA.md: 12-slide presentation deck breakdown and Viva Voce defense Q&A guide.
   - DATASET_DICTIONARY.md: Complete variable definitions and preprocessing decisions.

---

## Summary of Empirical Findings and Statistical Justifications

- **1. Financial Incentive Elasticity (p < 0.001, beta = +0.0895):**  
  Financial incentives per worker exhibit a strong, statistically significant positive association with productivity, validating targeted bonus structures.
- **2. The 120-Minute Overtime Threshold (p = 0.029 for fatigue penalty):**  
  Overtime delivers productivity gains only up to 120 minutes/worker/shift (approximately 2 hours). Beyond 120 minutes, physical exhaustion triggers diminishing and negative returns.
- **3. Style Change Retooling Penalty (p = 0.026, beta = -0.0158):**  
  Each garment design change reduces shift productivity by approximately 10% to 14%, demonstrating the critical need for batched order scheduling.
- **4. Target Anchoring Effect (p < 0.001):**  
  Targeted productivity accounts for approximately 19.7% of total feature importance (SHAP), empirically confirming goal-setting theory.

---

## Quickstart and Reproduction

### 1. Activate the Virtual Environment:
```bash
source .venv/bin/activate
```

### 2. Launch the Executable Jupyter Notebook:
```bash
jupyter notebook Team_Performance_Analysis.ipynb
# or select the kernel "Python (Team Performance ML)" in VS Code / Cursor
```

### 3. Launch the Interactive Streamlit Dashboard:
```bash
streamlit run app.py
```
Open browser at: `http://localhost:8501`

---

## Repository Directory Structure
```
Team Performance Analysis/
├── Team_Performance_Analysis.ipynb     # Master executable notebook (all ML, EDA, and XAI)
├── app.py                              # Interactive Streamlit application
├── data/
│   ├── garments_worker_productivity.csv # Raw UCI dataset
│   └── processed_team_performance.csv  # Cleaned and engineered dataset
├── models/
│   ├── best_productivity_regressor.pkl # Tuned Gradient Boosting Regressor
│   ├── best_target_classifier.pkl     # Tuned Gradient Boosting Classifier
│   ├── preprocessor.pkl                # Fitted StandardScaler transformer
│   ├── shap_explainer.pkl              # TreeSHAP explainer
│   └── model_metadata.json             # Serialized evaluation metrics and feature lists
├── artifacts/
│   ├── eda_plots/                      # High-res visualization figures
│   ├── model_eval/                     # Residuals, ROC curves, confusion matrix
│   ├── statistical_justification.csv   # OLS p-values, t-stats, confidence intervals
│   └── shap_feature_importance.csv     # SHAP importance rankings
├── documentation/
│   ├── PROJECT_REPORT.md               # Full academic and technical report
│   ├── PRESENTATION_VIVA.md            # Presentation deck and Viva defense Q&A
│   └── DATASET_DICTIONARY.md           # Dataset attributes and engineered features
├── src/                                # Modular source code
│   ├── data_loader.py                  # Ingestion and anomaly corrections
│   ├── preprocessing.py                # Feature engineering and transformers
│   ├── models.py                       # Benchmarking and training functions
│   └── explainability.py               # OLS significance and SHAP analysis
├── requirements.txt                    # Project dependencies
└── README.md                           # Project guide and reproduction instructions
```
