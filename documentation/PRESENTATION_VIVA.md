# Presentation Deck & Viva Voce Defense Guide
## Project: Empirical Modeling and Determinant Analysis of Team Productivity

---

## Part 1: Presentation Slide Deck Outline (10-12 Slides)

### Slide 1: Title & Overview
- **Project Title:** Empirical Modeling and Determinant Analysis of Team Productivity: A Multi-Task Machine Learning and Explainable AI Framework
- **Problem Statement:** A team wants to understand which measurable factors are associated with stronger team results. *(With Proper Justification)*
- **Team / Lead:** Data Science & Machine Learning Research Team
- **Key Deliverables:** Executable Jupyter Notebook (`Team_Performance_Analysis.ipynb`) & Interactive Streamlit Dashboard (`app.py`).

### Slide 2: Real-World Motivation & Problem Formulation
- **The Challenge:** Teams allocate substantial resources (overtime, bonuses, line restructuring) without quantitative justification of what drives output or causes burnout.
- **Formulation:**
  - **Regression:** Continuous productivity estimation ($y \in [0.2, 1.2]$).
  - **Classification:** Quota attainment prediction ($y \ge \text{Target}$).
  - **Explainability:** Econometric OLS ($p$-values) and Game-Theoretic SHAP attribution.

### Slide 3: Dataset Overview & Quality Findings
- **Source:** UCI Machine Learning Repository (Garment Manufacturing in Dhaka, Bangladesh).
- **Scope:** 1,197 shift records, 12 production teams, 59 working days.
- **Critical Data Discoveries:**
  - Typo corrections (`sweing` $\to$ `sewing`).
  - WIP missingness: 100% of missing values (506 rows) belong to Finishing due to downstream batch processing.
  - Anomaly fix: On 2015-03-09, finishing overtime was erroneously entered into the incentive column.

### Slide 4: Key Exploratory Insights (EDA)
- **Mean Productivity:** 0.735 vs. Target 0.730 (73.1% shifts achieved quota).
- **The 120-Minute Overtime Threshold:** Overtime past 120 minutes/worker exhibits diminishing returns and fatigue drag.
- **Style Change Retooling Penalty:** Introducing 1 style change drops productivity by ~13.7%; 2 changes drop productivity by ~22.2%.

### Slide 5: Feature Engineering & Preprocessing Pipeline
- **Domain Normalization:** `overtime_per_worker`, `incentive_per_worker`, `smv_per_worker`.
- **Non-Linear Dynamics:** `overtime_fatigue_penalty = max(0, overtime - 120)`.
- **Workload Intensity:** $\text{SMV} \times \text{Targeted Productivity}$.
- **Preprocessing:** `ColumnTransformer` (StandardScaler + OneHotEncoder), 80/20 stratified split.

### Slide 6: Model Development & Benchmarking
- **Evaluated Architectures:** Multiple Linear Regression, Logistic Regression, K-Nearest Neighbors, Decision Trees, Bagging, AdaBoost, and Gradient Boosting (exact models taught in coursework).
- **Validation Strategy:** 10-Fold Cross-Validation on training data + 240 unseen holdout shifts.
- **Highlights:**
  - Gradient Boosting Regressor: Test RMSE = 0.1290, R² = 0.4645, MedAE = 0.0435.
  - Gradient Boosting Classifier: 10-Fold CV ROC-AUC = 0.8743, Test Accuracy = 79.17%, Recall = 93.14%.
  - Bagging Ensemble: Test Recall = 97.71%, 10-Fold CV ROC-AUC = 0.8564.

### Slide 7: Model Evaluation & Error Diagnostics
- **Homoskedasticity & Normality:** Residuals are centered symmetrically around 0 with no major funneling.
- **Classification Performance:** Recall of 93.1% on target attainment with strong ROC curve separation (AUC = 0.857).

### Slide 8: Determinant Analysis: Econometric Justification
- **Formal Hypothesis Testing ($p < 0.05$):**
  - Financial incentives per worker ($\beta = +0.0895, p < 0.001$): Statistically verified positive driver.
  - Targeted productivity ($\beta = +0.0535, p < 0.001$): Validates goal-setting theory.
  - Style change count ($\beta = -0.0158, p = 0.026$): Statistically verified operational disruption drag.
  - Overtime fatigue penalty ($\beta = -0.0214, p = 0.029$): Confirms negative impact of excessive overtime.

### Slide 9: Determinant Analysis: SHAP Explainability
- **Global Importance (TreeSHAP):** Target pace (19.7%), Incentive per worker (11.6%), Task complexity SMV (11.4%), Team size (9.6%).
- **Local Waterfall Analysis:** Live decomposition of any individual shift into exact feature contributions.

### Slide 10: Prescriptive Managerial Playbook
1. **Cap Daily Overtime at 120 Minutes/Worker.**
2. **Deploy Tiered Quality-Linked Bonuses.**
3. **Batch Orders to Mitigate Style Change Disruption.**
4. **Calibrate Targets Dynamically Between 0.70 and 0.80.**

### Slide 11: Interactive Streamlit Demonstration
- Live What-If shift simulator.
- Real-time productivity gauge & attainment probability.
- Dynamic waterfall factor attribution.

### Slide 12: Conclusion, Limitations & Future Work
- Successfully answers problem statement with dual empirical and statistical justification.
- Limitations: Domain transferability, unobserved machine health data.
- Future work: Real-time sensor stream integration.

---

## Part 2: Viva Voce Defense Q&A Guide

### Q1: Why did you frame this problem as both a regression and a classification task?
**Answer:** In manufacturing and team management, stakeholders require two distinct forms of intelligence:
1. *Continuous operational pacing:* How close will the team get to absolute standard output? (Regression target: `actual_productivity`).
2. *Contractual / Quota fulfillment:* Will the batch meet delivery commitments? (Binary target: `actual_productivity >= targeted_productivity`).
Providing both allows factory planners to estimate both exact unit yields and calculate financial risk of missing delivery targets.

### Q2: How did you provide "Proper Justification" as requested by the problem statement?
**Answer:** "Proper Justification" requires more than black-box feature importance. We deployed a dual framework:
1. *Econometric Justification:* We fitted an Ordinary Least Squares (OLS) model to perform formal two-tailed hypothesis testing ($H_0: \beta = 0$). Features such as `incentive_per_worker` ($p < 0.001$) and `no_of_style_change` ($p = 0.026$) demonstrated statistical significance at the $\alpha = 0.05$ and $\alpha = 0.001$ levels.
2. *Game-Theoretic Justification (SHAP):* Using Shapley values, we verified marginal contributions across all feature permutations, satisfying efficiency, symmetry, and monotonicity.

### Q3: Why did you impute 0 for missing values in WIP for finishing instead of using mean/median imputation?
**Answer:** Standard automated imputation (such as mean imputation) would have introduced severe bias. Domain inspection revealed that 100% of missing WIP values were in the Finishing department. In garment manufacturing, WIP represents intermediate work in the sewing line. Finishing operates as a downstream packaging and inspection station receiving finished bundles; it does not maintain intermediate sewing inventory. Imputing 0 reflects the actual physical assembly process.

### Q4: How did you detect the data-entry swap on March 9, 2015?
**Answer:** During exploratory outlier detection, we noticed extreme spikes in `incentive` (values up to 3,600 BDT) occurring solely on 2015-03-09 in finishing teams, while their `over_time` was recorded as 0. Because finishing teams have 8 to 15 workers, standard 120-minute overtime equals $8 \times 120 = 960$ and $12 \times 120 = 1440$ minutes. Cross-referencing proved the clerk swapped overtime minutes into the incentive column.

### Q5: What is the justification for capping overtime at 120 minutes per worker?
**Answer:** In our quadratic regression and SHAP interaction analysis, team productivity increases with overtime up to approximately 120 minutes/worker/day. Past this inflection point, worker physical fatigue causes attentional decline, error rates rise, and productivity drops ($p = 0.029$ for the fatigue penalty). Therefore, scheduling overtime beyond 2 hours/worker wastes payroll and depresses hourly output.
