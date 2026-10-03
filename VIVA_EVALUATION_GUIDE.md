# Viva & Evaluation Guide: Case Study - Team Performance Analysis
## Olympic Team Performance and Determinant Optimization Using Machine Learning
**ITM Skills University | School of Future Tech | B.Tech CSE Semester V**
**Student Name: R Virshin | Roll No: 150096724147**

---

## Quick Elevator Pitch (If the evaluator asks: "Tell me what you did in this project")
> *"In this project, I developed an empirical machine learning and statistical framework to solve the problem: 'Which measurable factors are associated with stronger team results (With Proper Justification)?' Using the official 120 Years of Olympic History dataset covering modern Summer Games (1960 - 2016), I cleaned and aggregated 271,000 athlete records into 2,264 national delegation participations across 223 countries. I engineered measurable structural factors including delegation roster size, sports diversity, female participation ratio, athlete age and BMI, host nation advantage, and historical track record. I evaluated all 6 coursework classification algorithms and 5 regression architectures. Random Forest achieved 90.51% classification accuracy and 0.8782 F1-score for predicting podium success, while Gradient Boosting predicted medal output with an R-squared of 0.8838 (MAE of 1.80 medals). To ensure rigorous justification, I performed econometric OLS hypothesis testing to prove statistical significance (p < 0.001) for roster depth, event breadth, and sports diversity, and deployed the final system into an interactive Streamlit application."*

---

## Direct Answers to Section 8: "Questions to Be Answered"

### 1. Can team performance / results be predicted using ML?
* **Answer**: **Yes, with high empirical accuracy and statistical significance.**
* **Explanation**: Olympic team achievements are not random occurrences; they reflect structural capacity, athlete investment, and sports breadth. Our models achieve:
  - **Classification (Podium Success)**: **90.51% Accuracy**, **90.12% Precision**, **85.64% Recall**, and **0.9584 ROC-AUC**.
  - **Regression (Total Medals Won)**: **0.8838 $R^2$**, Root Mean Squared Error (RMSE) of **5.01 medals**, and Mean Absolute Error (MAE) of **1.80 medals**.
  - This demonstrates that machine learning models reliably distinguish between medal-winning and non-medal-winning national teams on unseen future Olympic cycles.

### 2. Which features contribute most to prediction?
* **Answer**:
  1. **Total Athletes on Roster (`total_athletes`, 25.5% importance, p < 0.001)**: The single strongest operational predictor. Larger delegations maximize probabilistic bracket representation across events.
  2. **Events Contested (`num_events`, 25.0% importance, p < 0.001)**: Entering athletes into 50+ diverse events dramatically scales medal opportunities.
  3. **Athlete-Event Volume (`total_entries`, 16.5% importance)**: Captures multi-event athlete participation (e.g. swimming and gymnastics dual entrants).
  4. **Sports Diversity (`num_sports`, 11.9% importance, p = 0.002)**: Nations competing in 15+ sports avoid downside dependency on any single sport's qualification shocks.
  5. **Historical Track Record (`prev_medals`, 11.3% importance, p < 0.001)**: Previous Olympic medal counts capture unmeasured national athletic infrastructure, coaching quality, and long-term sports ministry budget allocations.
  6. **Demographics (`female_ratio`, `mean_age`, `mean_bmi`, ~8.8% combined)**: Female athlete participation has a statistically significant positive relationship with modern total medal share ($p = 0.038$).

### 3. Which model performs best?
* **Answer**: **Random Forest Classifier (for Podium Classification)** and **Gradient Boosting Regressor (for Medal Count Estimation)**.
* **Explanation**:
  - **Random Forest**:
    - Accuracy: **90.51%**
    - Precision: **90.12%**
    - Recall: **85.64%**
    - F1-Score: **0.8782**
    - ROC-AUC: **0.9584**
  - **Why Random Forest Wins**: By constructing 100 de-correlated decision trees with random feature subsets, Random Forest captures non-linear synergies (such as the interaction between large roster size and sports diversity) while eliminating the high variance that causes individual decision trees to overfit.
  - **Gradient Boosting Regressor**: Outperformed all models with an out-of-sample $R^2$ of **0.8838** and an MAE of **1.80 medals**, effectively minimizing residual errors through sequential gradient descent optimization.

### 4. Which model provides better precision / recall?
* **Answer**:
  - **Highest Precision**: **SVM with RBF Kernel (92.36%)** and **Logistic Regression (91.19%)**, closely followed by **Random Forest (90.12%)**.
  - **Highest Recall**: **Decision Tree (86.74%)** and **Random Forest (85.64%)**.
  - **Why Random Forest is the Operational Choice**:
    - In sports analytics and national budget planning, both types of errors carry significant costs:
      - A **False Negative** (predicting zero medals for a country that actually wins) causes underfunding and missed sponsorship opportunities.
      - A **False Positive** (over-promising a medal that fails to materialize) misallocates public athletic funds.
    - While SVM achieved 92.36% precision, its recall fell to 80.11% (missing 1 in 5 medal winners). Random Forest balances both sides, delivering the highest overall **F1-score of 0.8782**.

### 5. How does preprocessing affect results?
* **Answer**: Preprocessing was critical for data integrity, scale uniformity, and model convergence:
  1. **Event-Level Deduplication**: Raw Olympic records contain rows for every athlete. In team sports (like water polo or football), 15 athletes receive medals for 1 national medal. Without event-level deduplication, team sport medals would be artificially inflated by 10x-18x.
  2. **Missing Value Imputation**: Historical athlete height and weight had missing records in smaller delegations. Imputing with the **median grouped by sport and sex** preserved anthropometric distributions without introducing synthetic bias.
  3. **Feature Scaling (`StandardScaler`)**: Crucial for distance-based and gradient-based algorithms (KNN, Logistic Regression, SVM). For example, `total_athletes` ranges from 1 to 650, while `female_ratio` ranges from 0.0 to 1.0. Without scaling, KNN was completely dominated by athlete count, reducing accuracy from 88.08% to under 78%.
  4. **Stratified Splitting**: Preserved the 40/60 class balance across train and test sets, avoiding sample distribution shift.

### 6. Can the model be deployed for prediction?
* **Answer**: **Yes, deployed live on Streamlit Community Cloud and running locally via `app.py`.**
* **Live Cloud Application**: [https://virshin1-team-performance-analysis-app-kwvw1u.streamlit.app/](https://virshin1-team-performance-analysis-app-kwvw1u.streamlit.app/)
* **Deployment Architecture**:
  - The trained models, preprocessor, and feature names are serialized as lightweight `.pkl` files (`best_model.pkl`, `best_regressor.pkl`, `scaler.pkl`, `feature_names.pkl`).
  - The Streamlit dashboard allows Olympic planners, sports analysts, and athletic directors to configure delegation variables (roster size, events, sports, athlete demographics, host status) and immediately receive:
    1. Real-time projected total medal count.
    2. Podium probability score (e.g. 88.4% Confidence).
    3. Breakdown of the primary driving factors for that specific delegation.

---

## In-Depth Comparison: Distinguishing All Six Classification Models

### 1. Logistic Regression
* **How it works**: Calculates a linear combination of normalized inputs ($z = w^T x + b$) and squashes it through the sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$ to estimate class probabilities.
* **What it assumes**: An additive, monotonic relationship between the log-odds of winning a medal and the scaled delegation features.
* **On this dataset**: Achieved **88.96% accuracy**, **91.19% precision**, **80.11% recall**, and **0.9595 ROC-AUC**.
* **Why it did well**: The core signal in Olympic performance (roster size and prior medals) is strongly monotonic. A larger delegation consistently increases the log-odds of securing at least one medal.
* **Weakness**: Cannot model non-linear interactions, such as diminishing returns at very large team sizes or synergies between specific sports disciplines.

### 2. K-Nearest Neighbors (KNN, k=5)
* **How it works**: A non-parametric instance-based learner. Classifies an unseen delegation by computing Euclidean distances to all training delegations and taking the majority vote among the 5 nearest neighbors.
* **What it assumes**: That delegations with similar roster sizes, sports breadth, and demographic traits achieve similar podium outcomes.
* **On this dataset**: Achieved **88.08% accuracy**, **89.94% precision**, **79.01% recall**, and **0.8412 F1-Score**.
* **Why it scored lowest among the six**: In 11-dimensional feature space, distances suffer from metric dilution (the curse of dimensionality). Furthermore, small nations with specialized medalists (e.g., Jamaica in sprinting or Kenya in distance running) have small rosters that neighbor non-medal winning nations, confusing nearest-neighbor boundaries.

### 3. Decision Tree (max_depth = 5)
* **How it works**: Recursively splits the training data using greedy binary thresholds that maximize Gini impurity reduction.
* **On this dataset**: Achieved **89.40% accuracy**, **86.74% precision**, and the highest raw recall (**86.74%**), with an F1-Score of **0.8674**.
* **Why it scored as it did**: The tree quickly discovered that delegations with `total_athletes >= 45` or `prev_medals >= 2` have over an 85% probability of winning a medal. However, being a single unregularized tree, it is high-variance and less precise (86.74% precision vs 90.12% for Random Forest).

### 4. Random Forest (n=100, max_depth = 6) - *Selected Best Classifier*
* **How it works**: Trains 100 de-correlated decision trees on bootstrap resamples (bagging) with random feature selection ($\sqrt{p}$) at every split node. Final classification is determined by majority vote.
* **On this dataset**: **90.51% Accuracy**, **90.12% Precision**, **85.64% Recall**, and **0.8782 F1-Score**.
* **Why it is the winning model**: It eliminates the variance of the single decision tree through averaging, robustly captures complex feature interactions (roster depth multiplied by sports diversity), and is naturally resistant to outliers and overfitting.

### 5. Gradient Boosting (learning_rate = 0.08, n=100)
* **How it works**: A sequential ensemble method that trains shallow regression trees sequentially, where each new tree fits the negative gradient (pseudo-residuals) of the loss function.
* **On this dataset**: Achieved **90.29% accuracy**, **90.06% precision**, **85.08% recall**, and **0.8750 F1-Score**.
* **Performance**: Performed nearly identically to Random Forest in classification, and was the undisputed winner in regression ($R^2 = 0.8838$, $RMSE = 5.01$).

### 6. Support Vector Machine (SVM, RBF Kernel)
* **How it works**: Maps the 11 normalized features into a higher-dimensional reproducing kernel Hilbert space using a Radial Basis Function kernel to find the maximum-margin separating hyperplane.
* **On this dataset**: Achieved **89.40% accuracy**, **92.36% Precision (highest of all models)**, **80.11% Recall**, and **0.8580 F1-Score**.
* **Trade-off Analysis**: SVM draws a conservative decision boundary that yields the highest precision (92.36%), but this conservatism causes it to miss 19.9% of true medal winners (lower recall of 80.11%). For balanced operational decision-making, Random Forest is preferred.

---

## Statistical Justification Summary (OLS Regression Table)

| Feature | Coefficient ($\beta$) | Std. Error | $t$-statistic | $p$-value | Statistical Significance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Total Athletes** | **+4.8214** | **0.4120** | **11.70** | **< 0.0001** | Significant at 99.9% level |
| **Previous Medals** | **+7.1420** | **0.3150** | **22.67** | **< 0.0001** | Significant at 99.9% level |
| **Number of Sports** | **+1.4180** | **0.4520** | **3.14** | **0.0017** | Significant at 99% level |
| **Events Entered** | **+2.3150** | **0.4810** | **4.81** | **< 0.0001** | Significant at 99.9% level |
| **Host Nation Status** | **+4.1200** | **1.2100** | **3.40** | **0.0007** | Significant at 99% level |
| **Female Ratio** | **+0.8920** | **0.4300** | **2.07** | **0.0384** | Significant at 95% level |
| **Mean Athlete Age** | +0.3120 | 0.2450 | 1.27 | 0.2041 | Not statistically significant |
| **Mean Athlete BMI** | -0.1540 | 0.2210 | -0.70 | 0.4840 | Not statistically significant |
