# Olympic Team Performance Analysis Using Machine Learning
**Case Study: Team Performance Analysis | B.Tech CSE Semester V**
**ITM Skills University | School of Future Tech**
**Student Name: R Virshin | Roll No: 150096724147**

An empirical machine learning framework and sports decision support system that analyzes measurable athletic, demographic, and operational factors to predict national Olympic team medal outcomes and podium attainment (with proper statistical justification).

---

## Problem Statement
> **A team wants to understand which measurable factors are associated with stronger team results. (With Proper Justification)**

---

## Selected Champion Model: Random Forest
Instead of deploying multiple redundant algorithms, this project selects and focuses on **Random Forest** as the single best machine learning algorithm for this problem statement.

### Why Random Forest Was Selected Over All Other Algorithms:
1. **Directly Answers "Which Measurable Factors Matter"**: Random Forest natively computes Mean Decrease in Impurity (MDI) across all decision trees. This provides mathematically sound, objective feature rankings that isolate the exact factors driving team success.
2. **Models Complex Non-Linear Synergies**: Sports team performance is non-linear (e.g. moving from 5 to 50 athletes provides massive exponential gains, while moving from 400 to 450 exhibits diminishing returns). Random Forest captures these non-linear thresholds and multi-factor interactions without restrictive linear assumptions.
3. **Robust Against Severe Sports Skewness**: Olympic data is heavily skewed (a few powerhouse nations win 70+ medals while over half win 0). Single decision trees overfit, and boosting can over-index on outliers. Random Forest uses bootstrap aggregation (bagging) and random feature subsampling ($\sqrt{p}$), making it resistant to noise and variance.
4. **Scale-Invariant (No Feature Distortion)**: Unlike KNN or SVM which require Euclidean distance normalization, Random Forest operates directly on original feature scales without distorting physical units.
5. **Highest Empirical Performance**: Achieved **90.51% Accuracy**, **90.12% Precision**, **85.64% Recall**, and an **F1-Score of 0.8782** (ROC-AUC: **0.9584**, 10-Fold CV: **88.18%**).

---

## Performance Summary (Random Forest)

| Metric | Score | Practical Meaning for National Olympic Delegations |
| :--- | :---: | :--- |
| **Accuracy** | **90.51%** | Accurately predicts 9 out of 10 delegation outcomes on unseen future Olympic Games. |
| **Precision** | **90.12%** | When the model predicts a nation will win a medal, it is correct 90.1% of the time (low false alarms). |
| **Recall** | **85.64%** | Captures 85.6% of all true medal-winning nations (avoids missing potential podium contenders). |
| **F1-Score** | **0.8782** | Harmonic balance between precision and recall, optimizing athletic resource allocation. |
| **Test ROC-AUC** | **0.9584** | Outstanding discriminative ability across all classification decision thresholds. |
| **10-Fold CV Mean** | **88.18%** | Validates model generalizability across 10 distinct subsets of historical Olympic cycles. |

---

## Key Measurable Factors (With Proper Justification)

| Rank | Measurable Factor | Importance | Statistical Justification ($p$-value) | Practical Impact on Team Results |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **Total Athletes on Roster (`total_athletes`)** | **25.5%** | **$p < 0.0001$ ($\beta = +4.82$)** | The primary structural driver. Larger delegations provide probabilistic coverage across competition brackets. |
| **2** | **Contested Events (`num_events`)** | **25.0%** | **$p < 0.0001$ ($\beta = +2.32$)** | Expanding entries across 50+ diverse events scales medal conversion opportunities. |
| **3** | **Athlete-Event Volume (`total_entries`)** | **16.5%** | **$p < 0.0001$** | Measures multi-event specialization (dual entrants in swimming, track, and gymnastics). |
| **4** | **Sports Diversity (`num_sports`)** | **11.9%** | **$p = 0.0017$ ($\beta = +1.42$)** | Competing in 15+ sports hedges against single-sport qualification shocks. |
| **5** | **Historical Track Record (`prev_medals`)** | **11.3%** | **$p < 0.0001$ ($\beta = +7.14$)** | Captures unmeasured institutional funding, coaching caliber, and training infrastructure. |
| **6** | **Gender Inclusivity (`female_ratio`)** | **1.9%** | **$p = 0.0384$ ($\beta = +0.89$)** | Delegations with balanced gender representation capture significantly higher overall medal shares. |

---

## How to Run

### 1. Run the Master Jupyter Notebook:
```bash
jupyter notebook olympic_team_performance.ipynb
```

### 2. Launch the Streamlit Web Application:
```bash
streamlit run app.py
```
Open browser at: `http://localhost:8502`
