import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import altair as alt

st.set_page_config(
    page_title="Olympic Team Performance Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom minimal typography and layout styles
st.markdown("""
<style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    h1, h2, h3, h4, p, span, label {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .header-tag {
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 0.25rem;
        font-weight: 600;
    }
    .title-text {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.35rem;
        line-height: 1.25;
    }
    .subtitle-text {
        font-size: 0.95rem;
        color: #475569;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    .stat-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .stat-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
        font-weight: 600;
    }
    .stat-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.2rem;
    }
    .stat-caption {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.2rem;
    }
    .verdict-card {
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #e2e8f0;
        background-color: #f8fafc;
    }
    .verdict-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    .verdict-desc {
        font-size: 0.92rem;
        color: #334155;
        line-height: 1.55;
    }
    @media (prefers-color-scheme: dark) {
        .title-text { color: #f8fafc; }
        .subtitle-text { color: #94a3b8; }
        .stat-box { background-color: #0f172a; border-color: #1e293b; }
        .stat-value { color: #f8fafc; }
        .verdict-card { background-color: #0f172a; border-color: #1e293b; }
        .verdict-desc { color: #cbd5e1; }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    base_dir = Path(__file__).resolve().parent
    clf = joblib.load(base_dir / 'best_model.pkl')
    reg = joblib.load(base_dir / 'best_regressor.pkl')
    scaler = joblib.load(base_dir / 'scaler.pkl')
    cols = joblib.load(base_dir / 'feature_names.pkl')
    feat_imp = pd.read_csv(base_dir / 'feature_importance.csv')
    return clf, reg, scaler, cols, feat_imp

model_clf, model_reg, scaler, feature_names, feat_importance_df = load_assets()

# Sidebar: Preset Profiles and Delegation Parameters
st.sidebar.markdown("### Delegation Presets")
preset = st.sidebar.selectbox(
    "Load Benchmark Profile",
    options=["Custom Configuration", "Global Powerhouse (e.g. USA / China)", "Mid-Tier Contender (e.g. Netherlands / Brazil)", "Emerging Delegation (e.g. Small Contingent)"]
)

# Set defaults based on preset
if preset == "Global Powerhouse (e.g. USA / China)":
    d_athletes, d_sports, d_entries, d_events = 520, 28, 750, 220
    d_age, d_female, d_height, d_weight = 26.2, 0.52, 178.5, 73.0
    d_host, d_prev = "Visiting Delegation (0)", 95
elif preset == "Mid-Tier Contender (e.g. Netherlands / Brazil)":
    d_athletes, d_sports, d_entries, d_events = 160, 18, 220, 105
    d_age, d_female, d_height, d_weight = 26.0, 0.48, 176.0, 71.5
    d_host, d_prev = "Visiting Delegation (0)", 16
elif preset == "Emerging Delegation (e.g. Small Contingent)":
    d_athletes, d_sports, d_entries, d_events = 12, 4, 15, 10
    d_age, d_female, d_height, d_weight = 24.5, 0.35, 172.0, 67.0
    d_host, d_prev = "Visiting Delegation (0)", 0
else:
    d_athletes, d_sports, d_entries, d_events = 85, 14, 120, 65
    d_age, d_female, d_height, d_weight = 25.5, 0.48, 176.5, 71.0
    d_host, d_prev = "Visiting Delegation (0)", 6

st.sidebar.markdown("---")
st.sidebar.markdown("### 1. Delegation Scale & Breadth")
total_athletes = st.sidebar.number_input("Total Athletes", min_value=1, max_value=800, value=d_athletes, step=5)
num_sports = st.sidebar.number_input("Distinct Sports Contested", min_value=1, max_value=35, value=d_sports, step=1)
total_entries = st.sidebar.number_input("Total Event Registrations", min_value=1, max_value=1200, value=d_entries, step=5)
num_events = st.sidebar.number_input("Distinct Events Entered", min_value=1, max_value=300, value=d_events, step=2)

st.sidebar.markdown("---")
st.sidebar.markdown("### 2. Physical & Demographics")
mean_age = st.sidebar.number_input("Average Athlete Age (yrs)", min_value=16.0, max_value=45.0, value=d_age, step=0.5)
female_ratio = st.sidebar.slider("Female Proportion", min_value=0.0, max_value=1.0, value=d_female, step=0.02)
mean_height = st.sidebar.number_input("Average Height (cm)", min_value=150.0, max_value=210.0, value=d_height, step=0.5)
mean_weight = st.sidebar.number_input("Average Weight (kg)", min_value=40.0, max_value=140.0, value=d_weight, step=0.5)
calculated_bmi = round(mean_weight / ((mean_height / 100) ** 2), 2)
st.sidebar.caption(f"Computed Delegation BMI: **{calculated_bmi} kg/m²**")

st.sidebar.markdown("---")
st.sidebar.markdown("### 3. Historical Track Record")
prev_medals = st.sidebar.number_input("Previous Olympic Medals", min_value=0, max_value=150, value=d_prev, step=1)
is_host_sel = st.sidebar.selectbox("Host Country Status", options=["Visiting Delegation (0)", "Host Nation (1)"], index=0 if d_host == "Visiting Delegation (0)" else 1)
is_host = 1 if "Host Nation" in is_host_sel else 0

# Header
st.markdown('<div class="header-tag">Machine Learning &middot; Sports Analytics Case Study &middot; <a href="https://virshin1-team-performance-analysis-app-kwvw1u.streamlit.app/" target="_blank" style="color: #2563eb; font-weight: 600; text-decoration: underline;">Live Cloud App</a></div>', unsafe_allow_html=True)
st.markdown('<div class="title-text">Olympic Team Performance Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Empirical analysis and machine learning evaluation of the measurable operational, demographic, and historical factors associated with Olympic podium success.</div>', unsafe_allow_html=True)

# Main Tabbed Layout
tab_assess, tab_factors, tab_docs = st.tabs([
    "Team Assessment & Prediction",
    "Measurable Factors & Importance",
    "Dataset & Methodology"
])

# Perform ML Inference
raw_input = pd.DataFrame([[
    total_athletes, total_entries, num_sports, num_events,
    female_ratio, mean_age, mean_height, mean_weight, calculated_bmi,
    is_host, prev_medals
]], columns=feature_names)

scaled_input = scaler.transform(raw_input)
pred_clf = model_clf.predict(scaled_input)[0]
prob_clf = model_clf.predict_proba(scaled_input)[0]
podium_prob = prob_clf[1] * 100
no_podium_prob = prob_clf[0] * 100
pred_medals = max(0.0, float(model_reg.predict(scaled_input)[0]))

with tab_assess:
    if pred_clf == 1:
        border_col = "#15803d"
        verdict_head = "Podium Success Predicted (Medal Winner Profile)"
        verdict_body = (
            f"The Random Forest model classifies this delegation configuration as a probable Olympic medal winner with "
            f"a confidence probability of {podium_prob:.1f}%. The continuous regressor estimates approximately "
            f"{pred_medals:.1f} total medals."
        )
    else:
        border_col = "#b91c1c"
        verdict_head = "Non-Podium Profile Predicted"
        verdict_body = (
            f"The Random Forest model classifies this delegation profile below the medal threshold ({no_podium_prob:.1f}% confidence for zero medals). "
            f"Projected medal volume is {pred_medals:.1f} medals. Increasing qualifying roster breadth across diverse disciplines is the primary path to podium viability."
        )

    st.markdown(f"""
    <div class="verdict-card" style="border-left: 5px solid {border_col};">
        <div class="verdict-title" style="color: {border_col};">{verdict_head}</div>
        <div class="verdict-desc">{verdict_body}</div>
    </div>
    """, unsafe_allow_html=True)

    # Key Performance Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">Projected Total Medals</div>
            <div class="stat-value">{pred_medals:.1f}</div>
            <div class="stat-caption">Gradient Boosting (Test R2: 0.8838)</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">Podium Probability</div>
            <div class="stat-value">{podium_prob:.1f}%</div>
            <div class="stat-caption">Odds of winning 1 or more medals</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">Classification Verdict</div>
            <div class="stat-value" style="font-size: 1.35rem; color: {border_col};">{'Medal Contender' if pred_clf == 1 else 'Non-Podium'}</div>
            <div class="stat-caption">Random Forest (Test Acc: 90.51%)</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">Model Certainty</div>
            <div class="stat-value">{max(podium_prob, no_podium_prob):.1f}%</div>
            <div class="stat-caption">Ensemble tree consensus</div>
        </div>
        """, unsafe_allow_html=True)

    # Probability Distribution (Native Streamlit, Zero White-Bar Artifacts)
    st.markdown("#### Outcome Probability Breakdown")
    c_p1, c_p2 = st.columns(2)
    with c_p1:
        st.markdown(f"**Medal Winner (Podium)**: `{podium_prob:.1f}%`")
        st.progress(min(1.0, max(0.0, float(podium_prob / 100.0))))
    with c_p2:
        st.markdown(f"**No Medals**: `{no_podium_prob:.1f}%`")
        st.progress(min(1.0, max(0.0, float(no_podium_prob / 100.0))))

    # Diagnostic Factor Benchmark Table (Native dataframe, theme-adaptive)
    st.markdown("---")
    st.markdown("#### Operational Factor Comparison with Historical Benchmarks")
    st.markdown("Comparing configured parameters against the median profile of Olympic medal-winning delegations (1960-2016):")

    bench_data = pd.DataFrame({
        "Operational Dimension": [
            "Total Athletes (Roster Size)",
            "Distinct Sports Contested",
            "Distinct Events Entered",
            "Prior Olympic Medals Won",
            "Female Athlete Proportion",
            "Average Athlete Age (yrs)"
        ],
        "Configured Value": [
            f"{total_athletes}",
            f"{num_sports}",
            f"{num_events}",
            f"{prev_medals}",
            f"{female_ratio * 100:.1f}%",
            f"{mean_age:.1f}"
        ],
        "Median Medal-Winning Team": [
            "142",
            "16",
            "108",
            "12",
            "44.0%",
            "25.2"
        ],
        "Median Non-Winning Team": [
            "11",
            "4",
            "10",
            "0",
            "25.0%",
            "24.8"
        ],
        "Factor Impact Weight": [
            "25.5% (Very High)",
            "11.9% (High)",
            "25.0% (Very High)",
            "11.3% (High)",
            "1.9% (Low)",
            "2.8% (Low)"
        ]
    })
    st.dataframe(bench_data, use_container_width=True, hide_index=True)

with tab_factors:
    st.markdown("### Measurable Factors Associated with Stronger Results")
    st.markdown(
        "To answer the problem statement, the Random Forest model quantifies the relative importance "
        "of all 11 factors using Mean Decrease in Impurity (Gini MDI across 200 estimators):"
    )

    c_left, c_right = st.columns([3, 2])
    with c_left:
        chart_data = feat_importance_df.sort_values(by="Importance", ascending=True)
        chart = alt.Chart(chart_data).mark_bar(color="#2563eb", cornerRadiusEnd=3).encode(
            x=alt.X("Importance:Q", title="Relative Predictive Importance", axis=alt.Axis(format="%")),
            y=alt.Y("Feature:N", title=None, sort="-x"),
            tooltip=["Feature", alt.Tooltip("Importance:Q", format=".2%")]
        ).properties(height=360)
        st.altair_chart(chart, use_container_width=True, theme="streamlit")

    with c_right:
        st.markdown("#### Key Factor Hierarchy")
        st.markdown("""
        1. **Roster Size & Opportunity Breadth (78.9% cumulative)**:
           `total_athletes` (25.5%), `num_events` (25.0%), `total_entries` (16.5%), and `num_sports` (11.9%) dictate the vast majority of outcomes. More qualification slots provide multiple non-correlated paths to victory.
        2. **Historical Momentum (11.3%)**:
           `prev_medals` ($r = 0.901$ with total medal volume) reflects multi-quadrennial infrastructure, coaching stability, and funding pipelines.
        3. **Athlete Biometrics (9.8% cumulative)**:
           `mean_age` (2.8%), `female_ratio` (1.9%), `mean_height` (1.8%), `mean_bmi` (1.7%), and `mean_weight` (1.6%) provide secondary fine-tuning.
        """)

    st.markdown("---")
    st.markdown("#### Pearson Linear Correlation Ranking")
    corr_summary = pd.DataFrame({
        "Rank": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "Measurable Factor": [
            "prev_medals", "total_entries", "total_athletes", "num_events",
            "num_sports", "is_host", "female_ratio", "mean_height",
            "mean_weight", "mean_age", "mean_bmi"
        ],
        "Correlation with Total Medals (r)": [
            "+0.901", "+0.834", "+0.834", "+0.777",
            "+0.622", "+0.304", "+0.159", "+0.110",
            "+0.071", "+0.012", "-0.006"
        ],
        "Correlation with Podium Occurrence (r)": [
            "+0.370", "+0.572", "+0.575", "+0.635",
            "+0.702", "+0.100", "+0.119", "+0.196",
            "+0.128", "+0.074", "+0.0001"
        ],
        "Interpretation": [
            "Strongest predictor of medal volume",
            "High athlete workload",
            "Foundation of qualification power",
            "Breadth of podium opportunities",
            "Highest correlation to winning any medal",
            "Quota entry & crowd benefit",
            "Program maturity indicator",
            "Discipline-specific advantage",
            "Discipline-specific advantage",
            "Tournament experience balance",
            "No direct linear relationship"
        ]
    })
    st.dataframe(corr_summary, use_container_width=True, hide_index=True)

with tab_docs:
    st.markdown("### Problem Definition & Methodology")
    st.markdown("""
    - **Problem Statement**: A team wants to understand which measurable factors are associated with stronger team results (With Proper Justification).
    - **Dataset Scope**: Modern Summer Olympic Games (1960–2016), aggregating 2,264 team-year observations across 223 National Olympic Committees (NOCs). Sourced from the historical Olympic athlete dataset (Kaggle / 120 Years of Olympic History).
    - **Primary Machine Learning Model**: Random Forest Classifier (`n_estimators=200`, `max_depth=12`, `random_state=42`).
    - **Evaluation Metrics (Normal R2 & Classification)**:
      - Classification Accuracy: **90.51%**
      - Precision: **90.12%**
      - Recall: **85.64%**
      - F1-Score: **0.8782**
      - ROC-AUC: **0.9584**
      - Continuous Regression: **Train R2: 0.9863**, **Test R2: 0.8838** (Gradient Boosting Regressor, MAE: 1.80, RMSE: 5.01).
    """)
