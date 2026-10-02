"""
Explainability and Statistical Factor Justification Module.
Provides SHAP (SHapley Additive exPlanations), econometric OLS significance testing,
and managerial factor attribution.
"""

import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt

def compute_ols_statistical_justification(
    X_train_df: pd.DataFrame,
    y_train_series: pd.Series
) -> Tuple[pd.DataFrame, Any]:
    """
    Fits an Ordinary Least Squares (OLS) model to obtain formal statistical
    justifications (coefficients, standard errors, t-statistics, and p-values).
    """
    # Add constant for intercept
    X_const = sm.add_constant(X_train_df)
    ols_model = sm.OLS(y_train_series, X_const).fit()

    summary_df = pd.DataFrame({
        'Feature': ols_model.params.index,
        'Coefficient': ols_model.params.values,
        'Std_Error': ols_model.bse.values,
        't_stat': ols_model.tvalues.values,
        'p_value': ols_model.pvalues.values,
        'CI_Lower': ols_model.conf_int()[0].values,
        'CI_Upper': ols_model.conf_int()[1].values
    })

    # Significance flags
    def flag_significance(p):
        if p < 0.001:
            return '*** (p < 0.001)'
        elif p < 0.01:
            return '** (p < 0.01)'
        elif p < 0.05:
            return '* (p < 0.05)'
        elif p < 0.1:
            return '. (p < 0.1)'
        else:
            return 'Not Significant'

    summary_df['Significance'] = summary_df['p_value'].apply(flag_significance)
    summary_df = summary_df[summary_df['Feature'] != 'const'].sort_values(by='p_value').reset_index(drop=True)

    return summary_df, ols_model

def compute_shap_explanations(
    tree_model: Any,
    X_data: pd.DataFrame
) -> Tuple[np.ndarray, shap.Explainer, pd.DataFrame]:
    """
    Computes TreeSHAP values and global feature importance ranking.
    """
    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(X_data)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature': X_data.columns,
        'Mean_Absolute_SHAP': mean_abs_shap
    }).sort_values(by='Mean_Absolute_SHAP', ascending=False).reset_index(drop=True)

    # Relative importance percentage
    total_shap = importance_df['Mean_Absolute_SHAP'].sum()
    importance_df['Relative_Importance_%'] = (importance_df['Mean_Absolute_SHAP'] / total_shap) * 100

    return shap_values, explainer, importance_df

def generate_local_factor_narrative(
    feature_names: List[str],
    raw_row: pd.Series,
    shap_row: np.ndarray,
    base_val: float,
    predicted_val: float
) -> Dict[str, Any]:
    """
    Generates a natural-language managerial diagnostic for an individual team outcome.
    """
    attributions = pd.DataFrame({
        'Feature': feature_names,
        'SHAP_Impact': shap_row
    }).sort_values(by='SHAP_Impact', key=abs, ascending=False)

    top_positive = attributions[attributions['SHAP_Impact'] > 0].head(3)
    top_negative = attributions[attributions['SHAP_Impact'] < 0].head(3)

    pos_drivers = []
    for _, r in top_positive.iterrows():
        pos_drivers.append(f"{r['Feature']} (+{r['SHAP_Impact']:.3f} productivity lift)")

    neg_drivers = []
    for _, r in top_negative.iterrows():
        neg_drivers.append(f"{r['Feature']} ({r['SHAP_Impact']:.3f} productivity drag)")

    return {
        'base_productivity': round(float(base_val), 4),
        'predicted_productivity': round(float(predicted_val), 4),
        'top_positive_drivers': pos_drivers,
        'top_negative_drivers': neg_drivers,
        'attributions_table': attributions
    }

if __name__ == "__main__":
    from data_loader import get_cleaned_data
    from preprocessing import engineer_features, prepare_modeling_data
    from sklearn.ensemble import GradientBoostingRegressor

    print("Running explainability test...")
    df_clean = get_cleaned_data()
    df_eng = engineer_features(df_clean)
    bundle = prepare_modeling_data(df_eng)

    X_train_trans = bundle['X_train_trans']
    y_reg_train = bundle['y_reg_train']

    ols_df, ols_model = compute_ols_statistical_justification(X_train_trans, y_reg_train)
    print("\n--- Top Statistically Significant Features (OLS) ---")
    print(ols_df.head(10)[['Feature', 'Coefficient', 'p_value', 'Significance']])

    gbr = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    gbr.fit(X_train_trans, y_reg_train)
    shap_vals, explainer, imp_df = compute_shap_explanations(gbr, X_train_trans)
    print("\n--- Top SHAP Features ---")
    print(imp_df.head(10))

