"""
Preprocessing and Feature Engineering Module for Team Performance Analysis.
Constructs operational, workload, human-factor, and temporal features.
Provides train/test splitters and transformation pipelines.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates domain-driven features reflecting team dynamics, operational load,
    fatigue factors, and calendar timing.
    """
    df = df.copy()

    # 1. Target Met Indicator (Classification Target)
    df['target_met'] = (df['actual_productivity'] >= df['targeted_productivity']).astype(int)

    # 2. Performance Margin
    df['productivity_gap'] = df['actual_productivity'] - df['targeted_productivity']

    # 3. Work In Progress (WIP) Handling:
    # Finishing has 0 WIP by process design; sewing has inventory buffer.
    df['wip_tracked'] = df['wip'].notnull().astype(int)
    sewing_wip_median = df.loc[df['department'] == 'sewing', 'wip'].median()
    df['wip_clean'] = df['wip'].fillna(0)
    # Log-transformed WIP to handle right-skewed inventory volume
    df['log_wip'] = np.log1p(df['wip_clean'])

    # 4. Normalized Resource Allocations per Worker
    # Prevents confounding between team size (8 in finishing vs 60 in sewing)
    safe_workers = np.maximum(df['no_of_workers'], 1.0)
    df['overtime_per_worker'] = df['over_time'] / safe_workers
    df['incentive_per_worker'] = df['incentive'] / safe_workers
    df['smv_per_worker'] = df['smv'] / safe_workers

    # 5. Workload Intensity & Quota Load
    # Expected standard minute output per minute of work
    df['workload_intensity'] = df['smv'] * df['targeted_productivity']
    df['smv_to_team_ratio'] = df['smv'] / safe_workers

    # 6. Fatigue & Disruption Indicators
    # Overtime fatigue non-linear proxy: beyond 120 mins/worker fatigue escalates
    df['overtime_fatigue_penalty'] = np.maximum(0, df['overtime_per_worker'] - 120)
    df['has_idle_time'] = (df['idle_time'] > 0).astype(int)
    df['has_idle_men'] = (df['idle_men'] > 0).astype(int)
    df['has_style_change'] = (df['no_of_style_change'] > 0).astype(int)
    df['idle_intensity'] = df['idle_time'] * df['idle_men']

    # 7. Temporal & Calendar Dynamics
    df['day_of_week'] = df['date'].dt.day_name()
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    # Weekend adjacent days (Saturday and Thursday, since Friday is factory weekend)
    df['is_weekend_adjacent'] = df['day_of_week'].isin(['Thursday', 'Saturday']).astype(int)

    return df

def get_feature_lists() -> Dict[str, List[str]]:
    """
    Returns grouped lists of numerical and categorical predictor columns.
    """
    numeric_features = [
        'targeted_productivity',
        'smv',
        'overtime_per_worker',
        'incentive_per_worker',
        'smv_per_worker',
        'workload_intensity',
        'overtime_fatigue_penalty',
        'log_wip',
        'idle_time',
        'idle_men',
        'no_of_style_change',
        'has_style_change',
        'no_of_workers',
        'day_of_month',
        'week_of_year'
    ]

    categorical_features = [
        'department',
        'quarter',
        'day_of_week',
        'team'
    ]

    return {
        'numeric': numeric_features,
        'categorical': categorical_features
    }

def prepare_modeling_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Prepares train and test splits for regression and classification.
    Returns feature matrices, targets, and preprocessor.
    """
    feature_meta = get_feature_lists()
    numeric_cols = feature_meta['numeric']
    categorical_cols = feature_meta['categorical']

    # Convert team to string category for proper categorical handling
    df_model = df.copy()
    df_model['team'] = df_model['team'].astype(str)

    all_features = numeric_cols + categorical_cols
    X = df_model[all_features].copy()
    y_reg = df_model['actual_productivity'].copy()
    y_clf = df_model['target_met'].copy()

    # Stratify by classification target and department for balanced evaluation
    stratify_col = df_model['department'] + "_" + df_model['target_met'].astype(str)

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col
    )

    # Build transformer pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
        ]
    )

    # Fit preprocessor on training data
    preprocessor.fit(X_train)

    # Get feature names post-transformation
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_names = list(cat_encoder.get_feature_names_out(categorical_cols))
    transformed_feature_names = numeric_cols + encoded_cat_names

    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    X_train_trans_df = pd.DataFrame(X_train_trans, columns=transformed_feature_names, index=X_train.index)
    X_test_trans_df = pd.DataFrame(X_test_trans, columns=transformed_feature_names, index=X_test.index)

    return {
        'X_train_raw': X_train,
        'X_test_raw': X_test,
        'X_train_trans': X_train_trans_df,
        'X_test_trans': X_test_trans_df,
        'y_reg_train': y_reg_train,
        'y_reg_test': y_reg_test,
        'y_clf_train': y_clf_train,
        'y_clf_test': y_clf_test,
        'preprocessor': preprocessor,
        'feature_names': transformed_feature_names,
        'raw_feature_names': all_features,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols
    }

if __name__ == "__main__":
    from data_loader import get_cleaned_data
    df_clean = get_cleaned_data()
    df_eng = engineer_features(df_clean)
    bundle = prepare_modeling_data(df_eng)
    print("Preprocessing successful!")
    print(f"Engineered dataset shape: {df_eng.shape}")
    print(f"Transformed train shape: {bundle['X_train_trans'].shape}")
    print(f"Transformed test shape: {bundle['X_test_trans'].shape}")
    print(f"Number of encoded features: {len(bundle['feature_names'])}")
