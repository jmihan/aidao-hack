import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from typing import Tuple, Dict, Any


def generate_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df.sort_values(['id', 'date'], inplace=True)

    for lag in [1, 3, 5, 10]:
        df[f"E_mu_Z_lag_{lag}"] = df.groupby('id')['E_mu_Z'].shift(lag)
        df[f"f_EC_lag_{lag}"] = df.groupby('id')['f_EC'].shift(lag)

    for window in [5, 10, 20]:
        grouped = df.groupby('id')['E_mu_Z']
        df[f'E_mu_Z_roll_mean_{window}'] = grouped.transform(lambda x: x.shift(1).rolling(window).mean())
        df[f'E_mu_Z_roll_std_{window}'] = grouped.transform(lambda x: x.shift(1).rolling(window).std())

    df['M_mu_XX_div_N_mu_X'] = df['M_mu_XX'] / (df['N_mu_X'] + 1e-6)
    df['M_nu1_XX_div_N_nu1_X'] = df['M_nu1_XX'] / (df['N_nu1_X'] + 1e-6)

    df['temp_1_x_bias_1'] = df['temp_1'] * df['biasVoltage_1']
    
    return df


def prepare_classification_data(
    df: pd.DataFrame,
    target_col: str = 'R'
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    
    df_featured = generate_base_features(df.copy())
    #print(df_featured.columns)
    df_featured.dropna(subset=[target_col], inplace=True)

    df_featured['R_class'] = pd.Categorical(df_featured[target_col]).codes

    r_to_class_map = dict(enumerate(pd.Categorical(df_featured[target_col]).categories))
    class_to_r_map = {v: k for k, v in r_to_class_map.items()}
    
    y = df_featured['R_class']

    cols_to_drop = [
        'id', 'date', 'estimator_name', 'R', 's', 'p', 'R_class', 
        'E_mu_Z_est', 'E_mu_phys_est'
    ]
    #print(df_featured['R_class'])
    feature_cols = [col for col in df_featured.columns if col not in cols_to_drop and df_featured[col].dtype in [np.int64, np.float64]]
    #print(feature_cols)
    X = df_featured[feature_cols]
    X = X.bfill().ffill()
    
    artifacts = {
        "class_to_r_map": class_to_r_map,
        "feature_cols": feature_cols
    }
    
    return X, y, artifacts

def split_by_block_id(X: pd.DataFrame, y: pd.Series, test_size=0.2):
    splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=X['id']))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    X_train = X_train.drop(columns=['id'])
    X_test = X_test.drop(columns=['id'])

    return X_train, X_test, y_train, y_test


from imblearn.over_sampling import SMOTE

def resample_training_data(X_train: pd.DataFrame, y_train: pd.Series):
    print("Применение SMOTE для балансировки обучающих классов...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"Размер до SMOTE: {X_train.shape}, после SMOTE: {X_train_resampled.shape}")
    print(f"Новое распределение классов:\n{y_train_resampled.value_counts(normalize=True)}")
    return X_train_resampled, y_train_resampled