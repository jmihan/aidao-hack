

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from Levenshtein import distance as lev_distance
from sklearn.model_selection import StratifiedKFold
from preprocess_text import clean_description, clean_name_rus, clean_brand_commertial4_name
from util import delete_rows_without_images
from scipy.stats import entropy

# --- Новая функция для оптимизации памяти ---
def optimize_memory_usage(df):
    """
    Итерируется по всем колонкам DataFrame и изменяет тип данных
    для уменьшения использования памяти.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Использование памяти до оптимизации: {start_mem:.2f} MB')

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and col_type.name != 'category':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # Используем float32 вместо float16 для большей стабильности вычислений
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Использование памяти после оптимизации: {end_mem:.2f} MB')
    print(f'Сокращение: {100 * (start_mem - end_mem) / start_mem:.1f}%')
    
    return df

# --- Класс для Target Encoding ---
class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=20.0, n_splits=5, random_state=42):
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.random_state = random_state
        self.global_mean_ = 0
        self.mappings_ = {}
        self.encoded_cols_ = []

    def fit(self, X, y):
        self.global_mean_ = np.mean(y)
        self.encoded_cols_ = [f"{c}_te" for c in X.columns]
        
        # 1. Обучаем "боевые" маппинги на ВСЕХ данных для будущего transform (на тесте)
        for col in X.columns:
            full_mapping = y.groupby(X[col]).mean()
            n = y.groupby(X[col]).count()
            self.mappings_[col] = (full_mapping * n + self.global_mean_ * self.smoothing) / (n + self.smoothing)
        
        # 2. Создаем OOF (Out-of-Fold) признаки для ТРЕНИРОВОЧНОГО набора
        oof_encodings = pd.DataFrame(index=X.index)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for col in X.columns:
            encoded_col = pd.Series(index=X.index, dtype=float)
            for train_idx, val_idx in skf.split(X, y):
                # Считаем среднее только на обучающем фолде
                fold_mapping = y.iloc[train_idx].groupby(X[col].iloc[train_idx]).mean()
                n_fold = y.iloc[train_idx].groupby(X[col].iloc[train_idx]).count()
                smooth_fold_mapping = (fold_mapping * n_fold + self.global_mean_ * self.smoothing) / (n_fold + self.smoothing)
                encoded_col.iloc[val_idx] = X[col].iloc[val_idx].map(smooth_fold_mapping)
            
            oof_encodings[f'{col}_te'] = encoded_col.fillna(self.global_mean_)
        
        self.oof_encodings_ = oof_encodings
        return self

    def transform(self, X):
        # Если мы на этапе fit, то возвращаем уже посчитанные OOF-признаки
        if hasattr(self, 'oof_encodings_') and X.index.equals(self.oof_encodings_.index):
            return self.oof_encodings_
        # Если на этапе predict (тестовые данные), применяем "боевые" маппинги
        else:
            X_encoded = pd.DataFrame(index=X.index)
            for col in X.columns:
                X_encoded[f'{col}_te'] = X[col].map(self.mappings_[col]).fillna(self.global_mean_)
            return X_encoded

    def get_feature_names_out(self, input_features=None):
        return self.encoded_cols_

# --- НОВАЯ, ОБЪЕДИНЕННАЯ ФУНКЦИЯ ДЛЯ СОЗДАНИЯ ПРИЗНАКОВ ---
def generate_all_features(df):
    
    # Заполняем пропуски в новых колонках нулями (на случай, если где-то деление на 0 дало NaN)
    new_feature_cols = [
        
    ]
    for col in new_feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    print(f"Создано {len(new_feature_cols)} новых признаков.")
    return df


def process_data(is_train=True):
    """Основная функция для обработки данных и сохранения артефактов."""
    
    base_data_path = './data/'
    
    if is_train:
        print("Обработка TRAIN данных...")
        train_file_path = os.path.join(base_data_path, 'train', 'ml_ozon_сounterfeit_train.csv')
        df = pd.read_csv(train_file_path)
    
    key_cols = ['id', 'name_rus', 'ItemID']
    if is_train:
        key_cols.append('resolution')

    print("Явная обработка пропусков...")
    num_cols_to_fill = []
    df[num_cols_to_fill] = df[num_cols_to_fill].fillna(0)

    # [ИЗМЕНЕНО] Вызываем новую единую функцию для генерации всех признаков
    df = generate_all_features(df)
    
    target_encode_cols = []
    
    # Определяем числовые колонки для масштабирования. Теперь включаем все новые сгенерированные признаки.
    numeric_cols = list(df.select_dtypes(include=np.number).columns)
    cols_to_remove_from_numeric = ['id', 'resolution', 'ItemID', 'SellerID']
    numeric_cols_to_scale = [col for col in numeric_cols if col not in cols_to_remove_from_numeric]

    print(f"Количество признаков для масштабирования: {len(numeric_cols_to_scale)}")

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    target_transformer = Pipeline(steps=[
        ('target_encoder', TargetEncoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols_to_scale),
            ('target', target_transformer, target_encode_cols)
        ],
        remainder='drop' # Явно указываем, что остальные колонки нужно отбросить
    )

    if is_train:
        print("Обучение препроцессора...")
        cols_to_drop_for_fit = ['resolution', 'description', 'name_rus', 'ItemID', 'id']
        X = df.drop(columns=cols_to_drop_for_fit, errors='ignore')
        y = df['resolution']
        preprocessor.fit(X, y)
        
        print("Сохранение артефакта препроцессора...")
        os.makedirs('artifacts', exist_ok=True)
        with open("artifacts/preprocessor.pkl", "wb") as f:
            pickle.dump(preprocessor, f)
    else:
        print("Загрузка обученного препроцессора...")
        with open("artifacts/preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)

    print("Трансформация данных...")
    X_to_transform = df.drop(columns=['resolution', 'description', 'name_rus', 'ItemID', 'id'], errors='ignore')
    X_transformed = preprocessor.transform(X_to_transform)
    
    scaled_cols = [f"{c}_scaled" for c in numeric_cols_to_scale]
    encoded_cols = preprocessor.named_transformers_['target'].get_feature_names_out()
    
    # [ИЗМЕНЕНО] Собираем финальный список колонок. `remainder` теперь 'drop', поэтому `passthrough_cols` не нужны.
    new_cols = scaled_cols + list(encoded_cols)
    df_processed = pd.DataFrame(X_transformed, columns=new_cols, index=df.index)
    
    df_final = pd.merge(df_keys, df_processed, left_index=True, right_index=True)

    print("\nОптимизация использования памяти...")
    df_final = optimize_memory_usage(df_final)
    
    if is_train:
        output_path = './data/processed/train.csv'
    else:
        output_path = './data/processed/test.csv'
         
    os.makedirs('data/processed', exist_ok=True)
    # Сохраняем в CSV
    df_final.to_csv(output_path, index=False)
    
    print(f"Обработка завершена! Результат сохранен в {output_path}")
    print(f"Итоговая форма DataFrame: {df_final.shape}")
    print("Пример колонок:", list(df_final.columns[:5]), "...", list(df_final.columns[-5:]))



if __name__ == '__main__':
    # Устанавливаем правильную рабочую директорию, если запускаем из ozon-services/srcs/utils
    # Это сделано для того, чтобы пути 'data/' и 'artifacts/' работали корректно
    if os.getcwd().endswith('utils'):
        os.chdir('../../')
        print(f"Рабочая директория изменена на: {os.getcwd()}")

    # process_data(is_train=True)
    process_data(is_train=False)