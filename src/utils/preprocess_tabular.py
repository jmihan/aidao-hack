"""
Предобработка данных и Feature Engineering.

Этот скрипт выполняет следующие шаги:
1. Загружает сырые данные.
2. Проводит базовую очистку и предобработку (удаление колонок, обработка пропусков).
3. Разбивает данные на обучающую, валидационную и тестовую выборки.
4. Генерирует новые признаки на основе временной динамики (лаги, скользящие статистики) для каждой выборки.
5. Генерирует признаки-взаимодействия (дельты, отношения) для каждой выборки.
6. Выполняет кластеризацию и масштабирование для каждой выборки.
7. Сохраняет обработанные датасеты, готовые для обучения модели.
"""
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle # Для сохранения scaler

# ==============================================================================
# КОНФИГУРАЦИЯ
# ==============================================================================

# --- Пути к файлам ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'frames_errors.csv'
OUTPUT_TRAIN_DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'featured_train_dataset.csv'
OUTPUT_VALID_DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'featured_valid_dataset.csv'
OUTPUT_TEST_DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'featured_test_dataset.csv'
SCALER_OUTPUT_PATH = PROJECT_ROOT / 'data' / 'processed' / 'scaler.pkl'

# --- Параметры обработки ---
COLUMNS_TO_DROP_INITIAL = ['nTot', 'estimator_name', "E_mu_phys_est", "f_EC",]

COLS_TO_DROP_FROM_FEATURES = [
  'id',
  'date',
  'E_mu_phys_est', # уже должна быть удалена
  'synErr',
  'N_EC_rounds',
  'maintenance_flag',
  'estimator_name', # уже должна быть удалена
  'f_EC', # уже должна быть удалена
  'E_mu_Z_est',
  'R',
  's',
  'p'
]

# Ключевые предикторы, для которых будем генерировать временные признаки.
KEY_PREDICTORS = [
    'E_mu_Z',     
    'E_nu1_Z',     
    'E_mu_Z_est',
    'M_nu1_XX',
    'M_nu2_XX',
    'M_mu_XX',
    'polarizerVoltages[1]',
    'polarizerVoltages[2]',
    'temp_1',
    'biasVoltage_1',
    'N_EC_rounds',
    's',
    'p',
]

# --- Параметры генерации признаков ---
LAG_STEPS = [1, 2, 3, 5]
ROLLING_WINDOWS = [5, 10]

# --- Параметры разбиения на выборки ---
HORIZON = 8
HISTORY = 160

TARGET_COLUMN = 'E_mu_Z'


# ==============================================================================
# ЗАГРУЗКА ДАННЫХ
# ==============================================================================
print("1. Загрузка сырых данных...")
try:
    df_raw = pd.read_csv(INPUT_DATA_PATH)
    print(f"  - Данные успешно загружены. Форма: {df_raw.shape}")
    df_raw.columns = [
        "block_id",
        "frame_idx",
        "E_mu_Z",
        "E_mu_phys_est",
        "E_mu_X",
        "E_nu1_X",
        "E_nu2_X",
        "E_nu1_Z",
        "E_nu2_Z",
        "N_mu_X",
        "M_mu_XX",
        "M_mu_XZ",
        "M_mu_X",
        "N_mu_Z",
        "M_mu_ZZ",
        "M_mu_Z",
        "N_nu1_X",
        "M_nu1_XX",
        "M_nu1_XZ",
        "M_nu1_X",
        "N_nu1_Z",
        "M_nu1_ZZ",
        "M_nu1_Z",
        "N_nu2_X",
        "M_nu2_XX",
        "M_nu2_XZ",
        "M_nu2_X",
        "N_nu2_Z",
        "M_nu2_ZZ",
        "M_nu2_Z",
        "nTot",
        "bayesImVoltage",
        "opticalPower",
        "polarizerVoltages[0]",
        "polarizerVoltages[1]",
        "polarizerVoltages[2]",
        "polarizerVoltages[3]",
        "temp_1",
        "biasVoltage_1",
        "temp_2",
        "biasVoltage_2",
        "synErr",
        "N_EC_rounds",
        "maintenance_flag",
        "estimator_name",
        "f_EC",
        "E_mu_Z_est",
        "R",
        "s",
        "p",
    ]
except FileNotFoundError:
    print(f"Ошибка: Файл не найден по пути '{INPUT_DATA_PATH}'. Прерывание работы.")
    exit()

# ==============================================================================
# ПРЕДОБРАБОТКА И ОЧИСТКА (Базовая)
# ==============================================================================
print("\n2. Базовая предобработка и очистка данных...")

# --- Удаление ненужных колонок ---
df_raw.drop(columns=COLUMNS_TO_DROP_INITIAL, inplace=True, errors='ignore')
print(f"  - Колонки {COLUMNS_TO_DROP_INITIAL} удалены.")

df_raw = df_raw.rename(
    columns={
        "block_id": "id",
        "frame_idx": "date",
    }
)
df_raw.sort_values(by=['id', 'date'], inplace=True)

# ==============================================================================
# РАЗБИЕНИЕ НА TRAIN/VALID/TEST
# ==============================================================================
print("\n3. Разбиение данных на Train/Validation/Test...")

train_dfs = []
valid_dfs = []
test_dfs = []

for current_id in df_raw["id"].unique():
    current_df = df_raw[df_raw["id"] == current_id].copy()
    
    if len(current_df) > HISTORY + HORIZON * 2: 
        test_split_idx = len(current_df) - HORIZON
        df_for_train_val = current_df.iloc[:test_split_idx]
        df_test = current_df.iloc[test_split_idx - HISTORY:]
        
        valid_split_idx = len(df_for_train_val) - HORIZON
        df_train = df_for_train_val.iloc[:valid_split_idx]
        df_valid = df_for_train_val.iloc[valid_split_idx - HISTORY:]
        
        train_dfs.append(df_train)
        valid_dfs.append(df_valid)
        test_dfs.append(df_test)
    elif len(current_df) > HISTORY + HORIZON: 
        df_train = current_df.iloc[:-HORIZON]
        df_valid = current_df.iloc[-HORIZON - HISTORY:]
        train_dfs.append(df_train)
        valid_dfs.append(df_valid)
    else:
        print(f"  - Пропускаем id={current_id}: недостаточно данных для разбиения (длина: {len(current_df)})")


df_train = pd.concat(train_dfs, ignore_index=True)
df_valid = pd.concat(valid_dfs, ignore_index=True)
df_test = pd.concat(test_dfs, ignore_index=True)

print(f"  - Данные разделены. Train: {len(df_train)} строк, Valid: {len(df_valid)} строк, Test: {len(df_test)} строк.")

# ==============================================================================
# ФУНКЦИЯ ДЛЯ ГЕНЕРАЦИИ ПРИЗНАКОВ (Применяется отдельно к каждой выборке)
# ==============================================================================
def generate_features_for_subset(df_subset, name="Subset"):
    print(f"\n4. Генерация новых признаков для {name}...")
    df = df_subset.copy() 

    # --- 4.1 Контекстные признаки (Режимы работы) ---
    print(f"  - Создание контекстных признаков для {name}...")
    # Проверяем наличие 'maintenance_flag' перед использованием
    if 'maintenance_flag' in df.columns:
        df['time_since_maintenance'] = df.groupby('id')['maintenance_flag'].transform(
            lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1
        )
        df.loc[df['maintenance_flag'] == 0, 'time_since_maintenance'] = 0
    else:
        print(f"    - Колонка 'maintenance_flag' отсутствует в {name} данных, контекстный признак 'time_since_maintenance' не будет создан.")
        df['time_since_maintenance'] = 0 # Заполняем нулями или другим значением по умолчанию

    physical_features = [
        'opticalPower', 'polarizerVoltages[0]', 'polarizerVoltages[1]',
        'polarizerVoltages[2]', 'polarizerVoltages[3]', 'temp_1',
        'biasVoltage_1', 'temp_2', 'biasVoltage_2'
    ]
    # Фильтруем physical_features, оставляя только те, что есть в текущем df
    physical_features = [f for f in physical_features if f in df.columns]

    if name == "Train":
        if physical_features: # Кластеризуем только если есть физические признаки
            scaler_physical = StandardScaler()
            df_physical_scaled = pd.DataFrame(scaler_physical.fit_transform(df[physical_features]), index=df.index)
            kmeans_model = KMeans(n_clusters=4, random_state=42, n_init='auto')
            df['physical_cluster'] = kmeans_model.fit_predict(df_physical_scaled)
            anomalous_cluster_id = df['physical_cluster'].value_counts().idxmin()
            df['is_anomalous_cluster'] = (df['physical_cluster'] == anomalous_cluster_id).astype(int)
            
            generate_features_for_subset.scaler_physical = scaler_physical
            generate_features_for_subset.kmeans_model = kmeans_model
            generate_features_for_subset.anomalous_cluster_id = anomalous_cluster_id
        else:
            print(f"    - В {name} отсутствуют физические признаки для кластеризации.")
            df['physical_cluster'] = -1
            df['is_anomalous_cluster'] = 0
    else:
        if physical_features and hasattr(generate_features_for_subset, 'kmeans_model'):
            df_physical_scaled = pd.DataFrame(generate_features_for_subset.scaler_physical.transform(df[physical_features]), index=df.index)
            df['physical_cluster'] = generate_features_for_subset.kmeans_model.predict(df_physical_scaled)
            df['is_anomalous_cluster'] = (df['physical_cluster'] == generate_features_for_subset.anomalous_cluster_id).astype(int)
        else:
            print(f"    - В {name} отсутствуют физические признаки или модель KMeans не обучена.")
            df['physical_cluster'] = -1
            df['is_anomalous_cluster'] = 0


    # --- 4.2 Временные признаки (Динамика) ---
    print(f"  - Создание временных признаков для {name}...")
    all_numeric_features = df.select_dtypes(include=np.number).columns.tolist()
    
    # Расширяем список исключаемых признаков для динамики
    dynamic_exclude_cols = ['id', 'date'] + COLS_TO_DROP_FROM_FEATURES
    features_for_dynamics = [
        f for f in all_numeric_features if f not in dynamic_exclude_cols
    ]
    # 's' может быть полезным для предсказания самого себя, если он не был исключен COLS_TO_DROP_FROM_FEATURES
    if 's' in df.columns and 's' not in features_for_dynamics and 's' not in COLS_TO_DROP_FROM_FEATURES:
        features_for_dynamics.append('s') 

    for col in tqdm(features_for_dynamics, desc=f"  Динамика ({name})"):
        if col not in df.columns:
            continue # Пропускаем, если колонка отсутствует после предыдущих удалений
        # Лаги
        for lag in LAG_STEPS:
            df[f'{col}_lag_{lag}'] = df.groupby('id')[col].shift(lag)
        
        # Скользящие статистики
        for win in ROLLING_WINDOWS:
            rolling_group = df.groupby('id')[col].rolling(window=win, min_periods=1)
            df[f'{col}_roll_mean_{win}'] = rolling_group.mean().reset_index(level=0, drop=True)
            df[f'{col}_roll_std_{win}'] = rolling_group.std().reset_index(level=0, drop=True)
            df[f'{col}_roll_max_{win}'] = rolling_group.max().reset_index(level=0, drop=True)
            df[f'{col}_roll_min_{win}'] = rolling_group.min().reset_index(level=0, drop=True)

        # Разница с предыдущим значением (моментум)
        df[f'{col}_diff'] = df.groupby('id')[col].diff()
        
        # EWMA (Экспоненциально взвешенное скользящее среднее)
        ewm = df.groupby('id')[col].ewm(span=5).mean()
        df[f'{col}_ewm_5'] = ewm.reset_index(level=0, drop=True)


    # --- 4.3 Признаки-взаимодействия (Отношения и Дельты) ---
    print(f"  - Создание признаков-взаимодействий для {name}...")
    epsilon = 1e-9

    # Проверяем наличие колонок перед созданием взаимодействий
    if 'E_mu_Z' in df.columns and 'E_mu_Z_est' in df.columns:
        df['delta_err_est'] = df['E_mu_Z'] - df['E_mu_Z_est']
    if 's' in df.columns and 'p' in df.columns:
        df['s_p_ratio'] = df['s'] / (df['p'] + epsilon)

    error_pairs = [
        ('E_mu_Z_est', 'E_nu1_Z'), ('E_mu_Z_est', 'E_nu2_Z'),
        ('E_mu_X', 'E_nu1_X'), ('E_mu_X', 'E_nu2_X')
    ]
    for e1, e2 in error_pairs:
        if e1 in df.columns and e2 in df.columns:
            df[f'ratio_{e1}_div_{e2}'] = df[e1] / (df[e2] + epsilon)

    count_pairs = [
        ('N_mu_X', 'N_nu1_X'), ('N_mu_Z', 'N_nu1_Z'),
        ('M_mu_XX', 'M_nu1_XX'), ('M_mu_ZZ', 'M_nu1_ZZ')
    ]
    for c1, c2 in count_pairs:
        if c1 in df.columns and c2 in df.columns:
            df[f'ratio_{c1}_div_{c2}'] = df[c1] / (df[c2] + epsilon)

    # Взаимодействия ключевых физических параметров с оценкой ошибки
    physical_interaction_cols = ['temp_1', 'temp_2', 'opticalPower', 'biasVoltage_1', 'biasVoltage_2']
    if 'E_mu_Z_est' in df.columns:
        for col in physical_interaction_cols:
            if col in df.columns:
                df[f'inter_{col}_x_E_mu_Z_est'] = df[col] * df['E_mu_Z_est']


    # --- 4.4 Признаки на уровне сессии (Агрегаты по 'id') ---
    print(f"  - Создание признаков на уровне сессии для {name}...")
    session_features = [
        'E_mu_Z_est', 'E_mu_X', 'temp_1', 'temp_2', 'opticalPower', 's'
    ]
    # Фильтруем session_features, оставляя только те, что есть в текущем df
    session_features = [f for f in session_features if f in df.columns]

    for col in tqdm(session_features, desc=f"  Агрегаты по ID ({name})"):
        grouped = df.groupby('id')[col]
        df[f'{col}_mean_per_id'] = grouped.transform('mean')
        df[f'{col}_std_per_id'] = grouped.transform('std')
        df[f'{col}_max_per_id'] = grouped.transform('max')
        df[f'{col}_min_per_id'] = grouped.transform('min')
        df[f'{col}_norm_by_id_mean'] = df[col] / (df[f'{col}_mean_per_id'] + epsilon)
        df[f'{col}_dist_from_id_mean'] = df[col] - df[f'{col}_mean_per_id']

    print(f"  - Генерация признаков для {name} завершена.")
    
    # Удаляем исходные физические признаки, т.к. мы их использовали для кластеризации
    df.drop(columns=physical_features, errors='ignore', inplace=True)

    return df

# Применяем функцию генерации признаков к каждой выборке
df_train_processed = generate_features_for_subset(df_train, name="Train")
df_valid_processed = generate_features_for_subset(df_valid, name="Validation")
df_test_processed = generate_features_for_subset(df_test, name="Test")


# ==============================================================================
# ФИНАЛЬНАЯ ОБРАБОТКА И МАСШТАБИРОВАНИЕ
# ==============================================================================
print("\n5. Финальная обработка и масштабирование...")

# Список всех обработанных датафреймов
processed_dfs = {
    'train': df_train_processed, 
    'valid': df_valid_processed, 
    'test': df_test_processed
}

all_numeric_features_after_fe = df_train_processed.select_dtypes(include=np.number).columns.tolist()

feature_cols_for_scaling = [
    col for col in all_numeric_features_after_fe 
    if col not in COLS_TO_DROP_FROM_FEATURES and col != TARGET_COLUMN
]

if TARGET_COLUMN in df_train_processed.columns and TARGET_COLUMN not in feature_cols_for_scaling:
    feature_cols_for_scaling.insert(0, TARGET_COLUMN) 
elif TARGET_COLUMN not in df_train_processed.columns:
    print(f"  - Внимание: Целевая колонка '{TARGET_COLUMN}' отсутствует в данных. Масштабирование может быть некорректным.")


print(f"  - Определены признаки для масштабирования ({len(feature_cols_for_scaling)}): {feature_cols_for_scaling[:5]}...")

scaler = StandardScaler()

for name, df_subset in processed_dfs.items():
    print(f"  - Обработка и масштабирование для {name}...")
    
    df_subset.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_subset.ffill(inplace=True)
    df_subset.dropna(inplace=True)

    if feature_cols_for_scaling: 
        if name == 'train':
            df_subset.loc[:, feature_cols_for_scaling] = scaler.fit_transform(df_subset[feature_cols_for_scaling])
        else:
            df_subset.loc[:, feature_cols_for_scaling] = scaler.transform(df_subset[feature_cols_for_scaling])
    else:
        print(f"    - Внимание: Нет признаков для масштабирования в {name}.")
    
    processed_dfs[name] = df_subset

print("  - Финальная обработка и масштабирование завершены.")


# ==============================================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ==============================================================================
print("\n6. Сохранение обработанных датасетов и скейлера...")

OUTPUT_TRAIN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
processed_dfs['train'].to_csv(OUTPUT_TRAIN_DATA_PATH, index=False)
print(f"  - Обучающий датасет сохранен в: '{OUTPUT_TRAIN_DATA_PATH}'. Форма: {processed_dfs['train'].shape}")

processed_dfs['valid'].to_csv(OUTPUT_VALID_DATA_PATH, index=False)
print(f"  - Валидационный датасет сохранен в: '{OUTPUT_VALID_DATA_PATH}'. Форма: {processed_dfs['valid'].shape}")

processed_dfs['test'].to_csv(OUTPUT_TEST_DATA_PATH, index=False)
print(f"  - Тестовый датасет сохранен в: '{OUTPUT_TEST_DATA_PATH}'. Форма: {processed_dfs['test'].shape}")

SCALER_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(SCALER_OUTPUT_PATH, 'wb') as f:
    pickle.dump(scaler, f)
print(f"  - StandardScaler сохранен в: '{SCALER_OUTPUT_PATH}'")

print("\nВсе шаги завершены.")