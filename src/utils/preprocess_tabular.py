import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'frames_errors.csv'
OUTPUT_DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'featured_dataset.csv'

COLUMNS_TO_DROP = ['nTot', 'estimator_name', "E_mu_phys_est", "f_EC",]

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

LAG_STEPS = [1, 2, 3, 5]

ROLLING_WINDOWS = [5, 10]

print("1. Загрузка сырых данных...")
try:
    df = pd.read_csv(INPUT_DATA_PATH)
    print(f"  - Данные успешно загружены. Форма: {df.shape}")
    df.columns = [
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

print("\n2. Предобработка и очистка данных...")

df.drop(columns=COLUMNS_TO_DROP, inplace=True)
print(f"  - Колонки {COLUMNS_TO_DROP} удалены.")

df = df.rename(
    columns={
        "block_id": "id",
        "frame_idx": "date",
    }
)

print("\n3. Генерация новых признаков...")
tqdm.pandas(desc="Прогресс")

df.sort_values(by=['id', 'date'], inplace=True)

# --- Контекстные признаки (Режимы работы) ---
print("  - Создание контекстных признаков...")
df['time_since_maintenance'] = df.groupby('id')['maintenance_flag'].transform(
    lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1
)
df.loc[df['maintenance_flag'] == 0, 'time_since_maintenance'] = 0

physical_features = [
    'opticalPower', 'polarizerVoltages[0]', 'polarizerVoltages[1]',
    'polarizerVoltages[2]', 'polarizerVoltages[3]', 'temp_1',
    'biasVoltage_1', 'temp_2', 'biasVoltage_2'
]
scaler = StandardScaler()
df_physical_scaled = pd.DataFrame(scaler.fit_transform(df[physical_features]), index=df.index)

kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
df['physical_cluster'] = kmeans.fit_predict(df_physical_scaled)
anomalous_cluster_id = df['physical_cluster'].value_counts().idxmin()
df['is_anomalous_cluster'] = (df['physical_cluster'] == anomalous_cluster_id).astype(int)


# --- Временные признаки (Динамика) ---
print("  - Создание временных признаков...")
all_numeric_features = df.select_dtypes(include=np.number).columns.tolist()
features_for_dynamics = [
    f for f in all_numeric_features if f not in [
        'id', 'date', 'R', 's', 'p', 'N_EC_rounds', 'E_mu_Z'
    ]
]
features_for_dynamics.append('s') 

for col in tqdm(features_for_dynamics, desc="  Динамика"):
    for lag in LAG_STEPS:
        df[f'{col}_lag_{lag}'] = df.groupby('id')[col].shift(lag)
    
    for win in ROLLING_WINDOWS:
        rolling_group = df.groupby('id')[col].rolling(window=win, min_periods=1)
        df[f'{col}_roll_mean_{win}'] = rolling_group.mean().reset_index(level=0, drop=True)
        df[f'{col}_roll_std_{win}'] = rolling_group.std().reset_index(level=0, drop=True)
        df[f'{col}_roll_max_{win}'] = rolling_group.max().reset_index(level=0, drop=True)
        df[f'{col}_roll_min_{win}'] = rolling_group.min().reset_index(level=0, drop=True)

    df[f'{col}_diff'] = df.groupby('id')[col].diff()
    
    ewm = df.groupby('id')[col].ewm(span=5).mean()
    df[f'{col}_ewm_5'] = ewm.reset_index(level=0, drop=True)


# --- Признаки-взаимодействия (Отношения и Дельты) ---
print("  - Создание признаков-взаимодействий...")
epsilon = 1e-9

df['delta_err_est'] = df['E_mu_Z'] - df['E_mu_Z_est']
df['s_p_ratio'] = df['s'] / (df['p'] + epsilon)

error_pairs = [
    ('E_mu_Z_est', 'E_nu1_Z'), ('E_mu_Z_est', 'E_nu2_Z'),
    ('E_mu_X', 'E_nu1_X'), ('E_mu_X', 'E_nu2_X')
]
for e1, e2 in error_pairs:
    df[f'ratio_{e1}_div_{e2}'] = df[e1] / (df[e2] + epsilon)

count_pairs = [
    ('N_mu_X', 'N_nu1_X'), ('N_mu_Z', 'N_nu1_Z'),
    ('M_mu_XX', 'M_nu1_XX'), ('M_mu_ZZ', 'M_nu1_ZZ')
]
for c1, c2 in count_pairs:
    df[f'ratio_{c1}_div_{c2}'] = df[c1] / (df[c2] + epsilon)

for col in ['temp_1', 'temp_2', 'opticalPower', 'biasVoltage_1', 'biasVoltage_2']:
    df[f'inter_{col}_x_E_mu_Z_est'] = df[col] * df['E_mu_Z_est']


# --- Признаки на уровне сессии (Агрегаты по 'id') ---
print("  - Создание признаков на уровне сессии...")
session_features = [
    'E_mu_Z_est', 'E_mu_X', 'temp_1', 'temp_2', 'opticalPower', 's'
]

for col in tqdm(session_features, desc="  Агрегаты по ID"):
    grouped = df.groupby('id')[col]
    df[f'{col}_mean_per_id'] = grouped.transform('mean')
    df[f'{col}_std_per_id'] = grouped.transform('std')
    df[f'{col}_max_per_id'] = grouped.transform('max')
    df[f'{col}_min_per_id'] = grouped.transform('min')
    df[f'{col}_norm_by_id_mean'] = df[col] / (df[f'{col}_mean_per_id'] + epsilon)
    df[f'{col}_dist_from_id_mean'] = df[col] - df[f'{col}_mean_per_id']

print("  - Генерация признаков завершена.")

print("\n4. Завершение...")

final_na_count = df.isnull().sum().sum()
print(f"  - В итоговом датасете {final_na_count} пропусков (это ожидаемо и будет обработано CatBoost).")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.drop(columns=physical_features, errors='ignore')

OUTPUT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_DATA_PATH, index=False)
print(f"  - Обработанный датасет сохранен в: '{OUTPUT_DATA_PATH}'")
print(f"  - Итоговая форма датасета: {df.shape}")
