"""
Предобработка данных и Feature Engineering.

Этот скрипт выполняет следующие шаги:
1. Загружает сырые данные.
2. Проводит базовую очистку и предобработку (удаление колонок, обработка пропусков).
3. Генерирует новые признаки на основе временной динамики (лаги, скользящие статистики).
4. Генерирует признаки-взаимодействия (дельты, отношения).
5. Сохраняет обработанный датасет, готовый для обучения модели.
"""
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path

# ==============================================================================
# КОНФИГУРАЦИЯ
# ==============================================================================

# --- Пути к файлам ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'frames_errors.csv'
OUTPUT_DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'featured_dataset.csv'


# --- Параметры обработки ---
COLUMNS_TO_DROP = ['nTot', 'estimator_name', "E_mu_phys_est", "f_EC",]

# Ключевые предикторы, для которых будем генерировать временные признаки.
# Выбраны на основе EDA (высокая корреляция, "режимные" признаки).
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

# Размеры окон для скользящих статистик (короткое и среднее окно)
ROLLING_WINDOWS = [5, 10]


# ==============================================================================
# ЗАГРУЗКА ДАННЫХ
# ==============================================================================
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


# ==============================================================================
# ПРЕДОБРАБОТКА И ОЧИСТКА
# ==============================================================================
print("\n2. Предобработка и очистка данных...")

# --- Удаление ненужных колонок ---
df.drop(columns=COLUMNS_TO_DROP, inplace=True)
print(f"  - Колонки {COLUMNS_TO_DROP} удалены.")

df = df.rename(
    columns={
        "block_id": "id",
        "frame_idx": "date",
    }
)

# ==============================================================================
# ГЕНЕРАЦИЯ ПРИЗНАКОВ (FEATURE ENGINEERING)
# ==============================================================================
print("\n3. Генерация новых признаков...")
tqdm.pandas(desc="Генерация признаков")

# Сортируем данные на всякий случай, чтобы временные операции были корректны
df.sort_values(by=['id', 'date'], inplace=True)

# --- Временные признаки (Лаги) ---
print("  - Создание лаговых признаков...")
for col in tqdm(KEY_PREDICTORS, desc="  Лаги"):
    for lag in LAG_STEPS:
        df[f'{col}_lag_{lag}'] = df.groupby('id')[col].shift(lag)

# --- Временные признаки (Скользящие статистики) ---
print("  - Создание скользящих статистик...")
for col in tqdm(KEY_PREDICTORS, desc="  Скользящие стат."):
    for win in ROLLING_WINDOWS:
        rolling_group = df.groupby('id')[col].rolling(window=win, min_periods=1)
        
        df[f'{col}_roll_mean_{win}'] = rolling_group.mean().reset_index(level=0, drop=True)
        df[f'{col}_roll_std_{win}'] = rolling_group.std().reset_index(level=0, drop=True)
        df[f'{col}_roll_max_{win}'] = rolling_group.max().reset_index(level=0, drop=True)

# --- Признаки-взаимодействия ("Дельты" и "Отношения") ---
print("  - Создание признаков-взаимодействий...")
# Разница между реальной ошибкой и ее оценкой
df['delta_err_est'] = df['E_mu_Z'] - df['E_mu_Z_est']

# Отношение между счетчиками в разных состояниях (может помочь выявить "режим")
df['ratio_M_mu_nu1'] = df['M_mu_XX'] / (df['M_nu1_XX'] + 1e-6)
df['ratio_M_mu_nu2'] = df['M_mu_XX'] / (df['M_nu2_XX'] + 1e-6)

print("  - Генерация признаков завершена.")


# ==============================================================================
# ЗАВЕРШЕНИЕ И СОХРАНЕНИЕ
# ==============================================================================
print("\n4. Завершение...")

# --- Обработка пропусков, созданных после генерации признаков ---
final_na_count = df.isnull().sum().sum()
print(f"  - В датасете {final_na_count} пропусков, появившихся после генерации временных признаков.")

# --- Сохранение результата ---
df.to_csv(OUTPUT_DATA_PATH, index=False)
print(f"  - Обработанный датасет сохранен в: '{OUTPUT_DATA_PATH}'")
print(f"  - Итоговая форма датасета: {df.shape}")
