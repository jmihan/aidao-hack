"""
Скрипт для Фазы 2Б: Обучение модели CatBoost для предсказания s.

Этот скрипт:
1. Загружает датасет с признаками.
2. Загружает уже обученную модель для R.
3. Генерирует предсказания R и добавляет их как новый признак.
4. Выполняет стратифицированное разделение данных (по R_grouped).
5. Обучает модель CatBoostRegressor для предсказания s.
6. Оценивает качество регрессионной модели и сохраняет артефакты.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostClassifier, CatBoostRegressor # Импортируем регрессор
import optuna

optuna.logging.set_verbosity(optuna.logging.INFO)

# ==============================================================================
# КОНФИГУРАЦИЯ (почти без изменений)
# ==============================================================================
SESSION_ID_COL = 'id'
TARGET_COLUMN_R = 'R'
TARGET_COLUMN_S = 's' # Наша новая цель

# Признаки, которые не являются предикторами
# E_mu_Z_est удаляем, так как его не будет в тестовых данных
COLS_TO_DROP_FROM_FEATURES = [TARGET_COLUMN_R, TARGET_COLUMN_S, 'p', SESSION_ID_COL, 'date', 'E_mu_Z_est']

VALIDATION_SIZE = 0.2
RANDOM_STATE = 42

def load_and_prepare_data_for_s(input_path: Path, r_model_path: Path):
    """
    Загружает данные, генерирует предсказания R как признак и разделяет данные.
    """
    print("1. Загрузка данных и генерация признака R_pred...")
    try:
        df = pd.read_csv(input_path)
        r_model = CatBoostClassifier()
        r_model.load_model(r_model_path)
        print(f"  - Данные и модель R ('{r_model_path.name}') успешно загружены.")
    except Exception as e:
        print(f"Ошибка при загрузке данных или модели R: {e}")
        return None, None, None, None

    # Шаг 1: Подготовить признаки для R-модели.
    features_for_r_model = df.drop(columns=[TARGET_COLUMN_R, 's', 'p', SESSION_ID_COL, 'date'])
    features_for_r_model = features_for_r_model[r_model.feature_names_] 
    
    # Шаг 2: Сделать предсказание R и добавить его как новый признак.
    predicted_r = r_model.predict(features_for_r_model)
    df['R_pred'] = predicted_r[:, 0]
    print("  - Признак 'R_pred' успешно сгенерирован.")

    # Шаг 3: Создать колонку для стратификации (как и раньше).
    df['R_grouped'] = df[TARGET_COLUMN_R]
    classes_to_merge = [0.55, 0.60, 0.65, 0.70]
    df.loc[df['R_grouped'].isin(classes_to_merge), 'R_grouped'] = 0.60
    
    # Шаг 4: Выполнить стратифицированное разделение.
    print("\n2. Выполнение стратифицированного разделения по id...")
    block_modes = df.groupby(SESSION_ID_COL)['R_grouped'].agg(lambda x: x.mode().iloc[0]).reset_index()
    # Теперь `block_modes` имеет колонки ['id', 'R_grouped']
    
    # --- НАЧАЛО ИСПРАВЛЕНИЯ ---
    
    # Используем правильное имя колонки 'R_grouped' вместо 'R_mode'
    mode_counts = block_modes['R_grouped'].value_counts()
    
    single_instance_modes = mode_counts[mode_counts == 1].index
    
    unique_session_blocks = block_modes[block_modes['R_grouped'].isin(single_instance_modes)]
    
    safe_session_blocks = block_modes[~block_modes['R_grouped'].isin(single_instance_modes)]
    
    if not unique_session_blocks.empty:
        print(f"  - Обнаружено {len(unique_session_blocks)} сессий с уникальными модальными классами, добавлены в train.")

    train_safe_blocks, val_blocks = train_test_split(
        safe_session_blocks, 
        test_size=VALIDATION_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=safe_session_blocks['R_grouped']  # Стратифицируем тоже по 'R_grouped'
    )
    
    # --- КОНЕЦ ИСПРАВЛЕНИЯ ---
    
    train_blocks = pd.concat([train_safe_blocks, unique_session_blocks], ignore_index=True)
    train_ids = train_blocks[SESSION_ID_COL]
    val_ids = val_blocks[SESSION_ID_COL]
    train_df = df[df[SESSION_ID_COL].isin(train_ids)].copy()
    val_df = df[df[SESSION_ID_COL].isin(val_ids)].copy()
    
    print(f"  - Количество сессий в train: {len(train_ids)}")
    print(f"  - Количество сессий в validation: {len(val_ids)}")
    
    # Шаг 5: Сформировать X и y для модели s.
    final_cols_to_drop = [
        TARGET_COLUMN_R, 
        'R_grouped', 
        TARGET_COLUMN_S, 
        'p', 
        SESSION_ID_COL, 
        'date'
    ]
    
    X_train = train_df.drop(columns=final_cols_to_drop)
    y_train = train_df[TARGET_COLUMN_S]
    X_val = val_df.drop(columns=final_cols_to_drop)
    y_val = val_df[TARGET_COLUMN_S]
    
    print(f"  - Количество признаков для модели s: {X_train.shape[1]}")
    return X_train, y_train, X_val, y_val

# ==============================================================================
# НОВЫЙ БЛОК: ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ ДЛЯ S С OPTUNA
# ==============================================================================

def objective_s(trial, X_train, y_train, X_val, y_val):
    """Целевая функция для Optuna (регрессия s)."""
    params = {
        'iterations': 2000,
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': RANDOM_STATE,
        'verbose': 0,
        'early_stopping_rounds': 100,
        'task_type': "GPU",
        'devices': '0'
    }
    
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    
    return rmse # Optuna будет минимизировать RMSE

def find_best_params_for_s(X_train, y_train, X_val, y_val, n_trials=50):
    """Запускает исследование Optuna для поиска лучших параметров для модели s."""
    print(f"\n3. Запуск Optuna для поиска гиперпараметров s ({n_trials} попыток)...")
    # direction='minimize', так как мы минимизируем ошибку RMSE
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective_s(trial, X_train, y_train, X_val, y_val), n_trials=n_trials)
    
    print("  - Поиск завершен!")
    print(f"  - Лучший RMSE: {study.best_value:.4f}")
    print("  - Лучшие параметры:")
    for key, value in study.best_params.items():
        print(f"    - {key}: {value}")
        
    return study.best_params


def train_final_s_model(X_train, y_train, X_val, y_val, best_params):
    """Обучает финальную модель CatBoostRegressor на лучших параметрах."""
    print("\n4. Обучение финальной модели CatBoost для s на лучших параметрах...")
    
    final_params = {
        'iterations': 3000,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': RANDOM_STATE,
        'verbose': 200,
        'early_stopping_rounds': 150,
        'task_type': "GPU",
        'devices': '0'
    }
    final_params.update(best_params)

    model = CatBoostRegressor(**final_params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    return model

def evaluate_s_model(model, X_val, y_val, output_dir: Path):
    """Оценивает регрессионную модель и сохраняет артефакты."""
    print("\n4. Оценка модели s и сохранение артефактов...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    y_pred = model.predict(X_val)
    
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    
    print(f"  - RMSE на валидации: {rmse:.4f}")
    print(f"  - R^2 на валидации: {r2:.4f}")
    
    # Сохраняем метрики в файл
    report_path = output_dir / 's_regression_report_optuna.txt'
    with open(report_path, 'w') as f:
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R^2: {r2:.4f}\n")
    print(f"  - Отчет о регрессии сохранен в: {report_path}")

    # График "Предсказания vs. Факт"
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=y_val, y=y_pred, alpha=0.3)
    plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'r--', lw=2) # Идеальная линия
    plt.title('Предсказания vs. Фактические значения для s')
    plt.xlabel('Фактические значения s')
    plt.ylabel('Предсказанные значения s')
    plt.savefig(output_dir / 's_predictions_vs_actual_optuna.png')
    plt.close()
    
    # Сохранение модели
    model_path = output_dir / 'catboost_s_model_optuna.cbm'
    model.save_model(str(model_path))
    print(f"  - Обученная модель для s сохранена в: {model_path}")

def main():
    """Главная функция пайплайна обучения модели s с Optuna."""
    parser = argparse.ArgumentParser(description="Скрипт для обучения модели s с Optuna")
    
    project_root = Path(__file__).resolve().parent.parent
    
    parser.add_argument('--input', type=str, 
                        default=str(project_root / 'data' / 'processed' / 'featured_dataset.csv'))
    parser.add_argument('--r_model', type=str, 
                        default=str(project_root / 'models' / 'catboost_R_model_optuna.cbm'))
    parser.add_argument('--output', type=str, 
                        default=str(project_root / 'models'))
    parser.add_argument('--n-trials', type=int, default=50, help='Количество итераций для подбора параметров Optuna.')
    
    args = parser.parse_args()
    input_path = Path(args.input)
    r_model_path = Path(args.r_model)
    output_dir = Path(args.output)
    n_trials = args.n_trials

    # --- Запуск пайплайна ---
    X_train, y_train, X_val, y_val = load_and_prepare_data_for_s(input_path, r_model_path)
    
    if X_train is not None:
        # Шаг 1: Найти лучшие параметры
        best_s_params = find_best_params_for_s(X_train, y_train, X_val, y_val, n_trials)
        # Шаг 2: Обучить финальную модель
        final_s_model = train_final_s_model(X_train, y_train, X_val, y_val, best_s_params)
        # Шаг 3: Оценить и сохранить
        evaluate_s_model(final_s_model, X_val, y_val, output_dir)
        print("\nПайплайн обучения модели s с Optuna успешно завершен!")

if __name__ == '__main__':
    main()