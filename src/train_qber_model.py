import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
import optuna

optuna.logging.set_verbosity(optuna.logging.INFO)

SESSION_ID_COL = 'id'
TARGET_COLUMN_QBER = 'E_mu_Z'

COLS_TO_DROP_FROM_FEATURES = [
    TARGET_COLUMN_QBER, 
    'R', 's', 'p',          
    'N_EC_rounds',          
    'delta_err_est',        
    SESSION_ID_COL, 
    'date'
]
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42

def load_and_prepare_data(input_path: Path):
    """Загружает данные и выполняет разделение по сессиям (id)"""

    print("1. Загрузка и подготовка данных...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Ошибка: Файл '{input_path}' не найден.")
        return None, None, None, None
    print(f"  - Данные успешно загружены. Форма: {df.shape}")

    print("\n2. Выполнение разделения по сессиям (id)...")
    session_ids = df[SESSION_ID_COL].unique()
    train_ids, val_ids = train_test_split(session_ids, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE)

    train_df = df[df[SESSION_ID_COL].isin(train_ids)].copy()
    val_df = df[df[SESSION_ID_COL].isin(val_ids)].copy()

    print(f"  - Количество сессий в train: {len(train_ids)}")
    print(f"  - Количество сессий в validation: {len(val_ids)}")

    X_train = train_df.drop(columns=COLS_TO_DROP_FROM_FEATURES)
    y_train = train_df[TARGET_COLUMN_QBER]
    X_val = val_df.drop(columns=COLS_TO_DROP_FROM_FEATURES)
    y_val = val_df[TARGET_COLUMN_QBER]
    
    print(f"  - Количество признаков: {X_train.shape[1]}")
    return X_train, y_train, X_val, y_val

def objective(trial, X_train, y_train, X_val, y_val):
    """Целевая функция для Optuna"""

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
        'devices': '0',
    }
    
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    
    return rmse

def find_best_params(X_train, y_train, X_val, y_val, n_trials=10):
    """Запускает исследование Optuna для поиска лучших параметров"""

    print(f"\n3. Запуск Optuna для поиска лучших гиперпараметров ({n_trials} попыток)...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=n_trials)
    
    print("  - Поиск завершен!")
    print(f"  - Лучший RMSE: {study.best_value:.6f}")
    print("  - Лучшие параметры:")
    for key, value in study.best_params.items():
        print(f"    - {key}: {value}")
        
    return study.best_params

def train_final_model(X_train, y_train, X_val, y_val, best_params):
    """Обучает модель CatBoostRegressor на лучших параметрах"""

    print("\n4. Обучение регрессионной модели для E_mu_Z на лучших параметрах...")
    
    final_params = {
        'iterations': 4000,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': RANDOM_STATE,
        'verbose': 200,
        'early_stopping_rounds': 150,
        'task_type': "GPU",
        'devices': '0',
    }
    final_params.update(best_params)

    model = CatBoostRegressor(**final_params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    
    return model

def evaluate_and_save_artifacts(model, X_val, y_val, output_dir: Path):
    """Оценивает регрессионную модель и сохраняет артефакты"""

    print("\n5. Оценка финальной модели E_mu_Z и сохранение артефактов...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    y_pred = model.predict(X_val)
    
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    metrics_report = f"""
    --- Отчет по метрикам регрессии для E_mu_Z ---
    RMSE (Root Mean Squared Error): {rmse:.6f}
    MAE (Mean Absolute Error):      {mae:.6f}
    R^2 (Coefficient of Det.):      {r2:.6f}
    """
    report_path = output_dir / 'QBER_regression_report_optuna.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(metrics_report)
    print(f"  - Отчет по метрикам сохранен в: {report_path}")
    print(metrics_report)

    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=y_val, y=y_pred, alpha=0.3, s=10)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], '--r', linewidth=2)
    plt.title('Предсказания vs. Реальные значения для E_mu_Z')
    plt.xlabel('Реальные значения')
    plt.ylabel('Предсказанные значения')
    scatter_path = output_dir / 'QBER_predicted_vs_actual_optuna.png'
    plt.savefig(scatter_path)
    plt.close()
    print(f"  - График 'Предсказания vs. Реальные' сохранен в: {scatter_path}")

    residuals = y_val - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=50, kde=True)
    plt.title('Распределение остатков (ошибок)')
    plt.xlabel('Ошибка (Реальное - Предсказанное)')
    residuals_path = output_dir / 'QBER_residuals_distribution_optuna.png'
    plt.savefig(residuals_path)
    plt.close()
    print(f"  - График распределения остатков сохранен в: {residuals_path}")

    feature_importance_df = pd.DataFrame({'feature': model.feature_names_, 'importance': model.get_feature_importance()}).sort_values(by='importance', ascending=False)
    plt.figure(figsize=(12, 10))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(25))
    plt.title('Топ-25 признаков для модели E_mu_Z')
    fi_path = output_dir / 'QBER_feature_importance_optuna.png'
    plt.savefig(fi_path, bbox_inches='tight')
    plt.close()
    print(f"  - График важности признаков сохранен в: {fi_path}")

    model_path = output_dir / 'catboost_QBER_model_optuna.cbm'
    model.save_model(str(model_path))
    print(f"  - Обученная модель для E_mu_Z сохранена в: {model_path}")

def main():
    """Главная функция для запуска всего пайплайна обучения модели E_mu_Z"""

    parser = argparse.ArgumentParser(description="Скрипт для обучения регрессионной модели E_mu_Z")
    
    project_root = Path(__file__).resolve().parent.parent
    
    parser.add_argument('--input', type=str, default=str(project_root / 'data' / 'processed' / 'featured_dataset.csv'))
    parser.add_argument('--output', type=str, default=str(project_root / 'models'))
    parser.add_argument('--n-trials', type=int, default=10, help='Количество итераций для подбора параметров Optuna.')
    
    args = parser.parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    n_trials = args.n_trials

    X_train, y_train, X_val, y_val = load_and_prepare_data(input_path)
    
    if X_train is not None:
        best_params = find_best_params(X_train, y_train, X_val, y_val, n_trials)
        final_model = train_final_model(X_train, y_train, X_val, y_val, best_params)
        evaluate_and_save_artifacts(final_model, X_val, y_val, output_dir)
        print("\nПайплайн обучения модели E_mu_Z успешно завершен!")

if __name__ == '__main__':
    main()