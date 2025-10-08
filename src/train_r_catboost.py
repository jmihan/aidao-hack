"""
Обучение модели CatBoost для предсказания R с подбором гиперпараметров с помощью Optuna.

Этот скрипт:
1. Загружает датасет с признаками.
2. Выполняет стратифицированное разделение данных.
3. Запускает исследование Optuna для поиска лучших гиперпараметров.
4. Обучает финальную модель CatBoostClassifier на найденных параметрах.
5. Оценивает качество финальной модели и сохраняет артефакты.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from catboost import CatBoostClassifier
import optuna

optuna.logging.set_verbosity(optuna.logging.INFO)

SESSION_ID_COL = 'id'
TARGET_COLUMN_R = 'R'
COLS_TO_DROP_FROM_FEATURES = [TARGET_COLUMN_R, 's', 'p', SESSION_ID_COL, 'date', 'E_mu_Z_est']
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42

def load_and_prepare_data(input_path: Path):
    """Загружает данные, объединяет редкие классы R и выполняет стратифицированное разделение."""

    print("1. Загрузка и подготовка данных...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Ошибка: Файл '{input_path}' не найден.")
        return None, None, None, None, None
    print(f"  - Данные успешно загружены. Форма: {df.shape}")
    print("\n2. Объединение редких классов R...")
    df['R_grouped'] = df[TARGET_COLUMN_R]
    classes_to_merge = [0.55, 0.60, 0.65, 0.70]
    new_class_value = 0.60
    df.loc[df['R_grouped'].isin(classes_to_merge), 'R_grouped'] = new_class_value
    target_column_grouped = 'R_grouped'
    all_possible_classes = sorted(df[target_column_grouped].unique())
    print(f"  - Новые классы для предсказания: {all_possible_classes}")
    print("\n3. Выполнение стратифицированного разделения по id...")
    block_modes = df.groupby(SESSION_ID_COL)[target_column_grouped].agg(lambda x: x.mode().iloc[0]).reset_index()
    block_modes.columns = [SESSION_ID_COL, 'R_mode']
    mode_counts = block_modes['R_mode'].value_counts()
    single_instance_modes = mode_counts[mode_counts == 1].index
    unique_session_blocks = block_modes[block_modes['R_mode'].isin(single_instance_modes)]
    safe_session_blocks = block_modes[~block_modes['R_mode'].isin(single_instance_modes)]
    if not unique_session_blocks.empty:
        print(f"  - Обнаружено {len(unique_session_blocks)} сессий с уникальными модальными классами, добавлены в train.")
    train_safe_blocks, val_blocks = train_test_split(
        safe_session_blocks, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE, stratify=safe_session_blocks['R_mode'])
    train_blocks = pd.concat([train_safe_blocks, unique_session_blocks], ignore_index=True)
    train_ids = train_blocks[SESSION_ID_COL]
    val_ids = val_blocks[SESSION_ID_COL]
    train_df = df[df[SESSION_ID_COL].isin(train_ids)].copy()
    val_df = df[df[SESSION_ID_COL].isin(val_ids)].copy()
    print(f"  - Количество сессий в train: {len(train_ids)}")
    print(f"  - Количество сессий в validation: {len(val_ids)}")
    cols_to_drop = COLS_TO_DROP_FROM_FEATURES + ['R_grouped']
    if TARGET_COLUMN_R not in cols_to_drop: cols_to_drop.append(TARGET_COLUMN_R)
    X_train = train_df.drop(columns=cols_to_drop)
    y_train = train_df[target_column_grouped]
    X_val = val_df.drop(columns=cols_to_drop)
    y_val = val_df[target_column_grouped]
    print(f"  - Количество признаков: {X_train.shape[1]}")
    return X_train, y_train, X_val, y_val, all_possible_classes

def objective(trial, X_train, y_train, X_val, y_val):
    """Целевая функция для Optuna."""

    params = {
        'iterations': 2000,
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        'loss_function': 'MultiClass',
        'eval_metric': 'TotalF1',
        'random_seed': RANDOM_STATE,
        'verbose': 0, 
        'early_stopping_rounds': 100,
        'task_type': "GPU",
        'devices': '0',
        'auto_class_weights': 'Balanced'
    }
    
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    
    preds = model.predict(X_val)

    y_val_int = (y_val * 100).astype(int)
    preds_int = (preds[:, 0] * 100).astype(int)

    f1 = f1_score(y_val_int, preds_int, average='weighted', zero_division=0)
    
    return f1

def find_best_params(X_train, y_train, X_val, y_val, n_trials=10):
    """Запускает исследование Optuna для поиска лучших параметров."""

    print(f"\n4. Запуск Optuna для поиска лучших гиперпараметров ({n_trials} попыток)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=n_trials)
    
    print("  - Поиск завершен!")
    print(f"  - Лучший Weighted F1-score: {study.best_value:.4f}")
    print("  - Лучшие параметры:")
    for key, value in study.best_params.items():
        print(f"    - {key}: {value}")
        
    return study.best_params

def train_final_model(X_train, y_train, X_val, y_val, best_params):
    """Обучает модель CatBoostClassifier на лучших параметрах."""

    print("\n5. Обучение модели CatBoost для R на лучших параметрах...")
    
    final_params = {
        'iterations': 3000, 
        'loss_function': 'MultiClass',
        'eval_metric': 'TotalF1',
        'random_seed': RANDOM_STATE,
        'verbose': 200,
        'early_stopping_rounds': 150,
        'task_type': "GPU",
        'devices': '0',
        'auto_class_weights': 'Balanced'
    }
    final_params.update(best_params) 

    model = CatBoostClassifier(**final_params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    
    return model

def evaluate_and_save_artifacts(model, X_val, y_val, class_names, output_dir: Path):
    """Оценивает модель и сохраняет артефакты."""

    print("\n6. Оценка финальной модели R и сохранение артефактов...")
    output_dir.mkdir(parents=True, exist_ok=True)
    y_pred = model.predict(X_val)
    y_val_int = (y_val * 100).astype(int)
    y_pred_int = (y_pred[:, 0] * 100).astype(int)
    class_names_int = [(c * 100).astype(int) for c in class_names]
    target_names_str = [str(c) for c in class_names]
    report = classification_report(y_val_int, y_pred_int, labels=class_names_int, target_names=target_names_str, zero_division=0)
    report_path = output_dir / 'R_classification_report_optuna.txt'
    with open(report_path, 'w', encoding='utf-8') as f: f.write(report)
    print(f"  - Отчет о классификации сохранен в: {report_path}")
    print(report)
    cm = confusion_matrix(y_val_int, y_pred_int, labels=class_names_int)
    plt.figure(figsize=(10, 8)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names_str, yticklabels=target_names_str)
    plt.title('Матрица ошибок для модели R (Optuna)'); plt.ylabel('Истинный класс'); plt.xlabel('Предсказанный класс')
    cm_path = output_dir / 'R_confusion_matrix_optuna.png'; plt.savefig(cm_path); plt.close()
    print(f"  - Матрица ошибок сохранена в: {cm_path}")
    feature_importance_df = pd.DataFrame({'feature': model.feature_names_, 'importance': model.get_feature_importance()}).sort_values(by='importance', ascending=False)
    plt.figure(figsize=(12, 10)); sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20))
    plt.title('Топ-20 признаков для модели R (Optuna)'); fi_path = output_dir / 'R_feature_importance_optuna.png'; plt.savefig(fi_path, bbox_inches='tight'); plt.close()
    print(f"  - График важности признаков сохранен в: {fi_path}")
    model_path = output_dir / 'catboost_R_model_optuna.cbm'; model.save_model(str(model_path))
    print(f"  - Обученная модель для R сохранена в: {model_path}")

def main():
    """Главная функция для запуска всего пайплайна обучения модели R с Optuna."""

    parser = argparse.ArgumentParser(description="Скрипт для обучения модели R с Optuna")
    
    project_root = Path(__file__).resolve().parent.parent
    
    parser.add_argument('--input', type=str, default=str(project_root / 'data' / 'processed' / 'featured_dataset.csv'))
    parser.add_argument('--output', type=str, default=str(project_root / 'models'))
    parser.add_argument('--n-trials', type=int, default=50, help='Количество итераций для подбора параметров Optuna.')
    
    args = parser.parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    n_trials = args.n_trials

    X_train, y_train, X_val, y_val, class_names = load_and_prepare_data(input_path)
    
    if X_train is not None:
        best_params = find_best_params(X_train, y_train, X_val, y_val, n_trials)
        final_model = train_final_model(X_train, y_train, X_val, y_val, best_params)
        evaluate_and_save_artifacts(final_model, X_val, y_val, class_names, output_dir)
        print("\nПайплайн обучения модели R с Optuna успешно завершен!")

if __name__ == '__main__':
    main()