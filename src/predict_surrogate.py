import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from catboost import CatBoostRegressor
from tqdm import tqdm

def find_optimal_params_for_row(row, surrogate_model, feature_names, alpha: float):
    """
    Находит оптимальные {R, s} для одной строки данных (одного фрейма)
    путем поиска по сетке с использованием суррогатной модели
    """
    best_params = {'R': -1, 's': -1, 'score': float('inf')}
    
    base_features = row[feature_names].to_dict()
    
    R_candidates = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    s_candidates = np.arange(0, 4801, 100)

    search_space = []
    for r_cand in R_candidates:
        for s_cand in s_candidates:
            features = base_features.copy()
            features['R'] = r_cand
            features['s'] = s_cand
            features['p'] = 4800 - s_cand
            search_space.append(features)
    
    search_df = pd.DataFrame(search_space)
    
    predicted_log_rounds = surrogate_model.predict(search_df)

    cost = predicted_log_rounds - alpha * search_df['R'].values
    
    best_idx = np.argmin(cost)
    
    best_params['R'] = search_df.loc[best_idx, 'R']
    best_params['s'] = int(search_df.loc[best_idx, 's'])
    best_params['score'] = cost[best_idx]
    
    return best_params['R'], best_params['s']

def make_predictions(df_featured: pd.DataFrame, qber_model: CatBoostRegressor, surrogate_model: CatBoostRegressor, threshold_1: float, threshold_2: float) -> pd.DataFrame:
    """
    Выполняет предсказание E_mu_Z и поиск оптимальных {R, s}
    с использованием стратегии бюджета риска и пакетной обработки
    """
    print("  - Шаг 1: Предсказание E_mu_Z для всех фреймов...")
    features_for_qber = df_featured[qber_model.feature_names_]
    predicted_qber = qber_model.predict(features_for_qber)
    df_featured['E_mu_Z_pred'] = predicted_qber
    df_featured['E_mu_Z'] = predicted_qber
    print("    - Предсказания для E_mu_Z получены.")

    print(f"  - Шаг 2: Векторизованный поиск по 'Бюджету риска' (t1={threshold_1}, t2={threshold_2})...")

    R_CANDIDATES_LOW_RISK = np.array([0.75, 0.8, 0.85, 0.9])
    R_CANDIDATES_MID_RISK = np.array([0.6, 0.65, 0.7])
    R_CANDIDATES_HIGH_RISK = np.array([0.5, 0.55])
    S_CANDIDATES = np.arange(0, 4801, 100)

    low_risk_mask = df_featured['E_mu_Z_pred'] < threshold_1
    mid_risk_mask = (df_featured['E_mu_Z_pred'] >= threshold_1) & (df_featured['E_mu_Z_pred'] < threshold_2)
    high_risk_mask = df_featured['E_mu_Z_pred'] >= threshold_2

    df_low_risk = df_featured[low_risk_mask]
    df_mid_risk = df_featured[mid_risk_mask]
    df_high_risk = df_featured[high_risk_mask]

    all_results = []
    
    def find_best_in_batch(segment_df, r_candidates, segment_name):
        n_segment = len(segment_df)
        if n_segment == 0:
            return None
            
        print(f"    - Обработка {n_segment} строк в сегменте '{segment_name}'...")
        
        BATCH_SIZE = 1024
        segment_results = []
        
        total_candidates = len(r_candidates) * len(S_CANDIDATES)
        r_grid, s_grid = np.meshgrid(r_candidates, S_CANDIDATES)
        candidates_df = pd.DataFrame({'R': r_grid.ravel(), 's': s_grid.ravel(), 'p': 4800 - s_grid.ravel()})

        surrogate_features = surrogate_model.feature_names_
        base_feature_names = [f for f in surrogate_features if f not in ['R', 's', 'p']]

        for i in tqdm(range(0, n_segment, BATCH_SIZE), desc=f"      Батчи '{segment_name}'"):
            batch_df = segment_df.iloc[i:i+BATCH_SIZE]
            n_batch = len(batch_df)
            
            batch_base_features = batch_df[base_feature_names]
            repeated_features = pd.DataFrame(np.repeat(batch_base_features.values, total_candidates, axis=0), columns=base_feature_names)
            tiled_candidates = pd.concat([candidates_df] * n_batch, ignore_index=True)
            
            search_df = pd.concat([repeated_features, tiled_candidates], axis=1)[surrogate_features]
            
            predicted_log_rounds = surrogate_model.predict(search_df)
            
            cost_matrix = predicted_log_rounds.reshape(n_batch, total_candidates)
            best_indices = np.argmin(cost_matrix, axis=1)
            
            best_r = candidates_df['R'].iloc[best_indices].values
            best_s = candidates_df['s'].iloc[best_indices].values
            
            segment_results.append(pd.DataFrame({'R_pred': best_r, 's_pred': best_s}, index=batch_df.index))
        
        return pd.concat(segment_results)

    all_results.append(find_best_in_batch(df_low_risk, R_CANDIDATES_LOW_RISK, "Низкий риск"))
    all_results.append(find_best_in_batch(df_mid_risk, R_CANDIDATES_MID_RISK, "Средний риск"))
    all_results.append(find_best_in_batch(df_high_risk, R_CANDIDATES_HIGH_RISK, "Высокий риск"))
    
    final_results = pd.concat([res for res in all_results if res is not None]).sort_index()

    submission_df = pd.DataFrame(index=df_featured.index)
    submission_df['E_mu_Z_pred'] = df_featured['E_mu_Z_pred']
    submission_df['R_pred'] = final_results['R_pred']
    submission_df['s_pred'] = final_results['s_pred']
    submission_df['p_pred'] = 4800 - submission_df['s_pred']
    
    print("  - Финальный DataFrame с оптимальными параметрами для каждого фрейма сформирован.")
    return submission_df

def compress_and_format_submission(df_full_predictions: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Сжимает полные предсказания до 2000 строк и форматирует их
    """

    print("  - Сжатие предсказаний до 2000 строк...")
    df_full_predictions['id'] = original_df['id']
    
    ids = df_full_predictions['id'].unique().tolist()
    n_ids = len(ids)
    
    TOTAL = 2000
    base = TOTAL // n_ids
    rem = TOTAL % n_ids
    
    final_rows = []
    
    print(f"  - Всего сессий: {n_ids}. Каждая сессия будет сжата до ~{base} строк.")

    for idx, session_id in enumerate(tqdm(ids, desc="    Сжатие сессий")):
        k = base + (1 if idx < rem else 0)

        session_preds = df_full_predictions[df_full_predictions['id'] == session_id]
        
        e_mu_z_arr = session_preds['E_mu_Z_pred'].to_numpy()
        r_arr = session_preds['R_pred'].to_numpy()
        s_arr = session_preds['s_pred'].to_numpy()

        if len(e_mu_z_arr) < k:
            e_mu_z_arr = np.pad(e_mu_z_arr, (0, k - len(e_mu_z_arr)), mode='edge')
            r_arr = np.pad(r_arr, (0, k - len(r_arr)), mode='edge')
            s_arr = np.pad(s_arr, (0, k - len(s_arr)), mode='edge')

        e_mu_z_chunks = np.array_split(e_mu_z_arr, k)
        r_chunks = np.array_split(r_arr, k)
        s_chunks = np.array_split(s_arr, k)

        e_mu_z_means = [np.mean(c) for c in e_mu_z_chunks]
        r_modes = [pd.Series(c).mode().iloc[0] for c in r_chunks]
        s_means = [np.mean(c) for c in s_chunks]
        
        for i in range(k):
            e_val = e_mu_z_means[i]
            r_val = round(r_modes[i], 2)
            s_val = int(round(s_means[i])) 
            p_val = 4800 - s_val
            final_rows.append([e_val, r_val, s_val, p_val])

    final_df = pd.DataFrame(final_rows)
    assert len(final_df) == TOTAL, f"Ошибка: получилось {len(final_df)} строк вместо {TOTAL}"
    
    print(f"  - Сжатие успешно завершено. Получено {len(final_df)} строк.")
    return final_df

def main():
    """Главная функция для запуска пайплайна предсказания"""

    parser = argparse.ArgumentParser(description="Скрипт для генерации submission.csv со стратегией 'Бюджета риска'")
    
    project_root = Path(__file__).resolve().parent.parent
    
    parser.add_argument('--input', type=str, 
                        default=str(project_root / 'data' / 'processed' / 'featured_dataset.csv'))
    parser.add_argument('--qber_model', type=str, default=str(project_root / 'models' / 'catboost_QBER_model_optuna.cbm'))
    parser.add_argument('--surrogate_model', type=str, default=str(project_root / 'models' / 'catboost_SURROGATE_model_optuna.cbm'))
    parser.add_argument('--output', type=str, default=str(project_root / 'submissions' / 'submission_risk_budget.csv'))
    
    parser.add_argument('--t1', type=float, default=0.01, help='Порог E_mu_Z для перехода от низкого к среднему риску.')
    parser.add_argument('--t2', type=float, default=0.08, help='Порог E_mu_Z для перехода от среднего к высокому риску.')

    args = parser.parse_args()
    input_path = Path(args.input)
    qber_model_path = Path(args.qber_model)
    surrogate_model_path = Path(args.surrogate_model)
    output_path = Path(args.output)
    threshold_1 = args.t1
    threshold_2 = args.t2

    print(f"Запуск пайплайна с порогами t1={threshold_1}, t2={threshold_2}")

    print("1. Загрузка обработанных данных и моделей...")
    try:
        df_featured = pd.read_csv(input_path)
        qber_model = CatBoostRegressor().load_model(qber_model_path)
        surrogate_model = CatBoostRegressor().load_model(surrogate_model_path)
        print("  - Все файлы успешно загружены.")
    except Exception as e:
        print(f"Ошибка при загрузке файлов: {e}")
        return

    full_submission_df = make_predictions(df_featured, qber_model, surrogate_model, threshold_1, threshold_2)
    
    final_submission_df = compress_and_format_submission(full_submission_df, df_featured)

    print("\n4. Сохранение submission файла...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_filename = output_path.stem + f"_t1_{threshold_1}_t2_{threshold_2}" + output_path.suffix
    final_output_path = output_path.parent / output_filename

    with open(final_output_path, 'w') as f:
        for index, row in final_submission_df.iterrows():
            line = f"{row[0]:.16f},{row[1]:.2f},{int(row[2])},{int(row[3])}\n"
            f.write(line)
    
    print(f"  - Submission файл успешно сохранен в: '{final_output_path}'")
    
    print("\nПайплайн предсказания успешно завершен!")

if __name__ == '__main__':
    main()