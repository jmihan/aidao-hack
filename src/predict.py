import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from catboost import CatBoostClassifier, CatBoostRegressor
from tqdm import tqdm

def make_predictions(df_featured: pd.DataFrame, r_model: CatBoostClassifier, s_model: CatBoostRegressor) -> pd.DataFrame:
    """
    Выполняет двухступенчатое предсказание R и s, а также E_mu_Z
    """

    print("  - Начало предсказания...")

    features_for_r = df_featured[r_model.feature_names_]
    predicted_r = r_model.predict(features_for_r)
    df_featured['R_pred'] = predicted_r[:, 0]
    print("    - Предсказания для R получены.")
    
    features_for_s = df_featured[s_model.feature_names_]
    predicted_s = s_model.predict(features_for_s)
    predicted_s = np.round(predicted_s).astype(int)
    print("    - Предсказания для s получены.")
    
    predicted_e_mu_z = df_featured['E_mu_Z_roll_mean_10'].fillna(df_featured['E_mu_Z'])
    print("    - Предсказания для E_mu_Z получены.")
    
    submission_df = pd.DataFrame()
    submission_df['E_mu_Z_pred'] = predicted_e_mu_z
    submission_df['R_pred'] = df_featured['R_pred']
    submission_df['s_pred'] = predicted_s
    submission_df['p_pred'] = 4800 - submission_df['s_pred']
    
    print("  - Финальный DataFrame для submission сформирован.")
    return submission_df

def compress_and_format_submission(df_full_predictions: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Сжимает полные предсказания до 2000 строк и форматирует их
    согласно требованиям сабмишена
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
        if len(e_mu_z_arr) < k: e_mu_z_arr = np.pad(e_mu_z_arr, (0, k - len(e_mu_z_arr)), mode='edge')
        e_mu_z_chunks = np.array_split(e_mu_z_arr, k)
        e_mu_z_means = [np.mean(c) for c in e_mu_z_chunks]
        
        r_arr = session_preds['R_pred'].to_numpy()
        if len(r_arr) < k: r_arr = np.pad(r_arr, (0, k - len(r_arr)), mode='edge')
        r_chunks = np.array_split(r_arr, k)
        r_modes = [pd.Series(c).mode().iloc[0] for c in r_chunks]
        
        s_arr = session_preds['s_pred'].to_numpy()
        if len(s_arr) < k: s_arr = np.pad(s_arr, (0, k - len(s_arr)), mode='edge')
        s_chunks = np.array_split(s_arr, k)
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

    parser = argparse.ArgumentParser(description="Упрощенный скрипт для генерации submission.csv")
    
    project_root = Path(__file__).resolve().parent.parent
    
    parser.add_argument('--input', type=str, 
                        default=str(project_root / 'data' / 'processed' / 'featured_dataset.csv'))
    parser.add_argument('--r_model', type=str, default=str(project_root / 'models' / 'catboost_R_model_optuna.cbm'))
    parser.add_argument('--s_model', type=str, default=str(project_root / 'models' / 'catboost_s_model_optuna.cbm'))
    parser.add_argument('--output', type=str, default=str(project_root / 'submissions' / 'submission_final.csv'))
    
    args = parser.parse_args()
    input_path = Path(args.input)
    r_model_path = Path(args.r_model)
    s_model_path = Path(args.s_model)
    output_path = Path(args.output)
    
    print("1. Загрузка обработанных данных и моделей...")
    try:
        df_featured = pd.read_csv(input_path)
        r_model = CatBoostClassifier().load_model(r_model_path)
        s_model = CatBoostRegressor().load_model(s_model_path)
        print("  - Все файлы успешно загружены.")
    except Exception as e:
        print(f"Ошибка при загрузке файлов: {e}")
        return

    full_submission_df = make_predictions(df_featured, r_model, s_model)
    
    final_submission_df = compress_and_format_submission(full_submission_df, df_featured)

    print("\n4. Сохранение submission файла...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for index, row in final_submission_df.iterrows():
            line = f"{row[0]:.16f},{row[1]:.2f},{int(row[2])},{int(row[3])}\n"
            f.write(line)
    
    print(f"  - Submission файл успешно сохранен в: '{output_path}'")
    
    print("\nПайплайн предсказания успешно завершен!")

if __name__ == '__main__':
    main()