import pandas as pd
import numpy as np
import catboost as cb
import pickle
import os
import logging
from math import ceil

from utils.fe_for_catboost import generate_base_features

log = logging.getLogger(__name__)


def h(x):
    if x > 0 and x < 1:
        return -x * np.log2(x) - (1 - x) * np.log2(1 - x)
    return 0.0


def select_code_rate(e_mu, f_ec, rates, frame_len, sp_count):
    r_candidate = 1 - h(e_mu) * f_ec
    R_res = 0.50
    s_n, p_n = 0, 0
    for R in sorted(rates):
        p_n_candidate = int(ceil((1 - R) * frame_len - (1 - r_candidate) * (frame_len - sp_count)))
        s_n_candidate = int(sp_count - p_n_candidate)
        if p_n_candidate >= 0 and s_n_candidate >= 0:
            R_res, s_n, p_n = R, s_n_candidate, p_n_candidate
    return round(R_res, 2), s_n, p_n


def predict_classifier(model_dir: str = "."):
    log.info("="*50)
    log.info("Запуск пайплайна предсказания КЛАССИФИКАТОРОМ...")
    log.info("="*50)

    log.info("--- Шаг 1/4: Загрузка артефактов ---")
    model = cb.CatBoostClassifier()
    model.load_model(os.path.join(model_dir, "catboost_model.cbm"))
    
    with open(os.path.join(model_dir, "classifier_artifacts.pkl"), "rb") as f:
        artifacts = pickle.load(f)
    
    index_to_r_map = artifacts['index_to_r_map']
    class_to_r_map = artifacts['class_to_r_map']
    feature_cols = artifacts['feature_cols']
    r_to_s_map = artifacts['r_to_s_map']
    r_to_e_mu_z_map = artifacts['r_to_e_mu_z_map']

    log.info("--- Шаг 2/4: Подготовка тестовых данных ---")
    data_path = "data/frames_errors.csv"
    if not os.path.exists(data_path):
        log.error(f"Файл данных {data_path} не найден.")
        return
    
    df_raw = pd.read_csv(data_path, header=None)
    df_raw.columns = [
        "block_id", "frame_idx", "E_mu_Z", "E_mu_phys_est", "E_mu_X", "E_nu1_X", "E_nu2_X", "E_nu1_Z", "E_nu2_Z", "N_mu_X", "M_mu_XX", "M_mu_XZ", "M_mu_X",
          "N_mu_Z", "M_mu_ZZ", "M_mu_Z", "N_nu1_X", "M_nu1_XX", "M_nu1_XZ", "M_nu1_X", "N_nu1_Z", "M_nu1_ZZ", "M_nu1_Z", "N_nu2_X", "M_nu2_XX", "M_nu2_XZ",
            "M_nu2_X", "N_nu2_Z", "M_nu2_ZZ", "M_nu2_Z", "nTot", "bayesImVoltage", "opticalPower", "polarizerVoltages[0]", "polarizerVoltages[1]",
              "polarizerVoltages[2]", "polarizerVoltages[3]", "temp_1", "biasVoltage_1", "temp_2", "biasVoltage_2", "synErr", "N_EC_rounds",
                "maintenance_flag", "estimator_name", "f_EC", "E_mu_Z_est", "R", "s", "p",
    ]
    df_raw = df_raw.rename(columns={"block_id": "id", "frame_idx": "date"})

    df_test = generate_base_features(df_raw)
    
    X_test = df_test[feature_cols].bfill().ffill()
    test_ids = df_test['id']

    log.info("--- Шаг 3/4: Генерация предсказаний ---")
    predicted_classes = model.predict(X_test).flatten()

    predicted_R_series = pd.Series(predicted_classes).map(index_to_r_map)
    
    predictions_df = pd.DataFrame({
        'id': test_ids.values,
        'predicted_R': predicted_R_series.values
    }).dropna()

    predictions_df['current_E_mu_Z'] = df_raw.loc[predictions_df.index, 'E_mu_Z'].values
    predictions_df['current_s'] = df_raw.loc[predictions_df.index, 's'].values

    log.info("--- Шаг 4/4: Создание submission.csv ---")
    unique_ids = predictions_df['id'].unique()
    n_ids = len(unique_ids)
    TOTAL = 2000
    
    base = TOTAL // n_ids
    rem = TOTAL % n_ids
    
    final_predictions_list = []
    
    for idx, current_id in enumerate(unique_ids):
        k = base + (1 if idx < rem else 0)
        if k == 0: continue
            
        subset_df = predictions_df.loc[predictions_df['id'] == current_id].tail(k)
        
        if len(subset_df) < k:
            last_row = subset_df.iloc[[-1]] # Берем последнюю строку как DataFrame
            padding = pd.concat([last_row] * (k - len(subset_df)), ignore_index=True)
            subset_df = pd.concat([subset_df, padding], ignore_index=True)
            
        final_predictions_list.append(subset_df)

    final_predictions_df = pd.concat(final_predictions_list, ignore_index=True)

    assert len(final_predictions_df) == 2000, f"Получилось {len(final_predictions_df)} предсказаний вместо 2000"
    d = 4800
    rows = []

    for _, row in final_predictions_df.iterrows():
        R_pred = row['predicted_R']

        correction_factor_E = 1.0 
        E_mu_Z_pred = row['current_E_mu_Z'] * correction_factor_E

        correction_factor_s = 1.0
        s_pred = int(row['current_s'] * correction_factor_s)

        p_pred = d - s_pred
        s_pred = max(0, min(s_pred, d))
        p_pred = d - s_pred

        rows.append([f"{E_mu_Z_pred:.16f}", R_pred, s_pred, p_pred])

    submission_df = pd.DataFrame(rows)
    submission_path = "submission.csv"
    submission_df.to_csv(submission_path, header=False, index=False)
    log.info(f"Файл {submission_path} успешно создан!")

if __name__ == '__main__':
    predict_classifier()