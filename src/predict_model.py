# predict.py

import os
import pickle
import argparse
import logging
from argparse import Namespace
from math import ceil

import torch
import pandas as pd
import numpy as np
from omegaconf import OmegaConf

from model.patchtst.model import Model
from utils.util import generate_features

log = logging.getLogger(__name__)

# --- baseline ---
def calculate_ema(prev_ema, current_value, alpha):
    if prev_ema is None:
        return current_value
    return alpha * current_value + (1 - alpha) * prev_ema

def h(x):
    if x > 0:
        return -x * np.log2(x) - (1 - x) * np.log2(1 - x)
    elif x == 0:
        return 0.0
    else:
        raise ValueError("Invalid x for binary entropy")

def select_code_rate(e_mu, f_ec, rates, frame_len, sp_count):
    r_candidate = 1 - h(e_mu) * f_ec
    R_res = 0.50
    s_n = sp_count
    p_n = 0
    for R in rates:
        p_n = int(
            ceil((1 - R) * frame_len - (1 - r_candidate) * (frame_len - sp_count))
        )
        s_n = int(sp_count - p_n)
        if p_n >= 0 and s_n >= 0:
            R_res = R
            return round(R_res, 2), s_n, p_n
    return round(R_res, 2), s_n, p_n


def predict(model_dir: str):
    """
    Основная функция для генерации предсказаний.
    """
    log.info("="*50)
    log.info("Запуск пайплайна предсказания...")
    log.info(f"Директория с моделью: {model_dir}")
    log.info("="*50)

    X = [[1, 1], [1, 2], [1, 3]]
    y = [1, 2, 3]
    X_t = torch.as_tensor(X, dtype=torch.float)
    y_t = torch.as_tensor(y, dtype=torch.float).reshape(-1,1)
    
    
    # --- 1. Загрузка артефактов ---
    log.info("--- Шаг 1/5: Загрузка артефактов (модель, скейлер, конфиг) ---")
    artifacts_path = os.path.join(model_dir, "artifacts.pkl")
    model_path = os.path.join(model_dir, "checkpoints", "model_best.pth")

    if not os.path.exists(artifacts_path) or not os.path.exists(model_path):
        log.error("Артефакты или чекпоинт не найдены. Убедитесь, что путь к директории верный.")
        return

    with open(artifacts_path, "rb") as f:
        artifacts = pickle.load(f)
    
    scaler = artifacts['scaler']
    feature_cols = artifacts['feature_cols']
    target_channel_idx = artifacts['target_channel_idx']
    cfg = artifacts['config']
    
    log.info(f"Конфигурация загружена. Количество признаков: {len(feature_cols)}.")

    # --- 2. Инициализация модели и загрузка весов ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_configs = Namespace(**cfg.model.arch.configs)
    print(model_configs)
    model = Model(configs=model_configs)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.eval()
    log.info(f"Модель успешно загружена на {device}.")

    # --- 3. Подготовка тестовых данных ---
    log.info("--- Шаг 2/5: Подготовка тестовых данных ---")
    csv_path = os.path.join(cfg.data.root_path, cfg.data.data_path)
    df_raw = pd.read_csv(csv_path, header=None)
    df_raw.columns = [
        "block_id", "frame_idx", "E_mu_Z", "E_mu_phys_est", "E_mu_X", "E_nu1_X", "E_nu2_X", "E_nu1_Z", "E_nu2_Z", "N_mu_X", "M_mu_XX", "M_mu_XZ", "M_mu_X", "N_mu_Z", "M_mu_ZZ", "M_mu_Z", "N_nu1_X", "M_nu1_XX", "M_nu1_XZ", "M_nu1_X", "N_nu1_Z", "M_nu1_ZZ", "M_nu1_Z", "N_nu2_X", "M_nu2_XX", "M_nu2_XZ", "M_nu2_X", "N_nu2_Z", "M_nu2_ZZ", "M_nu2_Z", "nTot", "bayesImVoltage", "opticalPower", "polarizerVoltages[0]", "polarizerVoltages[1]", "polarizerVoltages[2]", "polarizerVoltages[3]", "temp_1", "biasVoltage_1", "temp_2", "biasVoltage_2", "synErr", "N_EC_rounds", "maintenance_flag", "estimator_name", "f_EC", "E_mu_Z_est", "R", "s", "p",
    ]
    df_raw = df_raw.rename(columns={"block_id": "id", "frame_idx": "date"})
    
    # --- Генерация новых признаков ---
    df_raw = generate_features(df_raw)

    df_processed = df_raw[['id'] + feature_cols].copy()
    df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_processed = df_processed.ffill().bfill()

    test_inputs = []
    ids_order = []
    for group_id, group_df in df_processed.groupby('id'):
        if len(group_df) >= cfg.data.context_length:
            test_inputs.append(group_df.tail(cfg.data.context_length)[feature_cols].values)
            ids_order.append(group_id)

    test_inputs_scaled = np.array(test_inputs)
    sh = test_inputs_scaled.shape
    test_inputs_scaled = scaler.transform(test_inputs_scaled.reshape(-1, sh[-1])).reshape(sh)

    log.info(f"Подготовлено {len(test_inputs_scaled)} сэмплов для предсказания.")

    # --- 4. Получение предсказаний ---
    log.info("--- Шаг 3/5: Генерация предсказаний моделью ---")
    all_predictions = []
    with torch.no_grad():
        for i in range(len(test_inputs_scaled)):
            sample = torch.FloatTensor(test_inputs_scaled[i]).unsqueeze(0).to(device)
            output = model(sample)
            all_predictions.append(output.cpu().numpy())

    predictions_scaled = np.concatenate(all_predictions, axis=0)
    #target_predictions_scaled = predictions_scaled[:, :, target_channel_idx].flatten()

    # --- 5. Обратное масштабирование и агрегация ---
    log.info("--- Шаг 4/5: Обратное масштабирование и агрегация ---")

    pred_len = predictions_scaled.shape[1]
    reshaped_preds = predictions_scaled.reshape(-1, predictions_scaled.shape[-1])
    
    inversed_full = scaler.inverse_transform(reshaped_preds)
    
    final_predictions_flat = inversed_full[:, target_channel_idx]
    
    predictions_by_id = {}
    for i, group_id in enumerate(ids_order):
        start_idx = i * pred_len
        end_idx = start_idx + pred_len
        predictions_by_id[group_id] = final_predictions_flat[start_idx:end_idx]

    TOTAL = 2000
    n_ids = len(ids_order)
    base = TOTAL // n_ids
    rem = TOTAL % n_ids
    
    aggregated_predictions = []
    for idx, group_id in enumerate(ids_order):
        k = base + (1 if idx < rem else 0)
        if k == 0: continue

        arr = predictions_by_id[group_id]

        if len(arr) >= k:
            selected_arr = arr[:k]
        else:
            padding = np.full(k - len(arr), arr[-1])
            selected_arr = np.concatenate([arr, padding])

        aggregated_predictions.extend(selected_arr.tolist())

    assert len(aggregated_predictions) == 2000, f"Получилось {len(aggregated_predictions)} предсказаний вместо 2000"
    target_df = pd.DataFrame({"value": aggregated_predictions})

    # --- 6. Генерация submission.csv ---
    log.info("--- Шаг 5/5: Применение финальной логики и создание submission.csv ---")
    alpha = 0.33
    f_ec = 1.15
    R_range = [round(0.50 + 0.05 * x, 2) for x in range(9)]
    n = 32000
    d = 4800

    E_series = pd.to_numeric(target_df.iloc[:, 0], errors="coerce").dropna().reset_index(drop=True)
    prev_ema = None
    rows = []
    for E_mu_Z in E_series:
        ema_value = calculate_ema(prev_ema, float(E_mu_Z), alpha)
        prev_ema = ema_value
        R, s_n, p_n = select_code_rate(ema_value, f_ec, R_range, n, d)
        rows.append([f"{E_mu_Z:.16f}", R, s_n, p_n])

    submission_df = pd.DataFrame(rows)
    submission_path = "submission.csv"
    submission_df.to_csv(submission_path, header=False, index=False)
    log.info(f"Файл {submission_path} успешно создан!")
    log.info("="*50)
    log.info("Пайплайн предсказания завершен.")
    log.info("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate predictions from a trained PatchTST model.")
    parser.add_argument("model_dir", type=str, help="Path to the Hydra output directory containing the model artifacts.")
    args = parser.parse_args()
    
    predict(args.model_dir)