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
from tqdm.auto import tqdm
from omegaconf import OmegaConf

from model.patchtst.model import Model
from data_loader.data_loaders import Dataset_Custom
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

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

    log.info("--- Шаг 1/6: Загрузка артефактов (модель, конфиг, скейлер) ---")
    artifacts_path = os.path.join(model_dir, "artifacts.pkl")
    model_path = os.path.join(model_dir, "checkpoints", "model_best.pth")

    if not os.path.exists(artifacts_path) or not os.path.exists(model_path):
        log.error("Артефакты или чекпоинт не найдены. Убедитесь, что путь к директории верный.")
        return

    with open(artifacts_path, "rb") as f:
        artifacts = pickle.load(f)
    
    cfg = artifacts['config']
    feature_cols = artifacts['feature_cols']
    target_channel_idx = artifacts['target_channel_idx']
    
    scaler_path = os.path.join(cfg.data.root_path, 'data', 'processed', 'scaler.pkl')
    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        log.info(f"StandardScaler успешно загружен из: {scaler_path}")
    except FileNotFoundError:
        log.error(f"Ошибка: Scaler не найден по пути {scaler_path}. Прерывание.")
        return
    
    log.info(f"Конфигурация загружена. Количество признаков: {len(feature_cols)}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_configs = Namespace(**cfg.model.arch.configs)
    model = Model(configs=model_configs)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    log.info(f"Модель успешно загружена на {device}.")

    log.info("--- Шаг 2/6: Подготовка тестовых данных ---")
    test_csv_path = os.path.join(cfg.data.root_path, 'data', 'processed', 'featured_test_dataset.csv')
    try:
        df_test = pd.read_csv(test_csv_path)
        log.info(f"Тестовые данные загружены из: {test_csv_path}. Форма: {df_test.shape}")
    except FileNotFoundError:
        log.error(f"Ошибка: Тестовый датасет не найден по пути {test_csv_path}. Убедитесь, что preprocess_tabular.py был запущен.")
        return

    test_dataset = Dataset_Custom(df_test, features_list=feature_cols,
                                  size=[cfg.data.context_length, cfg.data.label_len, cfg.data.prediction_length],
                                  features='M')
    test_loader = DataLoader(test_dataset, batch_size=cfg.hparams.batch_size, shuffle=False, num_workers=cfg.get('n_cpu', 20))
    log.info(f"Подготовлено {len(test_dataset)} сэмплов для предсказания.")

    log.info("--- Шаг 3/6: Генерация предсказаний моделью ---")
    all_predictions = []
    with torch.no_grad():
        for (batch_x, _, batch_x_mark, _) in tqdm(test_loader, desc="Предсказание"):
            batch_x = batch_x.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            
            output = model(batch_x, batch_x_mark) 
            all_predictions.append(output.cpu().numpy())

    predictions_scaled = np.concatenate(all_predictions, axis=0)
    target_predictions_scaled = predictions_scaled[:, 0, target_channel_idx]
    log.info(f"Сгенерировано {len(target_predictions_scaled)} предсказаний в масштабированном виде.")

    log.info("--- Шаг 4/6: Обратное масштабирование и агрегация ---")
    dummy_array = np.zeros((len(target_predictions_scaled), len(feature_cols)))
    dummy_array[:, target_channel_idx] = target_predictions_scaled
    
    inversed_full = scaler.inverse_transform(dummy_array)
    final_predictions = inversed_full[:, target_channel_idx]
    
    TOTAL = 2000
    if len(final_predictions) < TOTAL:
        log.warning(f"Сгенерировано {len(final_predictions)} предсказаний, меньше {TOTAL}. Дополняем последним значением.")
        final_predictions = np.pad(final_predictions, (0, TOTAL - len(final_predictions)), 'edge')
    elif len(final_predictions) > TOTAL:
        log.info(f"Сгенерировано {len(final_predictions)} предсказаний. Агрегируем до {TOTAL} точек.")
        original_indices = np.linspace(0, len(final_predictions) - 1, len(final_predictions))
        target_indices = np.linspace(0, len(final_predictions) - 1, TOTAL)
        final_predictions = np.interp(target_indices, original_indices, final_predictions)

    target_df = pd.DataFrame({"value": final_predictions})
    log.info(f"Итоговое количество предсказаний после агрегации: {len(target_df)}")

    log.info("--- Шаг 5/6: Применение финальной логики и создание submission.csv ---")
    alpha = 0.33; f_ec = 1.15; R_range = [round(0.50 + 0.05 * x, 2) for x in range(9)]; n = 32000; d = 4800
    E_series = pd.to_numeric(target_df.iloc[:, 0], errors="coerce").dropna()
    prev_ema = None
    rows = []
    for E_mu_Z in tqdm(E_series, desc="Генерация submission"):
        ema_value = calculate_ema(prev_ema, float(E_mu_Z), alpha)
        prev_ema = ema_value
        R, s_n, p_n = select_code_rate(ema_value, f_ec, R_range, n, d)
        rows.append([f"{E_mu_Z:.16f}", R, s_n, p_n])

    submission_df = pd.DataFrame(rows)
    submission_path = "submission.csv"
    submission_df.to_csv(submission_path, header=False, index=False)
    log.info(f"Файл {submission_path} успешно создан в {os.getcwd()}!")
    log.info("="*50)
    log.info("Пайплайн предсказания завершен.")
    log.info("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate predictions from a trained PatchTST model.")
    parser.add_argument("model_dir", type=str, help="Path to the Hydra output directory containing the model artifacts.")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    predict(args.model_dir)