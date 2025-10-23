# src/data_loader/data_loaders.py

import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from utils.util import generate_features
from utils import get_logger

log = get_logger(__name__)


def get_data_loaders(config, batch_size):
    # --- 1. Загрузка и полная предобработка данных ---
    log.info(f"Загрузка данных из {config.data_path}...")
    try:
        csv_path = os.path.join(config.root_path, config.data_path)
        df_raw = pd.read_csv(csv_path, header=None)
        df_raw.columns = [
            "block_id", "frame_idx", "E_mu_Z", "E_mu_phys_est", "E_mu_X", "E_nu1_X", "E_nu2_X", "E_nu1_Z", "E_nu2_Z",
            "N_mu_X", "M_mu_XX", "M_mu_XZ", "M_mu_X", "N_mu_Z", "M_mu_ZZ", "M_mu_Z", "N_nu1_X", "M_nu1_XX",
            "M_nu1_XZ", "M_nu1_X", "N_nu1_Z", "M_nu1_ZZ", "M_nu1_Z", "N_nu2_X", "M_nu2_XX", "M_nu2_XZ",
            "M_nu2_X", "N_nu2_Z", "M_nu2_ZZ", "M_nu2_Z", "nTot", "bayesImVoltage", "opticalPower",
            "polarizerVoltages[0]", "polarizerVoltages[1]", "polarizerVoltages[2]", "polarizerVoltages[3]",
            "temp_1", "biasVoltage_1", "temp_2", "biasVoltage_2", "synErr", "N_EC_rounds",
            "maintenance_flag", "estimator_name", "f_EC", "E_mu_Z_est", "R", "s", "p",
        ]
        df_raw = df_raw.rename(columns={"block_id": "id", "frame_idx": "date"})
        log.info("Данные успешно загружены.")
    except FileNotFoundError:
        log.error(f"Ошибка: Файл данных не найден по пути {config.data_path}")
        return None, None, None, None, -1

    df_raw = generate_features(df_raw)

    # --- 2. Выбор признаков для модели ---
    feature_cols = [col for col in df_raw.columns if col not in config.cols_to_drop_from_features and df_raw[col].dtype in [np.int64, np.float64]]
    
    if config.target_column in feature_cols:
        feature_cols.insert(0, feature_cols.pop(feature_cols.index(config.target_column)))
    else:
        log.error(f"Целевая колонка {config.target_column} не найдена среди числовых признаков!")
        return None, None, None, None, -1
        
    target_channel_idx = feature_cols.index(config.target_column)
    
    log.info(f"Выбрано {len(feature_cols)} признаков для модели. Целевая колонка: '{config.target_column}' (индекс {target_channel_idx}).")
    log.debug(f"Список признаков: {feature_cols}")

    df_processed = df_raw[['id', 'date'] + feature_cols].copy()
    df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_processed = df_processed.ffill()
    df_processed.dropna(inplace=True, ignore_index=True)

    # --- 3. Разделение данных (бейзлайн) ---
    log.info("Разделение данных на train/valid...")
    train_dfs = []
    valid_dfs = []
    
    HORIZON = config.prediction_length # 8
    HISTORY = config.context_length   # 160

    for current_id in df_processed["id"].unique():
        current_df = df_processed[df_processed["id"] == current_id]
        if len(current_df) > HORIZON + HISTORY:
            train_dfs.append(current_df.iloc[:-HORIZON])
            valid_dfs.append(current_df.iloc[-HORIZON - HISTORY:])

    train_df = pd.concat(train_dfs, ignore_index=True)
    valid_df = pd.concat(valid_dfs, ignore_index=True)
    
    log.info(f"Данные разделены. Train: {len(train_df)} строк, Valid: {len(valid_df)} строк.")

    # --- 4. Масштабирование ---
    log.info("Масштабирование всех признаков с помощью StandardScaler...")
    scaler = StandardScaler()
    train_df.loc[:, feature_cols] = scaler.fit_transform(train_df[feature_cols])
    valid_df.loc[:, feature_cols] = scaler.transform(valid_df[feature_cols])
    log.info("Масштабирование завершено.")

    # --- 5. Создание датасетов и загрузчиков ---
    train_dataset = Dataset_Custom(train_df, features_list=feature_cols, size=[config.context_length, config.label_len, config.prediction_length], features='M')
    valid_dataset = Dataset_Custom(valid_df, features_list=feature_cols, size=[config.context_length, config.label_len, config.prediction_length], features='M')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config.get('n_cpu', 20), pin_memory=True, persistent_workers=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=config.get('n_cpu', 20), pin_memory=True, persistent_workers=True)
    log.info("Загрузчики данных (DataLoader) успешно созданы.")

    return train_loader, valid_loader, scaler, feature_cols, target_channel_idx


class Dataset_Custom(Dataset):
    def __init__(self, df, features_list, size=None, features='S', timeenc=1, freq='h'):
        if size is None:
            self.seq_len, self.label_len, self.pred_len = 96, 48, 24
        else:
            self.seq_len, self.label_len, self.pred_len = size
        
        self.features = features
        self.features_list = features_list
        self.timeenc = timeenc
        self.freq = freq

        self.__read_data__(df)

    def __read_data__(self, df_raw):
        df_stamp = df_raw[['date']].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        
        if self.timeenc == 1:
            datetime_index = pd.DatetimeIndex(df_stamp['date'])
            data_stamp = time_features(datetime_index, freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            df_stamp['month'] = df_stamp.date.dt.month
            df_stamp['day'] = df_stamp.date.dt.day
            df_stamp['weekday'] = df_stamp.date.dt.weekday
            df_stamp['hour'] = df_stamp.date.dt.hour
            data_stamp = df_stamp.drop(['date'], axis=1).values
        
        self.data_x = df_raw[self.features_list].values
        self.data_y = self.data_x
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return (torch.FloatTensor(seq_x), torch.FloatTensor(seq_y),
                torch.FloatTensor(seq_x_mark), torch.FloatTensor(seq_y_mark))

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1