# src/data_loader/data_loaders.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from ..utils.timefeatures import time_features
from ..utils import get_logger

log = get_logger(__name__)

def get_data_loaders(config, batch_size):
    # --- 1. Загрузка и базовая предобработка данных ---
    log.info(f"Загрузка данных из {config.data_path}...")
    try:
        df_raw = pd.read_csv(config.data_path, header=None)
        df_raw.columns = [
            "block_id", "frame_idx", "E_mu_Z", "E_mu_phys_est", "E_mu_X", "E_nu1_X", "E_nu2_X", "E_nu1_Z", "E_nu2_Z",
            "N_mu_X", "M_mu_XX", "M_mu_XZ", "M_mu_X", "N_mu_Z", "M_mu_ZZ", "M_mu_Z", "N_nu1_X", "M_nu1_XX",
            "M_nu1_XZ", "M_nu1_X", "N_nu1_Z", "M_nu1_ZZ", "M_nu1_Z", "N_nu2_X", "M_nu2_XX", "M_nu2_XZ",
            "M_nu2_X", "N_nu2_Z", "M_nu2_ZZ", "M_nu2_Z", "nTot", "bayesImVoltage", "opticalPower",
            "polarizerVoltages[0]", "polarizerVoltages[1]", "polarizerVoltages[2]", "polarizerVoltages[3]",
            "temp_1", "biasVoltage_1", "temp_2", "biasVoltage_2", "synErr", "N_EC_rounds",
            "maintenance_flag", "estimator_name", "f_EC", "E_mu_Z_est", "R", "s", "p",
        ]
        df_raw = df_raw.rename(columns={"block_id": "id", "E_mu_Z": "value", "frame_idx": "date"})
        df_raw = df_raw[["id", "date", "value"]].dropna(subset=["value"])
        log.info("Данные успешно загружены и предобработаны.")

    except FileNotFoundError:
        log.error(f"Ошибка: Файл данных не найден по пути {config.data_path}")
        return None, None, None

    # --- 2. Разделение данных на обучающую и валидационную выборки ---
    unique_dates = sorted(df_raw['date'].unique())
    split_idx = int(len(unique_dates) * 0.8)
    split_date = unique_dates[split_idx]

    train_df = df_raw[df_raw['date'] < split_date]
    valid_df = df_raw[df_raw['date'] >= split_date]
    
    # Заменяем print на log.info для информационных сообщений
    log.info(f"Данные разделены. Train: {len(train_df)} строк, Valid: {len(valid_df)} строк.")

    # --- 3. Масштабирование данных ---
    log.info("Масштабирование данных с помощью StandardScaler...")
    scaler = StandardScaler()
    train_df['value'] = scaler.fit_transform(train_df[['value']])
    valid_df['value'] = scaler.transform(valid_df[['value']])
    log.info("Масштабирование завершено. Скейлер обучен на train данных.")

    # --- 4. Создание датасетов и загрузчиков ---
    train_dataset = Dataset_Custom(train_df, size=[config.context_length, config.label_len, config.prediction_length])
    valid_dataset = Dataset_Custom(valid_df, size=[config.context_length, config.label_len, config.prediction_length])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config.get('n_cpu', 2))
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=config.get('n_cpu', 2))
    log.info("Загрузчики данных (DataLoader) успешно созданы.")

    return train_loader, valid_loader, scaler


class Dataset_Custom(Dataset):
    def __init__(self, df, size=None, features='S', target='value', scale=False, timeenc=1, freq='h'):
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 24
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.__read_data__(df)

    def __read_data__(self, df_raw):
        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        if self.timeenc == 1:
            data_stamp = time_features(df_stamp['date'].values, freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
            
        if self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            df_data = df_raw.drop(columns=['id', 'date'])
        
        self.data_x = df_data.values
        self.data_y = df_data.values
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        if s_end + self.pred_len > len(self.data_x):
            return self.__getitem__(0)
            
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return (torch.FloatTensor(seq_x), torch.FloatTensor(seq_y),
                torch.FloatTensor(seq_x_mark), torch.FloatTensor(seq_y_mark))

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1