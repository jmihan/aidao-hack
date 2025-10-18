# src/data_loader/data_loaders.py

import pandas as pd
import numpy as np
import torch
import os
import pickle
from torch.utils.data import Dataset, DataLoader
from utils.timefeatures import time_features
from utils import get_logger

log = get_logger(__name__)


def get_data_loaders(config, batch_size):
    # --- 1. Загрузка ПРЕДОБРАБОТАННЫХ и РАЗДЕЛЕННЫХ данных ---
    log.info("Загрузка предварительно обработанных данных (train, valid, test)...")
    
    train_csv_path = os.path.join(config.root_path, 'processed', 'featured_train_dataset.csv')
    valid_csv_path = os.path.join(config.root_path, 'processed', 'featured_valid_dataset.csv')
    test_csv_path = os.path.join(config.root_path, 'processed', 'featured_test_dataset.csv')
    scaler_path = os.path.join(config.root_path, 'processed', 'scaler.pkl')

    try:
        df_train = pd.read_csv(train_csv_path)
        df_valid = pd.read_csv(valid_csv_path)
        df_test = pd.read_csv(test_csv_path)
        log.info("Предварительно обработанные данные успешно загружены.")
        log.info(f"  - Train shape: {df_train.shape}")
        log.info(f"  - Valid shape: {df_valid.shape}")
        log.info(f"  - Test shape: {df_test.shape}")

    except FileNotFoundError as e:
        log.error(f"Ошибка: Файл данных не найден: {e}. Убедитесь, что скрипт preprocess_tabular.py был запущен.")
        return None, None, None, None, None, -1

    # --- 2. Загрузка скейлера ---
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        log.info(f"StandardScaler успешно загружен из: {scaler_path}")
    except FileNotFoundError:
        log.error(f"Ошибка: Файл scaler.pkl не найден по пути {scaler_path}. Прерывание.")
        return None, None, None, None, None, -1

    # --- 3. Определение признаков и целевой колонки ---
    
    TARGET_COLUMN = config.target_column
    
    feature_cols = [col for col in df_train.columns if df_train[col].dtype in [np.float64, np.int64]]
    
    if TARGET_COLUMN in feature_cols:
        feature_cols.insert(0, feature_cols.pop(feature_cols.index(TARGET_COLUMN)))
    else:
        log.error(f"Целевая колонка '{TARGET_COLUMN}' не найдена среди числовых признаков в предобработанных данных!")
        return None, None, None, None, None, -1
        
    target_channel_idx = feature_cols.index(TARGET_COLUMN)
    
    log.info(f"Выбрано {len(feature_cols)} признаков для модели. Целевая колонка: '{TARGET_COLUMN}' (индекс {target_channel_idx}).")
    log.debug(f"Список признаков: {feature_cols}")

    # --- 4. Создание датасетов и загрузчиков ---
    log.info("Создание датасетов и загрузчиков данных (DataLoader)...")
    
    train_dataset = Dataset_Custom(df_train, features_list=feature_cols, 
                                   size=[config.context_length, config.label_len, config.prediction_length], 
                                   features='M')
    valid_dataset = Dataset_Custom(df_valid, features_list=feature_cols, 
                                   size=[config.context_length, config.label_len, config.prediction_length], 
                                   features='M')
    test_dataset = Dataset_Custom(df_test, features_list=feature_cols, 
                                   size=[config.context_length, config.label_len, config.prediction_length], 
                                   features='M')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config.get('n_cpu', 20))
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=config.get('n_cpu', 20))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=config.get('n_cpu', 20))
    
    log.info("Загрузчики данных (DataLoader) успешно созданы.")

    return train_loader, valid_loader, test_loader, scaler, feature_cols, target_channel_idx


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
        df_stamp['date'] = pd.to_datetime(df_stamp['date'], errors='coerce')

        if self.timeenc == 1 and pd.api.types.is_datetime64_any_dtype(df_stamp['date']):
            datetime_index = pd.DatetimeIndex(df_stamp['date'])
            data_stamp = time_features(datetime_index, freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            log.warning("Колонка 'date' не является datetime. Временные признаки (time_features) не будут сгенерированы.")
            num_time_features = 4
            data_stamp = np.zeros((len(df_raw), num_time_features))
        
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