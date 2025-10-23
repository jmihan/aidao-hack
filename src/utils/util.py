import os
import yaml
import logging
from functools import partial, update_wrapper
from importlib import import_module
from itertools import repeat
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm
import pandas as pd


def is_master():
    return not dist.is_initialized() or dist.get_rank() == 0

def get_logger(name=None):
    return logging.getLogger(name)


def collect(scalar):
    """
    util function for DDP.
    syncronize a python scalar or pytorch scalar tensor between GPU processes.
    """
    # move data to current device
    if not isinstance(scalar, torch.Tensor):
        scalar = torch.tensor(scalar)
    scalar = scalar.to(dist.get_rank())

    # average value between devices
    dist.reduce(scalar, 0, dist.ReduceOp.SUM)
    return scalar.item() / dist.get_world_size()

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def instantiate(config, *args, is_func=False, **kwargs):
    """
    wrapper function for hydra.utils.instantiate.
    1. return None if config.class is None
    2. return function handle if is_func is True
    """
    assert '_target_' in config, f'Config should have \'_target_\' for class instantiation.'
    target = config['_target_']
    if target is None:
        return None
    if is_func:
        # get function handle
        modulename, funcname = target.rsplit('.', 1)
        mod = import_module(modulename)
        func = getattr(mod, funcname)

        # make partial function with arguments given in config, code
        kwargs.update({k: v for k, v in config.items() if k != '_target_'})
        partial_func = partial(func, *args, **kwargs)

        # update original function's __name__ and __doc__ to partial function
        update_wrapper(partial_func, func)
        return partial_func
    return hydra.utils.instantiate(config, *args, **kwargs)

def write_yaml(content, fname):
    with fname.open('wt') as handle:
        yaml.dump(content, handle, indent=2, sort_keys=False)

def write_conf(config, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    config_dict = OmegaConf.to_container(config, resolve=True)
    write_yaml(config_dict, save_path)

def get_logits(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    logits_list = []
    targets_list = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            logits_list.append(logits)
            targets_list.append(labels)

    logits = torch.cat(logits_list, dim=0)
    targets = torch.cat(targets_list, dim=0)

    logits = logits.cpu()
    targets = targets.cpu()
    return logits, targets



def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Генерирует новые признаки для датафрейма.
    """
    print("Генерация новых признаков...")
    
    df.sort_values(['id', 'date'], inplace=True)

    # --- БЛОК 1: Признаки для целевой переменной (оставляем как было) ---
    lags_target = [1, 3, 5, 10]
    for lag in lags_target:
        df[f'E_mu_Z_lag_{lag}'] = df.groupby('id')['E_mu_Z'].shift(lag)

    windows_target = [5, 10, 20]
    for window in windows_target:
        grouped = df.groupby('id')['E_mu_Z']
        df[f'E_mu_Z_roll_mean_{window}'] = grouped.transform(lambda x: x.shift(1).rolling(window).mean())
        df[f'E_mu_Z_roll_std_{window}'] = grouped.transform(lambda x: x.shift(1).rolling(window).std())

    # --- БЛОК 2: НОВЫЙ - Признаки для ключевых физических предикторов ---
    # [Аргументация]: Выбираем признаки, которые по физическому смыслу или по результатам
    # анализа CatBoost могут сильно влиять на QBER.
    key_predictors = [
        'temp_1', 
        'biasVoltage_1',
        'temp_2',
        'biasVoltage_2',
        'opticalPower',
        'synErr'
    ]

    # --- 2.1: Скользящие статистики для ключевых предикторов ---
    # [Аргументация]: Даем модели информацию о недавнем тренде (mean) и
    # волатильности (std) этих важных параметров.
    windows_predictors = [10, 20] 
    for col in key_predictors:
        for window in windows_predictors:
            grouped = df.groupby('id')[col]
            df[f'{col}_roll_mean_{window}'] = grouped.transform(lambda x: x.shift(1).rolling(window).mean())
            df[f'{col}_roll_std_{window}'] = grouped.transform(lambda x: x.shift(1).rolling(window).std())

    # --- 2.2: Дельта-признаки (скорость изменения) для ключевых предикторов ---
    # [Аргументация]: `diff(1)` - это самый сильный сигнал о том, что в системе
    # что-то меняется ПРЯМО СЕЙЧАС. Очень полезно для предсказания аномалий.
    for col in key_predictors:
        df[f'{col}_diff_1'] = df.groupby('id')[col].diff(1)


    # --- БЛОК 3: НОВЫЙ - Признаки-взаимодействия ---
    # [Аргументация]: Создаем простые взаимодействия, которые могут отражать
    # нелинейные зависимости, которые модель может не уловить сама.
    df['temp_1_x_bias_1'] = df['temp_1'] * df['biasVoltage_1']
    df['temp_2_x_bias_2'] = df['temp_2'] * df['biasVoltage_2']
    
    # Рассматриваем отношение M_mu_XX к N_mu_X как "частоту ошибок" на сигнальных состояниях
    df['M_mu_XX_div_N_mu_X'] = df['M_mu_XX'] / (df['N_mu_X'] + 1e-6)


    new_cols_count = (
        len(lags_target) + len(windows_target)*2 +
        len(key_predictors) * len(windows_predictors) * 2 +
        len(key_predictors) + 3 # interaction features
    )
    
    print(f"Примерно {new_cols_count} новых признаков было сгенерировано.")
    
    return df