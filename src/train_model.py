# train_model.py

import os
import pickle
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from argparse import Namespace

from data_loader.data_loaders import get_data_loaders
from trainer.trainer import Trainer
from model.patchtst.model import Model
from model.metric import mae, mse

log = logging.getLogger(__name__)

@hydra.main(config_path="../conf", config_name="train", version_base=None)
def train(cfg: DictConfig) -> None:
    log.info("="*50)
    log.info("Запуск пайплайна обучения...")
    log.info(f"Конфигурация:\n{OmegaConf.to_yaml(cfg)}")
    log.info("="*50)

    # --- 1. Загрузка данных ---
    log.info("--- Шаг 1/5: Загрузка и подготовка данных ---")
    train_loader, valid_loader, scaler, feature_cols, target_channel_idx = get_data_loaders(
        config=cfg.data, 
        batch_size=cfg.hparams.batch_size
    )
    if train_loader is None:
        log.error("Не удалось создать загрузчики данных. Прерывание.")
        return
        
    # --- 2. Динамическое обновление конфига ---
    n_features = len(feature_cols)
    log.info(f"Динамическое обновление конфига: n_features = {n_features}, target_channel_idx = {target_channel_idx}")
    OmegaConf.set_struct(cfg, False)
    cfg.model.arch.configs.enc_in = n_features
    cfg.model.target_channel_idx = target_channel_idx
    OmegaConf.set_struct(cfg, True)

    # --- 3. Инициализация модели ---
    log.info("--- Шаг 2/5: Инициализация модели PatchTST ---")
    model_configs = Namespace(**cfg.model.arch.configs)
    log.info(f"{model_configs}")
    model = Model(configs=model_configs)
    log.info(f"Модель создана с {n_features} входными каналами.")

    # --- 4. Инициализация компонентов обучения ---
    log.info("--- Шаг 3/5: Инициализация компонентов обучения ---")
    criterion = hydra.utils.instantiate(cfg.model.loss)
    optimizer = hydra.utils.instantiate(cfg.model.optimizer, params=model.parameters())
    lr_scheduler = hydra.utils.instantiate(cfg.model.lr_scheduler, optimizer=optimizer)

    # --- 5. Инициализация тренера ---
    log.info("--- Шаг 4/5: Инициализация кастомного тренера ---")
    metric_ftns = [mae, mse]
    
    trainer = Trainer(
        model=model,
        epochs=cfg.hparams.epochs,
        criterion=criterion,
        metric_ftns=metric_ftns,
        optimizer=optimizer,
        config=cfg,
        data_loader=train_loader,
        valid_data_loader=valid_loader,
        lr_scheduler=lr_scheduler
    )

    # --- 6. Запуск обучения ---
    log.info("--- Шаг 5/5: Запуск процесса обучения ---")
    trainer.train()
    log.info("="*50)
    log.info("Обучение успешно завершено!")
    log.info("="*50)

    # --- 7. Сохранение артефактов для предсказания ---
    if cfg.hparams.trainer.save_topk > 0:
        log.info("--- Шаг 6/6: Сохранение артефактов для инференса ---")

        artifacts = {
            'scaler': scaler,
            'feature_cols': feature_cols,
            'target_channel_idx': target_channel_idx,
            'config': cfg 
        }
        
        artifact_path = "artifacts.pkl"
        with open(artifact_path, "wb") as f:
            pickle.dump(artifacts, f)
        log.info(f"Артефакты (скейлер, конфиг) сохранены в: {os.getcwd()}/{artifact_path}")

    log.info("="*50)
    log.info("Пайплайн обучения успешно завершен!")
    log.info("="*50)

if __name__ == '__main__':
    train()