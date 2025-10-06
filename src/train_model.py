# train_model.py

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from argparse import Namespace

from src.data_loader.data_loaders import get_data_loaders
from src.trainer.trainer import Trainer
from src.model.patchtst.model import Model

log = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="train", version_base=None)
def train(cfg: DictConfig) -> None:
    log.info("="*50)
    log.info("Запуск пайплайна обучения...")
    log.info(f"Конфигурация:\n{OmegaConf.to_yaml(cfg)}")
    log.info("="*50)

    # --- 1. Загрузка данных ---
    log.info("--- Шаг 1/5: Загрузка и подготовка данных ---")
    train_loader, valid_loader, scaler = get_data_loaders(
        config=cfg.data, 
        batch_size=cfg.hparams.batch_size
    )
    if train_loader is None:
        log.error("Не удалось создать загрузчики данных. Прерывание.")
        return
        
    # --- 2. Инициализация модели ---
    log.info("--- Шаг 2/5: Инициализация модели PatchTST ---")
    # Модель ожидает объект с атрибутами, а не словарь.
    # Преобразуем конфиг модели в такой объект.
    model_configs = Namespace(**cfg.model.arch.configs)
    model = Model(configs=model_configs)
    log.info(f"Модель создана:\n{model}")

    # --- 3. Инициализация функции потерь, оптимизатора и планировщика ---
    log.info("--- Шаг 3/5: Инициализация компонентов обучения ---")
    criterion = hydra.utils.instantiate(cfg.model.loss)
    optimizer = hydra.utils.instantiate(cfg.model.optimizer, params=model.parameters())
    lr_scheduler = hydra.utils.instantiate(cfg.model.lr_scheduler, optimizer=optimizer)

    # --- 4. Инициализация тренера ---
    log.info("--- Шаг 4/5: Инициализация кастомного тренера ---")
    
    # Метрики для регрессии
    from src.model.metric import mae, mse
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

    # --- 5. Запуск обучения ---
    log.info("--- Шаг 5/5: Запуск процесса обучения ---")
    trainer.train()
    log.info("="*50)
    log.info("Обучение успешно завершено!")
    log.info("="*50)


if __name__ == '__main__':
    train()