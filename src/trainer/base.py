# src/trainer/base.py

import os
import signal
import torch
from abc import abstractmethod, ABCMeta
from pathlib import Path
from shutil import copyfile
from numpy import inf

from utils.util import write_conf, is_master, get_logger
from logger import TensorboardWriter, EpochMetrics


class BaseTrainer(metaclass=ABCMeta):
    """
    Base class for all trainers
    """
    def __init__(self, model, epochs, criterion, metric_ftns, optimizer, config):
        self.config = config
        self.logger = get_logger('trainer')

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.device = torch.device('cuda')
            self.logger.info('Модель будет обучаться на GPU.')
        else:
            self.device = torch.device('cpu')
            self.logger.info('Модель будет обучаться на CPU.')
        self.model = model.to(self.device)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config.hparams.trainer 
        self.epochs = config.hparams.epochs

        self.log_step = cfg_trainer['logging_step']
        self.monitor = cfg_trainer.get('monitor', 'off')

        metric_names = ['loss'] + [met.__name__ for met in self.metric_ftns]
        self.ep_metrics = EpochMetrics(metric_names, phases=('train', 'valid'), monitoring=self.monitor)

        self.checkpt_top_k = cfg_trainer.get('save_topk', -1)
        self.early_stop = cfg_trainer.get('early_stop', inf)

        output_dir = os.getcwd()
        self.checkpt_dir = Path(output_dir) / 'checkpoints'
        log_dir_tensorboard = Path(output_dir) / 'tensorboard'

        write_conf(self.config, os.path.join(output_dir, 'config.yaml'))

        self.start_epoch = 1
        
        if is_master():
            self.checkpt_dir.mkdir(parents=True, exist_ok=True)
            log_dir_tensorboard.mkdir(parents=True, exist_ok=True)
            self.writer = TensorboardWriter(log_dir_tensorboard, cfg_trainer['tensorboard'])
        else:
            self.writer = TensorboardWriter(log_dir_tensorboard, False)

        if "resume" in config and config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            self.ep_metrics.update(epoch, result)

            max_line_width = max(len(line) for line in str(self.ep_metrics).splitlines())
            self.logger.info('=' * max_line_width)
            self.logger.info(f'\n{self.ep_metrics.latest()}')
            self.logger.info('=' * max_line_width)

            is_best = False
            improved = self.ep_metrics.is_improved()
            if improved:
                not_improved_count = 0
                is_best = True
            else:
                not_improved_count += 1

            if self.early_stop < float('inf') and not_improved_count > self.early_stop:
                self.logger.info(f"Валидационная метрика не улучшалась {self.early_stop} эпох. Остановка обучения.")
                break

            using_topk_save = self.checkpt_top_k > 0
            if is_master():
                self._save_checkpoint(epoch, save_best=is_best, save_latest=using_topk_save)
                if using_topk_save:
                    self.ep_metrics.keep_topk_checkpt(self.checkpt_dir, self.checkpt_top_k)

            self.ep_metrics.to_csv('epoch-results.csv')

            self.logger.info('*' * max_line_width)

    def _save_checkpoint(self, epoch, save_best=False, save_latest=True):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch_metrics': self.ep_metrics,
            'config': self.config
        }
        filename = str(self.checkpt_dir / f'checkpoint-epoch{epoch}.pth')
        torch.save(state, filename)
        self.logger.info(f"Сохранен чекпоинт: {filename}")
        
        if save_latest:
            latest_path = str(self.checkpt_dir / 'model_latest.pth')
            torch.save(state, latest_path)
            
        if save_best:
            best_path = str(self.checkpt_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info(f"Обновлен лучший чекпоинт: {best_path}")

    def _resume_checkpoint(self, resume_path):
        self.logger.info(f"Загрузка чекпоинта: {resume_path} ...")
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.ep_metrics = checkpoint['epoch_metrics']
        self.model.load_state_dict(checkpoint['state_dict'])

        if checkpoint['config']['model']['optimizer']['_target_'] == self.config.model.optimizer._target_:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            self.logger.warning("Тип оптимизатора в конфиге отличается от чекпоинта. Состояние оптимизатора не загружено.")

        self.logger.info(f"Чекпоинт загружен. Обучение будет продолжено с эпохи {self.start_epoch}")