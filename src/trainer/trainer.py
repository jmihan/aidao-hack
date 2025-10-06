# src/trainer/trainer.py

import torch
import numpy as np
from ..trainer.base import BaseTrainer
from ..logger import BatchMetrics
from ..model.metric import mae, mse


class Trainer(BaseTrainer):
    def __init__(self, model, epochs, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, epochs, criterion, metric_ftns, optimizer, config)
        
        self.config = config
        self.data_loader = data_loader
        self.epochs = epochs
        self.len_epoch = len(self.data_loader) if len_epoch is None else len_epoch
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler

        metric_names = [m.__name__ for m in self.metric_ftns]
        self.train_metrics = BatchMetrics('loss', *metric_names, postfix='/train', writer=self.writer)
        self.valid_metrics = BatchMetrics('loss', *metric_names, postfix='/valid', writer=self.writer)

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (seq_x, seq_y, seq_x_mark, seq_y_mark) in enumerate(self.data_loader):
            seq_x = seq_x.to(self.device)
            seq_y = seq_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            output = self.model(seq_x)
            
            target = seq_y[:, -self.config.model.arch.configs.pred_len:, :]
            
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            if batch_idx % self.log_step == 0:
                self.logger.info(f'Train Epoch: {epoch} [{batch_idx}/{self.len_epoch}] Loss: {loss.item():.6f}')
        
        with torch.no_grad():
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))
        
        log = self.train_metrics.result()

        if self.valid_data_loader is not None:
            val_log = self._valid_epoch(epoch)
            log.update(**val_log)
        
        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(val_log['loss/valid'])
            else:
                self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()
        
        total_loss = 0
        all_outputs = []
        all_targets = []
        
        try:
            with torch.no_grad():
                for batch_idx, (seq_x, seq_y, seq_x_mark, seq_y_mark) in enumerate(self.valid_data_loader):
                    seq_x = seq_x.to(self.device)
                    seq_y = seq_y.to(self.device)

                    output = self.model(seq_x)
                    target = seq_y[:, -self.config.model.arch.configs.pred_len:, :]
                    
                    loss = self.criterion(output, target)
                    total_loss += loss.item()

                    all_outputs.append(output.cpu())
                    all_targets.append(target.cpu())

            all_outputs = torch.cat(all_outputs)
            all_targets = torch.cat(all_targets)

            self.valid_metrics.update('loss', total_loss / len(self.valid_data_loader))
            for met in self.metric_ftns:
                self.valid_metrics.update(met.__name__, met(all_outputs, all_targets))
            
            return self.valid_metrics.result()
            
        except Exception as e:
            self.logger.error(f"Ошибка во время валидации: {e}", exc_info=True)
            return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        total = self.len_epoch
        current = batch_idx
        return base.format(current, total, 100.0 * current / total)