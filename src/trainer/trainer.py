from srcs.utils.util import is_master
import torch
import numpy as np
import torch.distributed as dist
from srcs.trainer.base import BaseTrainer
from srcs.utils import inf_loop
from transformers import get_cosine_schedule_with_warmup
from srcs.logger import BatchMetrics
import logging

class Trainer(BaseTrainer):
    def __init__(self, model, epochs, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, epochs, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.epochs = epochs
        if len_epoch is None:
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler

        self.train_metrics = BatchMetrics('loss', *[m.__name__ for m in self.metric_ftns], postfix='/train',
                                          writer=self.writer)
        self.valid_metrics = BatchMetrics('loss', *[m.__name__ for m in self.metric_ftns], postfix='/valid',
                                          writer=self.writer)

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        self.mixup_alpha = 0.4
        if hasattr(self.model, 'transformer_encoder'):
            transformer_params = self.model.transformer_encoder.parameters()
            optimizer_transformer = torch.optim.AdamW(transformer_params, lr=5e-5, weight_decay=1e-2)

            num_training_steps = len(self.data_loader) * self.epochs
            num_warmup_steps = int(num_training_steps * 0.1)
            
            scheduler_transformer = get_cosine_schedule_with_warmup(
                optimizer_transformer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )

        for batch_idx, (batch_data, target) in enumerate(self.data_loader):
            if hasattr(self.model, 'transformer_encoder'):
                batch_data.pop('tabular')
            else:
                batch_data.pop('x_numerical')
                batch_data.pop('x_categorical')                
            batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
            target = target.to(self.device).squeeze(1).float() 

            if hasattr(self.model, 'transformer_encoder'):
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                batch_size = target.size(0)
                index = torch.randperm(batch_size).to(self.device)

                mixed_image = lam * batch_data['image'] + (1 - lam) * batch_data['image'][index, :]
                mixed_input_ids = batch_data['input_ids']
                mixed_attention_mask = batch_data['attention_mask']
                mixed_tabular = lam * batch_data['x_numerical'] + (1 - lam) * batch_data['x_numerical'][index, :]

                mixed_data = {
                    'image': mixed_image,
                    'input_ids': mixed_input_ids,
                    'attention_mask': mixed_attention_mask,
                    'x_numerical': mixed_tabular,
                    'x_categorical': batch_data['x_categorical']
                }

                mixed_target = lam * target + (1 - lam) * target[index]
                self.optimizer.zero_grad()
                optimizer_transformer.zero_grad()
                logits = self.model(**mixed_data)
                loss = self.criterion(logits, mixed_target)
                loss.backward()
                self.optimizer.step()
                optimizer_transformer.step()
                scheduler_transformer.step()

            else:
                #
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                batch_size = target.size(0)
                index = torch.randperm(batch_size).to(self.device)

                mixed_image = lam * batch_data['image'] + (1 - lam) * batch_data['image'][index, :]
                mixed_input_ids = batch_data['input_ids']
                mixed_attention_mask = batch_data['attention_mask']
                mixed_tabular = lam * batch_data['tabular'] + (1 - lam) * batch_data['tabular'][index, :]

                mixed_data = {
                    'image': mixed_image,
                    'input_ids': mixed_input_ids,
                    'attention_mask': mixed_attention_mask,
                    'tabular': mixed_tabular,
                }

                mixed_target = lam * target + (1 - lam) * target[index]

                n_features_to_cut = 8
                aug_mask = torch.rand(mixed_tabular.size(0)) < 0.2
    
                aug_indices = torch.where(aug_mask)[0]
                
                if aug_indices.numel() > 0:
                    for i in aug_indices:
                        features_to_cut = np.random.choice(mixed_tabular.size(1), n_features_to_cut, replace=False)
                        mixed_tabular[i, features_to_cut] = 0.0
                        
                self.optimizer.zero_grad()
                output_logits, tab_m_loss = self.model(**mixed_data)

                lambda_sparse = 1e-3 
                loss = self.criterion(output_logits, mixed_target) + lambda_sparse * tab_m_loss.mean()

                loss.backward()
                self.optimizer.step()
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            if batch_idx % self.log_step == 0:
                self.logger.info(f'Train Epoch: {epoch} {self._progress(batch_idx)} Loss: {loss.item():.6f}')
                
                with torch.no_grad():
                    if hasattr(self.model, 'transformer_encoder'):
                        metric_output = torch.round(torch.sigmoid(logits))
                    else:
                        metric_output = torch.round(torch.sigmoid(output_logits))
                    for met in self.metric_ftns:
                        self.train_metrics.update(met.__name__, met(metric_output.cpu(), target.cpu()))

            if batch_idx == self.len_epoch:
                break
        
        log = self.train_metrics.result()

        if self.valid_data_loader is not None:
            val_log = self._valid_epoch(epoch)
            log.update(**val_log)
        
        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                val_loss = val_log['loss/valid']
                self.lr_scheduler.step(val_loss)
            else:
                self.lr_scheduler.step()

        self.writer.set_step(epoch)
        # if epoch == 1 and is_master():
        #     keys_in_order = ['image', 'input_ids', 'attention_mask', 'tabular']
        #     input_to_graph = tuple(batch_data[k] for k in keys_in_order)
        #     self.writer.add_graph(self.model, input_to_graph)
        for k, v in log.items():
            self.writer.add_scalar(k + '/epoch', v)
        return log

    def _valid_epoch(self, epoch):
        try:
            self.model.eval()
            self.valid_metrics.reset()
            with torch.no_grad():
                for batch_idx, (batch_data, target) in enumerate(self.valid_data_loader):
                    if hasattr(self.model, 'transformer_encoder'):
                        batch_data.pop('tabular')
                    else:
                        batch_data.pop('x_numerical')
                        batch_data.pop('x_categorical') 
                    batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
                    target = target.to(self.device).squeeze(1).float() 
                    output_logits = self.model(**batch_data)
                    if isinstance(output_logits, tuple):
                        output_logits, m_loss = output_logits
                        lambda_loss_m = 1e-3
                        loss = self.criterion(output_logits, target) + lambda_loss_m * m_loss
                    else:
                        loss = self.criterion(output_logits, target)

                    self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx)
                    self.valid_metrics.update('loss', loss.item())
                    
                    metric_output = torch.round(torch.sigmoid(output_logits))
                    for met in self.metric_ftns:
                        self.valid_metrics.update(met.__name__, met(metric_output.cpu(), target.cpu()))

            return self.valid_metrics.result()
        except Exception as e:
            logging.error(msg=f"{e}")
    

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        try:
            # epoch-based training
            total = len(self.data_loader.dataset)
            current = batch_idx * self.data_loader.batch_size
            if dist.is_initialized():
                current *= dist.get_world_size()
        except AttributeError:
            # iteration-based training
            total = self.len_epoch
            current = batch_idx
        return base.format(current, total, 100.0 * current / total)