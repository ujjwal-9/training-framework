import warnings

import munch
import numpy as np
import pandas as pd

import qtrain
import monai
import torch
import torch.nn as nn
import torch.optim as optim
import monai.losses as L
import monai.metrics as M
import torch.nn.functional as F
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from sklearn import metrics
from collections import defaultdict


class qSegmentation(pl.LightningModule):
    def __init__(self, args, train=True):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        
        if "mode" not in self.args.keys():
            warnings.warn(" `mode` not in arguments file, setting it to `multilabel` ")
            self.args["mode"] = "multilabel"

        self.setup_model(train)
        self.setup_losses()
        self.setup_metrics()
        
    def setup_model(self, train=True):
        if self.args.model_params_star_pass:
            self.model = self.args.model(**self.args.model_params).double()
        else:
            self.model = self.args.model(self.args).double()
        
        if train:
            self.model.train()
        else:
            self.model.eval()

    def setup_losses(self):
        self.loss_contrib = self.args.loss_contrib
        self.losses = self.args.losses

    def setup_metrics(self):
        self.metrics = self.args.metrics

    def forward(self, z):
        z = self.model(z)
        return z

    def loss_fn(self, model_output, gt_segmentation_map, prefix='train_'):
        computed_losses = {}
        gt_segmentation_map = gt_segmentation_map.double()
        for i, loss in enumerate(self.losses):
            computed_losses[prefix+loss] = self.args.loss_contrib[loss]*self.losses[loss](model_output, gt_segmentation_map)
        loss = torch.sum(torch.stack(list(computed_losses.values())))
        computed_losses[prefix+"loss"] = loss
        self.log_dict(computed_losses)
        return loss
        

    def metric_fn(self, model_output, gt_segmentation_map, prefix='train_'):

        #model_output: batch_size x n_classes  x H x W
        #gt_segmentation_map : batch_size x 1  x H x W for multiclass, and batch_size x n_classes  x H x W
        # for multilabel and binary
        computed_metric = {}
        gt_segmentation_map = gt_segmentation_map.double()
        for metric in self.metrics:
            self.metrics[metric](model_output, gt_segmentation_map)
            computed_metric[prefix+metric] = self.metrics[metric].aggregate().item()
        self.log_dict(computed_metric)
        return computed_metric

    def compute_batch(self, batch, batch_idx, prefix='train_'):
        ct, gt_segmentation_map = batch
        if self.args.dataset_type == "3D":
            output = self(ct)[:,0,...].permute(0,3,1,2).contiguous().double()
        elif self.args.dataset_type == "2D":
            output = self(ct)["masks"].double()
        loss = self.loss_fn(output, gt_segmentation_map, prefix)
        metric = self.metric_fn(output, gt_segmentation_map, prefix)
        return loss, metric

    def training_step(self, batch, batch_idx):
        loss, metric = self.compute_batch(batch, batch_idx)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, metric = self.compute_batch(batch, batch_idx, "valid_")
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        ct, gt_segmentation_map = batch
        y_hat = self.model(ct)
        return y_hat

    def configure_optimizers(self):

        optimizer = self.args.optimizer(self.model.parameters(), **self.args.optimizer_params)
        scheduler = self.args.scheduler(optimizer, **self.args.scheduler_params)
        if 'ReduceLROnPlateau' in str(scheduler):
            return [optimizer], [{'scheduler': scheduler, 'monitor': 'valid_loss'}]
        if 'CyclicLR' in str(scheduler):
            scheduler.state_dict().pop("_scale_fn_ref")
            
        return [optimizer], [scheduler]


class qClassification(pl.LightningModule):
    def __init__(self, args, train=True):
        super().__init__()
        self.save_hyperparameters()
        self.args = args

        self.training_step_outputs = defaultdict(list)
        self.validation_step_outputs = defaultdict(list)
        
        if "mode" not in self.args.keys():
            warnings.warn(" `mode` not in arguments file, setting it to `multilabel` ")
            self.args["mode"] = "multilabel"

        self.setup_model(train)
        self.setup_losses()
        self.setup_metrics()
        
    def setup_model(self, train=True):
        if self.args.model_params_star_pass:
            self.model = self.args.model(**self.args.model_params).double()
        else:
            self.model = self.args.model(self.args).double()
        
        if train:
            self.model.train()
        else:
            self.model.eval()

    def setup_losses(self):
        self.loss_contrib = self.args.loss_contrib
        self.losses = self.args.losses

    def setup_metrics(self):
        self.metrics = self.args.metrics

    def forward(self, z):
        z = self.model(z)
        return z

    def loss_fn(self, model_output, gt_target, prefix='train_batch_'):
        computed_losses = {}
        for i, loss in enumerate(self.losses):
            computed_losses[prefix+loss] = self.args.loss_contrib[loss]*self.losses[loss](model_output, gt_target)
        loss = torch.sum(torch.stack(list(computed_losses.values())))
        computed_losses[prefix+"loss"] = loss
        self.log_dict(computed_losses)
        return loss
        

    def metric_fn(self, model_output, gt_target, prefix='train_batch_'):

        #model_output: batch_size x n_classes  x H x W
        #gt_segmentation_map : batch_size x 1  x H x W for multiclass, and batch_size x n_classes  x H x W
        # for multilabel and binary
        computed_metric = {}
        gt_segmentation_map = gt_segmentation_map.double()
        for metric in self.metrics:
            self.metrics[metric](model_output, gt_segmentation_map)
            computed_metric[prefix+metric] = self.metrics[metric].aggregate().item()
        self.log_dict(computed_metric)
        return computed_metric

    def compute_batch(self, batch, batch_idx, prefix='train_batch_'):
        ct, gt_segmentation_map = batch
        if self.args.dataset_type == "3D":
            output = self(ct)[:,0,...].permute(0,3,1,2).contiguous().double()
        elif self.args.dataset_type == "2D":
            output = self(ct)["masks"].double()
        loss = self.loss_fn(output, gt_segmentation_map, prefix)
        metric = self.metric_fn(output, gt_segmentation_map, prefix)
        return loss, metric

    def training_step(self, batch, batch_idx):
        loss, metric = self.compute_batch(batch, batch_idx)
        self.training_step_outputs["loss"].append(loss)
        self.training_step_outputs["metric"].append(metric)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, metric = self.compute_batch(batch, batch_idx, "valid_batch_")
        self.validation_step_outputs["loss"].append(loss)
        self.validation_step_outputs["metric"].append(metric)
        return loss
    
    def process_batch_for_epoch(self, batch_numbers, prefix='train'):
        epoch_prefix = prefix+"_epoch_"
        batch_prefix = prefix+"_batch_"
        epoch_end = defaultdict(list)
        for key in batch_numbers.keys():
            batch_numbers[key] = pd.DataFrame(batch_numbers[key]).to_dict(orient="list")
        for loss in self.losses:
            epoch_end[epoch_prefix+loss] = torch.stack(batch_numbers["loss"][batch_prefix+loss]).mean()
        for metric in self.metrics:
            epoch_end[epoch_prefix+metric] = torch.stack(batch_numbers["metric"][batch_prefix+metric]).mean()
        
        self.log_dict(epoch_end)

    def on_validation_epoch_end(self) -> None:
        self.process_batch_for_epoch(self.validation_step_outputs, prefix='valid')
        self.validation_step_outputs.clear()  # free memory
        return super().on_validation_epoch_end()

    def on_train_epoch_end(self) -> None:
        self.process_batch_for_epoch(self.training_step_outputs, prefix='train')
        self.training_step_outputs.clear()  # free memory
        return super().on_train_epoch_end()

    def configure_optimizers(self):
        optimizer = self.args.optimizer(self.model.parameters(), **self.args.optimizer_params)
        scheduler = self.args.scheduler(optimizer, **self.args.scheduler_params)
        if 'ReduceLROnPlateau' in str(scheduler):
            return [optimizer], [{'scheduler': scheduler, 'monitor': 'valid_loss'}]
        if 'CyclicLR' in str(scheduler):
            scheduler.state_dict().pop("_scale_fn_ref")
            
        return [optimizer], [scheduler]