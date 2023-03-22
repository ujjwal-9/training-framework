import warnings

import munch
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import monai.losses as L
import monai.metrics as M
import torch.nn.functional as F
import pytorch_lightning as pl
import segmentation_models_pytorch as smp


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
        
    def setup_model(self, train):
        if self.args.dataset_type == "2D":
            raise ValueError("Model not supported")
            exit(0)
        
        if self.args.dataset_type == "3D":
            if self.args.model == "3dunet":
                from monai.networks.nets import UNet
                self.model = UNet(**self.args.model_params).double()
            else:
                raise ValueError("Model not supported")
                exit(0)

        if train:
            self.model.train()
        else:
            self.model.eval()

    def setup_losses(self):
        self.loss_contrib = self.args.loss_contrib
        self.losses = {}
        for loss in self.args.losses:
            key = str(loss).split("(")[0]
            self.losses[key] = loss
        return

    def setup_metrics(self):
        self.metrics = {}
        for metric in self.args.metrics:
            key = str(metric).split(" ")[0].split(".")[-1]
            self.metrics[key] = metric
        return

    def forward(self, z):
        z = self.model(z)
        return z

    def loss_fn(self, model_output, gt_segmentation_map, prefix='train_'):
        computed_losses = {}
        gt_segmentation_map = gt_segmentation_map.double()
        for i, loss in enumerate(self.losses):
            computed_losses[prefix+loss] = self.args.loss_contrib[i]*self.losses[loss](model_output, gt_segmentation_map)
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
        output = self(ct)[:,0,...].permute(0,3,1,2).contiguous().double()
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
            
        return [optimizer], [scheduler]