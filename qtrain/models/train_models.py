import warnings

import munch
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
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
            if self.args.model == "unet":
                from qtrain.models.unet import Unet_Modified
                self.model = Unet_Modified(self.args).double()

            elif self.args.model == "unetpp":
                from qtrain.models.unet import UnetPlusPlus
                self.model = UnetPlusPlus(self.args).double()
            
            elif self.args.model == "transunet":
                from qtrain.models.transunet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
                margs = self.args.model_args
                margs.config_vit = CONFIGS[args.vit_name]
                margs.config_vit.n_classes = self.args.num_classes
                if args.vit_name.find('R50') != -1:
                    margs.config_vit.patches.grid = [int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size)]
                self.model = ViT_seg(margs.config_vit, img_size=self.args.img_size, num_classes=margs.config_vit.n_classes).double()
                self.model.load_from(weights=np.load(margs.config_vit.pretrained_path))

            elif self.args.model == "linknet":
                from qtrain.models.linknet import LinkNet
                self.model = LinkNet(self.args).double()
            
            elif self.args.model == "deeplabv3":
                from qtrain.models.deeplab import DeepLabV3
                self.model = DeepLabV3(self.args).double()
            
            elif self.args.model == "fcbformer":
                from qtrain.models.fcbformer.models.models import FCBFormer
                self.model = FCBFormer(self.args.img_size).double()

            else:
                raise ValueError("Model not supported")
                exit(0)
        
        if self.args.dataset_type == "3D":
            if self.args.model == "3dunet":
                from monai.networks.nets import UNet
                margs = self.args.model_args
                self.model = UNet(spatial_dims=margs.spatial_dims,
                                in_channels=margs.in_channels,
                                out_channels=margs.out_channels,
                                channels=margs.channels,
                                strides=margs.strides,
                                num_res_units=margs.num_res_units
                                ).double()
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
        if "focal" in self.args.losses:
            self.losses["focal"] = smp.losses.FocalLoss(mode=self.args.mode)
        if "dice" in self.args.losses:
            self.losses["dice"] = smp.losses.DiceLoss(mode=self.args.mode)
        if "mcc" in self.args.losses:
            if self.args.mode == "binary":
                self.losses["mcc"] = smp.losses.MCCLoss(mode=self.args.mode)
            else:
                warnings.warn(" 'non-binary' mode not supported for MCCLoss")
        

        if "boundary" in self.args.losses:
            from qtrain.losses.boundary_loss import BoundaryLoss
            self.losses["boundary"] = BoundaryLoss()
        if "iou" in self.args.losses:
            from qtrain.losses.dice_loss import IoULoss
            self.losses["iou"] = IoULoss()
        if "dice_v2" in self.args.losses:
            from qtrain.losses.dice_loss import GDiceLossV2
            self.losses["dice_v2"] = GDiceLossV2()
        

        if "crossentropy" in self.args.losses:
            self.losses["crossentropy"] = nn.CrossEntropyLoss()
        if "mse" in self.args.losses:
            self.losses["mse"] = nn.MSELoss()
        
        return

    def setup_metrics(self):
        self.metrics = {}
        if "dice" in self.args.metrics:
            from qtrain.metrics.dice import DiceCoeff
            self.metrics["dice"] = DiceCoeff(mode=self.args.mode)
            if self.args.metrics_nbg:
                self.metrics["dice_wbg"] = DiceCoeff(mode=self.args.mode, ignore_index=self.args.background_index)
        
        if "jaccard" in self.args.metrics:
            from qtrain.metrics.jaccard import JaccardIndex
            self.metrics["jaccard"] = JaccardIndex(mode=self.args.mode)

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
        computed_losses[prefix+"total_loss"] = loss
        self.log_dict(computed_losses)
        return loss
        

    def metric_fn(self, model_output, gt_segmentation_map, prefix='train_'):

        #model_output: batch_size x n_classes  x H x W
        #gt_segmentation_map : batch_size x 1  x H x W for multiclass, and batch_size x n_classes  x H x W
        # for multilabel and binary
        computed_metric = {}
        gt_segmentation_map = gt_segmentation_map.double()
        for metric in self.metrics:
            computed_metric[prefix+metric] = self.metrics[metric](model_output, gt_segmentation_map)
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
        loss, metric = self.compute_batch(batch, batch_idx, "valid")
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        ct, gt_segmentation_map = batch
        y_hat = self.model(ct)
        return y_hat

    def configure_optimizers(self):
        lr = self.args.lr
        
        if self.args.optimizer == "sgd":
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
            scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=0.2)
        elif self.args.optimizer == "adam":
            decay_step = [25, 75, 150, 230]
            decay_factor = 0.1
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=4e-5, eps=1e-4)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_step, gamma=decay_factor)
        elif self.args.optimizer == "adamw":
            decay_step = [25, 75, 150, 230]
            decay_factor = 0.1
            optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_step, gamma=decay_factor)
        elif self.args.optimizer == "sam":
            from qtrain.optimizers import SAM, optimize_with_SAM
            optimizer = SAM(self.model.parameters(), optim.SGD, lr=lr, momentum=0.9)
            scheduler = optim.lr_scheduler.StepLR(optimizer, lr, self.args.max_epoch)
            
        return [optimizer], [scheduler]