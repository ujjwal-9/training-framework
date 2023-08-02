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
from qtrain.models.unet.unet2dseqattn import UnetSeqAttn
from qtrain.models.unet.multitasker import MultiTaskSeqAttn
from torchmetrics.classification import JaccardIndex, AUROC

import torch
import torchmetrics as tm
import torch.nn.functional as F



class qMultiTasker(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        
        if "mode" not in self.args.keys():
            warnings.warn(" `mode` not in arguments file, setting it to `multilabel` ")
            self.args["mode"] = "multilabel"

        self.setup_model()
        if "model" not in self.args:
            self.args.model = "se_multitasker"
        self.nan_score = 0.0
        self.ignore_index = self.args.ignore_index

        self.seg_focal_loss = smp.losses.FocalLoss(mode="multiclass", ignore_index=self.ignore_index, reduction='none')
        self.seg_miou_metric = JaccardIndex(task="binary", num_classes=2, ignore_index=self.ignore_index)
        self.seg_ap_metric = tm.AveragePrecision(task="binary", num_classes=2, ignore_index=self.ignore_index)
        
        self.cls_ce_loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([2.,1.]), ignore_index=self.ignore_index)
        self.cls_auc_metric = AUROC(task="binary", num_classes=2, ignore_index=self.ignore_index)
        self.cls_ap_metric = tm.AveragePrecision(task="binary", num_classes=2, ignore_index=self.ignore_index)
        self.cls_sens_spec_metric = tm.StatScores(task="binary", num_classes=2, ignore_index=self.ignore_index)

        self.slc_bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.slc_auc_metric = AUROC(task="multiclass", num_classes=2, ignore_index=self.ignore_index)
        self.slc_ap_metric = tm.AveragePrecision(task="multiclass", num_classes=2, ignore_index=self.ignore_index)
        self.slc_sens_spec_metric = tm.StatScores(task="multiclass", num_classes=2, ignore_index=self.ignore_index)
        self.infarct_sens_spec_metric = tm.StatScores(task="multilabel", num_labels=2, ignore_index=self.ignore_index)
        
    def setup_model(self):
        if self.args.model == "multitask_qer":
            from qtrain.models.qer_multitask.multitask import MultiTaskNet
            self.model = MultiTaskNet(self.args.model_params)
        else:
            self.model = MultiTaskSeqAttn(self.args.model_params)

    def forward(self, z):
        z = self.model(z)
        return z
    
    def seg_loss_criterion(self, pred, gt, series):  
        seg_losses = {"focal": self.seg_focal_loss}

        total_loss = 0.0
        loss_dict = {}
        for key in self.args.seg_loss_wts:
            if torch.all(gt == self.ignore_index):
                loss_dict[key] = self.args.seg_loss_wts[key] * seg_losses[key](pred.view(-1, *pred.size()[2:]), torch.ones_like(gt.view(-1, *gt.shape[2:])))
                loss_dict[key] = torch.mean(loss_dict[key] * (gt.flatten()>=0))
            else:    
                loss_dict[key] = self.args.seg_loss_wts[key] * seg_losses[key](pred.view(-1, *pred.size()[2:]), gt.view(-1, *gt.shape[2:]))
                loss_dict[key] = torch.mean(loss_dict[key])
            total_loss += loss_dict[key]

        gt = gt.clone()
        gt[gt.sum(axis=(2,3))<=0] = self.ignore_index
        seg_metric = {"miou": self.seg_miou_metric(F.softmax(pred).argmax(2).detach(), gt),
                    #   "avg_precision": self.seg_ap_metric(pred_[:,:,0].detach(), gt),
                      }
        
        # print("seg: ", total_loss)
        return loss_dict, total_loss, seg_metric

    def cls_loss_criterion(self, pred, gt, series):
        cls_losses = {"ce": self.cls_ce_loss}
        total_loss = 0.0
        loss_dict = {}
        for key in self.args.cls_loss_wts:
            loss_dict[key] = self.args.cls_loss_wts[key] * cls_losses[key](pred, gt[:,0].to(torch.long))
            total_loss += loss_dict[key]
        tp, fp, tn, fn, sup = self.cls_sens_spec_metric(pred.detach().argmax(1), gt[:,0].detach().to(torch.long))
        cls_metric = {"auc": self.cls_auc_metric(pred.detach().argmax(1), gt[:,0].detach().to(torch.long)), 
                    #   "avg_precision": self.cls_ap_metric(pred.detach()[:,1], gt.detach()),
                      "sensitivity": tp/(tp+fn),
                      "specificity": tn/(tn+fp),
                      "youden": (tp/(tp+fn)) + (tn/(tn+fp)) - 1
                      }
        
        # print("cls: ", total_loss)
        return loss_dict, total_loss, cls_metric
        
    def slc_loss_criterion(self, pred, gt, series):
        slc_losses = {"bce": self.slc_bce_loss}
        target_score = gt.sum(axis=(2,3))
        
        trg_index = torch.where((gt.sum(axis=(2,3))>=0).sum(axis=1)==0)[0].tolist()
        for idx in trg_index:
            target_score[idx] = torch.ones_like(target_score[idx])*-100
        
        target_score[target_score>=1] = 1
        target_score[target_score==0] = 0
        target_score[target_score<0] = -100

        total_loss = 0.0
        loss_dict = {}
        for key in self.args.slc_loss_wts:
            loss_dict[key] = self.args.slc_loss_wts[key] * slc_losses[key](pred[:,:,1].double(), target_score.double()) * (target_score>=0)
            loss_dict[key] = torch.mean(loss_dict[key])
            total_loss += loss_dict[key]

        tp, fp, tn, fn, sup = self.slc_sens_spec_metric(F.softmax(pred).detach().permute(0,2,1), target_score.detach())   
        slc_metric = {
                    # "auc": self.slc_auc_metric(pred_.detach().permute(0,2,1), target_score.detach()), 
                    # "avg_precision": self.slc_ap_metric(pred_.detach().permute(0,2,1), target_score.detach()),
                    "sensitivity": tp/(tp+fn),
                    "specificity": tn/(tn+fp),
                    "youden": (tp/(tp+fn)) + (tn/(tn+fp)) - 1
                    }
        
        # print("slc: ", total_loss)
        return loss_dict, total_loss, slc_metric
    
    def infarct_loss_criterion(self, pred, target_score, series):
        slc_losses = {"bce": self.slc_bce_loss}

        total_loss = 0.0
        loss_dict = {}
        for key in self.args.slc_loss_wts:
            loss_dict[key] = self.args.slc_loss_wts[key] * slc_losses[key](pred.double(), target_score.double())*(target_score>=0)
            loss_dict[key] = torch.mean(loss_dict[key])
            total_loss += loss_dict[key]

        tp, fp, tn, fn, sup = self.infarct_sens_spec_metric(F.sigmoid(pred).detach(), target_score.detach())   
        slc_metric = {
                    # "auc": self.slc_auc_metric(pred_.detach().permute(0,2,1), target_score.detach()), 
                    # "avg_precision": self.slc_ap_metric(pred_.detach().permute(0,2,1), target_score.detach()),
                    "sensitivity": tp/(tp+fn),
                    "specificity": tn/(tn+fp),
                    "youden": (tp/(tp+fn)) + (tn/(tn+fp)) - 1
                    }
        
        # print("infarct: ", total_loss)
        return loss_dict, total_loss, slc_metric

    def loss_criterion(self, series, pred, gt, trg_cls, infarct_cls, prefix="train"):
        torch.nan_to_num_(gt, nan=self.nan_score, posinf=self.nan_score, neginf=self.nan_score)
        torch.nan_to_num_(trg_cls, nan=self.nan_score, posinf=self.nan_score, neginf=self.nan_score)
        normal_loss_dict, normal_loss, normal_metric = self.cls_loss_criterion(pred["normal_logits"], trg_cls, series)
        slc_loss_dict, slc_loss, slc_metric = self.slc_loss_criterion(pred["slc_logits"], gt, series)        
        infarct_type_loss_dict, infarct_type_loss, infarct_type_metric = self.infarct_loss_criterion(pred["acute_chronic_logits"], infarct_cls, series)
        seg_loss_dict, seg_loss, seg_metric = self.seg_loss_criterion(pred["masks"], gt, series)
        
        loss = seg_loss + slc_loss + normal_loss + infarct_type_loss
        
        losses = {
            "seg": seg_loss_dict,
            "slc": slc_loss_dict,
            "normal": normal_loss_dict,
            "infarct": infarct_type_loss_dict,
        }

        metrics = {
            "seg": seg_metric,
            "slc": slc_metric,
            "normal": normal_metric,
            "infarct": infarct_type_metric
        }
        
        log_losses = {}
        log_metrics = {}
        
        for key in losses:
            if losses[key] is None:
                continue
            else:
                for l in losses[key]:
                    log_losses[prefix+"_"+key+"_"+l] = losses[key][l]

        for key in metrics:
            if metrics[key] is None:
                continue
            else:
                for l in metrics[key]:
                    if torch.isnan(metrics[key][l]):
                        metrics[key][l] = torch.tensor(self.nan_score, device=self.device)
                    log_metrics[prefix+"_"+key+"_"+l] = metrics[key][l]
        
        return loss, log_losses, log_metrics
    
    def compute_batch(self, batch, batch_idx, prefix='train'):
        ct, gt_segmentation_map, trg_cls, infarct_cls, series = batch
        torch.nan_to_num_(ct, nan=self.nan_score, posinf=self.nan_score, neginf=self.nan_score)
        output = self(ct)
        loss, log_losses, log_metrics = self.loss_criterion(series, output, gt_segmentation_map, trg_cls, infarct_cls, prefix)
        return loss, log_losses, log_metrics

    def training_step(self, batch, batch_idx):
        loss, log_losses, log_metrics = self.compute_batch(batch, batch_idx)
        self.log_dict(log_losses, on_step=False,
                      on_epoch=True, batch_size=self.args.batch_size, 
                      prog_bar=True,rank_zero_only=True)
        self.log("train_loss", loss, on_step=False,
                      on_epoch=True, batch_size=self.args.batch_size, 
                      prog_bar=True,rank_zero_only=True)
        self.log_dict(log_metrics, on_step=False,
                      on_epoch=True, batch_size=self.args.batch_size, 
                      prog_bar=True,rank_zero_only=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, log_losses, log_metrics = self.compute_batch(batch, batch_idx, prefix="valid")
        self.log_dict(log_losses, on_step=False,
                      on_epoch=True, batch_size=self.args.batch_size, 
                      prog_bar=True,rank_zero_only=True)
        self.log("valid_loss", loss, on_step=False,
                      on_epoch=True, batch_size=self.args.batch_size, 
                      prog_bar=True,rank_zero_only=True)
        self.log_dict(log_metrics, on_step=False,
                      on_epoch=True, batch_size=self.args.batch_size, 
                      prog_bar=True,rank_zero_only=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        ct, gt_segmentation_map, trg_cls = batch
        y_hat = self.model(ct)
        return y_hat

    def configure_optimizers(self):
        optimizer = self.args.optimizer(self.model.parameters(), **self.args.optimizer_params)
        scheduler = self.args.scheduler(optimizer, **self.args.scheduler_params)
        if 'ReduceLROnPlateau' in str(scheduler):
            return [optimizer], [{'scheduler': scheduler, 'monitor': 'valid/loss'}]
        if 'CyclicLR' in str(scheduler):
            scheduler.state_dict().pop("_scale_fn_ref")
            
        return [optimizer], [scheduler]


class qSegAndClass(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        
        if "mode" not in self.args.keys():
            warnings.warn(" `mode` not in arguments file, setting it to `multilabel` ")
            self.args["mode"] = "multilabel"

        self.setup_model()
        self.nan_score = 0.0

        self.seg_focal_loss = smp.losses.FocalLoss(mode="multiclass", ignore_index=-100)
        self.seg_miou_metric = JaccardIndex(task="binary", num_classes=2, ignore_index=-100)
        self.seg_auc_metric = AUROC(task="binary", num_classes=2, ignore_index=-100)
        self.seg_ap_metric = tm.AveragePrecision(task="binary", num_classes=2, ignore_index=-100)
        
        self.cls_ce_loss = nn.CrossEntropyLoss()
        self.cls_auc_metric = AUROC(task="binary", num_classes=2, ignore_index=-100)
        self.cls_ap_metric = tm.AveragePrecision(task="binary", num_classes=2, ignore_index=-100)
        self.cls_sens_spec_metric = tm.StatScores(task="binary", num_classes=2, ignore_index=-100)

        self.slc_bce_loss = nn.BCEWithLogitsLoss(size_average=True)
        self.slc_auc_metric = AUROC(task="multiclass", num_classes=2, ignore_index=-100)
        self.slc_ap_metric = tm.AveragePrecision(task="multiclass", num_classes=2, ignore_index=-100)
        self.slc_sens_spec_metric = tm.StatScores(task="multiclass", num_classes=2, ignore_index=-100)
        
    def setup_model(self):
        self.model = UnetSeqAttn(self.args.model_params)

    def forward(self, z):
        z = self.model(z)
        return z
    
    def seg_loss_criterion(self, pred, gt):  
        seg_losses = {"focal": self.seg_focal_loss,
                      "bce": F.binary_cross_entropy_with_logits}

        total_loss = 0.0
        loss_dict = {}
        for key in self.args.seg_loss_wts:
            loss_dict[key] = self.args.seg_loss_wts[key] * seg_losses[key](pred.view(-1, *pred.size()[2:]), gt.view(-1, *gt.shape[2:]))
            if torch.isnan(loss_dict[key]):
                # loss_dict[key] = torch.tensor(self.nan_score, device=self.device, requires_grad=True)
                loss_dict[key][torch.isnan(loss_dict[key])] = self.nan_score
            total_loss += loss_dict[key]

        gt = gt.clone()
        gt[gt.sum(axis=(2,3))<=0] = -100
        seg_metric = {"miou": self.seg_miou_metric(F.softmax(pred).argmax(2).detach(), gt),
                    #   "avg_precision": self.seg_ap_metric(pred_[:,:,0].detach(), gt),
                    #   "auc": self.seg_auc_metric(pred_[:,:,0].detach(), gt)
                      }
        
        # indices = torch.where(trg_cls>0)[0]
        # pred_ = torch.index_select(pred, 0, indices)[:,:,0]
        # gt_ = torch.index_select(gt, 0, indices)
        
        # if torch.numel(pred_) == 0:
        #     return loss_dict, total_loss, {"miou": None}
        
        # metric = {"miou": self.metric((pred_>0.5).to(torch.int), gt_)}
        # return loss_dict, total_loss, metric
        # print("focal: ", total_loss)
        return loss_dict, total_loss, seg_metric

    def cls_loss_criterion(self, pred, gt):
        cls_losses = {"ce": self.cls_ce_loss}
        total_loss = 0.0
        loss_dict = {}
        for key in self.args.cls_loss_wts:
            loss_dict[key] = self.args.cls_loss_wts[key] * cls_losses[key](pred, gt[:,0].to(torch.long))
            if torch.isnan(loss_dict[key]):
                # loss_dict[key] = torch.tensor(self.nan_score, device=self.device, requires_grad=True)
                loss_dict[key][torch.isnan(loss_dict[key])] = self.nan_score
            total_loss += loss_dict[key]
        tp, fp, tn, fn, sup = self.cls_sens_spec_metric(pred.detach().argmax(1), gt[:,0].detach().to(torch.long))
        cls_metric = {"auc": self.cls_auc_metric(pred.detach().argmax(1), gt[:,0].detach().to(torch.long)), 
                    #   "avg_precision": self.cls_ap_metric(pred.detach()[:,1], gt.detach()),
                      "sensitivity": tp/(tp+fn),
                      "specificity": tn/(tn+fp),
                      "youden": (tp/(tp+fn)) + (tn/(tn+fp)) - 1
                      }
        
        # print("ce: ", total_loss)
        return loss_dict, total_loss, cls_metric

    # def slc_loss_criterion(self, pred, gt, trg_cls):
    #     NULL_SUM = torch.numel(torch.Tensor(gt.shape[1:]))*-100
    #     indices = torch.where((gt.sum(axis=(1,2,3)) == NULL_SUM) != True, gt, -100)[0]
    #     target_score = torch.index_select(gt, 0, indices)
    #     if torch.numel(target_score) == 0:
    #         return None, None, None
    #     else:
    #         target_score = (target_score.sum(axis=(2,3))>0).to(torch.float).contiguous()
    #         pred = torch.index_select(pred, 0, indices)
    #         slc_losses = {"bce": nn.BCEWithLogitsLoss(size_average=True)}
    #         total_loss = 0.0
    #         loss_dict = {}
    #         for key in self.args.slc_loss_wts:
    #             loss_dict[key] = self.args.slc_loss_wts[key] * slc_losses[key](pred[:,:,1], target_score)
    #             total_loss += loss_dict[key]

    #         slc_metric = {"auc": self.slc_auc_metric(pred[:,:,1].detach(), target_score.detach()), 
    #                   "precision": self.slc_loss_criterionc_precision_metric(pred[:,:,1].detach(), target_score.detach()),
    #                   "recall": self.slc_recall_metric(pred[:,:,1].detach(), target_score.detach())}
    #         return loss_dict, total_loss, slc_metric
        
    def slc_loss_criterion(self, pred, gt):
        slc_losses = {"bce": self.slc_bce_loss}
        target_score = gt.sum(axis=(2,3))
        
        trg_index = torch.where((gt.sum(axis=(2,3))>=0).sum(axis=1)==0)[0].tolist()
        for idx in trg_index:
            target_score[idx] = torch.ones_like(target_score[idx])*-100
        
        target_score[target_score>=1] = 1
        target_score[target_score==0] = 0
        target_score[target_score<0] = -100

        total_loss = 0.0
        loss_dict = {}
        for key in self.args.slc_loss_wts:
            loss_dict[key] = self.args.slc_loss_wts[key] * slc_losses[key](pred[:,:,1][target_score>=0].double(), target_score[target_score>=0].double())
            if torch.isnan(loss_dict[key]):
                # loss_dict[key] = torch.tensor(self.nan_score, device=self.device, requires_grad=True)
                loss_dict[key][torch.isnan(loss_dict[key])] = self.nan_score
            total_loss += loss_dict[key]

        tp, fp, tn, fn, sup = self.slc_sens_spec_metric(F.softmax(pred).detach().permute(0,2,1), target_score.detach())   
        slc_metric = {
                    # "auc": self.slc_auc_metric(pred_.detach().permute(0,2,1), target_score.detach()), 
                    # "avg_precision": self.slc_ap_metric(pred_.detach().permute(0,2,1), target_score.detach()),
                    "sensitivity": tp/(tp+fn),
                    "specificity": tn/(tn+fp),
                    "youden": (tp/(tp+fn)) + (tn/(tn+fp)) - 1
                    }
        
        # print("bce: ", total_loss)
        return loss_dict, total_loss, slc_metric

    def loss_criterion(self, pred, gt, trg_cls, prefix="train"):
        torch.nan_to_num_(gt)
        torch.nan_to_num_(trg_cls)
        cls_loss_dict, cls_loss, cls_metric = self.cls_loss_criterion(pred["cls_logits"], trg_cls)
        slc_loss_dict, slc_loss, slc_metric = self.slc_loss_criterion(pred["slc_logits"], gt)
        seg_loss_dict, seg_loss, seg_metric = self.seg_loss_criterion(pred["masks"], gt)
        
        losses_ = [slc_loss, cls_loss]
        loss = seg_loss
        for loss_ in losses_:
            if loss_ is not None:
                loss += loss_
        losses = {
            "seg": seg_loss_dict,
            "slc": slc_loss_dict,
            "cls": cls_loss_dict
        }

        metrics = {
            "seg": seg_metric,
            "slc": slc_metric,
            "cls": cls_metric
        }
        
        log_losses = {}
        log_metrics = {}
        
        for key in losses:
            if losses[key] is None:
                continue
            else:
                for l in losses[key]:
                    log_losses[prefix+"_"+key+"_"+l] = losses[key][l]

        for key in metrics:
            if metrics[key] is None:
                continue
            else:
                for l in metrics[key]:
                    if torch.isnan(metrics[key][l]):
                        metrics[key][l] = torch.tensor(self.nan_score, device=self.device)
                    log_metrics[prefix+"_"+key+"_"+l] = metrics[key][l]
        
        return loss, log_losses, log_metrics
    
    def compute_batch(self, batch, batch_idx, prefix='train'):
        ct, gt_segmentation_map, trg_cls = batch
        torch.nan_to_num_(ct)
        output = self(ct)
        loss, log_losses, log_metrics = self.loss_criterion(output, gt_segmentation_map, trg_cls, prefix)
        return loss, log_losses, log_metrics

    def training_step(self, batch, batch_idx):
        loss, log_losses, log_metrics = self.compute_batch(batch, batch_idx)
        self.log_dict(log_losses, on_step=False,
                      on_epoch=True, batch_size=self.args.batch_size, 
                      prog_bar=True,rank_zero_only=True)
        self.log("train_loss", loss, on_step=False,
                      on_epoch=True, batch_size=self.args.batch_size, 
                      prog_bar=True,rank_zero_only=True)
        self.log_dict(log_metrics, on_step=False,
                      on_epoch=True, batch_size=self.args.batch_size, 
                      prog_bar=True,rank_zero_only=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, log_losses, log_metrics = self.compute_batch(batch, batch_idx, prefix="valid")
        self.log_dict(log_losses, on_step=False,
                      on_epoch=True, batch_size=self.args.batch_size, 
                      prog_bar=True,rank_zero_only=True)
        self.log("valid_loss", loss, on_step=False,
                      on_epoch=True, batch_size=self.args.batch_size, 
                      prog_bar=True,rank_zero_only=True)
        self.log_dict(log_metrics, on_step=False,
                      on_epoch=True, batch_size=self.args.batch_size, 
                      prog_bar=True,rank_zero_only=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        ct, gt_segmentation_map, trg_cls = batch
        y_hat = self.model(ct)
        return y_hat

    def configure_optimizers(self):
        optimizer = self.args.optimizer(self.model.parameters(), **self.args.optimizer_params)
        scheduler = self.args.scheduler(optimizer, **self.args.scheduler_params)
        if 'ReduceLROnPlateau' in str(scheduler):
            return [optimizer], [{'scheduler': scheduler, 'monitor': 'valid/loss'}]
        if 'CyclicLR' in str(scheduler):
            scheduler.state_dict().pop("_scale_fn_ref")
            
        return [optimizer], [scheduler]



class qSegmentation(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        
        if "mode" not in self.args.keys():
            warnings.warn(" `mode` not in arguments file, setting it to `multilabel` ")
            self.args["mode"] = "multilabel"

        self.setup_model()
        self.setup_losses()
        self.setup_metrics()
        
    def setup_model(self):
        if self.args.model_params_star_pass:
            self.model = self.args.model(**self.args.model_params).double()
        else:
            self.model = self.args.model(self.args).double()

    def setup_losses(self):
        self.loss_contrib = self.args.loss_contrib
        self.losses = self.args.losses

    def setup_metrics(self):
        self.metrics = self.args.metrics

    def forward(self, z):
        z = self.model(z)
        return z

    
    def seg_loss_criterion(pred, gt):  
        seg_losses = {"focal": smp.losses.FocalLoss(mode="binary"),
                "dice": GDiceLossV2(),
                "mcc": smp.losses.MCCLoss()}

        total_loss = 0.0
        loss_dict = {}
        for key in args_terminal.seg_loss_wts:
            loss_dict[key] = args_terminal.seg_loss_wts[key] * seg_losses[key](pred, gt)
            total_loss += loss_dict[key]

        metric = miou((pred>0.5).to(torch.int), gt)
        return loss_dict, total_loss, metric

    def cls_loss_criterion(pred, gt):
        target = ((gt.sum(axis=(1,2,3)))>0).to(torch.long)
        cls_losses = {"ce": nn.CrossEntropyLoss(weight=ce_wts)}
        total_loss = 0.0
        loss_dict = {}
        for key in args_terminal.cls_loss_wts:
            loss_dict[key] = args_terminal.cls_loss_wts[key] * cls_losses[key](pred, target)
            total_loss += loss_dict[key]

        return loss_dict, total_loss, target

    def slc_loss_criterion(pred, gt):
        target_score = (gt.sum(axis=(2,3))>0).to(torch.float).contiguous()
        slc_losses = {"bce": nn.BCEWithLogitsLoss(size_average=True)}
        total_loss = 0.0
        loss_dict = {}
        for key in args_terminal.slc_loss_wts:
            loss_dict[key] = args_terminal.slc_loss_wts[key] * slc_losses[key](pred[:,:,1], target_score)
            total_loss += loss_dict[key]

        return loss_dict, total_loss
    
    
    def loss_fn(self, model_output, gt_segmentation_map, prefix='train/'):
        computed_losses = {}
        gt_segmentation_map = gt_segmentation_map.double()
        for i, loss in enumerate(self.losses):
            computed_losses[prefix+loss] = self.args.loss_contrib[loss]*self.losses[loss](model_output, gt_segmentation_map)
        loss = torch.sum(torch.stack(list(computed_losses.values())))
        computed_losses[prefix+"loss"] = loss
        self.log_dict(computed_losses)
        return loss
        

    def metric_fn(self, model_output, gt_segmentation_map, prefix='train/'):

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

    def compute_batch(self, batch, batch_idx, prefix='train/'):
        ct, gt_segmentation_map = batch
        if self.args.dataset_type == "3D":
            output = self(ct.permute(0,1,4,2,3))[-1].contiguous().double()
        elif self.args.dataset_type == "2D":
            output = self(ct)["masks"].double()
        loss = self.loss_fn(output, gt_segmentation_map, prefix)
        metric = self.metric_fn(output, gt_segmentation_map, prefix)
        return loss, metric

    def training_step(self, batch, batch_idx):
        loss, metric = self.compute_batch(batch, batch_idx)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, metric = self.compute_batch(batch, batch_idx, "valid/")
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        ct, gt_segmentation_map = batch
        y_hat = self.model(ct)
        return y_hat

    def configure_optimizers(self):

        optimizer = self.args.optimizer(self.model.parameters(), **self.args.optimizer_params)
        scheduler = self.args.scheduler(optimizer, **self.args.scheduler_params)
        if 'ReduceLROnPlateau' in str(scheduler):
            return [optimizer], [{'scheduler': scheduler, 'monitor': 'valid/loss'}]
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

    def loss_fn(self, model_output, gt_target, prefix='train/batch_'):
        computed_losses = {}
        for i, loss in enumerate(self.losses):
            computed_losses[prefix+loss] = self.args.loss_contrib[loss]*self.losses[loss](model_output, gt_target)
        loss = torch.sum(torch.stack(list(computed_losses.values())))
        computed_losses[prefix+"loss"] = loss
        self.log_dict(computed_losses)
        return loss
        

    def metric_fn(self, model_output, gt_target, prefix='train/batch_'):

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

    def compute_batch(self, batch, batch_idx, prefix='train/batch_'):
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
        loss, metric = self.compute_batch(batch, batch_idx, "valid/batch_")
        self.validation_step_outputs["loss"].append(loss)
        self.validation_step_outputs["metric"].append(metric)
        return loss
    
    def process_batch_for_epoch(self, batch_numbers, prefix='train/'):
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