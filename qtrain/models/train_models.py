import warnings

import qtrain
import munch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from collections import defaultdict
from qtrain.utils import get_sens_spec_youden, put_torchmetric_to_device
from qtrain.models.utils import freeze_layers, unfreeze_layers


class qMultiTasker(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.nan_score = 0.0
        self.ignore_index = self.args.ignore_index
        if "mode" not in self.args.keys():
            warnings.warn(" `mode` not in arguments file, setting it to `multilabel` ")
            self.args["mode"] = "multilabel"

        if "model" not in self.args:
            self.args.model = "se_multitasker"

        self.setup_model()
        self.setup_epoch_storage()
        self.point_estimates = self.args.point_estimates
        self.infarcts_type = ["acute", "chronic"]

    def setup_epoch_storage(self):
        self.seg_storage = defaultdict(dict)
        self.slc_storage = defaultdict(dict)
        self.infarcts_storage = defaultdict(dict)

    def setup_model(self):
        if self.args.model == "multitask_qer":
            from qtrain.models.qer_multitask.multitask import MultiTaskNet

            self.model = MultiTaskNet(self.args.model_params)
        elif self.args.model == "se_multitasker":
            from qtrain.models.unet.multitasker import MultiTaskSeqAttn

            self.model = MultiTaskSeqAttn(self.args.model_params)

    def forward(self, z):
        z = self.model(z)
        return z

    def seg_loss_criterion(self, pred, gt, series, prefix):
        total_loss = 0.0
        loss_dict = defaultdict()
        for key in self.point_estimates["seg"]["loss_fns"]:
            if torch.all(gt == self.ignore_index):
                loss_dict[key] = self.point_estimates["seg"]["loss_wts"][
                    key
                ] * put_torchmetric_to_device(
                    self.point_estimates["seg"]["loss_fns"][key], self.device
                )(
                    pred.view(-1, *pred.size()[2:]),
                    torch.ones_like(gt.view(-1, *gt.shape[2:])),
                )
                loss_dict[key] = torch.mean(loss_dict[key] * (gt.flatten() >= 0))
            else:
                loss_dict[key] = self.point_estimates["seg"]["loss_wts"][
                    key
                ] * self.point_estimates["seg"]["loss_fns"][key](
                    pred.view(-1, *pred.size()[2:]), gt.view(-1, *gt.shape[2:])
                )
                loss_dict[key] = torch.mean(loss_dict[key])
            total_loss += loss_dict[key]

        gt = gt.clone()
        gt[gt.sum(axis=(2, 3)) <= 0] = self.ignore_index
        pred_ = F.softmax(pred.detach()).argmax(2)

        if len(self.seg_storage[prefix].items()) == 0:
            self.seg_storage[prefix] = defaultdict(list)

        for key in self.point_estimates["seg"]["batch_metrics"]:
            self.seg_storage[prefix][key].extend(
                put_torchmetric_to_device(
                    self.point_estimates["seg"]["batch_metrics"][key], self.device
                )(
                    pred_.view(-1, *pred_.size()[2:]), gt.view(-1, *gt.size()[2:])
                ).tolist()
            )

        seg_metric = defaultdict()
        return loss_dict, total_loss, seg_metric

    # def cls_loss_criterion(self, pred, gt, series, prefix):
    #     total_loss = 0.0
    #     loss_dict = defaultdict()
    #     for key in self.point_estimates["cls"]["loss_fns"]:
    #         loss_dict[key] = self.point_estimates["cls"]["loss_wts"][
    #             key
    #         ] * self.point_estimates["cls"]["loss_fns"][key](
    #             pred, gt[:, 0].to(torch.long)
    #         )
    #         total_loss += loss_dict[key]

    #     tp, fp, tn, fn, sup = self.cls_sens_spec_metric(
    #         pred.detach().argmax(1), gt[:, 0].detach().to(torch.long)
    #     )
    #     cls_metric = {
    #         "auc": self.cls_auc_metric(
    #             pred.detach().argmax(1), gt[:, 0].detach().to(torch.long)
    #         ),
    #         "sensitivity": tp / (tp + fn),
    #         "specificity": tn / (tn + fp),
    #         "youden": (tp / (tp + fn)) + (tn / (tn + fp)) - 1,
    #     }

    #     # print("cls: ", total_loss)
    #     return loss_dict, total_loss, cls_metric

    def slc_loss_criterion(self, pred, gt, series, prefix):
        target_score = gt.sum(axis=(2, 3))
        trg_index = torch.where((gt.sum(axis=(2, 3)) >= 0).sum(axis=1) == 0)[0].tolist()
        for idx in trg_index:
            target_score[idx] = torch.ones_like(target_score[idx]) * -100

        target_score[target_score >= 1] = 1
        target_score[target_score == 0] = 0
        target_score[target_score < 0] = -100

        total_loss = 0.0
        loss_dict = defaultdict()
        for key in self.point_estimates["slc"]["loss_fns"]:
            loss_dict[key] = (
                self.point_estimates["slc"]["loss_wts"][key]
                * put_torchmetric_to_device(
                    self.point_estimates["slc"]["loss_fns"][key], self.device
                )(pred[:, :, 1].double(), target_score.double())
                * (target_score >= 0)
            )
            loss_dict[key] = torch.mean(loss_dict[key])
            total_loss += loss_dict[key]

        if len(self.slc_storage[prefix].items()) == 0:
            self.slc_storage[prefix]["scores"] = []
            self.slc_storage[prefix]["gt"] = []
            self.slc_storage[prefix]["confusion_matrix"] = defaultdict(int)

        pred_ = F.softmax(pred.detach()).permute(0, 2, 1)
        self.slc_storage[prefix]["scores"].append(pred_.detach())
        self.slc_storage[prefix]["gt"].append(target_score.detach())

        for key in self.point_estimates["slc"]["batch_metrics"]:
            if key == "stats":
                tp, fp, tn, fn, sup = put_torchmetric_to_device(
                    self.point_estimates["slc"]["batch_metrics"][key], self.device
                )(pred_, target_score.detach())
                self.slc_storage[prefix]["confusion_matrix"]["tp"] += tp.item()
                self.slc_storage[prefix]["confusion_matrix"]["fp"] += fp.item()
                self.slc_storage[prefix]["confusion_matrix"]["tn"] += tn.item()
                self.slc_storage[prefix]["confusion_matrix"]["fn"] += fn.item()

        slc_metric = defaultdict()
        # print("slc: ", total_loss)
        return loss_dict, total_loss, slc_metric

    def infarct_loss_criterion(self, pred, target_score, series, prefix):
        total_loss = 0.0
        loss_dict = defaultdict()
        for key in self.point_estimates["infarct"]["loss_wts"]:
            loss_dict[key] = (
                self.point_estimates["infarct"]["loss_wts"][key]
                * put_torchmetric_to_device(
                    self.point_estimates["infarct"]["loss_fns"][key], self.device
                )(pred.double(), target_score.double())
                * (target_score >= 0)
            )
            loss_dict[key] = torch.mean(loss_dict[key])
            total_loss += loss_dict[key]

        pred_ = F.sigmoid(pred.detach())
        if len(self.infarcts_storage[prefix].items()) == 0:
            for k in self.infarcts_type:
                self.infarcts_storage[prefix][f"{k}_scores"] = []
                self.infarcts_storage[prefix][f"{k}_gt"] = []
                self.infarcts_storage[prefix][f"{k}_confusion_matrix"] = defaultdict(
                    int
                )

        for key in self.point_estimates["infarct"]["batch_metrics"]:
            if key == "stats":
                stats = put_torchmetric_to_device(
                    self.point_estimates["infarct"]["batch_metrics"][key], self.device
                )(pred_, target_score.detach())
                for i, k in enumerate(self.infarcts_type):
                    tp, fp, tn, fn, sup = stats[i]
                    self.infarcts_storage[prefix][f"{k}_confusion_matrix"][
                        "tp"
                    ] += tp.item()
                    self.infarcts_storage[prefix][f"{k}_confusion_matrix"][
                        "fp"
                    ] += fp.item()
                    self.infarcts_storage[prefix][f"{k}_confusion_matrix"][
                        "tn"
                    ] += tn.item()
                    self.infarcts_storage[prefix][f"{k}_confusion_matrix"][
                        "fn"
                    ] += fn.item()

        for i, k in enumerate(self.infarcts_type):
            self.infarcts_storage[prefix][f"{k}_scores"].append(pred_[:, i])
            self.infarcts_storage[prefix][f"{k}_gt"].append(target_score[:, i].detach())

        infarct_metric = defaultdict()
        # print("infarct: ", total_loss)
        return loss_dict, total_loss, infarct_metric

    def loss_criterion(self, series, pred, gt, trg_cls, infarct_cls, prefix="train"):
        torch.nan_to_num_(
            gt, nan=self.nan_score, posinf=self.nan_score, neginf=self.nan_score
        )
        torch.nan_to_num_(
            trg_cls, nan=self.nan_score, posinf=self.nan_score, neginf=self.nan_score
        )

        log_losses = defaultdict()
        loss, losses = 0.0, {}

        if "infarct" in self.args.tasks:
            (
                infarct_type_loss_dict,
                infarct_type_loss,
                infarct_type_metric,
            ) = self.infarct_loss_criterion(
                pred["acute_chronic_logits"], infarct_cls, series, prefix
            )
            loss += infarct_type_loss
            losses["infarct"] = infarct_type_loss_dict

        if "slc" in self.args.tasks:
            slc_loss_dict, slc_loss, slc_metric = self.slc_loss_criterion(
                pred["slc_logits"], gt, series, prefix
            )
            loss += slc_loss
            losses["slc"] = slc_loss_dict

        if "seg" in self.args.tasks:
            seg_loss_dict, seg_loss, seg_metric = self.seg_loss_criterion(
                pred["masks"], gt, series, prefix
            )
            loss += seg_loss
            losses["seg"] = seg_loss_dict

        for key in losses:
            if losses[key] is None:
                continue
            else:
                for l in losses[key]:
                    log_losses[prefix + "_" + key + "_" + l] = losses[key][l]

        log_metrics = defaultdict()
        # metrics = {
        #     "seg": seg_metric,
        #     "slc": slc_metric,
        # "normal": normal_metric,
        #     "infarct": infarct_type_metric
        # }
        # for key in metrics:
        #     if metrics[key] is None:
        #         continue
        #     else:
        #         for l in metrics[key]:
        #             if type(metrics[key][l]) == list:
        #                 metrics[key][l] = torch.mean(torch.tensor(metrics[key][l]))
        #             # if torch.isnan(metrics[key][l]):
        #             #     metrics[key][l] = torch.tensor(self.nan_score, device=self.device)
        #             log_metrics[prefix+"_"+key+"_batch_"+l] = metrics[key][l]

        return loss, log_losses, log_metrics

    def compute_epoch_metric(self, prefix):
        epoch_metric = defaultdict()
        if "seg" in self.args.tasks:
            epoch_metric[f"{prefix}_seg_epoch_miou"] = torch.mean(
                torch.tensor(self.seg_storage[prefix]["miou"])
            )

        if "slc" in self.args.tasks:
            for key in self.point_estimates["slc"]["epoch_metrics"]:
                epoch_metric[f"{prefix}_slc_epoch_{key}"] = put_torchmetric_to_device(
                    self.point_estimates["slc"]["epoch_metrics"][key], self.device
                )(
                    torch.vstack(self.slc_storage[prefix]["scores"]),
                    torch.vstack(self.slc_storage[prefix]["gt"]),
                )

            (
                epoch_metric[f"{prefix}_slc_epoch_sensitivity"],
                epoch_metric[f"{prefix}_slc_epoch_specificity"],
                epoch_metric[f"{prefix}_slc_epoch_youden"],
            ) = get_sens_spec_youden(self.slc_storage[prefix]["confusion_matrix"])

        if "infarct" in self.args.tasks:
            for k in self.infarcts_type:
                for key in self.point_estimates["infarct"]["epoch_metrics"]:
                    epoch_metric[
                        f"{prefix}_infarct_epoch_{k}_{key}"
                    ] = put_torchmetric_to_device(
                        self.point_estimates["infarct"]["epoch_metrics"][key],
                        self.device,
                    )(
                        torch.vstack(self.infarcts_storage[prefix][f"{k}_scores"]),
                        torch.vstack(self.infarcts_storage[prefix][f"{k}_gt"]),
                    )

                (
                    epoch_metric[f"{prefix}_infarct_epoch_{k}_sensitivity"],
                    epoch_metric[f"{prefix}_infarct_epoch_{k}_specificity"],
                    epoch_metric[f"{prefix}_infarct_epoch_{k}_youden"],
                ) = get_sens_spec_youden(
                    self.infarcts_storage[prefix][f"{k}_confusion_matrix"]
                )

        self.log_dict(
            epoch_metric,
            on_step=False,
            on_epoch=True,
            batch_size=self.args.batch_size,
            prog_bar=True,
            rank_zero_only=True,
        )
        self.seg_storage[prefix].clear()
        self.slc_storage[prefix].clear()
        self.infarcts_storage[prefix].clear()

    def compute_batch(self, batch, batch_idx, prefix="train"):
        ct, gt_segmentation_map, trg_cls, infarct_cls, series = batch
        torch.nan_to_num_(
            ct, nan=self.nan_score, posinf=self.nan_score, neginf=self.nan_score
        )
        output = self(ct)
        loss, log_losses, log_metrics = self.loss_criterion(
            series, output, gt_segmentation_map, trg_cls, infarct_cls, prefix
        )
        return loss, log_losses, log_metrics

    def training_step(self, batch, batch_idx):
        if "freeze_task_epoch_index" in self.args:
            if self.current_epoch in self.args.freeze_task_epoch_index["start"]:
                freeze_index = self.args.freeze_task_epoch_index["start"].index(
                    self.current_epoch
                )
                freeze_layers(
                    self.args.freeze_task_epoch_index["layers"][freeze_index],
                    self.model,
                )

            if self.current_epoch in self.args.freeze_task_epoch_index["end"]:
                freeze_index = self.args.freeze_task_epoch_index["end"].index(
                    self.current_epoch
                )
                unfreeze_layers(
                    self.args.freeze_task_epoch_index["layers"][freeze_index],
                    self.model,
                )

        loss, log_losses, log_metrics = self.compute_batch(batch, batch_idx)
        self.log_dict(
            log_losses,
            on_step=False,
            on_epoch=True,
            batch_size=self.args.batch_size,
            prog_bar=True,
            rank_zero_only=True,
        )
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=self.args.batch_size,
            prog_bar=True,
            rank_zero_only=True,
        )
        self.log_dict(
            log_metrics,
            on_step=False,
            on_epoch=True,
            batch_size=self.args.batch_size,
            prog_bar=True,
            rank_zero_only=True,
        )
        return loss

    def on_train_epoch_end(self):
        self.compute_epoch_metric(prefix="train")

    def validation_step(self, batch, batch_idx):
        loss, log_losses, log_metrics = self.compute_batch(
            batch, batch_idx, prefix="valid"
        )
        self.log_dict(
            log_losses,
            on_step=False,
            on_epoch=True,
            batch_size=self.args.batch_size,
            prog_bar=True,
            rank_zero_only=True,
        )
        self.log(
            "valid_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=self.args.batch_size,
            prog_bar=True,
            rank_zero_only=True,
        )
        self.log_dict(
            log_metrics,
            on_step=False,
            on_epoch=True,
            batch_size=self.args.batch_size,
            prog_bar=True,
            rank_zero_only=True,
        )
        return loss

    def on_validation_epoch_end(self):
        self.compute_epoch_metric(prefix="valid")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    def configure_optimizers(self):
        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer)

        if isinstance(scheduler, type(torch.optim.lr_scheduler.ReduceLROnPlateau)):
            return [optimizer], [{"scheduler": scheduler, "monitor": self.args.monitor}]

        if isinstance(scheduler, type(torch.optim.lr_scheduler.OneCycleLR)):
            scheduler.state_dict().pop("_scale_fn_ref")
            return [optimizer], [scheduler]

        return [optimizer], [scheduler]

    def _create_optimizer(self):
        return self.args.optimizer(
            self.model.parameters(), **self.args.optimizer_params
        )

    def _create_scheduler(self, optimizer):
        return self.args.scheduler(optimizer, **self.args.scheduler_params)
