import warnings

warnings.filterwarnings("ignore")
import logging

logging.getLogger("lightning").setLevel(logging.ERROR)

import os
import json
import yaml
import torch
import munch
import argparse
import pytorch_lightning as pl

from glob import glob
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from qtrain.dataset.infarct import InfarctDataModule
from qtrain.models.train_models import qMultiTasker
from clearml import Task

import ssl

ssl._create_default_https_context = ssl._create_unverified_context
# torch.multiprocessing.set_sharing_strategy("file_system")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:21"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument("--exp", type=str, help="Name of experiment")
parser.add_argument("--config", type=str, help="Path to config file")
parser.add_argument("--resume", type=str, help="Resume training")
init_args = parser.parse_args()


with open(init_args.config) as f:
    if init_args.config.split(".")[-1] == "json":
        args = json.load(f)
    elif init_args.config.split(".")[-1] == "yaml":
        args = yaml.load(f, Loader=yaml.Loader)

args = munch.munchify(args)
args.experiment = init_args.exp
pl.seed_everything(args.seed, workers=True)
task = Task.init(project_name="Infarcts Segmentation", task_name=args.experiment)

print("Training parameters:\n", args)

dm = InfarctDataModule(args)
args.train_iters = dm.get_num_training_samples() // args.batch_size
if ("total_steps" in args.scheduler_params) and (
    args.scheduler_params.total_steps is None
):
    args.scheduler_params.total_steps = args.train_iters // args.accumulate_grad_batches

model = qMultiTasker(args)
callbacks_to_minitor = []

metrics_to_monitor = [
    {
        "monitor": "valid_loss",
        "mode": "min",
        "filename": "{valid_loss:.2f}-{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}",
    },
    {
        "monitor": "train_loss",
        "mode": "min",
        "filename": "{train_loss:.2f}-{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}",
    },
    {
        "monitor": "valid_infarct_bce",
        "mode": "min",
        "filename": "{valid_infarct_bce:.2f}-{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}",
    },
    {
        "monitor": "valid_slc_bce",
        "mode": "min",
        "filename": "{valid_slc_bce:2f}-{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}",
    },
    {
        "monitor": "valid_seg_focal",
        "mode": "min",
        "filename": "{valid_seg_focal:.2f}-{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}",
    },
    {
        "monitor": "valid_infarct_epoch_acute_sensitivity",
        "mode": "max",
        "filename": "{valid_infarct_epoch_acute_sensitivity:2f}-{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}",
    },
    {
        "monitor": "valid_infarct_epoch_acute_specificity",
        "mode": "max",
        "filename": "{valid_infarct_epoch_acute_specificity:2f}-{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}",
    },
    {
        "monitor": "valid_infarct_epoch_acute_youden",
        "mode": "max",
        "filename": "{valid_infarct_epoch_acute_youden:2f}-{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}",
    },
    {
        "monitor": "valid_infarct_epoch_acute_auc",
        "mode": "max",
        "filename": "{valid_infarct_epoch_acute_auc:2f}-{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}",
    },
    {
        "monitor": "valid_infarct_epoch_chronic_sensitivity",
        "mode": "max",
        "filename": "{valid_infarct_epoch_chronic_sensitivity:2f}-{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}",
    },
    {
        "monitor": "valid_infarct_epoch_chronic_specificity",
        "mode": "max",
        "filename": "{valid_infarct_epoch_chronic_specificity:2f}-{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}",
    },
    {
        "monitor": "valid_infarct_epoch_chronic_youden",
        "mode": "max",
        "filename": "{valid_infarct_epoch_chronic_youden:2f}-{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}",
    },
    {
        "monitor": "valid_infarct_epoch_chronic_auc",
        "mode": "max",
        "filename": "{valid_infarct_epoch_chronic_auc:2f}-{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}",
    },
    {
        "monitor": "valid_seg_epoch_miou",
        "mode": "max",
        "filename": "{valid_seg_epoch_miou:.2f}-{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}",
    },
    {
        "monitor": "valid_slc_epoch_sensitivity",
        "mode": "max",
        "filename": "{valid_slc_epoch_sensitivity:.2f}-{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}",
    },
    {
        "monitor": "valid_slc_epoch_specificity",
        "mode": "max",
        "filename": "{valid_slc_epoch_specificity:.2f}-{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}",
    },
    {
        "monitor": "valid_slc_epoch_youden",
        "mode": "max",
        "filename": "{valid_slc_epoch_youden:.2f}-{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}",
    },
    {
        "monitor": "valid_slc_epoch_auc",
        "mode": "max",
        "filename": "{valid_slc_epoch_auc:.2f}-{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}",
    },
]

for metrics_ in metrics_to_monitor:
    callbacks_to_minitor.append(
        pl.callbacks.ModelCheckpoint(
            filename=metrics_["filename"],
            monitor=metrics_["monitor"],
            mode=metrics_["mode"],
            save_top_k=2,
        )
    )


if args.stochastic_weight_averaging:
    from pytorch_lightning.callbacks import StochasticWeightAveraging

    callbacks_to_minitor.append(
        StochasticWeightAveraging(
            swa_lrs=args.swa_lrs, swa_epoch_start=args.swa_epoch_start
        )
    )

early_stopping = EarlyStopping(monitor=args.monitor, patience=args.patience)
lr_monitor = LearningRateMonitor(logging_interval="epoch")

callbacks_to_minitor.extend([lr_monitor, early_stopping])

log_dir = args.save_checkpoints
tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir)


log_dirs = glob(os.path.join(log_dir, "lightning_logs", "*"))
ver = None
if len(log_dirs) == 0:
    ver = 0
else:
    for dir_ in log_dirs:
        ver_ = dir_.split("_")[-1]
        if ver is None:
            ver = int(ver_)
        else:
            if ver < int(ver_):
                ver = int(ver_)
print(f"\nLOG FOLDER: VERSION_{ver}\n")

trainer = pl.Trainer(
    accelerator="auto",
    devices=args.gpu,
    precision=args.precision,
    max_epochs=args.max_epoch,
    default_root_dir=log_dir,
    callbacks=callbacks_to_minitor,
    logger=[tb_logger],
    sync_batchnorm=args.sync_batchnorm,
    fast_dev_run=args.fast_dev_run,
    strategy=args.strategy,
    accumulate_grad_batches=args.accumulate_grad_batches,
    gradient_clip_val=args.gradient_clip_val,
    gradient_clip_algorithm=args.gradient_clip_algorithm,
    track_grad_norm=args.track_grad_norm,
    detect_anomaly=False,
    num_sanity_val_steps=0,
    # overfit_batches=5,
    # limit_val_batches=0.1,
)

if __name__ == "__main__":
    trainer.fit(model, dm, ckpt_path=init_args.resume)
