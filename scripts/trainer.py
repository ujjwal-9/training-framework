import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger("lightning").setLevel(logging.ERROR)

import torch
import os.path as osp
import json
import yaml
import munch
import argparse
import pytorch_lightning as pl

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from qtrain.dataset.infarct import InfarctDataModule
from qtrain.models.train_models import qMultiTasker

from clearml import Task

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:21"

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, help='Name of experiment')
parser.add_argument('--config', type=str, help='Path to config file')
parser.add_argument('--resume', type=str, help='Resume training')
init_args = parser.parse_args()


with open(init_args.config) as f:
    if init_args.config.split(".")[-1] == "json":
        args = json.load(f)
    elif init_args.config.split(".")[-1] == "yaml":
        args = yaml.load(f, Loader=yaml.Loader)

args = munch.munchify(args)
args.experiment = init_args.exp

task = Task.init(project_name="Infarcts Segmentation", task_name=args.experiment)

print("Training parameters:\n", args)

#dist.init_process_group(backend='gloo', rank=args.rank, world_size=len(args.gpu.split(",")))
pl.seed_everything(args.seed, workers=True)
dm = InfarctDataModule(args)
args.training_samples = dm.get_num_training_samples()

if "total_steps" in args.scheduler_params:
    args.scheduler_params.total_steps = args.training_samples // args.batch_size * args.max_epoch

model =  qMultiTasker(args)
if args.mode == "multiclass":
    checkpoint_callback_valid_loss = pl.callbacks.ModelCheckpoint(filename='{epoch}-{valid_metric_nbg:.2f}-{train_metric_nbg:.2f}-{valid_metric_bg:.2f}-{train_metric_bg:.2f}',
                                                       monitor=args.monitor,
                                                       mode='min',
                                                       save_last=True)
else:
    checkpoint_callback_valid_seg_loss = pl.callbacks.ModelCheckpoint(filename='{epoch}-{train_loss:.2f}-{valid_loss:.2f}-{valid_metric:.2f}-{train_metric:.2f}',
                                                       monitor='valid_loss',
                                                       mode='min',
                                                       save_last=True)
    
    checkpoint_callback_valid_cls_loss = pl.callbacks.ModelCheckpoint(filename='{epoch}-{valid_loss}-{valid_metric:.2f}-{train_metric:.2f}',
                                                       monitor='valid/cls/ce',
                                                       mode='min',
                                                       save_last=True)
    
    checkpoint_callback_valid_slc_loss = pl.callbacks.ModelCheckpoint(filename='{epoch}-{valid_metric:.2f}-{train_metric:.2f}',
                                                       monitor='valid/slc/bce',
                                                       mode='min',
                                                       save_last=True)
    
    # checkpoint_callback_valid_metric = pl.callbacks.ModelCheckpoint(filename='{epoch}-{valid_metric:.2f}-{train_metric:.2f}',
    #                                                    monitor='valid/miou',
    #                                                    mode='max',
    #                                                    save_last=True)

metrics_to_monitor = [
    {"monitor": "valid_loss", "mode": 'min', "filename": "valid_loss_{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}-{valid_loss:.2f}"},
    {"monitor": "train_loss", "mode": 'min', "filename": "train_loss_{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}-{train_loss:.2f}"},
    # {"monitor": "valid_normal_ce", "mode": 'min', "filename": "valid_normal_ce_{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}-{valid_normal_ce:.2f}"},
    {"monitor": "valid_infarct_bce", "mode": 'min', "filename": "valid_infarct_ce_{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}-{valid_infarct_bce:.2f}"},
    # {"monitor": "valid_slc_bce", "mode": 'min', "filename": "valid_slc_bce_{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}-{valid_slc_bce:2f}"},
    # {"monitor": "valid_seg_focal", "mode": 'min', "filename": "valid_seg_focal_{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}-{valid_seg_focal:.2f}"},
    # {"monitor": "valid_infarct_sensitivity", "mode": 'max', "filename": "valid_infarct_sensitivity_{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}-{valid_infarct_sensitivity:2f}"},
    # {"monitor": "valid_infarct_specificity", "mode": 'max', "filename": "valid_infarct_specificity_{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}-{valid_infarct_specificity:2f}"},
    # {"monitor": "valid_infarct_youden", "mode": 'max', "filename": "valid_infarct_youden_{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}-{valid_infarct_youden:2f}"},

    {"monitor": "valid_infarct_acute_sensitivity", "mode": 'max', "filename": "valid_infarct_acute_sensitivity_{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}-{valid_infarct_acute_sensitivity:2f}"},
    {"monitor": "valid_infarct_acute_specificity", "mode": 'max', "filename": "valid_infarct_acute_specificity_{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}-{valid_infarct_acute_specificity:2f}"},
    {"monitor": "valid_infarct_acute_youden", "mode": 'max', "filename": "valid_infarct_acute_youden_{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}-{valid_infarct_acute_youden:2f}"},

    {"monitor": "valid_infarct_chronic_sensitivity", "mode": 'max', "filename": "valid_infarct_chronic_sensitivity_{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}-{valid_infarct_chronic_sensitivity:2f}"},
    {"monitor": "valid_infarct_chronic_specificity", "mode": 'max', "filename": "valid_infarct_chronic_specificity_{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}-{valid_infarct_chronic_specificity:2f}"},
    {"monitor": "valid_infarct_chronic_youden", "mode": 'max', "filename": "valid_infarct_chronic_youden_{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}-{valid_infarct_chronic_youden:2f}"},

    # {"monitor": "valid_normal_sensitivity", "mode": 'max', "filename": "valid_normal_sensitivity_{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}-{valid_normal_sensitivity:2f}"},
    # {"monitor": "valid_normal_specificity", "mode": 'max', "filename": "valid_normal_specificity_{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}-{valid_normal_specificity:.2f}"},
    # {"monitor": "valid_normal_youden", "mode": 'max', "filename": "valid_normal_youden_{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}-{valid_normal_youden:.2f}"},
    # {"monitor": "valid_normal_auc", "mode": 'max', "filename": "valid_normal_auc_{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}-{valid_normal_auc:.2f}"},
    {"monitor": "valid_seg_miou", "mode": 'max', "filename": "valid_seg_miou_{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}-{valid_seg_miou:.2f}"},
    # {"monitor": "valid_slc_sensitivity", "mode": 'max', "filename": "valid_slc_sensitivity_{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}-{valid_slc_sensitivity:.2f}"},
    # {"monitor": "valid_slc_specificity", "mode": 'max', "filename": "valid_slc_specificity_{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}-{valid_slc_specificity:.2f}"},
    # {"monitor": "valid_slc_youden", "mode": 'max', "filename": "valid_slc_youden_{epoch:02d}-{valid_metric:.2f}-{train_metric:.2f}-{valid_loss:.2f}-{train_loss:.2f}-{valid_slc_youden:.2f}"},
]

callbacks_to_minitor = []
for metrics_ in metrics_to_monitor:
    callbacks_to_minitor.append(
        pl.callbacks.ModelCheckpoint(filename=metrics_["filename"],
                                    monitor=metrics_["monitor"],
                                    mode=metrics_["mode"],
                                    save_top_k=2,
                                    # save_last=True
                                    )
        )

# checkpoint_callback.FILE_EXTENSION = ".pth"
log_dir = args.save_checkpoints
tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir)
# csv_logger = pl_loggers.CSVLogger(save_dir="logs/")

lr_monitor = LearningRateMonitor(logging_interval='step')
early_stopping = EarlyStopping(monitor="valid_loss", patience=args.patience)

callbacks_to_minitor.extend([lr_monitor, early_stopping])

import os
from glob import glob
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
print(f"LOG FOLDER: VERSION_{ver}")

trainer = pl.Trainer(accelerator="auto",
                     devices=args.gpu, 
                     precision=args.precision,
                     max_epochs=args.max_epoch, 
                     default_root_dir=log_dir,
                     callbacks=callbacks_to_minitor,
                     logger=[tb_logger],
                     num_sanity_val_steps=0,
                     sync_batchnorm=True,
                     fast_dev_run=args.fast_dev_run,
                     strategy=args.strategy,
                     accumulate_grad_batches=args.accumulate_grad_batches,
                     gradient_clip_val=args.gradient_clip_val,
                     gradient_clip_algorithm = args.gradient_clip_algorithm,
                     detect_anomaly=False,
                    )

if __name__ == "__main__":
    trainer.fit(model, dm, ckpt_path=init_args.resume)