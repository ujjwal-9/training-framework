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
from qtrain.models.train_models import qSegmentation

from clearml import Task


parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, help='Name of experiment')
parser.add_argument('--config', type=str, help='Path to config file')
parser.add_argument('--resume', type=str, help='Resume training')
init_args = parser.parse_args()


with open(init_args.config) as f:
    if init_args.config.split(".")[-1] == "json":
        args = json.load(f)
    elif init_args.config.split(".")[-1] == "yaml":
        args = yaml.safe_load(f)
        args = args['args']

args = munch.munchify(args)
args.experiment = init_args.exp

task = Task.init(project_name="Infarcts", task_name=args.experiment)

print("Training parameters:\n", args)

#dist.init_process_group(backend='gloo', rank=args.rank, world_size=len(args.gpu.split(",")))
pl.seed_everything(args.seed, workers=True)
dm = InfarctDataModule(args)

args.num_classes = dm.num_classes

model =  qSegmentation(args)
if args.mode == "multiclass":
    checkpoint_callback = pl.callbacks.ModelCheckpoint(filename='{epoch}-{valid_metric_nbg:.2f}-{train_metric_nbg:.2f}-{valid_metric_bg:.2f}-{train_metric_bg:.2f}',
                                                       monitor='valid_loss',
                                                       mode='min',
                                                       save_last=True)
else:
    checkpoint_callback = pl.callbacks.ModelCheckpoint(filename='{epoch}-{valid_metric:.2f}-{train_metric:.2f}',
                                                       monitor='valid_loss',
                                                       mode='min',
                                                       save_last=True)

checkpoint_callback.FILE_EXTENSION = ".pth"
log_dir = osp.join(args.project_path, args.save_checkpoints)
tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir)
# csv_logger = pl_loggers.CSVLogger(save_dir="logs/")

lr_monitor = LearningRateMonitor(logging_interval='step')
early_stopping = EarlyStopping(monitor="valid_loss", patience=10)


trainer = pl.Trainer(gpus=args.gpu, 
                     precision=args.precision,
                     max_epochs=args.max_epoch, 
                     default_root_dir=log_dir,
                     callbacks=[checkpoint_callback, lr_monitor, early_stopping],
                     logger=[tb_logger],
                     num_sanity_val_steps=0,
                     fast_dev_run=args.fast_dev_run,
                    )

if __name__ == "__main__":
    trainer.fit(model, dm, ckpt_path=init_args.resume)

