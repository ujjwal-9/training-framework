import warnings
warnings.filterwarnings('ignore')
import os
import sys
import yaml
import glob
import munch
import random

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, roc_auc_score
# from torch.utils.tensorboard import SummaryWriter
import lightning as L
from lightning.fabric.loggers import TensorBoardLogger


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--log', required=True, type=str)
parser.add_argument('--devices', required=True, type=int)
parser.add_argument('--strategy', required=True, type=str)
parser.add_argument('--sample', required=True, type=int)
parser.add_argument('--bsz', default=8, type=int)
parser.add_argument('--backbone', default="seresnet101", type=str)
parser.add_argument('--attn_mode', default="softminmax", type=str)
parser.add_argument('--dropout', default=0.3, type=float)
parser.add_argument('--data', required=True, type=str)
parser.add_argument('--num_workers', required=True, type=int)
args_terminal = parser.parse_args()
model_params_to_save = {
    "backbone": args_terminal.backbone,
    "attn_mode": args_terminal.attn_mode,
    "dropout": args_terminal.dropout,
    "data": args_terminal.data,
    "strategy": args_terminal.strategy,
    "num_workers": args_terminal.num_workers,
}

LOG_DIR = os.path.join("logs", args_terminal.log)
MODEL_SAVE_DIR = os.path.join("/data_nas5/qer/ujjwal/models/fusion_resnet/", args_terminal.log)
print(f"Log: {LOG_DIR}")
print(f"Model save: {MODEL_SAVE_DIR}")

# Pick a logger and add it to Fabric

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

if not os.path.exists(MODEL_SAVE_DIR):
    print("Directory created")
    os.makedirs(MODEL_SAVE_DIR)

np.random.seed(0)
torch.manual_seed(0)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

logger = TensorBoardLogger(root_dir=LOG_DIR)

# device = torch.device(f"cuda:{args_terminal.gpu}")
# print(f"device: {device}")
fabric = L.Fabric(accelerator="cuda", devices=args_terminal.devices, strategy=args_terminal.strategy, precision="16-mixed", loggers=logger)
fabric.launch()


# In[6]:





# In[58]:


from qtrain.dataset.infarct import InfarctDataset3D_only_cls

with open("../configs/3dunet_22march.yaml", "r") as f:
    args = munch.munchify(yaml.load(f, Loader=yaml.Loader))

args.datapath = args_terminal.data
args.batch_size = args_terminal.bsz
args.val_batch_size = args_terminal.bsz
args.augmentation = True
args.extra_augs = True

args.n_slices = 32
args.windowing = "old"



import pandas as pd
df = pd.read_json(args.datapath)
wts = []
val_count = df[df['status']=="train"].annotation.value_counts()
n_samples = len(df[df['status']=="train"].annotation)
l = df[df['status']=="train"].annotation.to_list()
for i in l:
    if i == 0:
        wts.append(n_samples/val_count[0])
    else:
        wts.append(n_samples/val_count[1])
samples_weight = torch.tensor(wts)
sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), args_terminal.sample)
del df
train_dataloader = DataLoader(InfarctDataset3D_only_cls(args), batch_size=args.batch_size, sampler=sampler, num_workers=args_terminal.num_workers)
test_dataloader = DataLoader(InfarctDataset3D_only_cls(args, 'valid'), batch_size=args.val_batch_size, shuffle=False, num_workers=args_terminal.num_workers)

train_dataloader = fabric.setup_dataloaders(train_dataloader)
test_dataloader = fabric.setup_dataloaders(test_dataloader)


# from qer.common.multisoftmax_classifier.models import load_checkpoint
# model, args_old = load_checkpoint("/home/users/ujjwal.upadhyay/packages/qer/resources/checkpoints/infarcts/infarct_xentropy_incl_lacunar_resume.pth.tar")
# from qer.common.multisoftmax_classifier.models.fusion_resnet import FusionResnet
from fresnet import FusionResnet
model = FusionResnet(args_terminal.backbone, dropout=0.3, attn_mode="softminmax").to(dtype=torch.float)
model.train()

numEpochs = 200
lr = 1e-2
decay_step = [25, 75, 100]
decay_factor = 0.1



loss_criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=lr)
# scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=5e-4, max_lr=1e-2)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=len(train_dataloader), epochs=numEpochs)


model, optimizer = fabric.setup(model, optimizer)


train_iter = 0
val_iter = 0
min_accuracy = 0
min_loss = 10000

# writer = SummaryWriter(log_dir=LOG_DIR)

for epoch in tqdm(range(numEpochs)):
    
        epoch_loss = 0
        numCorrTrain = 0
        trainSamples = 0
        iterPerEpoch = 0
        tn, fp, fn, tp = 0,0,0,0
        preds = []
        targets_gt = []

        model.train()
        
        for i, (inputs, targets) in enumerate(tqdm(train_dataloader, leave=True)):
            image_sequences = torch.Tensor(inputs.permute(0,2,1,3,4)).to(dtype=torch.float)
            labels = targets
            
            optimizer.zero_grad()

            output_labels = model(image_sequences, only_return_scan_output=True)[0]

            loss = loss_criterion(output_labels, labels)
            fabric.backward(loss)
            
            optimizer.step()
            
            output_labels = torch.softmax(output_labels, dim=1)
            _, predicted = torch.max(output_labels.data, 1)
            corr__ = (predicted == targets).sum().item()
            numCorrTrain += corr__
            
            preds.append(output_labels[:,1].ravel().detach().cpu())
            targets_gt.append(targets.detach().cpu().ravel())
            
            c_matrix = confusion_matrix(targets.cpu().numpy(), predicted.detach().cpu().numpy()>0.5, labels=[0,1]).ravel()
            tn += c_matrix[0]
            fp += c_matrix[1]
            fn += c_matrix[2] 
            tp += c_matrix[3]
            
            epoch_loss += loss.item()
            
            train_iter += 1
            iterPerEpoch += 1
            trainSamples += inputs.size(0) 
            fabric.log('train/batch_loss', loss.item(), on_step=True)
            # writer.add_scalar('train/batch_loss', loss.item(), train_iter)
            # writer.add_scalar('train/learning_rate', scheduler.get_lr()[0], train_iter)
        
        scheduler.step()
        avg_loss = epoch_loss / iterPerEpoch
        trainAccuracy = float((numCorrTrain / trainSamples) * 100)
        
        train_sens = round((tp + 1e-6) / (tp+fn+1e-6),2)
        train_spec = round((tn + 1e-6) / (tn+fp+1e-6),2)
        ys = train_sens + train_spec - 1
        preds = torch.cat(preds).numpy()
        targets_gt = torch.cat(targets_gt).numpy()
        try:
            train_roc_auc = round(roc_auc_score(targets_gt, preds),2)
        except:
            train_roc_auc = 0

        print('Train: Epoch = {} | Loss = {} | Accuracy = {} | Correct = {} | Sens = {} | Spec = {} | Yoden = {} | Auc = {}'.format(
            epoch+1, 
            avg_loss, 
            round(trainAccuracy,3),
            int(numCorrTrain),
            train_sens, train_spec, ys, train_roc_auc
        ))

        fabric.log_dict({
                'train/epoch_loss': avg_loss,
                'train/n_correct': int(numCorrTrain),
                'train/sens': train_sens,
                'train/spec': train_spec,
                'train/yoden': ys,
                'train/auc': train_roc_auc
            })
        
        # writer.add_scalar('train/epoch_loss', avg_loss, train_iter)
        # writer.add_scalar('train/n_correct', int(numCorrTrain), train_iter)
        # writer.add_scalar('train/sens', train_sens, train_iter)
        # writer.add_scalar('train/spec', train_spec, train_iter)
        # writer.add_scalar('train/yoden', ys, train_iter)
        # writer.add_scalar('train/auc', train_roc_auc, train_iter)
        
        
        if (epoch+1) % 1 == 0:
            
            model.eval()
            val_loss_epoch = 0
            val_samples = 0
            numCorr = 0
            
            tn, fp, fn, tp = 0,0,0,0
            preds = []
            targets_gt = []
            
            with torch.no_grad():
                for j, (inputs, targets) in enumerate(tqdm(test_dataloader, leave=True)):
                    val_iter += 1
                    val_samples += inputs.size(0)

                    image_sequences = torch.Tensor(inputs.permute(0,2,1,3,4)).to(dtype=torch.float)
                    labels = targets

                    output_labels = model(image_sequences, only_return_scan_output=True)[0]

                    val_loss = loss_criterion(output_labels, labels)
                    val_loss_epoch += val_loss.item()

                    output_labels = torch.softmax(output_labels, dim=1)
                    _, predicted = torch.max(output_labels.data, 1)
                    corr__ = (predicted == targets).sum().item()
                    numCorr += corr__
                    
                    preds.append(output_labels[:,1].ravel().detach().cpu())
                    targets_gt.append(targets.detach().cpu().ravel())

                    c_matrix = confusion_matrix(targets.cpu().numpy(), predicted.detach().cpu().numpy()>0.5, labels=[0,1]).ravel()
                    tn += c_matrix[0]
                    fp += c_matrix[1]
                    fn += c_matrix[2] 
                    tp += c_matrix[3]
                    fabric.log('valid/batch_loss', val_loss.item(), on_step=True)
                    # writer.add_scalar('valid/batch_loss', val_loss.item(), val_iter)

                val_accuracy = float((numCorr / val_samples) * 100)
                avg_val_loss = val_loss_epoch / val_iter
                sens = round((tp + 1e-6) / (tp+fn+1e-6),2)
                spec = round((tn + 1e-6) / (tn+fp+1e-6),2)
                ys = sens + spec - 1
                preds = torch.cat(preds).numpy()
                targets_gt = torch.cat(targets_gt).numpy()
                try:
                    roc_auc = round(roc_auc_score(targets_gt, preds),2)
                except:
                    roc_auc = 0

                print('Val:   Epoch = {} | Loss = {} | Accuracy = {} | Correct = {} | Sens = {} | Spec = {} | Yoden = {} | Auc = {}'.format(
                    epoch + 1, 
                    avg_val_loss, 
                    round(val_accuracy,3),
                    int(numCorr),
                    sens, spec, ys, roc_auc
                ))
                
                fabric.log_dict({
                    'valid/epoch_loss': avg_val_loss,
                    'valid/n_correct': int(numCorr),
                    'valid/sens': sens,
                    'valid/spec': spec,
                    'valid/yoden': ys,
                    'valid/auc': roc_auc
                })

                # writer.add_scalar('valid/epoch_loss', avg_val_loss, val_iter)
                # writer.add_scalar('valid/n_correct', int(numCorr), val_iter)
                # writer.add_scalar('valid/sens', sens, val_iter)
                # writer.add_scalar('valid/spec', spec, val_iter)
                # writer.add_scalar('valid/yoden', ys, val_iter)
                # writer.add_scalar('valid/auc', roc_auc, val_iter)

                
                if val_accuracy > min_accuracy:
                    scheduler_state = scheduler.state_dict()
                    try:
                        scheduler_state.pop("_scale_fn_ref")
                    except:
                        pass
                    checkpoint = { 
                            'epoch': epoch,
                            'batch_size': args_terminal.bsz,
                            'sample': args_terminal.sample,
                            'model_params': model_params_to_save,
                            'valid_epoch_loss': avg_val_loss,
                            'valid_auc': roc_auc,
                            'valid_sens': sens,
                            'valid_spec': spec,
                            'train_epoch_loss': avg_loss,
                            'train_auc': train_roc_auc,
                            'train_sens': train_sens,
                            'train_spec': train_spec,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler_state
                        }
                    save_path_model = os.path.join(MODEL_SAVE_DIR, 'best_model.pth')
                    torch.save(checkpoint, save_path_model)
                    min_accuracy = val_accuracy
                
                if avg_val_loss < min_loss:
                    scheduler_state = scheduler.state_dict()
                    try:
                        scheduler_state.pop("_scale_fn_ref")
                    except:
                        pass
                    checkpoint = { 
                            'epoch': epoch,
                            'batch_size': args_terminal.bsz,
                            'sample': args_terminal.sample,
                            'model_params': model_params_to_save,
                            'valid_epoch_loss': avg_val_loss,
                            'valid_auc': roc_auc,
                            'valid_sens': sens,
                            'valid_spec': spec,
                            'train_epoch_loss': avg_loss,
                            'train_auc': train_roc_auc,
                            'train_sens': train_sens,
                            'train_spec': train_spec,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler_state
                        }
                    save_path_model = os.path.join(MODEL_SAVE_DIR, 'best_model_loss.pth')
                    torch.save(checkpoint, save_path_model)
                    min_loss = avg_val_loss

                if (epoch+1) % 10 == 0:
                    scheduler_state = scheduler.state_dict()
                    try:
                        scheduler_state.pop("_scale_fn_ref")
                    except:
                        pass
                    checkpoint = { 
                            'epoch': epoch,
                            'batch_size': args_terminal.bsz,
                            'sample': args_terminal.sample,
                            'model_params': model_params_to_save,
                            'valid_epoch_loss': avg_val_loss,
                            'valid_auc': roc_auc,
                            'valid_sens': sens,
                            'valid_spec': spec,
                            'train_epoch_loss': avg_loss,
                            'train_auc': train_roc_auc,
                            'train_sens': train_sens,
                            'train_spec': train_spec,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler_state
                        }
                    save_path_model = os.path.join(MODEL_SAVE_DIR, f'epoch_{str(epoch+1)}.pth')
                    torch.save(checkpoint, save_path_model)
