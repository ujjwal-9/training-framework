LOG_DIR = "uniformer_llgg_runs"
MODEL_SAVE_DIR = "/data_nas5/qer/ujjwal/models/uniformer/"
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.tensorboard import SummaryWriter

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--log', required=True, type=str)
parser.add_argument('--gpu', required=True, type=str)
parser.add_argument('--sample', required=True, type=int)
parser.add_argument('--bsz', default=4, type=int)
parser.add_argument('--opt', default="sgd", type=str)
args_terminal = parser.parse_args()

LOG_DIR = os.path.join("logs", args_terminal.log)
MODEL_SAVE_DIR = os.path.join("/data_nas5/qer/ujjwal/models/uniformer/", args_terminal.log)
print(f"Log: {LOG_DIR}")
print(f"Model save: {MODEL_SAVE_DIR}")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
writer = SummaryWriter(log_dir=LOG_DIR)

if not os.path.exists(MODEL_SAVE_DIR):
    print("Directory created")
    os.makedirs(MODEL_SAVE_DIR)

np.random.seed(0)
torch.manual_seed(0)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

device = torch.device(f"cuda:{args_terminal.gpu}")
print(f"device: {device}")



from qtrain.dataset.infarct import InfarctDataset3D_only_cls

with open("../configs/3dunet_22march.yaml", "r") as f:
    args = munch.munchify(yaml.load(f, Loader=yaml.Loader))

args.datapath = "/cache/fast_data_nas72/qer/ujjwal/npy_cached_all_studies/jsons/all_studies_old_style.json"
# import pandas as pd
# df = pd.read_json(args.datapath).iloc[:100,:]
# df.to_json("test.json")
# args.datapath = "test.json"
#args.datapath = "/home/users/ujjwal.upadhyay/projects/qtrain/data/dataset_15april23.json"
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

train_dataloader = DataLoader(InfarctDataset3D_only_cls(args), batch_size=args.batch_size, sampler=sampler)
test_dataloader = DataLoader(InfarctDataset3D_only_cls(args, 'valid'), batch_size=args.val_batch_size, shuffle=False)

from uniformer_pytorch import Uniformer
model = Uniformer(
    num_classes = 2,                 # number of output classes
    dims = (64, 128, 256, 64),         # feature dimensions per stage (4 stages)
    depths = (3, 7, 7, 3),              # depth at each stage
    mhsa_types = ('l', 'l', 'g', 'g')   # aggregation type at each stage, 'l' stands for local, 'g' stands for global
).to(device=device, dtype=torch.float)
# model.load_state_dict(torch.load("/data_nas5/qer/ujjwal/models/uniformer/exp15apr23/best_model.pth"))
model.train()


# In[126]:


learning_rate = 1e-2

optimizer = optim.SGD(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=5e-4, max_lr=1e-2)
loss_criterion = nn.CrossEntropyLoss().to(device)


# In[127]:


print("Model saved here: ", MODEL_SAVE_DIR)


train_iter = 0
val_iter = 0
min_accuracy = 0
numEpochs = 200

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
            image_sequences = torch.Tensor(inputs.permute(0,2,1,3,4)).to(device=device, dtype=torch.float)
            labels = targets.to(device=device)
            
            optimizer.zero_grad()

            output_labels = model(image_sequences)

            loss = loss_criterion(output_labels, labels)
            loss.backward()
            
            optimizer.step()
            
            output_labels = torch.softmax(output_labels, dim=1)
            _, predicted = torch.max(output_labels.data, 1)
            corr__ = (predicted == targets.to(device=device)).sum().item()
            numCorrTrain += corr__
            
            preds.append(output_labels[:,1].ravel().detach().cpu())
            targets_gt.append(targets.ravel())
            
            c_matrix = confusion_matrix(targets.cpu().numpy(), predicted.detach().cpu().numpy()>0.5, labels=[0,1]).ravel()
            tn += c_matrix[0]
            fp += c_matrix[1]
            fn += c_matrix[2] 
            tp += c_matrix[3]
            
            epoch_loss += loss.item()
            
            train_iter += 1
            iterPerEpoch += 1
            trainSamples += inputs.size(0)
            writer.add_scalar('train/batch_loss', loss.item(), train_iter)
        
        scheduler.step()
        avg_loss = epoch_loss / iterPerEpoch
        trainAccuracy = float((numCorrTrain / trainSamples) * 100)
        
        sens = round((tp + 1e-6) / (tp+fn+1e-6),2)
        spec = round((tn + 1e-6) / (tn+fp+1e-6),2)
        ys = sens + spec - 1
        preds = torch.cat(preds).numpy()
        targets_gt = torch.cat(targets_gt).numpy()
        try:
            roc_auc = round(roc_auc_score(targets_gt, preds),2)
        except:
            roc_auc = 0

        print('Train: Epoch = {} | Loss = {} | Accuracy = {} | Correct = {} | Sens = {} | Spec = {} | Yoden = {} | Auc = {}'.format(
            epoch+1, 
            avg_loss, 
            round(trainAccuracy,3),
            int(numCorrTrain),
            sens, spec, ys, roc_auc
        ))
        
        writer.add_scalar('train/epoch_loss', avg_loss, train_iter)
        writer.add_scalar('train/n_correct', int(numCorrTrain), train_iter)
        writer.add_scalar('train/sens', sens, train_iter)
        writer.add_scalar('train/spec', spec, train_iter)
        writer.add_scalar('train/yoden', ys, train_iter)
        writer.add_scalar('train/auc', roc_auc, train_iter)
        
        
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

                    image_sequences = torch.Tensor(inputs.permute(0,2,1,3,4)).to(device=device, dtype=torch.float)
                    labels = targets.to(device=device, non_blocking=True)

                    output_labels = model(image_sequences)

                    val_loss = loss_criterion(output_labels, labels)
                    val_loss_epoch += val_loss.item()

                    output_labels = torch.softmax(output_labels, dim=1)
                    _, predicted = torch.max(output_labels.data, 1)
                    corr__ = (predicted == targets.to(device=device)).sum().item()
                    numCorr += corr__
                    
                    preds.append(output_labels[:,1].ravel().detach().cpu())
                    targets_gt.append(targets.ravel())

                    c_matrix = confusion_matrix(targets.cpu().numpy(), predicted.detach().cpu().numpy()>0.5, labels=[0,1]).ravel()
                    tn += c_matrix[0]
                    fp += c_matrix[1]
                    fn += c_matrix[2] 
                    tp += c_matrix[3]
                    writer.add_scalar('valid/batch_loss', val_loss.item(), val_iter)

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
                
                writer.add_scalar('valid/epoch_loss', avg_val_loss, val_iter)
                writer.add_scalar('valid/n_correct', int(numCorr), val_iter)
                writer.add_scalar('valid/sens', sens, val_iter)
                writer.add_scalar('valid/spec', spec, val_iter)
                writer.add_scalar('valid/yoden', ys, val_iter)
                writer.add_scalar('valid/auc', roc_auc, val_iter)

                
                if val_accuracy > min_accuracy:
                    scheduler_state = scheduler.state_dict()
                    if args_terminal.opt != "lion":
                        scheduler_state.pop("_scale_fn_ref")
                    checkpoint = { 
                            'epoch': epoch,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler_state
                        }
                    save_path_model = os.path.join(MODEL_SAVE_DIR, 'best_model.pth')
                    torch.save(checkpoint, save_path_model)
                    min_accuracy = val_accuracy

                if (epoch+1) % 10 == 0:
                    scheduler_state = scheduler.state_dict()
                    if args_terminal.opt != "lion":
                        scheduler_state.pop("_scale_fn_ref")
                    checkpoint = { 
                            'epoch': epoch,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler_state
                        }
                    save_path_model = os.path.join(MODEL_SAVE_DIR, f'epoch_{str(epoch+1)}.pth')
                    torch.save(checkpoint, save_path_model)
