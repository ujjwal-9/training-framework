


MODEL_SAVE_DIR = "/data_nas5/qer/ujjwal/models/fusion_resnet/exp12apr23"


# In[3]:


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
("")


import torchvision
import torchvision.transforms as transforms


# In[4]:


os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

np.random.seed(0)
torch.manual_seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


# In[5]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

device = torch.device("cuda:2")
device


# In[6]:





# In[58]:


from qtrain.dataset.infarct import InfarctDataset3D_only_cls

with open("../configs/3dunet_22march.yaml", "r") as f:
    args = munch.munchify(yaml.load(f, Loader=yaml.Loader))

args.datapath = "/home/users/ujjwal.upadhyay/projects/qtrain/data/dataset_13april23.json"
args.batch_size = 4
args.val_batch_size = 4

args.n_slices = 32
args.windowing = "old"

train_dataset = InfarctDataset3D_only_cls(args)
test_dataset = InfarctDataset3D_only_cls(args, 'valid')

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.val_batch_size, shuffle=False)


# In[59]:


print(len(test_dataset))
print(len(train_dataset))


# In[60]:


from qer.common.multisoftmax_classifier.models import load_checkpoint
model, args_old = load_checkpoint("/home/users/ujjwal.upadhyay/packages/qer/resources/checkpoints/infarcts/infarct_xentropy_incl_lacunar_resume.pth.tar")
model = model.to(device=device, dtype=torch.float)
model.train()


# In[61]:


args_old


# In[62]:


learning_rate = 1e-2

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, amsgrad=False)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0, last_epoch=-1, verbose=False)
loss_criterion = nn.CrossEntropyLoss().to(device)


# In[51]:


print("Model saved here: ", MODEL_SAVE_DIR)


# In[16]:


# a,b = next(iter(train_dataloader))
# a = a.permute(0,2,1,3,4)
# out = model(a.to(device=device, dtype=torch.float))
# out[0][0].shape, out[1][0].shape


# In[57]:


start_time = datetime.now()
train_iter = 0
min_accuracy = 0
numEpochs = 200

for epoch in tqdm(range(numEpochs)):
    
        epoch_loss = 0
        numCorrTrain = 0
        trainSamples = 0
        iterPerEpoch = 0

        model.train()
        
        for i, (inputs, targets) in enumerate(tqdm(train_dataloader, leave=True)):
            image_sequences = torch.Tensor(inputs.permute(0,2,1,3,4)).to(device=device, dtype=torch.float)
            labels = targets.to(device=device)
            
            optimizer.zero_grad()

            output_labels = model(image_sequences)[0][0]

            loss = loss_criterion(output_labels, labels)
            loss.backward()
            
            optimizer.step()
            
            output_labels = torch.softmax(output_labels, dim=1)
            _, predicted = torch.max(output_labels.data, 1)
            numCorrTrain += (predicted == targets.to(device=device)).sum()
            
            epoch_loss += loss.item()
            
            train_iter += 1
            iterPerEpoch += 1
            trainSamples += inputs.size(0) 
        
        scheduler.step()
        avg_loss = epoch_loss / iterPerEpoch
        trainAccuracy = float((numCorrTrain.item() / trainSamples) * 100)

        print('Train: Epoch = {} | Loss = {} | Accuracy = {} | Correct = {}'.format(
            epoch+1, 
            avg_loss, 
            round(trainAccuracy,3),
            int(numCorrTrain.item())
        ))

        if (epoch+1) % 1 == 0:
            
            model.eval()
            val_loss_epoch = 0
            val_iter = 0
            val_samples = 0
            numCorr = 0
            with torch.no_grad():
                for j, (inputs, targets) in enumerate(tqdm(test_dataloader, leave=True)):
                    val_iter += 1
                    val_samples += inputs.size(0)

                    image_sequences = torch.Tensor(inputs.permute(0,2,1,3,4)).to(device=device, dtype=torch.float)
                    labels = targets.to(device=device, non_blocking=True)

                    output_labels = model(image_sequences)[0][0]

                    val_loss = loss_criterion(output_labels, labels)
                    val_loss_epoch += val_loss.item()

                    output_labels = torch.softmax(output_labels, dim=1)
                    _, predicted = torch.max(output_labels.data, 1)
                    numCorr += (predicted == targets.to(device=device)).sum()

                val_accuracy = float((numCorr.item() / val_samples) * 100)
                avg_val_loss = val_loss_epoch / val_iter

                print('Val:   Epoch = {} | Loss = {} | Accuracy = {} | Correct = {}'.format(
                    epoch + 1, 
                    avg_val_loss, 
                    round(val_accuracy,3),
                    int(numCorr.item())
                ))

                if val_accuracy > min_accuracy:
                    save_path_model = os.path.join(MODEL_SAVE_DIR, 'best_model.pth')
                    torch.save(model.state_dict(), save_path_model)
                    min_accuracy = val_accuracy

                if (epoch+1) % 10 == 0:
                    save_path_model = os.path.join(MODEL_SAVE_DIR, f'epoch_{str(epoch+1)}.pth')
                    torch.save(model.state_dict(), save_path_model)


# In[ ]:


end_time = datetime.now()
print(start_time, end_time)

