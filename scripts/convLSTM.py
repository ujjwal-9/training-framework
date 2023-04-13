feature_extractor = 'resnet18'
train_resnet = True

import os
import sys
import yaml
import glob
import munch
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms


from tqdm import tqdm
from PIL import Image
from resnet import attentionModel
from qtrain.dataset.infarct import InfarctDataset3D_only_cls


np.random.seed(0)
torch.manual_seed(0)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

device = torch.device("cuda:1")



with open("../configs/3dunet_22march.yaml", "r") as f:
    args = munch.munchify(yaml.load(f, Loader=yaml.Loader))

args.datapath = "/home/users/ujjwal.upadhyay/projects/qtrain/data/dataset_14april23.json"
args.batch_size = 4
args.val_batch_size = 4

args.n_slices = 32
args.windowing = "old"

train_dataloader = DataLoader(InfarctDataset3D_only_cls(args), batch_size=args.batch_size, shuffle=True)
test_dataloader = DataLoader(InfarctDataset3D_only_cls(args, 'valid'), batch_size=args.val_batch_size, shuffle=False)

num_classes = 2
memSize = 512
train_params = []


# In[19]:


if train_resnet is False:
    model = attentionModel(device, num_classes=num_classes, mem_size=memSize)
    model.train(False)
    for params in model.parameters():
        params.requires_grad = False
else:
    model = attentionModel(device, num_classes=num_classes, mem_size=memSize)
    model.train(False)
    for params in model.parameters():
        params.requires_grad = False
    #
    for params in model.resNet.layer4[0].conv1.parameters():
        params.requires_grad = True
        train_params += [params]

    for params in model.resNet.layer4[0].conv2.parameters():
        params.requires_grad = True
        train_params += [params]

    for params in model.resNet.layer4[1].conv1.parameters():
        params.requires_grad = True
        train_params += [params]

    for params in model.resNet.layer4[1].conv2.parameters():
        params.requires_grad = True
        train_params += [params]

    for params in model.resNet.layer4[2].conv1.parameters():
        params.requires_grad = True
        train_params += [params]
    #
    for params in model.resNet.layer4[2].conv2.parameters():
        params.requires_grad = True
        train_params += [params]
    #
    for params in model.resNet.fc.parameters():
        params.requires_grad = True
        train_params += [params]

    model.resNet.layer4[0].conv1.train(True)
    model.resNet.layer4[0].conv2.train(True)
    model.resNet.layer4[1].conv1.train(True)
    model.resNet.layer4[1].conv2.train(True)
    model.resNet.layer4[2].conv1.train(True)
    model.resNet.layer4[2].conv2.train(True)
    model.resNet.fc.train(True)


# In[20]:


for params in model.lstm_cell.parameters():
    params.requires_grad = True
    train_params += [params]

for params in model.classifier.parameters():
    params.requires_grad = True
    train_params += [params]


# In[23]:


import torch.optim as optim
lr = 1e-2
decay_step = [25, 75, 100]
decay_factor = 0.1

model.lstm_cell.train(True)

model.classifier.train(True)
model.to(device)

loss_criterion = nn.CrossEntropyLoss().to(device)

optimizer = optim.SGD(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=5e-4, max_lr=1e-2)


train_iter = 0
min_accuracy = 0
numEpochs = 200

for epoch in range(numEpochs):
    
        epoch_loss = 0
        numCorrTrain = 0
        trainSamples = 0
        iterPerEpoch = 0

        model.train(True)

        for i, (inputs, targets) in enumerate(tqdm(train_dataloader, leave=True)):
            image_sequences = torch.Tensor(inputs.permute(1, 0, 2, 3, 4)).to(device=device, dtype=torch.float)
            labels = targets.to(device=device)
            
            optimizer.zero_grad()

            output_labels, _ = model(image_sequences)

            loss = loss_criterion(output_labels, labels)
            loss.backward()
            
            optimizer.step()
            
            output_labels = F.softmax(output_labels)
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
            
            model.train(False)
            val_loss_epoch = 0
            val_iter = 0
            val_samples = 0
            numCorr = 0
            with torch.no_grad():
                for j, (inputs, targets) in enumerate(tqdm(test_dataloader, leave=True)):
                    val_iter += 1
                    val_samples += inputs.size(0)
                    
                    image_sequences = torch.Tensor(inputs.permute(1, 0, 2, 3, 4)).to(device=device, dtype=torch.float)
                    labels = targets.to(device=device, non_blocking=True)
                    
                    output_labels, _ = model(image_sequences)
                    
                    val_loss = loss_criterion(output_labels, labels)
                    val_loss_epoch += val_loss.item()
                    
                    output_labels = F.softmax(output_labels)
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
                    save_path_model = '/data_nas5/qer/ujjwal/models/infarcts_convlstm/models_14april/best_model.pth'
                    torch.save(model.state_dict(), save_path_model)
                    min_accuracy = val_accuracy
            
                if (epoch+1) % 10 == 0:
                    save_path_model = '/data_nas5/qer/ujjwal/models/infarcts_convlstm/models_14april/{}_{}.pth'.format(feature_extractor, str(epoch+1))
                    torch.save(model.state_dict(), save_path_model)

