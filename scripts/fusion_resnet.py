import warnings
warnings.filterwarnings('ignore')

import os
import yaml
import munch
import argparse
import numpy as np
import pandas as pd


from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader, WeightedRandomSampler

from qtrain.dataset.infarct import InfarctDataset3D_only_cls

# np.random.seed(0)
# torch.manual_seed(0)
torch.multiprocessing.set_sharing_strategy('file_system')

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

parser = argparse.ArgumentParser()
parser.add_argument('--log', required=True, type=str)
parser.add_argument('--gpu', required=True, type=str)
parser.add_argument('--sample', required=True, type=int)
parser.add_argument('--bsz', default=8, type=int)
parser.add_argument('--img_size', default=224, type=int)
parser.add_argument('--n_slices', default=32, type=int)
parser.add_argument('--windowing', default="old", type=str)
parser.add_argument('--augmentation', default=True, type=bool)
parser.add_argument('--augmentation_config', default="HEAVY_AUG", type=str)
parser.add_argument('--backbone', required=True, type=str)
parser.add_argument('--attn_mode', default="softminmax", type=str)
parser.add_argument('--dropout', default=0.3, type=float)
parser.add_argument('--data', required=True, type=str)
parser.add_argument('--num_workers', required=True, type=int)
parser.add_argument('--wts', default="1,1", type=str)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--pin_memory', default=False, type=bool)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--fast_dev_run', default=False, type=bool)
args_terminal = parser.parse_args()

if args_terminal.resume is not None:
    RESUME_CHECKPOINT = args_terminal.resume
    args_terminal_ = munch.munchify(torch.load(args_terminal.resume)["model_params"])
    args_terminal.resume = RESUME_CHECKPOINT
    for key in args_terminal.keys():
        if key not in args_terminal_.keys():
            args_terminal_[key] = args_terminal[key]
    args_terminal = args_terminal_
    del args_terminal_

if type(args_terminal.wts) == str:
    wts = args_terminal.wts.split(",")
    for i in range(len(wts)):
        wts[i] = float(wts[i])
    args_terminal.wts = wts


# model_params_to_save = {
#     "backbone": args_terminal.backbone,
#     "attn_mode": args_terminal.attn_mode,
#     "dropout": args_terminal.dropout,
#     "data": args_terminal.data,
#     "num_workers": args_terminal.num_workers,
#     "wts": args_terminal.wts,
#     "img_size": args_terminal.img_size,
# }


LOG_DIR = os.path.join("logs", args_terminal.log)
os.makedirs(LOG_DIR, exist_ok=True)
print(f"Log: {LOG_DIR}")

MODEL_SAVE_DIR = os.path.join("/data_nas5/qer/ujjwal/models/fusion_resnet/", args_terminal.log)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
print(f"Model save: {MODEL_SAVE_DIR}")

device = torch.device(f"cuda:{args_terminal.gpu}")
print(f"Device used: {device}")


model_params_to_save = vars(args_terminal)
import pprint
pprint.pprint(model_params_to_save)

# with open("../configs/3dunet_22march.yaml", "r") as f:
#     args = munch.munchify(yaml.load(f, Loader=yaml.Loader))



df = pd.read_json(args_terminal.data)
df = df[df['status']=="train"]
samples_weight = []
val_count = df.annotation.value_counts()
n_samples = len(df.annotation)
l = df.annotation.to_list()
for i in l:
    if i == 0:
        samples_weight.append(n_samples/val_count[0])
    else:
        samples_weight.append(n_samples/val_count[1])
del df, l, val_count, n_samples

samples_weight = torch.DoubleTensor(samples_weight)
sampler = WeightedRandomSampler(samples_weight, args_terminal.sample)
train_dataloader = DataLoader(InfarctDataset3D_only_cls(args_terminal), batch_size=args_terminal.bsz, sampler=sampler, num_workers=args_terminal.num_workers, pin_memory=args_terminal.pin_memory)
test_dataloader = DataLoader(InfarctDataset3D_only_cls(args_terminal, 'valid'), batch_size=args_terminal.bsz, shuffle=False, num_workers=args_terminal.num_workers, pin_memory=args_terminal.pin_memory)


# from qer.common.multisoftmax_classifier.models import load_checkpoint
# model, args_old = load_checkpoint("/home/users/ujjwal.upadhyay/packages/qer/resources/checkpoints/infarcts/infarct_xentropy_incl_lacunar_resume.pth.tar")
# from qer.common.multisoftmax_classifier.models.fusion_resnet import FusionResnet
from fresnet import FusionResnet
model = FusionResnet(args_terminal.backbone, dropout=args_terminal.dropout, attn_mode=args_terminal.attn_mode).to(device=device, dtype=torch.float)
model.train()

num_epochs = 200
lr = args_terminal.lr


loss_criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(args_terminal.wts)).to(device)
optimizer = optim.SGD(model.parameters(), lr=lr)
# scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=5e-4, max_lr=1e-2)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=len(train_dataloader), epochs=num_epochs)

start_epoch = 0
train_iter = 0
val_iter = 0
max_accuracy = 0
min_loss = 10000
max_auc = 0

if args_terminal.resume is not None:
    print("Loading checkpoint")
    checkpoint = torch.load(args_terminal.resume)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = checkpoint["epoch"]+1
    max_accuracy = checkpoint["valid_accuracy"]
    min_loss = checkpoint["valid_epoch_loss"]
    max_auc = checkpoint["valid_auc"]
    train_iter = checkpoint["train_iter"]
    val_iter = checkpoint["val_iter"]
    print(f"Loaded checkpoint from epoch {start_epoch}, max_accuracy {max_accuracy}, min_loss {min_loss}, max_auc {max_auc}\n")


writer = SummaryWriter(log_dir=LOG_DIR)

for epoch in tqdm(range(start_epoch, num_epochs)):
    
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
            targets = targets.to(device=device)
            
            optimizer.zero_grad()

            output_labels = model(image_sequences)[0][0]

            loss = loss_criterion(output_labels, targets)
            loss.backward()
            
            optimizer.step()
            
            output_labels = torch.softmax(output_labels, dim=1)
            _, predicted = torch.max(output_labels.data, 1)
            corr__ = (predicted == targets).sum().item()
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
        
        train_sens = round((tp + 1e-6) / (tp+fn+1e-6),2)
        train_spec = round((tn + 1e-6) / (tn+fp+1e-6),2)
        ys = train_sens + train_spec - 1
        preds = torch.cat(preds).numpy()
        targets_gt = torch.cat(targets_gt).detach().cpu().numpy()
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
        
        writer.add_scalar('train/learning_rate', scheduler.get_lr()[0], train_iter)
        writer.add_scalar('train/epoch_loss', avg_loss, train_iter)
        writer.add_scalar('train/n_correct', int(numCorrTrain), train_iter)
        writer.add_scalar('train/sens', train_sens, train_iter)
        writer.add_scalar('train/spec', train_spec, train_iter)
        writer.add_scalar('train/yoden', ys, train_iter)
        writer.add_scalar('train/auc', train_roc_auc, train_iter)
        
        
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
                    targets = targets.to(device=device, non_blocking=True)

                    output_labels = model(image_sequences)[0][0]

                    val_loss = loss_criterion(output_labels, targets)
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

                
                if val_accuracy > max_accuracy:
                    scheduler_state = scheduler.state_dict()
                    try:
                        scheduler_state.pop("_scale_fn_ref")
                    except:
                        pass
                    checkpoint = { 
                            'epoch': epoch,
                            'lr': scheduler.get_lr()[0],
                            'val_iter': val_iter,
                            'train_iter': train_iter,
                            'batch_size': args_terminal.bsz,
                            'sample': args_terminal.sample,
                            'model_params': model_params_to_save,
                            'valid_epoch_loss': avg_val_loss,
                            'valid_accuracy': val_accuracy,
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
                    save_path_model = os.path.join(MODEL_SAVE_DIR, 'best_model_acc.pth')
                    torch.save(checkpoint, save_path_model)
                    max_accuracy = val_accuracy
                    print("Saving model with best accuracy @ epoch {}: {}\n".format(epoch+1, max_accuracy))

                if roc_auc > max_auc:
                    scheduler_state = scheduler.state_dict()
                    try:
                        scheduler_state.pop("_scale_fn_ref")
                    except:
                        pass
                    checkpoint = { 
                            'epoch': epoch,
                            'val_iter': val_iter,
                            'train_iter': train_iter,
                            'lr': scheduler.get_lr()[0],
                            'batch_size': args_terminal.bsz,
                            'sample': args_terminal.sample,
                            'model_params': model_params_to_save,
                            'valid_epoch_loss': avg_val_loss,
                            'valid_accuracy': val_accuracy,
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
                    save_path_model = os.path.join(MODEL_SAVE_DIR, 'best_model_auc.pth')
                    torch.save(checkpoint, save_path_model)
                    max_auc = roc_auc
                    print("Saving model with best auc @ epoch {}: {}\n".format(epoch+1, max_auc))
                
                if avg_val_loss < min_loss:
                    scheduler_state = scheduler.state_dict()
                    try:
                        scheduler_state.pop("_scale_fn_ref")
                    except:
                        pass
                    checkpoint = { 
                            'epoch': epoch,
                            'val_iter': val_iter,
                            'train_iter': train_iter,
                            'lr': scheduler.get_lr()[0],
                            'batch_size': args_terminal.bsz,
                            'sample': args_terminal.sample,
                            'model_params': model_params_to_save,
                            'valid_epoch_loss': avg_val_loss,
                            'valid_accuracy': val_accuracy,
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
                    print("Saving model with best loss @ epoch {}: {}\n".format(epoch+1, min_loss))

                if (val_accuracy > max_accuracy) and (roc_auc > max_auc) and (avg_val_loss < min_loss):
                    scheduler_state = scheduler.state_dict()
                    try:
                        scheduler_state.pop("_scale_fn_ref")
                    except:
                        pass
                    checkpoint = { 
                            'epoch': epoch,
                            'lr': scheduler.get_lr()[0],
                            'val_iter': val_iter,
                            'train_iter': train_iter,
                            'batch_size': args_terminal.bsz,
                            'sample': args_terminal.sample,
                            'model_params': model_params_to_save,
                            'valid_epoch_loss': avg_val_loss,
                            'valid_accuracy': val_accuracy,
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
                    save_path_model = os.path.join(MODEL_SAVE_DIR, 'best_model_all.pth')
                    torch.save(checkpoint, save_path_model)
                    max_accuracy = val_accuracy
                    min_loss = avg_val_loss
                    max_auc = roc_auc
                    print("Saving model with best accuracy @ epoch {}: {}\n".format(epoch+1, max_accuracy))

                if (epoch+1) % 10 == 0:
                    scheduler_state = scheduler.state_dict()
                    try:
                        scheduler_state.pop("_scale_fn_ref")
                    except:
                        pass
                    checkpoint = { 
                            'epoch': epoch,
                            'val_iter': val_iter,
                            'train_iter': train_iter,
                            'lr': scheduler.get_lr()[0],
                            'batch_size': args_terminal.bsz,
                            'sample': args_terminal.sample,
                            'model_params': model_params_to_save,
                            'valid_epoch_loss': avg_val_loss,
                            'valid_accuracy': val_accuracy,
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
                    # print("Saving model at epoch: {}".format(epoch+1))
