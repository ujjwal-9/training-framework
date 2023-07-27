from clearml import Task

import warnings
warnings.filterwarnings('ignore')

import os
import json
import yaml
import munch
import random
import pprint
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmetrics.classification import BinaryConfusionMatrix, BinaryAUROC
from qtrain.dataset.infarct import InfarctDataset3D
from qtrain.losses.dice_loss import GDiceLossV2


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# np.random.seed(0)
# torch.manual_seed(0)
seed_everything(0)
torch.multiprocessing.set_sharing_strategy('file_system')

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



parser = argparse.ArgumentParser()
parser.add_argument('--mp', default=True, type=str2bool)
parser.add_argument('--log', required=True, type=str)
parser.add_argument('--gpu', required=True, type=str)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--val_batch_size', default=8, type=int)
parser.add_argument('--img_size', default=224, type=int)
parser.add_argument('--n_slices', default=32, type=int)
parser.add_argument('--windowing', default="old", type=str)
parser.add_argument('--only_annotated', default=False, type=str2bool)
parser.add_argument('--augmentation', default=True, type=str2bool)
parser.add_argument('--backbone', required=True, type=str)
parser.add_argument('--attn_mode', default="softminmax", type=str)
parser.add_argument('--dropout', default=0.3, type=float)
parser.add_argument('--datapath', required=True, type=str)
parser.add_argument('--num_workers', required=True, type=int)
parser.add_argument('--wts', default="1,1", type=str)
parser.add_argument('--seg_loss_wts', default={"focal": 3.0, "dice": 2.0, "mcc": 2.0}, type=json.loads)
parser.add_argument('--cls_loss_wts', default={"ce": 1.0}, type=json.loads)
parser.add_argument('--slc_loss_wts', default={"ce": 1.0}, type=json.loads)
parser.add_argument('--lr', default=5e-3, type=float)
parser.add_argument('--pin_memory', default=False, type=str2bool)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--fast_dev_run', default=False, type=str2bool)
parser.add_argument('--mode', default="binary", type=str)
parser.add_argument('--crop', default=True, type=str2bool)
parser.add_argument('--clearml', default=False, type=str2bool)
args_terminal = parser.parse_args()

if args_terminal.resume is not None:
    RESUME_CHECKPOINT = args_terminal.resume
    args_terminal_ = munch.munchify(torch.load(args_terminal.resume, map_location="cpu")["model_params"])
    args_terminal_.resume = RESUME_CHECKPOINT
    args_terminal_.log = args_terminal.log
    args_terminal_.gpu = args_terminal.gpu
    args_terminal_.num_workers = args_terminal.num_workers
    args_terminal_.n_slices = args_terminal.n_slices
    try:
        args_terminal_.mp = args_terminal.mp
    except:
        pass
    args_terminal = vars(args_terminal)
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

LOG_DIR = os.path.join("/data_nas5/qer/ujjwal/logs", args_terminal.log)
os.makedirs(LOG_DIR, exist_ok=True)
print(f"Log: {LOG_DIR}")

MODEL_SAVE_DIR = os.path.join("/data_nas5/qer/ujjwal/models/infarcts_seg/fusion_resnet", args_terminal.log)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
print(f"Model save: {MODEL_SAVE_DIR}")

device = torch.device(f"cuda:{args_terminal.gpu}")
print(f"Device used: {device}")

model_params_to_save = vars(args_terminal)
pprint.pprint(model_params_to_save)

# with open("../configs/3dunet_22march.yaml", "r") as f:
#     args = munch.munchify(yaml.load(f, Loader=yaml.Loader))
confusion_matrix = BinaryConfusionMatrix().to(device)
roc_auc_score = BinaryAUROC(thresholds=None).to(device)
ce_wts = torch.FloatTensor(args_terminal.wts).to(device)

if args_terminal.clearml:
    task = Task.init(project_name="Infarcts Segmentation", task_name=args_terminal.log)

# df = pd.read_json(args_terminal.data)
# df = df[df['status']=="train"]
# samples_weight = []
# val_count = df.annotation.value_counts()
# n_samples = len(df.annotation)
# l = df.annotation.to_list()
# for i in l:
#     if i == 0:
#         samples_weight.append(n_samples/val_count[0])
#     else:
#         samples_weight.append(n_samples/val_count[1])
# del df, l, val_count, n_samples

# samples_weight = torch.DoubleTensor(samples_weight)
# sampler = WeightedRandomSampler(samples_weight, args_terminal.sample)
train_dataloader = DataLoader(InfarctDataset3D(args_terminal), batch_size=args_terminal.batch_size, num_workers=args_terminal.num_workers, pin_memory=args_terminal.pin_memory)
if args_terminal.fast_dev_run == True:
    test_dataloader = DataLoader(InfarctDataset3D(args_terminal), batch_size=args_terminal.val_batch_size, num_workers=args_terminal.num_workers, pin_memory=args_terminal.pin_memory)
else:
    test_dataloader = DataLoader(InfarctDataset3D(args_terminal, 'valid'), batch_size=args_terminal.val_batch_size, shuffle=False, num_workers=args_terminal.num_workers, pin_memory=args_terminal.pin_memory)


# from qer.common.multisoftmax_classifier.models import load_checkpoint
# model, args_old = load_checkpoint("/home/users/ujjwal.upadhyay/packages/qer/resources/checkpoints/infarcts/infarct_xentropy_incl_lacunar_resume.pth.tar")
# from qer.common.multisoftmax_classifier.models.fusion_resnet import FusionResnet
from fresnet import FusionResnet
model = FusionResnet(args_terminal.backbone, dropout=args_terminal.dropout, attn_mode=args_terminal.attn_mode, num_classes=2, return_pixel_output=True).to(device=device, dtype=torch.float)
model.train()

num_epochs = 200
lr = args_terminal.lr

def dice_score_3d(predicted, ground_truth):
    intersection = torch.sum(predicted * ground_truth)
    total = torch.sum(predicted) + torch.sum(ground_truth)
    dice = (2.0 * intersection) / (total + 1e-7)  # Adding a small constant to avoid division by zero
    return dice

def dice_loss(predicted, ground_truth):
    intersection = torch.sum(predicted * ground_truth)
    total = torch.sum(predicted) + torch.sum(ground_truth)
    dice = (2.0 * intersection) / (total + 1e-7)  # Adding a small constant to avoid division by zero
    return 1-dice

import segmentation_models_pytorch as smp

def seg_loss_criterion(pred, gt):
    output_map = F.interpolate(pred, 
                               size=targets.shape[1:], 
                               mode="trilinear", 
                               align_corners=False)[:,0].contiguous()
    
    seg_losses = {"focal": smp.losses.FocalLoss(mode="binary"),
              "dice": GDiceLossV2(),
              "mcc": smp.losses.MCCLoss()}

    total_loss = 0.0
    loss_dict = {}
    for key in args_terminal.seg_loss_wts:
        loss_dict[key] = args_terminal.seg_loss_wts[key] * seg_losses[key](output_map, gt)
        total_loss += loss_dict[key]
    output_map = F.softmax(output_map, dim=1)
    metric = dice_score_3d((output_map>0.5).to(torch.int), gt)
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
    logits = [x - x.mean(dim=(0, 1), keepdim=True) for x in pred]
    slice_scores = logits[0].contiguous()
    target_score = (gt.sum(axis=(2,3))>0).to(torch.long).contiguous()
    slc_losses = {"ce": nn.CrossEntropyLoss(weight=ce_wts)}

    total_loss = 0.0
    loss_dict = {}
    for key in args_terminal.slc_loss_wts:
        loss_dict[key] = args_terminal.slc_loss_wts[key] * slc_losses[key](slice_scores, target_score)
        total_loss += loss_dict[key]

    return loss_dict, total_loss

def loss_criterion(pred, gt, prefix="train"):
    seg_loss_dict, seg_loss, metric = seg_loss_criterion(pred[2][0].contiguous(), gt)
    cls_loss_dict, cls_loss, target = cls_loss_criterion(pred[0][0].contiguous(), gt)
    slc_loss_dict, slc_loss = slc_loss_criterion(pred[1], gt)
    loss = seg_loss + cls_loss + slc_loss
    # loss = cls_loss + slc_loss
    losses = {
        "seg": seg_loss_dict,
        "slc": slc_loss_dict,
        "cls": cls_loss_dict
    }
    log_losses = {}
    for key in losses:
        for l in losses[key]:
            log_losses[prefix+"/"+key+"/"+l] = losses[key][l]
    return loss, log_losses, target, metric

optimizer = optim.SGD(model.parameters(), lr=lr)
# scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=5e-4, max_lr=1e-2)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=len(train_dataloader), epochs=num_epochs)

start_epoch = 0
train_iter = 0
val_iter = 0
max_accuracy = 0
min_loss = 10000
min_metric = 0
max_auc = 0
log_every_n_steps = 100

if args_terminal.mp == True:
    scaler = torch.cuda.amp.GradScaler()

if args_terminal.resume is not None:
    print("Loading checkpoint")
    checkpoint = torch.load(args_terminal.resume, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer = optim.SGD(model.parameters(), lr=checkpoint["lr"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"]+1
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=len(train_dataloader), epochs=num_epochs-start_epoch)
    scheduler.load_state_dict(checkpoint["scheduler"])
    try:
        scaler.load_state_dict(checkpoint["scaler"])
    except:
        pass
    if "valid_accuracy" in checkpoint.keys():
        max_accuracy = checkpoint["valid_accuracy"]
    else:
        max_accuracy = 0
    min_loss = checkpoint["valid_epoch_loss"]
    min_metric = checkpoint["valid_epoch_metric"]
    max_auc = checkpoint["valid_auc"]
    if "train_iter" in checkpoint.keys():
        train_iter = checkpoint["train_iter"]
        val_iter = checkpoint["val_iter"]
    else:
        train_iter = 0
        val_iter = 0
    print(f"Loaded checkpoint from epoch {start_epoch}, max_accuracy {max_accuracy}, min_loss {min_loss}, max_auc {max_auc}\n")


    writer = SummaryWriter(log_dir=LOG_DIR)

for epoch in tqdm(range(start_epoch, num_epochs)):
        torch.cuda.empty_cache()

        epoch_loss = 0
        epoch_metric = 0
        numCorrTrain = 0
        trainSamples = 0
        iterPerEpoch = 0
        tn, fp, fn, tp = 0,0,0,0
        preds = []
        targets_gt = []

        model.train()
        
        for i, (input_scan, targets) in enumerate(tqdm(train_dataloader, leave=True)):
            train_iter += 1
            iterPerEpoch += 1
            trainSamples += input_scan.size(0) 
            
            input_scan = torch.Tensor(input_scan.permute(0,2,1,3,4)).to(device=device, dtype=torch.float, non_blocking=True)
            targets = targets.to(device=device, non_blocking=True)
            optimizer.zero_grad()

            if args_terminal.mp == True:
                with torch.cuda.amp.autocast():
                    output = model(input_scan)
                    loss, log_losses, cls_target, metric = loss_criterion(output, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            else:
                output = model(input_scan)
                loss, log_losses, cls_target, metric = loss_criterion(output, targets)
                loss.backward()
                optimizer.step()
            
            output_labels = torch.softmax(output[0][0], dim=1)
            _, predicted = torch.max(output_labels.data, 1)
            batch_corr = (predicted == cls_target).sum().item()
            numCorrTrain += batch_corr
            
            preds.append(output_labels[:,1].ravel())
            targets_gt.append(cls_target.ravel())
            
            c_matrix = confusion_matrix(cls_target, predicted).ravel()
            tn += c_matrix[0].item()
            fp += c_matrix[1].item()
            fn += c_matrix[2].item()
            tp += c_matrix[3].item()
            
            epoch_loss += loss.item()
            epoch_metric += metric.item()
            
            if train_iter % log_every_n_steps == 0:
                writer.add_scalar('train/batch_loss', loss.item(), train_iter)
                writer.add_scalar('train/batch_dice_score', metric.item(), train_iter)
                for key in log_losses:
                    writer.add_scalar(key, log_losses[key].item(), train_iter)
        
        scheduler.step()
        train_avg_loss = epoch_loss / iterPerEpoch
        train_avg_metric = round(epoch_metric / iterPerEpoch, 3)
        trainAccuracy = float((numCorrTrain / trainSamples) * 100)
        
        train_sens = round((tp + 1e-6) / (tp+fn+1e-6),2)
        train_spec = round((tn + 1e-6) / (tn+fp+1e-6),2)
        train_ys = train_sens + train_spec - 1
        preds = torch.cat(preds)
        targets_gt = torch.cat(targets_gt)
        try:
            train_roc_auc = round(roc_auc_score(targets_gt, preds>0.5).item(),2)
        except:
            train_roc_auc = 0

        print('Train: Epoch = {} | Loss = {} | Accuracy = {} | Correct = {} | Sens = {} | Spec = {} | Yoden = {} | Auc = {} | Dice = {} |'.format(
            epoch+1, 
            round(train_avg_loss,5), 
            round(trainAccuracy,3),
            int(numCorrTrain),
            train_sens, train_spec, round(train_ys,2), train_roc_auc, train_avg_metric
        ))
        
        writer.add_scalar('train/learning_rate', scheduler.get_lr()[0], train_iter)
        writer.add_scalar('train/epoch-per-step', epoch, train_iter)
        writer.add_scalar('train/epoch_metric', train_avg_metric, train_iter)
        writer.add_scalar('train/epoch_loss', train_avg_loss, train_iter)
        writer.add_scalar('train/n_correct', int(numCorrTrain), train_iter)
        writer.add_scalar('train/sens', train_sens, train_iter)
        writer.add_scalar('train/spec', train_spec, train_iter)
        writer.add_scalar('train/yoden', train_ys, train_iter)
        writer.add_scalar('train/auc', train_roc_auc, train_iter)
        
        
        if (epoch+1) % 1 == 0:
            
            model.eval()
            val_epoch_loss = 0
            val_epoch_metric = 0
            val_samples = 0
            numCorrVal = 0
            
            tn, fp, fn, tp = 0,0,0,0
            preds = []
            targets_gt = []
            
            
            with torch.no_grad():
                for j, (input_scan, targets) in enumerate(tqdm(test_dataloader, leave=True)):
                    val_iter += 1
                    val_samples += input_scan.size(0)

                    input_scan = torch.Tensor(input_scan.permute(0,2,1,3,4)).to(device=device, dtype=torch.float, non_blocking=True)
                    targets = targets.to(device=device, non_blocking=True)
                    
                    
                    if args_terminal.mp == True:
                        with torch.cuda.amp.autocast():
                            output = model(input_scan)
                            val_loss, log_losses, cls_target, metric = loss_criterion(output, targets, prefix="valid")                  
                    else:
                        output = model(input_scan)
                        val_loss, log_losses, cls_target, metric = loss_criterion(output, targets, prefix="valid")
                        
                    val_epoch_loss += val_loss.item()
                    val_epoch_metric += metric.item()

                    output_labels = torch.softmax(output[0][0], dim=1)
                    _, predicted = torch.max(output_labels.data, 1)
                    batch_corr = (predicted == cls_target).sum().item()
                    numCorrVal += batch_corr
                    
                    preds.append(output_labels[:,1].ravel())
                    targets_gt.append(cls_target.ravel())

                    c_matrix = confusion_matrix(cls_target, predicted).ravel()
                    tn += c_matrix[0].item()
                    fp += c_matrix[1].item()
                    fn += c_matrix[2].item()
                    tp += c_matrix[3].item()
                    
                    if train_iter % log_every_n_steps == 0:
                        writer.add_scalar('train/batch_loss', loss.item(), train_iter)
                        writer.add_scalar('train/batch_dice_score', metric.item(), train_iter)
                        for key in log_losses:
                            writer.add_scalar(key, log_losses[key].item(), train_iter)

                    if val_iter % log_every_n_steps == 0:
                        writer.add_scalar('valid/batch_loss', val_loss.item(), val_iter)
                        writer.add_scalar('valid/batch_dice_score', metric.item(), val_iter)
                        for key in log_losses:
                            writer.add_scalar(key, log_losses[key].item(), train_iter)

                val_accuracy = float((numCorrVal / val_samples) * 100)
                val_avg_loss = val_epoch_loss / val_iter
                val_avg_metric = round(val_epoch_metric / val_iter, 3)
                val_sens = round((tp + 1e-6) / (tp+fn+1e-6),2)
                val_spec = round((tn + 1e-6) / (tn+fp+1e-6),2)
                val_ys = val_sens + val_spec - 1
                preds = torch.cat(preds)
                targets_gt = torch.cat(targets_gt)
                
                try:
                    val_roc_auc = round(roc_auc_score(targets_gt, preds>0.5).item(),2)
                except:
                    val_roc_auc = 0

                print('Val:   Epoch = {} | Loss = {} | Accuracy = {} | Correct = {} | Sens = {} | Spec = {} | Yoden = {} | Auc = {} | Dice = {} |'.format(
                    epoch + 1, 
                    round(val_avg_loss,5), 
                    round(val_accuracy,3),
                    int(numCorrVal),
                    val_sens, val_spec, round(val_ys,2), val_roc_auc, val_avg_metric
                ))
                
                writer.add_scalar('valid/epoch-per-step', epoch, val_iter)
                writer.add_scalar('valid/epoch_metric', val_avg_metric, val_iter)
                writer.add_scalar('valid/epoch_loss', val_avg_loss, val_iter)
                writer.add_scalar('valid/n_correct', int(numCorrVal), val_iter)
                writer.add_scalar('valid/sens', val_sens, val_iter)
                writer.add_scalar('valid/spec', val_spec, val_iter)
                writer.add_scalar('valid/yoden', val_ys, val_iter)
                writer.add_scalar('valid/auc', val_roc_auc, val_iter)

                
                if (val_accuracy > max_accuracy) or (val_roc_auc > max_auc) or (val_avg_loss < min_loss) or ((epoch+1) % 10 == 0):
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
                            'batch_size': args_terminal.batch_size,
                            'val_batch_size': args_terminal.val_batch_size,
                            'model_params': model_params_to_save,
                            'valid_epoch_loss': val_avg_loss,
                            'valid_epoch_metric': val_avg_metric,
                            'valid_accuracy': val_accuracy,
                            'valid_auc': val_roc_auc,
                            'valid_sens': val_sens,
                            'valid_spec': val_spec,
                            'train_epoch_loss': train_avg_loss,
                            'train_auc': train_roc_auc,
                            'train_sens': train_sens,
                            'train_spec': train_spec,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler_state,
                            'scaler': scaler.state_dict()
                        }
                    if val_accuracy > max_accuracy:
                        save_path_model = os.path.join(MODEL_SAVE_DIR, 'best_model_acc.pth')
                        torch.save(checkpoint, save_path_model)
                        max_accuracy = val_accuracy
                        print("Saving model with best accuracy @ epoch {}: {}\n".format(epoch+1, max_accuracy))

                    if val_roc_auc > max_auc:
                        save_path_model = os.path.join(MODEL_SAVE_DIR, 'best_model_auc.pth')
                        torch.save(checkpoint, save_path_model)
                        max_auc = val_roc_auc
                        print("Saving model with best auc @ epoch {}: {}\n".format(epoch+1, max_auc))
                
                    if val_avg_loss < min_loss:
                        save_path_model = os.path.join(MODEL_SAVE_DIR, 'best_model_loss.pth')
                        torch.save(checkpoint, save_path_model)
                        min_loss = val_avg_loss
                        print("Saving model with best loss @ epoch {}: {}\n".format(epoch+1, min_loss))

                    if val_avg_metric > min_metric:
                        save_path_model = os.path.join(MODEL_SAVE_DIR, 'best_model_metric.pth')
                        torch.save(checkpoint, save_path_model)
                        min_metric = val_avg_metric
                        print("Saving model with best metric @ epoch {}: {}\n".format(epoch+1, min_metric))

                    if (val_accuracy > max_accuracy) and (val_roc_auc > max_auc) and (val_avg_loss < min_loss) and (val_avg_metric > min_metric):
                        save_path_model = os.path.join(MODEL_SAVE_DIR, 'best_model_all.pth')
                        torch.save(checkpoint, save_path_model)
                        max_accuracy = val_accuracy
                        min_loss = val_avg_loss
                        max_auc = val_roc_auc
                        min_metric = val_avg_metric
                        print("Saving model with best everything @ epoch {}: AUC={}, Acc={}, Loss={}, Dice={}\n".format(epoch+1, max_auc, max_accuracy, min_loss, min_metric))

                    if (epoch+1) % 10 == 0:
                        save_path_model = os.path.join(MODEL_SAVE_DIR, f'epoch_{str(epoch+1)}.pth')
                        torch.save(checkpoint, save_path_model)
                        print("Saving model at epoch: {}".format(epoch+1))