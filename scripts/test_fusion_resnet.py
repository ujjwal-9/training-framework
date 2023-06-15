import warnings
warnings.filterwarnings("ignore")

import os
import torch
import munch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from qtrain.dataset.infarct import InfarctDataset3D_only_cls

torch.multiprocessing.set_sharing_strategy('file_system')
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

parser = argparse.ArgumentParser()
parser.add_argument('--save', required=True, type=str)
parser.add_argument('--checkpoint', required=True, type=str)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--test', default="/cache/fast_data_nas72/qer/ujjwal/q25k_cached/jsons/q25k_s224npy_25_04_23.json", type=str)
parser.add_argument('--complete', default=False, type=bool)
args_terminal = parser.parse_args()

checkpoint_name = os.path.dirname(args_terminal.checkpoint)
checkpoint_version = os.path.basename(args_terminal.checkpoint)
print("Evaluating: ", checkpoint_name)
args_terminal.checkpoint = os.path.join("/data_nas5/qer/ujjwal/models/fusion_resnet", args_terminal.checkpoint)
if not os.path.exists(args_terminal.checkpoint):
    raise Exception(f"Checkpoint not found: {args_terminal.checkpoint}")

print(f"Please get all checkpoint details from here: {args_terminal.checkpoint}\n\n")

checkpoint = torch.load(args_terminal.checkpoint, map_location="cpu")
print(print(f'Loaded checkpoint from epoch {checkpoint["epoch"]+1}, max_accuracy {checkpoint["valid_accuracy"]}, min_loss {checkpoint["valid_epoch_loss"]}, max_auc {checkpoint["valid_auc"]}\n'))

args_terminal_ = munch.munchify(checkpoint["model_params"])
del checkpoint
args_terminal_.checkpoint = args_terminal.checkpoint
args_terminal_.gpu = args_terminal.gpu
args_terminal_.save = args_terminal.save
args_terminal_.test = args_terminal.test
args_terminal_.complete = args_terminal.complete
args_terminal = args_terminal_
del args_terminal_



model_params_to_save = vars(args_terminal)
device = torch.device(f"cuda:{args_terminal.gpu}")

print(f"Model checkpoint: {args_terminal.checkpoint}")
print(f"Data source: {args_terminal.test}")
print(f"Device used: {device}\n")

defaults = {
    "attn_mode": "softminmax",
    "dropout": 0.0,
    "img_size": 224,
    "n_slices": 32,
    "windowing": "old",
    "augmentation": False,
    "fast_dev_run": False
}

for key in defaults:
    if key not in args_terminal:
        args_terminal[key] = defaults[key]
    
import pprint
pprint.pprint(model_params_to_save)

test_dataset = InfarctDataset3D_only_cls(args_terminal, 'valid')

from fresnet import FusionResnet
model = FusionResnet(args_terminal.backbone, dropout=args_terminal.dropout, attn_mode=args_terminal.attn_mode).to(device=device, dtype=torch.float)

#from qer.common.multisoftmax_classifier.models import load_checkpoint
#model, args_old = load_checkpoint("/home/users/ujjwal.upadhyay/packages/qer/resources/checkpoints/infarcts/infarct_xentropy_incl_lacunar_resume.pth.tar")
params = torch.load(args_terminal.checkpoint, map_location="cpu")
org_save_name = args_terminal.save
args_terminal.save = args_terminal.save + "_epoch_" + str(params["epoch"]) + '_' + str(checkpoint_version.split(".pth")[0]) + '.txt'
outpath = os.path.join("/data_nas5/qer/ujjwal/preds", checkpoint_name, args_terminal.save)
os.makedirs(os.path.dirname(outpath), exist_ok=True)

if not args_terminal.complete:
    from glob import glob
    already_there = glob(os.path.join("/data_nas5/qer/ujjwal/preds", checkpoint_name, "*.txt"))
    for file in already_there:
        file = file.split("/")[-1]
        print(file)
        if org_save_name in file:
            epoch = file.split("_epoch_")[1].split("_")[0]
            print(epoch, params["epoch"])
            if str(epoch) == str(params["epoch"]):
                print("HERE")
                raise Exception(f"Epoch {epoch} already evaluated. Please delete {file} and try again.")

    if os.path.exists(outpath):
        raise Exception(f"Save path already exists: {outpath}")

print(f"\n\nSaving predictions to: {outpath}\n")


model.load_state_dict(params["model"])
model = model.to(device=device, dtype=torch.float)
model.eval()


# df = pd.read_json("/cache/fast_data_nas72/qer/ujjwal/q25k_cached/jsons/q25k_s224npy_25_04_23.json")
df = pd.read_json(args_terminal.test)

def preprocess_cts(ct_scan):
    ct_scan = test_dataset.get_foreground_crop(ct_scan)
    ct_scan = test_dataset.initialize_transform()(ct_scan)
    if ct_scan.shape[0] <= args_terminal.n_slices:
        ct_scan = torch.cat([ct_scan, torch.zeros(args_terminal.n_slices - ct_scan.shape[0] + 1, *ct_scan.shape[1:])])
    ct_scan = test_dataset.get_window_channels(ct_scan)
    ct_scan = ct_scan.unsqueeze(0).permute(0, 1, 4, 2, 3).to(device=device, dtype=torch.float)
    return ct_scan


def process(df):
    preds = []
    done = []
    if not os.path.exists(outpath):
        os.system(f"touch {outpath}")
    with open(outpath, 'r') as f:
        for l in f:
            series = l.split(",")[0]
            done.append(series)
    for i in tqdm(range(len(df))):
        series = df.loc[i, "series"]
        if series in done:
            continue
        # try:
        sitk_arr = np.load(df.loc[i, "filepath"])
        input_tensor = preprocess_cts(sitk_arr)
        out = model(input_tensor)[0][0]
        out = torch.softmax(out, dim=1)[0,1].item()
        with open(outpath, 'a') as f:
            f.write(f"{series}, {out}\n")
        preds.append(out)
        done.append(series)
        # except:
        #     with open(outpath, 'a') as f:
        #         f.write(f"{series}, {None}\n")
    return preds, done

preds, done = process(df)
