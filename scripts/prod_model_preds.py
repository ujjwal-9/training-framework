import warnings
warnings.filterwarnings("ignore")

import os
import torch
import munch
import argparse
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from qtrain.dataset.infarct import InfarctDataset3D_only_cls

torch.multiprocessing.set_sharing_strategy('file_system')
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

parser = argparse.ArgumentParser()
parser.add_argument('--save', required=True, type=str)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--test', default="/cache/fast_data_nas72/qer/ujjwal/q25k_cached/jsons/q25k_s224npy_25_04_23.json", type=str)
parser.add_argument('--complete', default=False, type=bool)
args_terminal = parser.parse_args()


device = torch.device(f"cuda:{args_terminal.gpu}")

print(f"Data source: {args_terminal.test}")
print(f"Device used: {device}\n")

checkpoint_name = "prod_model"
 
from qer.common.multisoftmax_classifier.models import load_checkpoint
model, args_old = load_checkpoint("/home/users/ujjwal.upadhyay/packages/qer/resources/checkpoints/infarcts/infarct_xentropy_incl_lacunar_resume.pth.tar")
    
import pprint
pprint.pprint(args_old)

test_dataset = InfarctDataset3D_only_cls(args_old, 'valid')



params = torch.load(args_terminal.checkpoint, map_location="cpu")
org_save_name = args_terminal.save
args_terminal.save = args_terminal.save + "_epoch_" + str(params["epoch"]) + '.txt'
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
        result = model(input_tensor)
        img_output = result[0]
        softmax_output = [F.softmax(x, dim=1).data.cpu()[0, 1] for x in img_output]
        out = softmax_output[0].item()
        with open(outpath, 'a') as f:
            f.write(f"{series}, {out}\n")
        preds.append(out)
        done.append(series)
        # except:
        #     with open(outpath, 'a') as f:
        #         f.write(f"{series}, {None}\n")
    return preds, done

preds, done = process(df)
