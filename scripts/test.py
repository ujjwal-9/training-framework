import json, yaml, random, os
import munch
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from qtrain.utils.data import InfarctDataset
from skimage.segmentation import mark_boundaries
from pprint import pprint
from tqdm import tqdm
from shutil import copyfile
from collections import defaultdict
from qtrain.utils.dice_coeff import dice_coeff
import qtrain.models.transunet.networks.vit_seg_configs as configs

# hparams_file = 'checkpoints/lightning_logs/version_1/hparams.yaml'
# checkpoint_file = 'checkpoints/lightning_logs/version_1/checkpoints/epoch=66-valid_metric=-2185.28-train_metric=-686.26.pth'

# with open(hparams_file, 'r') as f:
#     args = yaml.safe_load(f)
# # args
parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default = 'version_15', help='version to test')
parser.add_argument('--chkpt', type=str, default = 'best', help='which checkpoint to use best or last')
parser.add_argument('--user', type=str, default = 'ujjwal.upadhyay', help='user name which stores the checkpoint')

init_args = parser.parse_args()

chkpt_root = '/home/users/{}/projects/infarcts/checkpoints/lightning_logs/{}/'.format(init_args.user,init_args.version)
hparams_file = os.path.join(chkpt_root, 'hparams.yaml')
chkpt_path = os.path.join(chkpt_root, 'checkpoints')
if init_args.chkpt == 'best':
    files = os.listdir(chkpt_path)
    for file in files:
        if file.startswith('epoch'):
            checkpoint_file = os.path.join(chkpt_path,file)
else:
    checkpoint_file = os.path.join(chkpt_path,'last.pth')
print(">>Loading Checkpoint from file: {}...".format(checkpoint_file))


def get_args(hparams_file):
    with open(hparams_file, 'r') as f:
        args = yaml.safe_load(f)
    args = args['args']
    args.only_annotated_slices = True
    print(">>Printing args...")
    pprint(args)
    return args

def get_model(hparams_file, checkpoint_file):
    
    model = InfarctSegmentation.load_from_checkpoint(
        checkpoint_path=checkpoint_file,
        hparams_file=hparams_file,
        map_location=None
    )
    return model.eval()

def thresholding(pred, threshold=0.01):
    pred[pred > threshold] = 1
    pred[pred < threshold] = 0

    return pred.astype("int")

args = get_args(hparams_file)

if args.model == "transunet":
    from qtrain.models.train_transunet import InfarctSegmentation
elif args.model == "unet":
    from qtrain.models.train_unet import InfarctSegmentation
elif args.model == "unetplusplus":
    from qtrain.models.train_unetplusplus import InfarctSegmentation
elif args.model == "linknet":
    from qtrain.models.train_linknet import InfarctSegmentation
else:
    print("Invalid Model")
    exit(0)


if __name__ == "__main__":
    
    # hparams_file = '../checkpoints/lightning_logs/version_1/hparams.yaml'
    # checkpoint_file = '../checkpoints/lightning_logs/version_1/checkpoints/epoch=66-valid_metric=-2185.28-train_metric=-686.26.pth'
    model = get_model(hparams_file, checkpoint_file)
    # args['datapath'] = 'data/datasetv0.csv'
    dataset = InfarctDataset('valid', args)

    threshold_values = np.arange(0.6,1.1,0.1)
    diff_thresholds = defaultdict(dict)

    for thresh in tqdm(threshold_values):
        regions = []
        total = 0
        count = 0
        for i in tqdm(range(len(dataset))):
            count += 1
            input_data = dataset[i][0]
            target = dataset[i][1].double()
            output_data = torch.Tensor(thresholding(model(input_data.unsqueeze(0))[0].detach().clone().numpy(), threshold=thresh)).double()
            #model(input_data.unsqueeze(0)) : (1, 2, 224, 224)
            #model(input_data.unsqueeze(0))[0] : (2, 224, 224)
            for j in range(output_data.shape[0]):
                if i == 0:
                    regions.append(dice_coeff(output_data[j], target[j]))
                else:
                    regions[j] += dice_coeff(output_data[j], target[j])
            total += dice_coeff(output_data, target)
#             print(">> total:{}".format(total))

        total = total / count
        print(total)
        for i in range(len(regions)):
            regions[i] = regions[i] / count
                
        for k, region in enumerate(args.regions[:output_data.shape[0]]):
            diff_thresholds[region][thresh] = regions[k].item()
            filename = 'assets/dice_score_{}_best_model_{}.json'.format(args.model,init_args.version)
            print(">>saving {}...".format(filename))
            with open(filename, 'w') as fp:
                json.dump(diff_thresholds, fp, indent=4)
