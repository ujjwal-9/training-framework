import segmentation_models_pytorch as smp
import torch
class JaccardIndex:
    def __init__(self, mode, jaccard=None):
        self.jaccard = jaccard
        self.mode = mode
    
    def __call__(self, input_tensor, target_tensor):
        if self.jaccard is not None:
            return 1-jaccard(input_tensor, target_tensor)
        else:
            return 1-smp.losses.JaccardLoss(mode=self.mode)(input_tensor, target_tensor)


def mean_iou_ignore_idx(predictions, targets, ignore_index=-100):
    valid_mask = targets != ignore_index
    idx = torch.where(valid_mask.sum(axis=(1,2))>0)

    intersection = (predictions[idx] &  targets[idx]).sum(dim=(1,2))
    union = (predictions[idx] | targets[idx]).sum(dim=(1,2))

    iou = (intersection.float() / union.float())
    return iou