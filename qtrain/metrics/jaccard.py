import segmentation_models_pytorch as smp

class JaccardIndex:
    def __init__(self, mode, ignore_index=None, jaccard=None):
        self.jaccard = dice
        self.mode = mode
        self.ignore_index = ignore_index
    
    def __call__(self, input_tensor, target_tensor):
        if self.jaccard is not None:
            return 1-jaccard(input_tensor, target_tensor)
        else:
            return 1-smp.losses.JaccardLoss(mode=mode, ignore_index=ignore_index)(input_tensor, target_tensor)