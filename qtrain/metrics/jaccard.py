import segmentation_models_pytorch as smp

class JaccardIndex:
    def __init__(self, mode, jaccard=None):
        self.jaccard = jaccard
        self.mode = mode
    
    def __call__(self, input_tensor, target_tensor):
        if self.jaccard is not None:
            return 1-jaccard(input_tensor, target_tensor)
        else:
            return 1-smp.losses.JaccardLoss(mode=self.mode)(input_tensor, target_tensor)