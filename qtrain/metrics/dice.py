import segmentation_models_pytorch as smp

class DiceCoeff:
    def __init__(self, mode, ignore_index=None, dice=None):
        self.dice = dice
        self.mode = mode
        self.ignore_index = ignore_index
    
    def __call__(self, input_tensor, target_tensor):
        if self.dice is not None:
            return 1-dice(input_tensor, target_tensor)
        else:
            return 1-smp.losses.DiceLoss(mode=self.mode, ignore_index=self.ignore_index)(input_tensor, target_tensor)