import torch
from torch import nn

import segmentation_models_pytorch as smp

class DeepLabV3(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.task = args.task
        self.model = smp.DeepLabV3(encoder_name=args.encoder_name,
                                      in_channels=args.windows,
                                      classes=len(args.regions),
                                      encoder_depth=args.encoder_depth, 
                                      encoder_weights='imagenet'
                                      )

    def forward(self, ct):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        masks = self.model(ct)
        return masks