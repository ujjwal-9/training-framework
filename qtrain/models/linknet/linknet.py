import torch
from torch import nn

import segmentation_models_pytorch as smp
# from segmentation_models_pytorch.unet.decoder import UnetDecoder
# from segmentation_models_pytorch.base import SegmentationHead, ClassificationHead
# from segmentation_models_pytorch.base import initialization as init

class LinkNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.task = args.task
        self.model = smp.Linknet(encoder_name=args.encoder_name,
                                      in_channels=args.windows,
                                      classes=len(args.regions),
                                      encoder_depth=args.encoder_depth, 
                                      encoder_weights='imagenet',
                                      decoder_use_batchnorm=args.decoder_use_batchnorm
                                      )

    def forward(self, ct):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        masks = self.model(ct)
        return masks