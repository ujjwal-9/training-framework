import torch
from torch import nn

import segmentation_models_pytorch as smp

class Domain(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model1 = smp.Unet(encoder_name=args.encoder_name,
                              in_channels=args.windows,
                              classes=len(args.regions),
                              decoder_channels=args.decoder_channels,
                              encoder_depth=args.encoder_depth, 
                              encoder_weights='imagenet',
                              decoder_use_batchnorm=args.decoder_use_batchnorm,
                              decoder_attention_type=None if "decoder_attention_type" not in args.keys() else args.decoder_attention_type,
                              activation='sigmoid'
                            )
        
        self.model2 = smp.Unet(encoder_name=args.encoder_name,
                              in_channels=args.windows,
                              classes=len(args.regions),
                              decoder_channels=args.decoder_channels,
                              encoder_depth=args.encoder_depth, 
                              encoder_weights='imagenet',
                              decoder_use_batchnorm=args.decoder_use_batchnorm,
                              decoder_attention_type=None if "decoder_attention_type" not in args.keys() else args.decoder_attention_type,
                              activation='sigmoid'
                            )
                                      

    def forward(self, ct):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        masks = self.model(ct)
        return masks