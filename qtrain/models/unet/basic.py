import torch
from torch import nn

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base import SegmentationHead, ClassificationHead
from segmentation_models_pytorch.base import initialization as init

class Unet_Modified(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.segmentation_head = None
        self.task = args.task

        self.encoder = smp.encoders.get_encoder(name=args.encoder_name,
                                                in_channels=args.windows,
                                                depth=args.encoder_depth, 
                                                weights='imagenet')

        if "segmentation" in self.task:
            self.decoder = UnetDecoder(
                encoder_channels=self.encoder.out_channels,
                decoder_channels=args.decoder_channels,
                n_blocks=args.encoder_depth,
                use_batchnorm=args.decoder_use_batchnorm,
                attention_type=None if "decoder_attention_type" not in args.keys() else args.decoder_attention_type,
                center=True if args.encoder_name.startswith("vgg") else False,
            )
            self.segmentation_head = SegmentationHead(
                in_channels=args.decoder_channels[-1],
                out_channels=args.n_maps,
                activation=None if "seg_activation" not in args.keys() else args.seg_activation,
                kernel_size=args.seg_kernel_size,
            )

        self.initialize()

    def initialize(self):
        if self.segmentation_head is not None:
            init.initialize_decoder(self.decoder)
            init.initialize_head(self.segmentation_head)

    def forward(self, ct):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        
        masks = None
        features = self.encoder(ct)
        
        if self.segmentation_head is not None:
            decoder_output = self.decoder(*features)
            masks = self.segmentation_head(decoder_output)

        return masks


        