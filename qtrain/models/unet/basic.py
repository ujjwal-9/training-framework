import torch
from torch import nn

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import initialization as init

class Unet_Modified(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.segmentation_head = None
        self.classification_head = None
        self.task = args.task
        self.args = args.model_params

        self.encoder = smp.encoders.get_encoder(name=self.args.encoder_name,
                                                in_channels=self.args.in_channels,
                                                depth=self.args.depth,
                                                output_stride=self.args.output_stride, 
                                                weights='imagenet')

        if "segmentation" in self.task:
            from segmentation_models_pytorch.base import SegmentationHead
            from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
            self.decoder = UnetDecoder(
                encoder_channels=self.encoder.out_channels,
                decoder_channels=self.args.decoder_channels,
                n_blocks=self.args.depth,
                use_batchnorm=self.args.use_batchnorm,
                attention_type=self.args.attention_type,
                center=True if self.args.encoder_name.startswith("vgg") else False,
            )
            self.segmentation_head = SegmentationHead(
                in_channels=self.args.decoder_channels[-1],
                out_channels=self.args.n_maps,
                activation=self.args.seg_activation,
                kernel_size=self.args.kernel_size,
            )

        if "classification" in args.task:
            from segmentation_models_pytorch.base import ClassificationHead
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1],
                classes=args.num_classes,
                pooling=self.args.cls_pooling,
                dropout=self.args.cls_dropout,
                activation=self.args.cls_activation,
            )

        self.initialize()

    def initialize(self):
        if self.segmentation_head is not None:
            init.initialize_decoder(self.decoder)
            init.initialize_head(self.segmentation_head)

    def forward(self, ct):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        output = {}
        masks = None
        
        features = self.encoder(ct)
        
        output["bottleneck"] = features[-1]
        
        if self.segmentation_head is not None:
            decoder_output = self.decoder(*features)
            masks = self.segmentation_head(decoder_output)
            output["decoder_output"] = decoder_output
            output["masks"] = masks

        if self.classification_head is not None:
            cls_logits = self.classification_head(features[-1])
            output["cls_logits"] = cls_logits

        return output


        