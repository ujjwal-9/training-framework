import torch
from torch import nn

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base import initialization as init
from segmentation_models_pytorch.base import ClassificationHead

class UnetSeq(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.segmentation_head = None
        self.classification_head = None
        self.args = args

        self.encoder = smp.encoders.get_encoder(name=self.args.encoder_name,
                                                in_channels=self.args.in_channels,
                                                depth=self.args.depth,
                                                output_stride=self.args.output_stride, 
                                                weights='imagenet')

            
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
            
        self.classification_head = ClassificationHead(
            in_channels=self.encoder.out_channels[-1],
            classes=args.num_classes,
            pooling=self.args.cls_pooling,
            dropout=self.args.cls_dropout,
            activation=None,
        )
        self.linear_classification_cls = nn.Linear(self.args.n_slices*self.args.batch_size, self.args.batch_size)
        
        self.multilabel_classification_head = ClassificationHead(
            in_channels=self.encoder.out_channels[-1],
            classes=self.args.n_slices,
            pooling=self.args.cls_pooling,
            dropout=self.args.cls_dropout,
            activation=None,
        )
        self.linear_classification_slc = nn.Linear(self.args.n_slices*self.args.batch_size, self.args.batch_size)

        self.initialize()

    def initialize(self):
        if self.segmentation_head is not None:
            init.initialize_decoder(self.decoder)
            init.initialize_head(self.segmentation_head)

    def forward(self, ct):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        
        batch_size = ct.size(0)
        z_size = ct.size(1)
        ct = ct.view(-1, *ct.size()[2:])
        
        features = self.encoder(ct)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        cls_logits = self.classification_head(features[-1])
        cls_logits = self.linear_classification_cls(cls_logits.T).T
        slice_logits = self.multilabel_classification_head(features[-1])
        slice_logits = self.linear_classification_slc(slice_logits.T).T
        
        output = {}
        # output["bottleneck"] = features[-1]
#         output["decoder_output"] = decoder_output
        output["masks"] = masks.view(batch_size, z_size, self.args.img_size, self.args.img_size)
        output["cls_logits"] = cls_logits
        output["slc_logits"] = slice_logits

        return output