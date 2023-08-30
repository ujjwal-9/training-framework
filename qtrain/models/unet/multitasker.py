import torch
from torch import nn

import qer_utils
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base import initialization as init
from segmentation_models_pytorch.base import ClassificationHead
from torchvision.ops import SqueezeExcitation
from qtrain.utils.pooling import lse_pooling
from collections import defaultdict


class MultiTaskSeqAttn(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        try:
            self.combined_embedding = self.args.combined_embedding
        except:
            self.combined_embedding = False

        assert (
            ("normal" in args.tasks)
            or ("stacked_normal" in args.tasks)
            or ("stacked_normal" not in args.tasks and "normal" not in args.tasks)
        ), "stacked normal and normal head can't be used together"

        if "stacked_normal" in args.tasks:
            assert (
                "infarct" in args.tasks
            ), "infarct head is required for stacked normal head"

        self.encoder = smp.encoders.get_encoder(
            name=self.args.encoder_name,
            in_channels=self.args.in_channels,
            depth=self.args.depth,
            output_stride=self.args.output_stride,
            weights="imagenet",
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=self.args.decoder_channels,
            n_blocks=self.args.depth,
            use_batchnorm=self.args.use_batchnorm,
            attention_type=self.args.attention_type,
            center=True if self.args.encoder_name.startswith("vgg") else False,
        )

        if "seg" in self.args.tasks:
            self.segmentation_head = SegmentationHead(
                in_channels=self.args.decoder_channels[-1],
                out_channels=self.args.n_segmaps,
                activation=self.args.seg_activation,
                kernel_size=self.args.kernel_size,
            )

        if "slc" in self.args.tasks:
            self.slc_classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1],
                classes=self.args.cls_nclasses,
                pooling=self.args.cls_pooling,
                dropout=self.args.cls_dropout,
                activation=None,
            )

        if "infarct" in self.args.tasks:
            self.acute_chronic_classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1],
                classes=self.args.cls_ac_nclasses,
                pooling=self.args.cls_ac_pooling,
                dropout=self.args.cls_ac_dropout,
                activation=None,
            )
            self.acute_chronic_fc = nn.Linear(self.args.n_slices, 1)

        if "normal" in self.args.tasks:
            self.normal_classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1],
                classes=self.args.cls_normal_nclasses,
                pooling=self.args.cls_normal_pooling,
                dropout=self.args.cls_normal_dropout,
                activation=None,
            )

        if "stacked_normal" in self.args.tasks:
            self.stacked_normal_classification_head = nn.Linear(
                self.args.cls_ac_nclasses, 2
            )

        self.se_blocks = nn.ModuleList(
            [SqueezeExcitation(self.args.n_slices, 5) for i in range(self.args.depth)]
        )

        if self.combined_embedding:
            self.aggregator_decoder = nn.Conv2d(
                self.args.decoder_channels[-1],
                self.encoder.out_channels[-1],
                3,
                stride=8,
                padding=1,
            )

        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        if "seg" in self.args.tasks:
            init.initialize_head(self.segmentation_head)
        if "slc" in self.args.tasks:
            init.initialize_head(self.slc_classification_head)
        if "infarct" in self.args.tasks:
            init.initialize_head(self.acute_chronic_classification_head)
            init.initialize_head(self.acute_chronic_fc)
        if "normal" in self.args.tasks:
            init.initialize_head(self.normal_classification_head)
        if "stacked_normal" in self.args.tasks:
            init.initialize_head(self.stacked_normal_classification_head)

    def forward(self, ct, embedding=False):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        output = defaultdict(list)
        batch_size = ct.size(0)
        # z_size = ct.size(1)
        for bsz in range(batch_size):
            features = self.encoder(ct[bsz])
            for i in range(1, len(features)):
                features[i] = self.se_blocks[i - 1](
                    features[i].permute(1, 0, 2, 3)
                ).permute(1, 0, 2, 3)

            decoder_output = self.decoder(*features)

            if "seg" in self.args.tasks:
                masks = self.segmentation_head(decoder_output)
                output["masks"].append(masks)

            if self.combined_embedding:
                features[-1] = self.aggregator_decoder(decoder_output) * features[-1]

            if "slc" in self.args.tasks:
                slc_logits = self.slc_classification_head(features[-1])
                output["slc_logits"].append(slc_logits)

            if "infarct" in self.args.tasks:
                acute_chronic_logits = self.acute_chronic_classification_head(
                    features[-1]
                )
                output["acute_chronic_logits"].append(acute_chronic_logits)

            if "normal" in self.args.tasks:
                normal_logits = self.normal_classification_head(features[-1])
                output["normal_logits"].append(normal_logits)

            if embedding:
                output["embedding"].append(features[-1])

        if "seg" in self.args.tasks:
            output["masks"] = torch.stack(output["masks"])
        if "slc" in self.args.tasks:
            output["slc_logits"] = torch.stack(output["slc_logits"])
        if "infarct" in self.args.tasks:
            output["acute_chronic_logits"] = self.acute_chronic_fc(
                torch.stack(output["acute_chronic_logits"]).permute(0, 2, 1)
            )[:, :, 0]
        if "normal" in self.args.tasks:
            output["normal_logits"] = lse_pooling(
                torch.stack(output["normal_logits"]).permute(0, 2, 1)
            )
        if "stacked_normal" in self.args.tasks:
            output["normal_logits"] = self.stacked_normal_classification_head(
                output["acute_chronic_logits"]
            )
        if embedding:
            output["embedding"] = torch.stack(output["embedding"])
        return output
