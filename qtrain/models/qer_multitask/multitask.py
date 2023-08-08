import logging

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from collections import defaultdict

import qer_utils

logger = logging.getLogger(__name__)


class MultiTaskNet(torch.nn.Module):
    def __init__(self,args):

        # get all the above args in a dict
        super().__init__()
        self.args = args
        # TODO: dropout

        super().__init__()
        self.backbone = getattr(smp, "Unet")(
            encoder_name=self.args.encoder_name,
            encoder_depth=self.args.depth,
            decoder_channels=self.args.decoder_channels,
            classes=self.args.n_segmaps,
            encoder_weights=None,
            decoder_attention_type=self.args.attention_type,
        )
        self.init()

        self.num_features = self.backbone.encoder.out_channels[-1]

        
        self.num_features += self.backbone.segmentation_head[0].in_channels
        self.multi_fc = qer_utils.nn.models.multilabel.MultiFC(self.num_features, self.args.cls_nclasses)
        self.acute_chronic_multi_fc = qer_utils.nn.models.multilabel.MultiFC(self.num_features, self.args.cls_ac_nclasses)
        self.acute_chronic_fc = nn.Linear(self.args.n_slices, 1)

        self.normal_multi_fc = qer_utils.nn.models.multilabel.MultiFC(self.num_features, self.args.cls_nclasses)
        self.normal_fc = nn.Linear(self.args.n_slices, 1)

        self.dropout = torch.nn.Dropout(self.args.cls_dropout)

    def init(self, expected_fraction=0.99):
        """Initialize model parameters.

        Initialize segmentation head so that background is predicted
        most of the time. This idea is from Focal Loss paper.
        """
        fill_0 = np.log(expected_fraction / (1 - expected_fraction)) / 2
        bias = self.backbone.segmentation_head[0].bias.data
        bias[0].fill_(fill_0)
        bias[1].fill_(-fill_0)

    def _forward(self, x):
        # x = (b, 1, 224, 224)
        inp_shape = x.shape[2:]

        # copy paste from backbone
        features = self.backbone.encoder(x)
        embeddings = qer_utils.nn.models.pooling.pool(features[-1], "avg")
        embeddings = self.dropout(embeddings)

        output = {}
        decoder_output = self.backbone.decoder(*features)
        masks = self.backbone.segmentation_head(decoder_output)
        masks = torch.nn.functional.interpolate(masks, size=inp_shape, mode="bilinear", align_corners=False)

        embeddings = torch.cat(
            [embeddings, qer_utils.nn.models.pooling.pool(decoder_output, "max")],
            dim=1,
        )
        output["masks"] = masks

        slice_out = self.multi_fc(embeddings)
        output["slc_logits"] = slice_out[0]
        output["acute_chronic_logits"] = self.acute_chronic_fc(self.acute_chronic_multi_fc(embeddings)[0].permute(1,0)).T
        output["normal_logits"] = self.normal_fc(self.normal_multi_fc(embeddings)[0].permute(1,0)).T

#         output["slice_embeddings"] = embeddings
        return output

    def forward(self, ct, chunk_size=None):
        output = defaultdict(list)
        batch_size = ct.size(0)
        for bsz in range(batch_size):
            out = self._forward(ct[bsz])
            for key in out:
                output[key].append(out[key])
        for key in output:
            output[key] = torch.stack(output[key])
            if key in ["acute_chronic_logits", "normal_logits"]:
                output[key] = output[key].squeeze(1)
        return output

    def load_state_dict_legacy(self, state_dict):
        assert set(self.args.tasks) == {"classification"}
        state_dict = qer_utils.nn.serialization.modify_model_state_dict(
            model_state_dict=state_dict,
            old_prefix="backbone",
            new_prefix="backbone.encoder",
            modify_conv1x1=True,
        )
        super().load_state_dict(state_dict)


class MultiTaskSeq(MultiTaskNet):
    def forward(self, x, chunk_size=None):
        # x = (b, z, 1, 224, 224)
        inp_shape = x.shape
        batch_size = inp_shape[0]
        z_size = inp_shape[1]

        # fold z in to batch dimension
        x = x.view(-1, *x.size()[2:])  # (b*z, 1, 224, 224)
        out = super().forward(x, chunk_size)

        final_output = {}
        if "pixel" in out:
            pixel_output = out["pixel"]
            pixel_output = pixel_output.view(batch_size, z_size, pixel_output.shape[1], *inp_shape[-2:]).transpose(1, 2)
            final_output["pixel"] = pixel_output

        if "slice" in out:
            slice_output = out["slice"]
            slice_output = [y.view(batch_size, z_size, 2).transpose(1, 2) for y in slice_output]  # [(b, 2, z)]

            scan_output = [qer_utils.nn.models.fusion.lse_pooling(y, beta=4) for y in slice_output]

            final_output["scan"] = scan_output
            final_output["slice"] = slice_output

        embeddings = out["slice_embeddings"]
        embeddings = embeddings.view(batch_size, z_size, *embeddings.shape[1:])
        final_output["slice_embeddings"] = embeddings
        return final_output