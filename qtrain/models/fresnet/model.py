import timm
import torch
from torch import nn
import torch.nn.functional as F
from .pooling import MultiConv1x1, MultiSoftmax, min_max, soft_attention, soft_pooling, lse_pooling


def net_from_name(name):
    return timm.create_model(name, pretrained=True)


class FusionResnet(nn.Module):
    """ResNet with MultiSoftmax.

    Final fc of resnet is removed and replaced with multi softmax.
    """

    def __init__(self, resnet, dropout=None, num_classes=2, img_size=[60,224,224], attn_mode=None, stride=None, pretrained=None, return_pixel_output=True):
        """
        Args:
            resnet: ResNet to be modified
            dropout: If provided, dropout of this much amount is inserted
                after block 3 and 4.
        """
        super().__init__()
        print(img_size)
        self.name = resnet
        self.img_size = img_size
        self.resnet = net_from_name(self.name)
        self.resnet.train()
        total_params = sum(param.numel() for param in self.resnet.parameters())
        print("Total params:", total_params)

        self.num_classes = num_classes
        self.use_dropout = dropout
        self.attn_mode = attn_mode
        self.stride = stride

        if self.name.startswith("res"):
            self.num_features = self.resnet.fc.in_features
            # remove fc, avgpool
            if self.stride:
                from qer.fractures.models.dilated_models import DilatedResNet as DilatedResnet
                self.resnet = DilatedResnet(self.resnet, num_channels=3, num_classes=num_classes, stride=stride).resnet
            else:
                del self.resnet._modules["fc"]
                del self.resnet._modules["avgpool"]
        elif self.name.startswith("se"):
            self.num_features = self.resnet.fc.in_features
            del self.resnet._modules["fc"]
        elif self.name.startswith("efficient"):
            self.num_features = self.resnet.classifier.in_features
            del self.resnet._modules["classifier"]
        else:
            raise ValueError(f"Unknow model {self.name}")

        self.multi_softmax = MultiSoftmax(self.num_features, self.num_classes)
        self.multi_softmax.train()
        # for segmentation
        if return_pixel_output:
            self.return_pixel_output = return_pixel_output
            self.multi_conv = MultiConv1x1(self.num_features, self.num_classes)

        if self.use_dropout:
            self.dropout = nn.Dropout2d(self.use_dropout)

    def _if_dropout(self, x):
        if self.use_dropout:
            x = self.dropout(x)

        return x

    def _pool(self, x, use_pooling):
        if use_pooling == "avg":
            return x.mean(dim=-1).mean(dim=-1)
        elif use_pooling == "max":
            return x.max(dim=-1)[0].max(dim=-1)[0]
        elif use_pooling == "soft":
            return soft_pooling(x)
        elif use_pooling == "gem":
            p = 3
            x = x.clamp(min=1e-6)
            x = x**p
            x = x.mean(dim=-1).mean(dim=-1)
            x = x ** (1.0 / p)
            return x
        else:
            raise NotImplementedError(f"{use_pooling} Pooling not supported")

    def ftrs(self, x):
        if self.name.startswith("res"):
            x = self.resnet.conv1(x)
            x = self.resnet.bn1(x)
            x = self.resnet.relu(x)
            x = self.resnet.maxpool(x)

            x = self.resnet.layer1(x)
            x = self.resnet.layer2(x)
            x = self.resnet.layer3(x)
            x = self._if_dropout(x)
            x = self.resnet.layer4(x)
            ftrs = self._if_dropout(x)
            pooled_output = self._pool(ftrs, "avg")
        elif self.name.startswith("se") or self.name.startswith("efficient"):
            ftrs = self.resnet.forward_features(x)
            pooled_output = self.resnet.global_pool(ftrs)
            pooled_output = self.multi_softmax(pooled_output)
        return ftrs, pooled_output

    def forward(self, x):
        return_pixel_output = self.return_pixel_output
        # x = (b, 3, z, 224, 224)
        batch_size = x.size(0)
        z_size = x.size(2)

        # fold z in to batch dimension
        x = x.transpose(1, 2).contiguous()  # (b, z, 3, 224, 224)
        x = x.view(-1, *x.size()[2:])  # (b*z, 3, 224, 224)
        x, pooled_output = self.ftrs(x)  # (b*z, num_features, 7, 7)

        # pooled_output = self.multi_softmax(pooled_output)  # list of (b*z, 2)
        slice_output = [y.view(batch_size, z_size, 2).transpose(1, 2) for y in pooled_output]  # list of (b, 2, z)

        if self.attn_mode == "hard":
            img_output = [min_max(y) for y in slice_output]  # list of b x 2
        elif self.attn_mode in ("soft", "softminmax"):
            img_output = [soft_attention(y) for y in slice_output]
        elif self.attn_mode == "lse":
            img_output = [lse_pooling(y) for y in slice_output]
        else:
            raise NotImplementedError
        
        if return_pixel_output:
            if not getattr(self, "multi_conv", None):
                return img_output, slice_output, None

            # list of (b*z, 2, 7, 7)
            pixel_output = self.multi_conv(x)
            # list of (b, 2, z, 7, 7)
            pixel_output = [y.view(batch_size, z_size, 2, *y.shape[-2:]).transpose(1, 2) for y in pixel_output]
            for i in range(len(pixel_output)):
                pixel_output[i] = F.softmax(pixel_output[i], dim=1)
                pixel_output[i] = F.interpolate(pixel_output[i], size=self.img_size, mode="trilinear", align_corners=False)[0,1]

            pixel_output = torch.stack(pixel_output)
            # pixel_output= pixel_output[0, 1].detach().cpu().numpy()
            # x = (x > 0.5).astype("int16")
            # print(x.shape)
            # pixel_output = F.softmax(pixel_output[1], dim=1)
            # pixel_output = F.interpolate(pixel_output, size=self.img_size, mode="trilinear", align_corners=False)
            # pixel_output = pixel_output[:,1]
            # print(pixel_output.shape)

            return img_output[0], slice_output[0], pixel_output
        else:
            return img_output, slice_output