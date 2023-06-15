import torch
import segmentation_models_pytorch as smp

class UnetContrast(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = args.model(args)
        self.args = args.contrast_params
        in_channels = self.model.encoder.out_channels[-1]
        self.conv2d = torch.nn.Conv2d(in_channels=in_channels, 
                                      out_channels=self.args.out_channels, 
                                      kernel_size=self.args.kernel_size)

    def forward(self, scana, scanb):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        x_a = self.model(scana)["bottleneck"]
        x_a = self.conv2d(x_a).flatten(1)

        x_b = self.model(scanb)["bottleneck"]
        x_b = self.conv2d(x_b).flatten(1)

        return x_a, x_b