import torch
import segmentation_models_pytorch as smp

class UnetPlusPlus(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.task = args.task
        if args.status == "test":
            self.model = smp.UnetPlusPlus(encoder_name=args.encoder_name,
                                      in_channels=args.windows,
                                      classes=len(args.regions),
                                      decoder_channels=args.decoder_channels,
                                      encoder_depth=args.encoder_depth, 
                                      encoder_weights=None,
                                      decoder_use_batchnorm=args.decoder_use_batchnorm,
                                      decoder_attention_type=None if "decoder_attention_type" not in args.keys() else args.decoder_attention_type,
                                      )
        elif args.status == "train":
            self.model = smp.UnetPlusPlus(encoder_name=args.encoder_name,
                                        in_channels=args.windows,
                                        classes=len(args.regions),
                                        decoder_channels=args.decoder_channels,
                                        encoder_depth=args.encoder_depth, 
                                        encoder_weights='imagenet',
                                        decoder_use_batchnorm=args.decoder_use_batchnorm,
                                        decoder_attention_type=None if "decoder_attention_type" not in args.keys() else args.decoder_attention_type,
                                        )

    def forward(self, ct):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        masks = self.model(ct)
        return masks