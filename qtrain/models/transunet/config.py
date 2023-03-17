import qtrain.models.transunet.networks.vit_seg_configs as configs
CONFIGS = {
        'ViT-B_16': configs.get_b16_config(args),
        'ViT-B_32': configs.get_b32_config(args),
        'ViT-L_16': configs.get_l16_config(args),
        'ViT-L_32': configs.get_l32_config(args),
        'ViT-H_14': configs.get_h14_config(args),
        'R50-ViT-B_16': configs.get_r50_b16_config(args),
        'R50-ViT-L_16': configs.get_r50_l16_config(args),
        'R26-ViT-B_32': configs.get_r26_b32_config(args),
        'testing': configs.get_testing(args),
    }