# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------


#from .deit import deit_tiny_patch16_224_repbn, deit_small_patch16_224_repbn, deit_tiny_patch16_224
from .SELA_ViT import deit_tiny_patch16_224
from timm.models import create_model


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'deits':
        model = create_model(
            config.MODEL.NAME,
            pretrained=False,   
            num_classes=config.MODEL.NUM_CLASSES,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            drop_block_rate=None,
        )

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
