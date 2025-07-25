from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from lib.models.ostrack.utils import combine_tokens, recover_tokens


class SELABackbone(nn.Module):
    def __init__(self):
        super().__init__()

        # for SELA-ViT (no global pos_embed, uses ConvPosEnc instead)
        self.pos_embed = None  
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 384

        self.cat_mode = 'direct'

        self.pos_embed_z = None
        self.pos_embed_x = None

        self.template_segment_pos_embed = None
        self.search_segment_pos_embed = None

        self.return_inter = False
        self.return_stage = [2, 5, 8, 11]

        self.add_cls_token = False  
        self.add_sep_seg = False

    def finetune_track(self, cfg, patch_start_index=0):  
        search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
        template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)
        new_patch_size = cfg.MODEL.BACKBONE.STRIDE

        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE
        self.return_inter = cfg.MODEL.RETURN_INTER
        self.return_stage = cfg.MODEL.RETURN_STAGES
        self.add_sep_seg = cfg.MODEL.BACKBONE.SEP_SEG

        if new_patch_size != self.patch_size:
            print('SELA-ViT patch size updated for tracking')
            self.patch_size = new_patch_size

        search_H, search_W = search_size
        template_H, template_W = template_size
        
        actual_stride = 16  
        search_patch_H, search_patch_W = search_H // actual_stride, search_W // actual_stride
        template_patch_H, template_patch_W = template_H // actual_stride, template_W // actual_stride

        self.pos_embed_z = nn.Parameter(torch.zeros(1, template_patch_H * template_patch_W, self.embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, search_patch_H * search_patch_W, self.embed_dim))

        # separate token and segment token
        if self.add_sep_seg:
            self.template_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.template_segment_pos_embed = trunc_normal_(self.template_segment_pos_embed, std=.02)
            self.search_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.search_segment_pos_embed = trunc_normal_(self.search_segment_pos_embed, std=.02)

        if self.return_inter:
            for i_layer in self.return_stage:
                if i_layer != 11:
                    norm_layer = partial(nn.LayerNorm, eps=1e-6)
                    layer = norm_layer(self.embed_dim)
                    layer_name = f'norm{i_layer}'
                    self.add_module(layer_name, layer)

    

    def forward_features(self, z, x):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        x = self.patch_embed(x)  
        z = self.patch_embed(z)

        lens_z = z.shape[1]
        lens_x = x.shape[1]

        z += self.pos_embed_z
        x += self.pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x = combine_tokens(z, x, mode=self.cat_mode)
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            output = blk(x, lens_z, lens_x)
            if isinstance(output, tuple) and len(output) == 2:
                x, _ = output
            else:
                x = output

        x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)

        aux_dict = {"attn": None}
        return self.norm(x), aux_dict



    def forward(self, z, x, **kwargs):
        """
        Joint feature extraction and relation modeling for the SELA-ViT backbone.
        Args:
            z (torch.Tensor): template feature, [B, C, H_z, W_z]
            x (torch.Tensor): search region feature, [B, C, H_x, W_x]

        Returns:
            x (torch.Tensor): merged template and search region feature, [B, L_z+L_x, C]
            attn : None
        """
        x, aux_dict = self.forward_features(z, x)
        return x, aux_dict