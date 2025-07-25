# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from torch import einsum
import math
from torch.nn import functional as F

from lib.models.ostrack.sela_backbone import SELABackbone


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvPosEnc(nn.Module):
    def __init__(self, dim, k=3):
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)

    def forward(self, x, lens_z=None, lens_x=None):
        B, N, C = x.shape
        
        if lens_z is not None and lens_x is not None:

            z = x[:, :lens_z, :]  
            search = x[:, lens_z:lens_z+lens_x, :]  
            
            z_processed = self._process_region(z, lens_z)
            search_processed = self._process_region(search, lens_x)
            
            x = torch.cat([z_processed, search_processed], dim=1)
            return x
        else:
            H = int(N ** 0.5)
            if H * H == N:
                W = H
                feat = x.transpose(1, 2).view(B, C, H, W)
                x = self.proj(feat) + feat
                x = x.flatten(2).transpose(1, 2)
                return x
            else:
                return x
    
    def _process_region(self, region, region_len):
        B, N, C = region.shape
        H = int(region_len ** 0.5)
        
        if H * H == region_len:
            W = H
            feat = region.transpose(1, 2).view(B, C, H, W)
            processed = self.proj(feat) + feat
            processed = processed.flatten(2).transpose(1, 2)
            return processed
        else:
            return region


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.expansion_ratio = 2    
        expanded_dim = int(dim * self.expansion_ratio)

        self.proj = nn.Linear(expanded_dim, dim)        
        self.proj_drop = nn.Dropout(proj_drop)

        self.lepe = nn.Conv2d(expanded_dim, expanded_dim, 3, 1, 1, groups=expanded_dim)

    def forward(self, x, lens_z=None, lens_x=None):
        B, N, C = x.shape
        expanded_dim = int(self.expansion_ratio * C)

        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)  # make torchscript happy (cannot use tensor as tuple) # B H N C/H
        q = self.q(x)

        expanded_q = self._expand_heads(q) ## B, N, C*r   # B, H*r, N, C/H
        
        if lens_z is not None and lens_x is not None:
            lepe = self._compute_lepe_separate(expanded_q, lens_z, lens_x, expanded_dim)
        else:
            H = int(N ** 0.5)
            if H * H == N:
                W = H
                lepe = self.lepe(expanded_q.transpose(2, 3).reshape(B, expanded_dim, H, W)).reshape(B, expanded_dim, N).transpose(1, 2)
            else:
                lepe = torch.zeros_like(expanded_q.transpose(1, 2))

        k_softmax = k.softmax(dim=3)
        k_softmax_T_dot_v = k_softmax.transpose(-2, -1) @ v  # (B, H, C/H, C/H)

        expanded_k_softmax_T_dot_v = self._expand_heads(k_softmax_T_dot_v.permute(0, 2, 1, 3).reshape(B, C // self.num_heads, C))  
        # B, H*r, C/H, C/H   --->   B, C/H, C*r

        attn = expanded_q @ expanded_k_softmax_T_dot_v  # B, H, N, C/H

        x = (self.scale * attn).transpose(1, 2).reshape(B, N, expanded_dim)

        x = self.proj(x + lepe)
        x = self.proj_drop(x)
        return x

    def _compute_lepe_separate(self, expanded_q, lens_z, lens_x, expanded_dim):
        B = expanded_q.shape[0]
        
        z_expanded_q = expanded_q[:, :, :lens_z, :]  # B, H*r, lens_z, C/H
        search_expanded_q = expanded_q[:, :, lens_z:lens_z+lens_x, :]  # B, H*r, lens_x, C/H
        
        z_lepe = self._compute_region_lepe(z_expanded_q, lens_z, expanded_dim)
        search_lepe = self._compute_region_lepe(search_expanded_q, lens_x, expanded_dim)
        
        lepe = torch.cat([z_lepe, search_lepe], dim=1)  # B, lens_z+lens_x, expanded_dim
        return lepe

    def _compute_region_lepe(self, region_expanded_q, region_len, expanded_dim):
        B, num_heads_expanded, N, head_dim = region_expanded_q.shape
        H = int(region_len ** 0.5)
        
        if H * H == region_len:
            W = H
            region_for_conv = region_expanded_q.transpose(2, 3).reshape(B, expanded_dim, H, W)
            region_lepe = self.lepe(region_for_conv).reshape(B, expanded_dim, N).transpose(1, 2)
            return region_lepe
        else:
            return torch.zeros(B, N, expanded_dim, device=region_expanded_q.device, dtype=region_expanded_q.dtype)

    def _expand_heads(self, tensor):
        B, *dim, C = tensor.shape   
        r = self.expansion_ratio
        r_int = math.ceil(r)

        expanded_fea = [tensor]

        for i in range(1, r_int):
            move_dim = int(C // self.num_heads * (r_int - i) // r_int)  
            rolled_tensor = torch.roll(tensor, shifts=-move_dim, dims=2)   
            expanded_fea.append(rolled_tensor)
        
        expanded_fea = torch.cat(expanded_fea, dim=-1)
        expanded_fea = expanded_fea[:, :, :int(C*r)]
    
        return expanded_fea.reshape(B, *dim, int(self.num_heads * r), C // self.num_heads).permute(0, 2, 1, 3)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, block_num=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.use_cpe = (0 < block_num < 6)
        self.cpe = ConvPosEnc(dim=dim, k=3)

    def forward(self, x, lens_z=None, lens_x=None):
        if self.use_cpe:
            x = self.cpe(x, lens_z, lens_x)

        feat = self.attn(self.norm1(x), lens_z, lens_x)
        x = x + self.drop_path(feat)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, feat


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim//8, 3, 2, 1),
            nn.BatchNorm2d(embed_dim//8),
            nn.GELU(),
            nn.Conv2d(embed_dim // 8, embed_dim // 4, 3, 2, 1),
            nn.BatchNorm2d(embed_dim//4),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim//2, 3, 2, 1),
            nn.BatchNorm2d(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim, 3, 2, 1),
            nn.BatchNorm2d(embed_dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return x


class VisionTransformer(SELABackbone):
    """ Vision Transformer with SELA improvements
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, block_num=i)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        if mode.startswith('jax'):
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = VisionTransformer(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
            print('Load pretrained model from: ' + pretrained)

    return model


def sela_tiny_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=192, depth=11, num_heads=3, **kwargs)
    model = _create_vision_transformer('sela_tiny_patch16_224_in1k', pretrained=pretrained, **model_kwargs)
    return model