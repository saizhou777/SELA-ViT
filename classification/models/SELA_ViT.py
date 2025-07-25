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

    def forward(self, x):
        B, N, C = x.shape
        H = int(N ** 0.5)
        assert H * H == N, f"{N} not a square number"
        W = H

        # Depthwise convolution.
        feat = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(feat) + feat
        x = x.flatten(2).transpose(1, 2)
        return x



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

    def forward(self, x):
        B, N, C = x.shape
        H = int(N ** 0.5)
        assert H * H == N, f"{N} not a square number"
        W = H
        expanded_dim = int(self.expansion_ratio * C)

        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)  # make torchscript happy (cannot use tensor as tuple) # B H N C/H
        q = self.q(x)

        expanded_q = self._expand_heads(q) ## B, N, C*r   # B, H*r, N, C/H
        
        #lepe = self.lepe(expanded_q.permute(0, 2, 1).reshape(B, expanded_dim, H, W)).reshape(B, expanded_dim, N).permute(0, 2, 1)
        lepe = self.lepe(expanded_q.transpose(2, 3).reshape(B, expanded_dim, H, W)).reshape(B, expanded_dim, N).transpose(1, 2)


        k_softmax = k.softmax(dim=3)
        k_softmax_T_dot_v = k_softmax.transpose(-2, -1) @ v  # (B, H, C/H, C/H)

        expanded_k_softmax_T_dot_v = self._expand_heads(k_softmax_T_dot_v.permute(0, 2, 1, 3).reshape(B, C // self.num_heads, C))  
        # B, H*r, C/H, C/H   --->   B, C/H, C*r

        attn = expanded_q @ expanded_k_softmax_T_dot_v  # B, H, N, C/H

        x = (self.scale * attn).transpose(1, 2).reshape(B, N, expanded_dim)

        x = self.proj(x + lepe)
        x = self.proj_drop(x)
        return x


    def _expand_heads(self, tensor):
    
        B, *dim, C =tensor.shape   
        r= self.expansion_ratio
        r_int = math.ceil(r)

        expanded_fea = [tensor]

        for i in range(1, r_int):
            move_dim =  int(C //self.num_heads * (r_int - i) // r_int)  
        
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


    def forward(self, x):
        #         import pdb
        #         pdb.set_trace()
        #x = x + self.drop_path(self.attn(self.norm1(x)))

        if self.use_cpe:
            x = self.cpe(x)

        feat = self.attn(self.norm1(x))
        x = x + self.drop_path(feat)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # x = x + self.attn(self.norm1(x))
        # x = x + self.mlp(self.norm2(x))
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
            nn.BatchNorm2d(embed_dim) #v1
            )


    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)

        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC

        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        #self.num_tokens = 1 if distilled else 0
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        #num_patches = self.patch_embed.num_patches

        #self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        #self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        #self.sim_loss_weight = nn.Parameter(torch.tensor(0.1))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, block_num=i)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size: #and not distilled:
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
        #if distilled:
            #self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        #trunc_normal_(self.pos_embed, std=.02)
        #if self.dist_token is not None:
            #trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            #trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}   #'cls_token', , 'dist_token'

    def get_classifier(self):
        return self.head
        '''if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist'''

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        #if self.num_tokens == 1:
            #self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()


    def forward_features(self, x):
        proj_features = [] 

        x = self.patch_embed(x)  # [B, N, D]
        
        # Pass through all blocks and accumulate the attention similarity loss
        for blk in self.blocks:
            x, proj_x = blk(x)
            proj_features.append(proj_x)
        
        x = self.norm(x)
        x = x.mean(dim=1)  # [B, D] by averaging along the sequence length (N)
        return x, proj_features  # Return both features and total attention similarity loss


    def compute_block_similarity_loss(self, proj_features):

        total_loss = 0
        num_blocks = len(proj_features)
        
        for i in range(num_blocks - 1):

            feat1 = proj_features[i].flatten(start_dim=1)  # [B, H*r*Ch*Ch]
            feat2 = proj_features[i+1].flatten(start_dim=1)
            
            cos_sim = F.cosine_similarity(feat1, feat2)
            total_loss += (cos_sim).mean()  

        return total_loss / (num_blocks - 1) #* torch.sigmoid(self.sim_loss_weight)


    def forward(self, x):
        features, attn_maps = self.forward_features(x)
        x = self.head(features)
        
        sim_loss = self.compute_block_similarity_loss(attn_maps)
        return x, sim_loss


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
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
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)



@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=11, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=nn.LayerNorm, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model



@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=11, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=nn.LayerNorm, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


if __name__ == "__main__":
    device = 'cuda'

    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=11, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=nn.LayerNorm).to(device)
    model.eval()
    inputs = torch.randn(1, 3, 224, 224).to(device)
    model(inputs)

    from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis, flop_count_table

    flops = FlopCountAnalysis(model, inputs)
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    acts = ActivationCountAnalysis(model, inputs)

    print(f"total flops : {flops.total()}")
    print(f"total activations: {acts.total()}")
    print(f"number of parameter: {param}")

    print(flop_count_table(flops, max_depth=1))