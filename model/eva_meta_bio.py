""" EVA

EVA from https://github.com/baaivision/EVA , paper: https://arxiv.org/abs/2211.07636

@article{EVA,
  title={EVA: Exploring the Limits of Masked Visual Representation Learning at Scale},
  author={Fang, Yuxin and Wang, Wen and Xie, Binhui and Sun, Quan and Wu, Ledell and Wang, Xinggang and Huang,
  Tiejun and Wang, Xinlong and Cao, Yue},
  journal={arXiv preprint arXiv:2211.07636},
  year={2022}
}

EVA-02: A Visual Representation for Neon Genesis - https://arxiv.org/abs/2303.11331
@article{EVA02,
  title={EVA-02: A Visual Representation for Neon Genesis},
  author={Fang, Yuxin and Sun, Quan and Wang, Xinggang and Huang, Tiejun and Wang, Xinlong and Cao, Yue},
  journal={arXiv preprint arXiv:2303.11331},
  year={2023}
}

This file contains EVA & EVA02 model implementations evolved from BEiT, additional models in vision_transformer.py.

Modifications by / Copyright 2023 Ross Wightman, original copyrights below
"""
# EVA models Copyright (c) 2022 BAAI-Vision
# EVA02 models Copyright (c) 2023 BAAI-Vision

import math
from typing import Callable, Optional, Tuple, Union
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import PatchEmbed, Mlp, GluMlp, SwiGLU, LayerNorm, DropPath, PatchDropout, RotaryEmbeddingCat, \
    apply_rot_embed_cat, apply_keep_indices_nlc, trunc_normal_, resample_patch_embed, resample_abs_pos_embed, \
    to_2tuple, use_fused_attn

from timm.models.helpers  import build_model_with_cfg
from timm.models.registry import generate_default_cfgs, register_model

from torchinfo import summary
from functools import partial

__all__ = ['Eva']


class EvaAttention(nn.Module):
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            attn_head_dim: Optional[int] = None,
            norm_layer: Optional[Callable] = None,
    ):
        """

        Args:
            dim:
            num_heads:
            qkv_bias:
            qkv_fused:
            attn_drop:
            proj_drop:
            attn_head_dim:
            norm_layer:
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        if qkv_fused:
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
            self.q_proj = self.k_proj = self.v_proj = None
            if qkv_bias:
                self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
                self.register_buffer('k_bias', torch.zeros(all_head_dim), persistent=False)
                self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
            else:
                self.q_bias = self.k_bias = self.v_bias = None
        else:
            self.q_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj = nn.Linear(dim, all_head_dim, bias=qkv_bias)
            self.qkv = None
            self.q_bias = self.k_bias = self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(all_head_dim) if norm_layer is not None else nn.Identity()
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            x,
            rope: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        B, N, C = x.shape

        if self.qkv is not None:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        else:
            q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)  # B, num_heads, N, C
            k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)

        if rope is not None:
            q = torch.cat([q[:, :, :1, :], apply_rot_embed_cat(q[:, :, 1:, :], rope)], 2).type_as(v)
            k = torch.cat([k[:, :, :1, :], apply_rot_embed_cat(k[:, :, 1:, :], rope)], 2).type_as(v)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            if attn_mask is not None:
                attn_mask = attn_mask.to(torch.bool)
                attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EvaBlock(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            scale_attn_inner: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            init_values: Optional[float] = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            attn_head_dim: Optional[int] = None,
    ):
        """

        Args:
            dim:
            num_heads:
            qkv_bias:
            qkv_fused:
            mlp_ratio:
            swiglu_mlp:
            scale_mlp:
            scale_attn_inner:
            proj_drop:
            attn_drop:
            drop_path:
            init_values:
            act_layer:
            norm_layer:
            attn_head_dim:
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = EvaAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_fused=qkv_fused,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_head_dim=attn_head_dim,
            norm_layer=norm_layer if scale_attn_inner else None,
        )
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        hidden_features = int(dim * mlp_ratio)
        if swiglu_mlp:
            if scale_mlp:
                # when norm in SwiGLU used, an impl with separate fc for gate & x is used
                self.mlp = SwiGLU(
                    in_features=dim,
                    hidden_features=hidden_features,
                    norm_layer=norm_layer if scale_mlp else None,
                    drop=proj_drop,
                )
            else:
                # w/o any extra norm, an impl with packed weights is used, matches existing GluMLP
                self.mlp = GluMlp(
                    in_features=dim,
                    hidden_features=hidden_features * 2,
                    norm_layer=norm_layer if scale_mlp else None,
                    act_layer=nn.SiLU,
                    gate_last=False,
                    drop=proj_drop,
                )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=hidden_features,
                act_layer=act_layer,
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
            )
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, rope: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None):
        if self.gamma_1 is None:
            x = x + self.drop_path1(self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))
            x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class EvaBlockDimChange(nn.Module):

    def __init__(
            self,
            dim: int,
            out_features:int,
            num_heads: int,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            scale_attn_inner: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            init_values: Optional[float] = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            attn_head_dim: Optional[int] = None,
    ):
        """

        Args:
            dim:
            num_heads:
            qkv_bias:
            qkv_fused:
            mlp_ratio:
            swiglu_mlp:
            scale_mlp:
            scale_attn_inner:
            proj_drop:
            attn_drop:
            drop_path:
            init_values:
            act_layer:
            norm_layer:
            attn_head_dim:
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = EvaAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_fused=qkv_fused,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_head_dim=attn_head_dim,
            norm_layer=norm_layer if scale_attn_inner else None,
        )
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim)) if init_values is not None else None
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        hidden_features = int(dim * mlp_ratio)
        if swiglu_mlp:
            if scale_mlp:
                # when norm in SwiGLU used, an impl with separate fc for gate & x is used
                self.mlp = SwiGLU(
                    in_features=dim,
                    hidden_features=hidden_features,
                    out_features=out_features,
                    norm_layer=norm_layer if scale_mlp else None,
                    drop=proj_drop,
                )
            else:
                # w/o any extra norm, an impl with packed weights is used, matches existing GluMLP
                self.mlp = GluMlp(
                    in_features=dim,
                    hidden_features=hidden_features * 2,
                    out_features=out_features,
                    norm_layer=norm_layer if scale_mlp else None,
                    act_layer=nn.SiLU,
                    gate_last=False,
                    drop=proj_drop,
                )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=hidden_features,
                out_features=out_features,
                act_layer=act_layer,
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
            )
        self.gamma_2 = nn.Parameter(init_values * torch.ones(out_features)) if init_values is not None else None
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, rope: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None):
        if self.gamma_1 is None:
            x = x + self.drop_path1(self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))
            x =self.drop_path2(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask))
            x =  self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class EvaBlockPostNorm(nn.Module):
    """ EVA block w/ post-norm and support for swiglu, MLP norm scale, ROPE. """
    def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            scale_attn_inner: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            init_values: Optional[float] = None,  # ignore for post-norm
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
            attn_head_dim: Optional[int] = None,
    ):
        """

        Args:
            dim:
            num_heads:
            qkv_bias:
            qkv_fused:
            mlp_ratio:
            swiglu_mlp:
            scale_mlp:
            scale_attn_inner:
            proj_drop:
            attn_drop:
            drop_path:
            init_values:
            act_layer:
            norm_layer:
            attn_head_dim:
        """
        super().__init__()
        self.attn = EvaAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_fused=qkv_fused,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_head_dim=attn_head_dim,
            norm_layer=norm_layer if scale_attn_inner else None,
        )
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        hidden_features = int(dim * mlp_ratio)
        if swiglu_mlp:
            if scale_mlp:
                # when norm in SwiGLU used, an impl with separate fc for gate & x is used
                self.mlp = SwiGLU(
                    in_features=dim,
                    hidden_features=hidden_features,
                    norm_layer=norm_layer if scale_mlp else None,
                    drop=proj_drop,
                )
            else:
                # w/o any extra norm, an impl with packed fc1 weights is used, matches existing GluMLP
                self.mlp = GluMlp(
                    in_features=dim,
                    hidden_features=hidden_features * 2,
                    norm_layer=norm_layer if scale_mlp else None,
                    act_layer=nn.SiLU,
                    gate_last=False,
                    drop=proj_drop,
                )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=hidden_features,
                act_layer=act_layer,
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
            )
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, rope: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.drop_path1(self.norm1(self.attn(x, rope=rope, attn_mask=attn_mask)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(torch.nn.Module):
    def __init__(self, alpha):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class GradientReversalClassifier(torch.nn.Module):
    def __init__(self, in_num,class_num,alpha=1.0,weight=None):
        super(GradientReversalClassifier, self).__init__()
        self.alpha=alpha
        self.grl = GradientReversalLayer(self.alpha)
        self.head_fc = nn.Linear(in_num, class_num, bias=False)
        self.apply(self._init_weights)
        if weight is not None:
            self.head_fc.weight.data = weight.clone()
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        x_reverse=self.grl(x)
        class_score=self.head_fc(x_reverse)
        return class_score

        
    
class GradientReversalMLP(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            alpha=1.0,
    ):
        super().__init__()
        self.alpha=alpha
        self.grl = GradientReversalLayer(self.alpha)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.grl(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x             
                  

class Eva(nn.Module):
    """ Eva Vision Transformer w/ Abs & Rotary Pos Embed

    This class implements the EVA and EVA02 models that were based on the BEiT ViT variant
      * EVA - abs pos embed, global avg pool
      * EVA02 - abs + rope pos embed, global avg pool, SwiGLU, scale Norm in MLP (ala normformer)
    """

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            embed_dim: int = 768,
            clip_feat_dim: int = 1024,
            last_layer: str = 'clipFc',
            global_pool: str = 'avg',
            depth: int = 12,
            num_heads: int = 12,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            scale_attn_inner: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            norm_layer: Callable = LayerNorm,
            init_values: Optional[float] = None,
            class_token: bool = True,
            use_abs_pos_emb: bool = True,
            use_rot_pos_emb: bool = False,
            use_post_norm: bool = False,
            ref_feat_shape: Optional[Union[Tuple[int, int], int]] = None,
            head_init_scale: float = 0.001,
            cloth: int = 0,
            camera_xishu: int = 3,
            camera_num: int = 0,
            add_nonBio: bool = True,
            nonBio_num: int = 3,
            reverse_bio: bool = True,
            CLIP_LOSS_TYPE='constant',
            subspace_dim: int = 0,
    ):
        """

        Args:
            img_size:
            patch_size:
            in_chans:
            num_classes:
            global_pool:
            embed_dim:
            depth:
            num_heads:
            qkv_bias:
            qkv_fused:
            mlp_ratio:
            swiglu_mlp:
            scale_mlp:
            scale_attn_inner:
            drop_rate:
            pos_drop_rate:
            proj_drop_rate:
            attn_drop_rate:
            drop_path_rate:
            norm_layer:
            init_values:
            class_token:
            use_abs_pos_emb:
            use_rot_pos_emb:
            use_post_norm:
            ref_feat_shape:
            head_init_scale:
        """
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.grad_checkpointing = False
        self.depth=depth
        self.last_layer=last_layer
        self.add_nonBio=add_nonBio
        self.nonBio_num=nonBio_num
        self.reverse_bio=reverse_bio
        self.CLIP_LOSS_TYPE=CLIP_LOSS_TYPE
        
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_prefix_tokens, embed_dim)) if use_abs_pos_emb else None
        #self.cloth_embed = nn.Parameter(torch.zeros(cloth, 1, embed_dim))
        self.camera_xishu = camera_xishu
        if self.camera_xishu > 0:
            self.camera_embed = nn.Parameter(torch.zeros(camera_num, 1, embed_dim))
            trunc_normal_(self.camera_embed, std=.02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
                return_indices=True,
            )
        else:
            self.patch_drop = None

        if use_rot_pos_emb:
            ref_feat_shape = to_2tuple(ref_feat_shape) if ref_feat_shape is not None else None
            self.rope = RotaryEmbeddingCat(
                embed_dim // num_heads,
                in_pixels=False,
                feat_shape=self.patch_embed.grid_size,
                ref_feat_shape=ref_feat_shape,
            )
        else:
            self.rope = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        block_fn = EvaBlockPostNorm if use_post_norm else EvaBlock
        
        if  self.last_layer in ['clipFc','clipMLP']:
            depth-=1
        self.blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qkv_fused=qkv_fused,
                mlp_ratio=mlp_ratio,
                swiglu_mlp=swiglu_mlp,
                scale_mlp=scale_mlp,
                scale_attn_inner=scale_attn_inner,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
            )
            for i in range(depth)])

        use_fc_norm = self.global_pool == 'avg'
        self.norm = nn.Identity() if use_fc_norm else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.clip_feat_dim=clip_feat_dim
        
       
        self.head = nn.Linear(embed_dim, num_classes)
       
        
        if self.last_layer=='fc':
            # self.head_bio = nn.Linear(embed_dim, clip_feat_dim) 
            # self.head_nonbio = nn.Linear(embed_dim, clip_feat_dim)
            self.weight_bio = torch.nn.Parameter(torch.Tensor(clip_feat_dim,embed_dim))
            init_weight = torch.eye(clip_feat_dim,embed_dim)
            self.weight_bio.data=init_weight
            #nn.init.kaiming_uniform_(self.weight_bio, a=math.sqrt(5))
            # self.batch_norm_bio = nn.BatchNorm1d(clip_feat_dim) 
            # self.act=nn.GELU()
            #
            if self.add_nonBio:
                self.weight_nonbio = torch.nn.Parameter(torch.Tensor(clip_feat_dim,embed_dim))
                self.weight_nonbio.data=init_weight.flip(dims=[1])
              
        elif  self.last_layer in ['clipFc','clipMLP']:
            del self.norm
            self.head_bio=block_fn(dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qkv_fused=qkv_fused,
                mlp_ratio=mlp_ratio,
                swiglu_mlp=swiglu_mlp,
                scale_mlp=scale_mlp,
                scale_attn_inner=scale_attn_inner,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[-1],
                norm_layer=norm_layer,
                init_values=init_values,)
            self.norm_bio = nn.Identity() if use_fc_norm else norm_layer(embed_dim)
            
            if self.CLIP_LOSS_TYPE=='constant':
                self.logit_scale_bio = 1
                self.logit_bias_bio = 0
            elif self.CLIP_LOSS_TYPE=='contrastive':
                self.logit_scale_bio = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
                self.logit_bias_bio = 0
            elif self.CLIP_LOSS_TYPE=='sigmoid':
                self.logit_scale_bio = nn.Parameter(torch.randn(1))
                self.logit_bias_bio = nn.Parameter(torch.randn(1))
            
            if self.last_layer=='clipFc':
                # del self.head
                # self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
                self.head_clip_bio = nn.Linear(embed_dim, clip_feat_dim) if clip_feat_dim > 0 else nn.Identity()
            elif self.last_layer=='clipMLP':
                self.head_clip_bio = Mlp(
                        in_features=embed_dim,
                        hidden_features=subspace_dim,
                        out_features=clip_feat_dim,
                    )
                
                
            if self.add_nonBio:
                self.head_clip_nonbio = nn.ModuleList([nn.Linear(embed_dim, clip_feat_dim) for x in range(self.nonBio_num)])
                
            if self.reverse_bio:
                if self.last_layer=='clipFc':
                    self.head_clip_bioReverse=nn.ModuleList([GradientReversalClassifier(in_num=embed_dim,class_num=clip_feat_dim) 
                                                            for x in range(self.nonBio_num)])
                elif self.last_layer=='clipMLP':
                    self.head_clip_bioReverse=nn.ModuleList([GradientReversalMLP(in_features=embed_dim,hidden_features=subspace_dim,
                                                                                out_features=clip_feat_dim,alpha=1.0)
                                                            for x in range(self.nonBio_num)])
                if self.CLIP_LOSS_TYPE=='constant':
                    self.logit_scale_nonbios = [1 for x in range(self.nonBio_num)]
                    self.logit_bias_nonbios=0
                elif self.CLIP_LOSS_TYPE=='contrastive':
                    self.logit_scale_nonbios = [nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) for x in range(self.nonBio_num)]
                    self.logit_bias_nonbios=0
                elif self.CLIP_LOSS_TYPE=='sigmoid':
                    self.logit_scale_nonbios = [nn.Parameter(torch.randn(1)) for x in range(self.nonBio_num)]
                    self.logit_bias_nonbios = [nn.Parameter(torch.randn(1))  for x in range(self.nonBio_num)]
                        
        self.apply(self._init_weights)

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        
        trunc_normal_(self.cls_token, std=.02)

        self.fix_init_weight()
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
            self.head.weight.data.mul_(head_init_scale)
            if self.head.bias is not None:
                nn.init.zeros_(self.head.bias)
            #self.head.bias.data.mul_(head_init_scale)
            
    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = {'pos_embed', 'cls_token'}
        return nwd

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))],
        )
        return matcher

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x,camera_id):
        x = self.patch_embed(x)

        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        # apply abs position embedding
        if self.pos_embed is not None:
            x = x + self.pos_embed + self.camera_xishu * self.camera_embed[camera_id]
        x = self.pos_drop(x)

        # obtain shared rotary position embedding and apply patch dropout
        rot_pos_embed = self.rope.get_embed() if self.rope is not None else None
        if self.patch_drop is not None:
            x, keep_indices = self.patch_drop(x)
            if rot_pos_embed is not None and keep_indices is not None:
                rot_pos_embed = apply_keep_indices_nlc(x, rot_pos_embed, keep_indices)

        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, rope=rot_pos_embed)
            else:
                x = blk(x, rope=rot_pos_embed)       
        if  self.last_layer in ['clipFc','clipMLP']:
            x_bio=self.head_bio(x,rope=rot_pos_embed)
            x_bio = self.norm_bio(x_bio)
            if self.add_nonBio:
                x_nonbio=self.head_nonbio(x,rope=rot_pos_embed)
                x_nonbio = self.norm_nonbio(x_nonbio)
                return [x_bio,x_nonbio]
            else:
                
                return [x_bio]
        else:
            x = self.norm(x)
            return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)

        return x

    def foward_score(self,x,norm=True):
        if norm:
            x=F.normalize(x,dim=1)
            self.head.weight.data=F.normalize(self.head.weight.data,dim=1)
        cls_score = self.head(x)
        return cls_score
            
    def forward(self, x,cam_label=3,norm=False):
        x = self.forward_features(x,cam_label)
        if self.add_nonBio:
            x_bio,x_nonbio=x
            feat_nonbio = self.forward_head(x_nonbio, pre_logits=True)
        else:
            x_bio=x[0]
        feat_bio = self.forward_head(x_bio, pre_logits=True)
       
            
        if not self.training:
            #if False:
            return feat_bio
        else:
            feat_ouput=[feat_bio]
            cls_score = self.head(feat_bio)
            
            score_output={'cls_score':cls_score}
            
            if self.reverse_bio:
                clip_bio_reverse_score=[layer(feat_bio) for layer in self.head_clip_bioReverse]
                clip_bio_reverse_score = [t.unsqueeze(1) for t in clip_bio_reverse_score]
                score_output['clip_bio_reverse_score']=torch.cat(clip_bio_reverse_score, dim=1)
                score_output['clip_nonbio_scale']=self.logit_scale_nonbios
                score_output['clip_nonbio_bias']=self.logit_bias_nonbios
                
                
            if self.last_layer in ['clipFc','clipMLP']:
                clip_bio_score=self.head_clip_bio(feat_bio)
                score_output['clip_bio_score']=clip_bio_score
                score_output['clip_bio_scale']=self.logit_scale_bio
                score_output['clip_bio_bias']=self.logit_bias_bio

                return score_output,feat_ouput
            
            elif self.last_layer=='fc':
                if self.add_nonBio:
                    return score_output, [feat_bio,feat_nonbio,self.weight_bio,self.weight_nonbio]
                else:
                    return score_output, [feat_bio,None,self.weight_bio,None]
            return score_output, feat_ouput
        
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

def checkpoint_filter_fn(
        state_dict,
        model,
        interpolation='bicubic',
        antialias=True,
):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    state_dict = state_dict.get('model_ema', state_dict)
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('module', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    # prefix for loading OpenCLIP compatible weights
    if 'visual.trunk.pos_embed' in state_dict:
        prefix = 'visual.trunk.'
    elif 'visual.pos_embed' in state_dict:
        prefix = 'visual.'
    else:
        prefix = ''
    mim_weights = prefix + 'mask_token' in state_dict
    no_qkv = prefix + 'blocks.0.attn.q_proj.weight' in state_dict

    len_prefix = len(prefix)
    for k, v in state_dict.items():
        if prefix:
            if k.startswith(prefix):
                k = k[len_prefix:]
            else:
                continue
      
            
        if 'rope' in k:
            # fixed embedding no need to load buffer from checkpoint
            continue
        
        if 'patch_embed.proj.weight' in k:
            _, _, H, W = model.patch_embed.proj.weight.shape
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(
                    v,
                    (H, W),
                    interpolation=interpolation,
                    antialias=antialias,
                    verbose=True,
                )
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            # To resize pos embedding when using model at different size from pretrained weights
            num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
            v = resample_abs_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )
        
        k = k.replace('mlp.ffn_ln', 'mlp.norm')
        k = k.replace('attn.inner_attn_ln', 'attn.norm')
        k = k.replace('mlp.w12', 'mlp.fc1')
        k = k.replace('mlp.w1', 'mlp.fc1_g')
        k = k.replace('mlp.w2', 'mlp.fc1_x')
        k = k.replace('mlp.w3', 'mlp.fc2')
        if no_qkv:
            k = k.replace('q_bias', 'q_proj.bias')
            k = k.replace('v_bias', 'v_proj.bias')


        if mim_weights and k in ('mask_token', 'lm_head.weight', 'lm_head.bias', 'norm.weight', 'norm.bias'):
            if k == 'norm.weight' or k == 'norm.bias':
                # try moving norm -> fc norm on fine-tune, probably a better starting point than new init
                k = k.replace('norm', 'fc_norm')
            else:
                # skip pretrain mask token & head weights
                continue
        
        if model.last_layer in ['clipFc','clipMLP'] and k.find(str(model.depth-1))>0:
                block_name=f'blocks.{model.depth-1}'
                
                k_bio=k.replace(block_name,'head_bio')
                out_dict[k_bio] = v
                
                if model.add_nonBio:
                    k_nonbio=k.replace(block_name,'head_nonbio')
                    out_dict[k_nonbio] = v
                
                
        else:
            out_dict[k] = v

    return out_dict


def _create_eva(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Eva models.')

    model = build_model_with_cfg(
        Eva, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)
    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': OPENAI_CLIP_MEAN, 'std': OPENAI_CLIP_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        'license': 'mit', **kwargs
    }


default_cfgs = generate_default_cfgs({
    'eva02_large_patch14_224.mim_in22k': _cfg(
        #hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/pt/eva02_L_pt_in21k_p14.pt',
        hf_hub_id='timm/',
        num_classes=0,
    ),
    'eva02_large_patch14_224.mim_m38m': _cfg(
        #hf_hub_id='Yuxin-CV/EVA-02', hf_hub_filename='eva02/pt/eva02_L_pt_m38m_p14.pt',
        hf_hub_id='timm/',
        num_classes=0,
    ),
    'eva02_large_patch14_clip_224.merged2b': _cfg(
        # hf_hub_id='QuanSun/EVA-CLIP', hf_hub_filename='EVA02_CLIP_L_psz14_s4B.pt',
        hf_hub_id='timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k',  # float16 weights
        hf_hub_filename='open_clip_pytorch_model.bin',
        num_classes=768,
    ),
})



@register_model
def eva02_large_patch14_clip_224_bio(pretrained=False,config={}, **kwargs) -> Eva:
    """ A EVA-CLIP specific variant that adds additional attn scale layernorm to eva02_large """
    #,,
    
    model_args = dict(
        img_size=224,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4 * 2 / 3,
        qkv_fused=False,
        swiglu_mlp=True,
        scale_mlp=True,
        scale_attn_inner=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),  # 224/14
        global_pool=kwargs.pop('global_pool', 'token'),
        camera_xishu=config.MODEL.CAMERA_XISHU,
        last_layer=config.MODEL.LAST_LAYER,
        add_nonBio=config.MODEL.NONBIO_HEAD,
        nonBio_num=len(config.DATA.NOBIO_INDEX),
        reverse_bio='clipBioReverse' in config.MODEL.LOSS_TYPE,
        clip_feat_dim=config.MODEL.CLIP_DIM,
        CLIP_LOSS_TYPE=config.MODEL.CLIP_LOSS_TYPE,
        subspace_dim=config.MODEL.SUBSPACE_DIM,
    )
    model = _create_eva('eva02_large_patch14_clip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


if __name__ == "__main__":
    x = torch.randn([2, 3, 224, 224])
    cloth_id = torch.tensor([2, 3])
    model = eva02_large_patch14_clip_224_bio()
    # import ipdb;ipdb.set_trace()
    model.train()
    output = model(x, cloth_id)
    #print(output[0].shape)
    # print(summary(model, input_size=((8, 3, 224, 224),(3,7))))

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('/home/xi860799/code/MADE/logs/ltcc/model/bio')
    writer.add_graph(model, [x,cloth_id])
    writer.close()