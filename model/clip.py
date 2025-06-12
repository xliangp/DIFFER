from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import hashlib
import os
import urllib
import warnings
from packaging import version
from typing import Union, List

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

from timm.layers import trunc_normal_
    
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
    
_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ClipViT-B32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ClipViT-B16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ClipViT-L14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ClipViT-L14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
            
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)

class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, 
                output_dim, 
                heads, 
                input_resolution=224, 
                width=64,
                num_classes: int = 77,
                clip_feat_dim: int = 1024,
                camera_num: int = 0,
                cloth: int = 0,
                cloth_xishu: int = 3,
                cameraNum: int = 0,
                last_layer: str = 'clipFc_clsFc',
                bioClip=False,
                add_nonBio: bool = False,
                nonBio_num: int = 3,
                reverse_bio: bool = False,
                ):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        
        self.num_classes = num_classes
        self.add_nonBio=add_nonBio
        self.nonBio_num=nonBio_num
        self.reverse_bio=reverse_bio
        self.clip_feat_dim=clip_feat_dim
        self.bioClip=bioClip

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        if self.bioClip:
            self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, clip_feat_dim)
        self.pool = nn.AvgPool2d(kernel_size=(input_resolution // 32,input_resolution // 32))

        self.bottleneck = nn.BatchNorm1d(output_dim)
        
        #x = nn.functional.avg_pool2d(x, x.shape[2:4]).view(x.shape[0], -1) 
        
        #self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.head = nn.Linear(output_dim, num_classes) if num_classes > 0 else nn.Identity()
            
        self.initialize_parameters()

    def initialize_parameters(self):
        head_init_scale = 0.001
        if isinstance(self.head, nn.Linear):
            nn.init.normal_(self.head.weight, std=.02)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)
            
        
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
            
        if self.bioClip:
            std = self.attnpool.c_proj.in_features ** -0.5
            nn.init.normal_(self.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.attnpool.c_proj.weight, std=std)

        for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward_features(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
    
    def forward_head(self, x):
        gm_pool=True
        if gm_pool:
            b, c, h, w = x.shape
            x = x.view(b, c, -1)
            p = 3.0
            x = (torch.mean(x**p, dim=-1) + 1e-12)**(1/p)
        else:
            x=self.pool(x)
            x=x.view(x.shape[0], -1) 
            #x = nn.functional.avg_pool2d(x, x.shape[2:4]).view(x.shape[0], -1) 
        if self.training:
            x = self.bottleneck(x)
        # x = self.fc_norm(x)
        # x = self.head_drop(x)
        #x = x / x.norm(dim=1, keepdim=True)
        
        
        return x

    def forward(self, x,camera_id):
        x = self.forward_features(x)
        feat_bio = self.forward_head(x)
        if not self.training:
            return feat_bio
        else:
            feat_ouput=[feat_bio,None]
            
            #self.head.weight.data = F.normalize(self.head.weight.data, p=2, dim=1)
            cls_score = self.head(feat_bio)
            score_output={'cls_score':cls_score}
            if self.bioClip:
                x_clip = self.attnpool(x)
                score_output['clip_bio_score']=x_clip
           
            if self.reverse_bio:
                # clip_bio_reverse=self.grl(feat_bio)
                # cip_bio_reverse_score=self.head_clip_bioReverse(clip_bio_reverse)
                #clip_bio_reverse_score=[self.head_clip_bioReverse[x](feat_bio) for x in range(self.nonBio_num)]
                clip_bio_reverse_score=[layer(feat_bio) for layer in self.head_clip_bioReverse]
                #score_output['clip_bio_reverse_score']=clip_bio_reverse_score
                clip_bio_reverse_score = [t.unsqueeze(1) for t in clip_bio_reverse_score]
                score_output['clip_bio_reverse_score']=torch.cat(clip_bio_reverse_score, dim=1)
              
                
            return score_output, feat_ouput
    


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None,drop=0., attn_drop=0.,
                 drop_path=0.,):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=attn_drop)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None,drop=0., attn_drop=0.,
                 drop_path=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask,drop, attn_drop,drop_path[i]) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, config:dict, input_resolution: tuple, patch_size: int, width: int, layers: int, heads: int,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., camera_xishu=0.0,
                 clip_output_dim: int=-1,num_classes: int=-1,head_init_scale: float = 0.001,
                 camera_num: int = 0, ):
        super().__init__()
        self.input_resolution = input_resolution
        self.width = width
        
        self.camera_xishu = camera_xishu
        #self.clip_output_dim = clip_output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        num_patches = (input_resolution[0] // patch_size) * (input_resolution[1] // patch_size)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(num_patches + 1, width))
        self.ln_pre = LayerNorm(width)
        
        if self.camera_xishu>0.1:
            self.camera_embedding =  nn.Parameter(torch.zeros(camera_num, 1, width))
            trunc_normal_(self.camera_embedding, std=.02)


        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]  # stochastic depth decay rule

        self.transformer = Transformer(width, layers, heads,drop=drop_rate, attn_drop=attn_drop_rate,drop_path=dpr)

        self.ln_post = LayerNorm(width)
        
        #bottleNeck=False
        # if bottleNeck:
        #     self.bottleneck = nn.BatchNorm1d(width)
        #     self.bottleneck.bias.requires_grad_(False)
        #     self.bottleneck.apply(weights_init_kaiming)
        # else:
        #     self.bottleneck=nn.Identity()

        # if clip_output_dim != -1:
        #     self.proj = nn.Parameter(scale * torch.randn(width, clip_output_dim))
        # if num_classes != -1:
        #     self.head = nn.Linear(width, num_classes)
            
        #     if isinstance(self.head, nn.Linear):
        #         trunc_normal_(self.head.weight, std=.02)
        #         self.head.weight.data.mul_(head_init_scale)
        #         self.head.bias.data.mul_(head_init_scale)
        


    def forward(self, x: torch.Tensor,cam_label):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        if self.camera_xishu>0.1:
            x = x + self.camera_xishu*self.camera_embedding[cam_label]
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        # if self.clip_output_dim!= -1:
        #     x = x @ self.proj
            
        x = x / x.norm(dim=1, keepdim=True)
            
        # if self.training:
        #     x_bn = self.bottleneck(x)
        #     cls_score = self.head(x_bn)
        #     score_output={'cls_score':cls_score}
        #     feat_output=[x,None]
        #     return score_output, feat_output

        return x




def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    

def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def ClipRN50(pretrained=False,config={}, **kwargs):
    
    name="RN50"
    if name in _MODELS:
        model_path = _download(_MODELS[name], os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
   

    with open(model_path, 'rb') as opened_file:
        try:
            # loading JIT archive
            pretrain_model = torch.jit.load(opened_file, map_location="cpu")
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            state_dict = torch.load(opened_file, map_location="cpu")

    state_dict=pretrain_model.state_dict()
    
    counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
    vision_layers = tuple(counts)
    vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
    output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
    vision_patch_size = None
    assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
    image_resolution = output_width * 32

 
    #embed_dim = 2048
    embed_dim = state_dict["text_projection"].shape[1]
    clip_feat_dim=state_dict["text_projection"].shape[1]
   
    vision_heads = vision_width * 32 // 64
    model = ModifiedResNet(
        layers=vision_layers,
        output_dim=embed_dim,
        heads=vision_heads,
        input_resolution=image_resolution,
        width=vision_width,
        cloth=300,
        cloth_xishu=config.MODEL.CLOTH_XISHU,
        last_layer=config.MODEL.LAST_LAYER,
        add_nonBio=config.MODEL.NONBIO_HEAD,
        nonBio_num=len(config.DATA.NOBIO_INDEX.strip('*').split('*')),
        bioClip='clipBio' in config.MODEL.METRIC_LOSS_TYPE,
        reverse_bio='clipBioReverse' in config.MODEL.METRIC_LOSS_TYPE,
        clip_feat_dim=clip_feat_dim, #config.MODEL.CLIP_DIM,
        **kwargs,
    )
        


    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
            
    out_dict = {}
    len_prefix=len('visual.')
    #has_layer = hasattr(model, 'logit_scale')
    for k, v in state_dict.items():
        #print(cc)
        if k.startswith('visual'):
            #print(k)
            k = k[len_prefix:]
            out_dict[k]=v
        if k.find('logit_scale')>=0 and  hasattr(model, 'logit_scale'):
            out_dict[k]=v
            

    #convert_weights(model)
    missing_keys, unexpected_keys = model.load_state_dict(out_dict, strict=False)
    print('missing_keys:',missing_keys)
    print('unexpected_keys:',unexpected_keys)
    
    
    
    #model = build_model(state_dict or model.state_dict())
    
    return model #, _transform(model.input_resolution)


import math
def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    print('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
      
    ntok_new = posemb_new.shape[0] #129,2048

    posemb_token, posemb_grid = posemb[:1], posemb[1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid))) #14
    print('Position embedding resize to height:{} width: {}'.format(hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2) 
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear') 
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid.squeeze()], dim=0)
    return posemb
 
def ClipViT(pretrained=False,config={}, **kwargs):
    
    name=config.MODEL.NAME
    if name in _MODELS:
        model_path = _download(_MODELS[name], os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
   
    img_size=(config.DATA.IMG_HEIGHT,config.DATA.IMG_WIDTH)
    #sie_xishu=config.MODEL.CLOTH_XISHU
    #camera=kwargs['cameraNum']
    drop_path_rate=0.0
    view=0
    stride_size=16
    drop_rate= 0.0
    attn_drop_rate=0.0
    
    with open(model_path, 'rb') as opened_file:
        try:
            # loading JIT archive
            pretrain_model = torch.jit.load(opened_file, map_location="cpu")
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            state_dict = torch.load(opened_file, map_location="cpu")

    state_dict=pretrain_model.state_dict()
    
    vit='ViT' in name
    
    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        #grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        #image_resolution = vision_patch_size * grid_size


    embed_dim =vision_width
    if 'clipBio' in config.MODEL.METRIC_LOSS_TYPE:
        clip_feat_dim=state_dict["text_projection"].shape[1]
    else:
        clip_feat_dim=-1

    
    clip_projection_weight=state_dict["visual.proj"]
    vision_heads =vision_width // 64
    model =VisionTransformer(config,
            input_resolution=img_size,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            clip_output_dim=clip_feat_dim,
            drop_path_rate=drop_path_rate, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            #emb_dropout=emb_dropout
            # cloth=300,
            camera_xishu=config.MODEL.CLOTH_XISHU,
            # last_layer=config.MODEL.LAST_LAYER,
            # add_nonBio=config.MODEL.NONBIO_HEAD,
            # nonBio_num=len(config.DATA.NOBIO_INDEX.strip('*').split('*')),
            # bioClip='clipBio' in config.MODEL.METRIC_LOSS_TYPE,
            # reverse_bio='clipBioReverse' in config.MODEL.METRIC_LOSS_TYPE,
            # clip_feat_dim=clip_feat_dim, #config.MODEL.CLIP_DIM,
            **kwargs,
    )
    
    num_patches_x = img_size[0] // vision_patch_size
    num_patches_y= img_size[1] // vision_patch_size

        
    state_dict["visual.positional_embedding"] = resize_pos_embed(state_dict["visual.positional_embedding"], model.positional_embedding, num_patches_x, num_patches_y)
    
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
            
    out_dict = {}
    len_prefix=len('visual.')
    #has_layer = hasattr(model, 'logit_scale')
    for k, v in state_dict.items():
        #print(cc)
        if k.startswith('visual'):
            #print(k)
            if k.startswith('visual.ln_post'):
                continue
            k = k[len_prefix:]
            out_dict[k]=v
        if k.find('logit_scale')>=0 : #and  hasattr(model, 'logit_scale'):
            out_dict[k]=v
            

    #convert_weights(model)
    missing_keys, unexpected_keys = model.load_state_dict(out_dict, strict=False)
    print('missing_keys:',missing_keys)
    print('unexpected_keys:',unexpected_keys)
    
    
    
    #model = build_model(state_dict or model.state_dict())
    
    return model, clip_projection_weight #, _transform(model.input_resolution)



if __name__ == "__main__":
    import sys
    import os
    
    preprocess=_transform(224)
    x = preprocess(Image.open("/home/xi860799/code/MADE/paddimg.png")).unsqueeze(0)
    cloth_id = torch.tensor([1])
    # Add the parent directory to sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    #config_file='/home/xi860799/code/MADE/logs/ltcc/315[ltcc][ClipRN50]/config.yml'
    config_file='/home/xi860799/code/MADE/configs/ltcc/clipViTB_bio.yml'
    from config import cfg
    # x = torch.randn([2, 3, 224, 224])
    # cloth_id = torch.tensor([2, 3])

    cfg.merge_from_file(config_file)
    #modelOg = ClipRN50(config=cfg)
    # import ipdb;ipdb.set_trace()
    model = ClipViT(config=cfg)
    
    
    #model = ClipRN50(config=cfg)
    # pth_file_path='/home/xi860799/code/MADE/logs/ltcc/315[ltcc][ClipRN50]/ClipRN50_best.pth'
    # state_dict = torch.load(pth_file_path)
    # model.load_state_dict(state_dict)
    # differences=[]
    # for name, param in modelOg.named_parameters():
        
    #     trained_weight=state_dict[name]
    #     l2_distance = torch.norm(param - trained_weight.cpu(), p=2)
    #     print(name,l2_distance)
    #     if l2_distance > 1e-6:
    #         print(l2_distance)
    #         differences.append((name, l2_distance.item()))

    # modelOg.eval()
    # model.eval()
    # outputOg = modelOg(x,cloth_id)
    # output = model(x,cloth_id)
    # l2_distance = torch.norm(outputOg- output, p=2)
    
    # def register_hooks(model, outputs):
    #     def get_hook_fn(name):
    #         def hook_fn(module, input, output):
    #             outputs.append((name, output))
    #         return hook_fn

    #     for name, layer in model.named_modules():
    #         if not isinstance(layer, nn.Sequential) and not isinstance(layer, nn.ModuleList) and layer != model:
    #             layer.register_forward_hook(get_hook_fn(name))

            
    # outputs_model1 = []
    # outputs_model2 = []

    # # Register hooks to capture the outputs
    # register_hooks(modelOg, outputs_model1)
    # register_hooks(model, outputs_model2)
    # with torch.no_grad():
    #     _ = modelOg(x,cloth_id)
    #     _ = model(x,cloth_id)
        
    # for (name1, out1), (name2, out2) in zip(outputs_model1, outputs_model2):
    #     l2_distance = torch.norm(out1 - out2, p=2).item()
    #     if l2_distance > -1:
    #         print(f"Layer {name1} has significant difference: L2 Distance = {l2_distance}")
    #         print(out1.shape)


        
    # for key in outputOg[1]:
    #     l2_distance = torch.norm(outputOg[key] - output[key], p=2)
    #     print(l2_distance)
    #print(output[0].shape)
    # print(summary(model, input_size=((8, 3, 224, 224),(3,7))))

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('/home/xi860799/code/Simple-CCReID/logs/model/resnetClip')
    writer.add_graph(model, x)
    writer.close()