import torch
from torch import nn
import sys
from .eva_vit_model import eva_clip
from .eva_embed import eva02_large_patch14_clip_224,eva02_base_patch16_clip_224
from .eva_meta_bio import eva02_large_patch14_clip_224_bio,GradientReversalClassifier
from .clip  import ClipRN50,ClipViT,weights_init_kaiming


from timm.layers import trunc_normal_
import numpy as np

factory = {
    'eva02_large':eva02_large_patch14_clip_224,
    'eva02_base':eva02_base_patch16_clip_224,
    'eva02_l_bio':eva02_large_patch14_clip_224_bio,
    'eva02':eva_clip,
}


def build_model(config,num_classes,camera_num):
    model_type = config.MODEL.TYPE
    if model_type in ['eva02']:
        model=build_transformer( num_classes=num_classes,camera_num=camera_num,cfg=config,)
    elif model_type=='' or model_type=='eva02_cloth':
        model = factory[config.MODEL.NAME](pretrained=True,config=config, num_classes=num_classes,camera_num=camera_num)
    else:
        model = factory[f"{config.MODEL.NAME}_{config.MODEL.TYPE}"](pretrained=True,config=config, num_classes=num_classes,camera_num=camera_num)

    return model

    
class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, cfg):
        super(build_transformer, self).__init__()
        
        nonBio_num=len(cfg.DATA.NOBIO_INDEX.strip('*').split('*'))
        bio_num=len(cfg.DATA.BIO_INDEX.strip('*').split('*'))
        
        reverse_bio='clipBioReverse' in cfg.MODEL.METRIC_LOSS_TYPE
        clip_feat_dim=cfg.MODEL.CLIP_DIM
        
        self.last_layer=cfg.MODEL.LAST_LAYER
        #self.add_nonBio=cfg.MODEL.NONBIO_HEAD
        self.bio_num=bio_num
        self.nonBio_num=nonBio_num
        self.reverse_bio=reverse_bio
        self.normalizeFeature=cfg.MODEL.NORM_FEAT
        self.num_classes=num_classes

        bottleNeck=cfg.MODEL.BOTTEL_NECK
        
        
        model_type = cfg.MODEL.TYPE

        if model_type in ['eva02']:
            self.base = factory[f"{cfg.MODEL.NAME}"](pretrained=True,config=cfg,camera_num=camera_num)
            self.feat_dim=self.base.num_features
            clip_projection_weight=None
        else:
            self.base ,clip_projection_weight= factory[model_type](pretrained=True, config=cfg, 
                            num_classes=num_classes, 
                            camera_num=camera_num)

            self.feat_dim=self.base.width
        
        head_init_scale = 0.001
       
        if bottleNeck:
            self.bottleneck = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck.bias.requires_grad_(False)
            self.bottleneck.apply(weights_init_kaiming)
        else:
            self.bottleneck=nn.Identity()
            
            
        # if clip_feat_dim != -1:
        #     self.proj = nn.Parameter(scale * torch.randn(self.feat_dim, clip_feat_dim))
        if num_classes != -1:
            self.head = nn.Linear(self.feat_dim, num_classes)
            
            if isinstance(self.head, nn.Linear):
                trunc_normal_(self.head.weight, std=.02)
                self.head.weight.data.mul_(head_init_scale)
                self.head.bias.data.mul_(head_init_scale)
                
        if clip_projection_weight is not None:      
            if clip_projection_weight.shape[1]!=clip_feat_dim:
                clip_projection_weight=None
            else:
                clip_projection_weight=clip_projection_weight.T.type('torch.FloatTensor')
        if self.last_layer in ['clipFc_clsFc']:
            self.head_clip_bio =  nn.ModuleList([nn.Linear(self.feat_dim, clip_feat_dim, bias=False) for x in range(bio_num)])
            for x in range(bio_num):
                if clip_projection_weight is None:
                    self.head_clip_bio[x].apply(weights_init_kaiming)  
                else:
                    self.head_clip_bio[x].weight.data = clip_projection_weight.clone()
        
        if 'clipBio' in cfg.MODEL.METRIC_LOSS_TYPE:
            if cfg.MODEL.CLIP_LOSS_TYPE=='constant':
                self.logit_scale_bio = 1
                self.logit_bias_bio = 0
            elif cfg.MODEL.CLIP_LOSS_TYPE=='contrastive':
                self.logit_scale_bio = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
                self.logit_bias_bio = 0
            elif cfg.MODEL.CLIP_LOSS_TYPE=='sigmoid':
                self.logit_scale_bio = nn.Parameter(torch.randn(1))
                self.logit_bias_bio = nn.Parameter(torch.randn(1))
            clip_feat_dim=cfg.MODEL.CLIP_DIM
        else:
            clip_feat_dim=-1
            
        if self.reverse_bio :  
            self.head_clip_bioReverse=nn.ModuleList([GradientReversalClassifier(in_num=self.feat_dim,
                                                                                class_num=clip_feat_dim,
                                                                                weight=clip_projection_weight) 
                                                    for x in range(self.nonBio_num)])
            if cfg.MODEL.CLIP_LOSS_TYPE=='constant':
                self.logit_scale_nonbios = [1 for x in range(self.nonBio_num)]
                self.logit_bias_nonbios=0
            elif cfg.MODEL.CLIP_LOSS_TYPE=='contrastive':
                self.logit_scale_nonbios = [nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) for x in range(self.nonBio_num)]
                self.logit_bias_nonbios=0
            elif cfg.MODEL.CLIP_LOSS_TYPE=='sigmoid':
                self.logit_scale_nonbios = [nn.Parameter(torch.randn(1)) for x in range(self.nonBio_num)]
                self.logit_bias_nonbios = [nn.Parameter(torch.randn(1))  for x in range(self.nonBio_num)]
                
    def foward_score(self,x,norm=True):
        if norm:
            x=nn.functional.normalize(x,dim=1)
            self.head.weight.data=nn.functional.normalize(self.head.weight.data,dim=1)
        cls_score = self.head(x)
        return cls_score        

    def forward(self, x, cam_label= None,norm=False):
        x = self.base(x,cam_label)

        if self.training:
            feat_output=[x,None]
            x_bn = self.bottleneck(x)
            
            if norm:
                x_bn=nn.functional.normalize(x_bn,dim=1)
                self.head.weight.data=nn.functional.normalize(self.head.weight.data,dim=1)
                
            cls_score = self.head(x_bn)
            score_output={'cls_score':cls_score}
                
            if self.last_layer in ['clipFc_clsFc']:
                clip_bio_score=[layer(x) for layer in self.head_clip_bio]
                clip_bio_score = [t.unsqueeze(1) for t in clip_bio_score]
                score_output['clip_bio_score']=torch.cat(clip_bio_score, dim=1)
                score_output['clip_bio_scale']=self.logit_scale_bio
                score_output['clip_bio_bias']=self.logit_bias_bio
                
            if self.reverse_bio:
                clip_bio_reverse_score=[layer(x) for layer in self.head_clip_bioReverse]
                clip_bio_reverse_score = [t.unsqueeze(1) for t in clip_bio_reverse_score]
                score_output['clip_bio_reverse_score']=torch.cat(clip_bio_reverse_score, dim=1)
                
                score_output['clip_nonbio_scale']=self.logit_scale_nonbios
                score_output['clip_nonbio_bias']=self.logit_bias_nonbios
                
            return score_output, feat_output

        else:
            return x



