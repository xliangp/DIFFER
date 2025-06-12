import torchvision
from torch import nn
from torch.nn import init 

import torch
from torch import nn
from torch.nn import functional as F


class GeMPooling(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), x.size()[2:]).pow(1./self.p)


class MaxAvgPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpooling = nn.AdaptiveMaxPool2d(1)
        self.avgpooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        max_f = self.maxpooling(x)
        avg_f = self.avgpooling(x)

        return torch.cat((max_f, avg_f), 1)
        
        
class ResNet50(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()

        resnet50 = torchvision.models.resnet50(pretrained=True)
        # if config.RES4_STRIDE == 1:
        #     resnet50.layer4[0].conv2.stride=(1, 1)
        #     resnet50.layer4[0].downsample[0].stride=(1, 1) 
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.reduction_layer = nn.Conv2d(2048, config.FEATURE_DIM, kernel_size=1, stride=1, padding=0,bias=False)


        if config.POOLING.NAME == 'avg':
            self.globalpooling = nn.AdaptiveAvgPool2d(1)
        elif config.POOLING.NAME == 'max':
            self.globalpooling = nn.AdaptiveMaxPool2d(1)
        elif config.POOLING.NAME == 'gem':
            self.globalpooling = GeMPooling(p=config.POOLING.P)
        elif config.POOLING.NAME == 'maxavg':
            self.globalpooling = MaxAvgPooling()
        else:
            raise KeyError("Invalid pooling: '{}'".format(config.POOLING.NAME))

        self.bn = nn.BatchNorm1d(config.FEATURE_DIM)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)
        
    def forward(self, x):
        x = self.base(x)
        x= self.reduction_layer(x)
        x = self.globalpooling(x)
        x = x.view(x.size(0), -1)
        f = self.bn(x)

        return f