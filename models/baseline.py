import torch
import torch.nn as nn
import math
from utils.builder import *

class Baseline(nn.Module):
    def __init__(self,
                 num_class=20,
                 num_clip=16,
                 dim_size=2048,
                 pretrain=None,
                 dropout=0):

        super(Baseline, self).__init__()

        self.num_clip = num_clip
        self.dim_size = dim_size
        
        module_builder = Builder(num_clip, pretrain, False, dim_size)
        
        self.backbone = module_builder.build_backbone()
        self.bottleneck = nn.Conv2d(2048, dim_size, 3, 1, 1)
        
        self.get_token = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                       nn.Flatten(),
                                       Reshape(-1, self.num_clip, dim_size))
        
        self.emb_head = nn.Sequential(Reshape(-1, self.num_clip * dim_size),
                                     nn.Linear(self.num_clip * dim_size, dim_size))
        
        self.cls_head = nn.Sequential(nn.Dropout(dropout),
                                     nn.Linear(dim_size, num_class))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
    
    
    def forward(self, x: torch.Tensor, embed=False):
        x = self.backbone(x)
        x = self.bottleneck(x)
        
        seq_features = self.get_token(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.emb_head(x)
        
        if embed:
            return x, seq_features

        x = self.cls_head(x)
        
        return x, seq_features
