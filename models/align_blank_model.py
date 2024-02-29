import torch
import torch.nn as nn
import math
from utils.builder import *


class Align(nn.Module):
    def __init__(self,
                 num_class=20,
                 num_clip=16,
                 dim_size=128,
                 pretrain=None,
                 dropout=0):

        super(Align, self).__init__()

        self.num_clip = num_clip
        self.dim_size = dim_size
        self.blank_padding = nn.Embedding(1, dim_size)
        module_builder = Builder(num_clip, pretrain, False, dim_size)
        
        self.backbone = module_builder.build_backbone()
        self.bottleneck = nn.Conv2d(2048, dim_size, 3, 1, 1)
        
        self.get_token = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                       nn.Flatten(),
                                       Reshape(-1, self.num_clip, dim_size))
        
        self.step_encoder = nn.Sequential(nn.Conv1d(dim_size, dim_size, kernel_size=3, padding=1),
                                          nn.ReLU(),
                                          nn.Conv1d(dim_size, dim_size, kernel_size=1, padding=0))
        
        self.global_net = nn.Sequential(Reshape(-1, self.num_clip * dim_size),
                                        nn.Linear(self.num_clip * dim_size, dim_size))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.cls_fc = nn.Linear(dim_size, num_class)
    
    
    def forward(self, x: torch.Tensor, embed=False):
        x = self.backbone(x)
        x = self.bottleneck(x)
        seq_features = self.get_token(x)
        
        seq_features = seq_features.permute(0,2,1)
        seq_features = self.step_encoder(seq_features)
        seq_features = seq_features.permute(0,2,1)
        
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.global_net(x)
        
        blank = self.blank_padding.weight.unsqueeze(0).expand(seq_features.shape[0], 1, -1)
        seq_features = torch.cat((blank, seq_features), dim=1)
        
        if embed:
            return x, seq_features
        
        x = self.dropout(x)
        x = self.cls_fc(x)
        
        return x, seq_features
