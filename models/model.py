import torch
import torch.nn as nn
import math
import torch.nn.init as init
from torch.autograd import Variable
from utils.builder import *

class CAT(nn.Module):
    def __init__(self,
                 num_class=20,
                 num_clip=16,
                 dim_embedding=128,
                 pretrain=None,
                 dropout=0,
                 use_TE=False,
                 use_SeqAlign=False,
                 freeze_backbone=False):

        super(CAT, self).__init__()

        self.num_clip = num_clip
        self.use_TE = use_TE
        self.use_SeqAlign = use_SeqAlign
        self.freeze_backbone = freeze_backbone

        module_builder = Builder(num_clip, pretrain, use_TE, dim_embedding)

        self.backbone = module_builder.build_backbone()

        self.seq_features_extractor = module_builder.build_seq_features_extractor()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.embed_head = module_builder.build_embed_head()
        self.dropout = nn.Dropout(dropout)
        self.cls_fc = nn.Linear(dim_embedding, num_class)


    def train(self, mode=True):
        """
        Override the default train() to freeze the backbone
        :return:
        """
        super(CAT, self).train(mode)

        if self.freeze_backbone:
            print('Freezeing backbone.')
            for param in self.backbone.parameters():
                param.requires_grad = False


    def forward(self, x, embed=False):
        x = self.backbone(x)  # [bs * num_clip, 2048, 6, 10]

        x = self.avgpool(x)
        x = x.flatten(1)  # [bs * num_clip, 2048]
        
        x = self.embed_head(x)
        
        if embed:
            return x

        x = self.dropout(x)
        x = self.cls_fc(x)

        return x


class Baseline(nn.Module):
    def __init__(self,
                 num_class=20,
                 num_clip=16,
                 dim_size=128,
                 pretrain=None,
                 dropout=0):

        super(Baseline, self).__init__()

        self.num_clip = num_clip
        self.dim_size = dim_size
        
        module_builder = Builder(num_clip, pretrain, False, dim_size)

        self.backbone = module_builder.build_backbone()
        self.feature2token = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                           nn.Flatten(),
                                           Reshape(-1, self.num_clip, 2048)
                                           )
        
        self.global_net = nn.Sequential(
                Reshape(-1, self.num_clip * 2048),
                nn.Linear(self.num_clip * 2048, dim_size),
            )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.dropout = nn.Dropout(dropout)
        self.cls_fc = nn.Linear(dim_size, num_class)


    def forward(self, x: torch.Tensor, embed=False):
        x = self.backbone(x)
        
        x = self.avgpool(x)
        x = x.flatten(1)
        
        x = self.global_net(x)
        if embed:
            return x
        
        x = self.dropout(x)
        preds = self.cls_fc(x)
        
        return preds


class Baseline_x3d(nn.Module):
    def __init__(self,
                 num_class=20,
                 num_clip=16,
                 dim_size=128,
                 iters=10,
                 pretrain=None,
                 dropout=0,
                 freeze_backbone=False):

        super(Baseline_x3d, self).__init__()

        self.num_clip = num_clip
        self.dim_size = dim_size
        self.iters = iters
        
        self.freeze_backbone = freeze_backbone
        
        module_builder = Builder(num_clip, pretrain, False, dim_size)

        self.backbone = module_builder.build_3d_backbone()
        self.feature2token = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                           nn.Flatten(),
                                           Reshape(-1, self.num_clip, 192)
                                           )
        
        self.global_net = nn.Sequential(
                Reshape(-1, self.num_clip * 192),
                nn.Linear(self.num_clip * 192, 192),
            )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.dropout = nn.Dropout(dropout)
        self.cls_fc = nn.Linear(192, num_class)
        
        
    def train(self, mode=True):
        """
        Override the default train() to freeze the backbone
        :return:
        """
        super(Baseline_x3d, self).train(mode)

        if self.freeze_backbone:
            print('Freezeing backbone.')
            for param in self.backbone.parameters():
                param.requires_grad = False


    def forward(self, x: torch.Tensor, embed=False):
        x = self.backbone(x)
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)
        
        ground_truth = self.feature2token(x)
        
        global_features = self.global_net(ground_truth)
        
        if embed:
            return global_features
        
        global_features = self.dropout(global_features)
        preds = self.cls_fc(global_features)
        
        return preds

