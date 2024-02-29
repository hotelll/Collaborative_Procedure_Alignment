import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.builder import *
from utils.dpm_decoder import seg_and_interpolate
from utils.loss import frame2learnedstep_dist


FIX_LEN = 3
STEP_NUM = 14


class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        weights = F.softmax(self.fc(x), dim=1) # [batch_size, seq_len, 1]
        weighted = weights.expand_as(x) * x # [batch_size, seq_len, input_dim]
        representations = weighted.sum(1) # [batch_size, input_dim]
        return representations, weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_dim):
        super(TransformerModel, self).__init__()
        self.pe = PositionalEncoding(input_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=1)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=1)
        self.attention = Attention(input_dim)

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        x = x.permute(1, 0, 2) # [seq_len, batch_size, input_dim]
        x = self.pe(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2) # [batch_size, seq_len, input_dim]
        x, weights = self.attention(x)
        return x, weights


class CPA_randomStep(nn.Module):
    def __init__(self,
                 num_class=20,
                 num_clip=16,
                 dim_size=2048,
                 pretrain=None,
                 dropout=0):

        super(CPA_randomStep, self).__init__()

        self.num_clip = num_clip
        self.dim_size = dim_size
        
        module_builder = Builder(num_clip, pretrain, False, dim_size)
        
        self.backbone = module_builder.build_backbone()
        self.bottleneck = nn.Conv2d(2048, 128, 3, 1, 1)
        
        self.get_token = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                       nn.Flatten(),
                                       Reshape(-1, self.num_clip, dim_size))
        
        self.step_encoder = nn.Sequential(nn.Conv1d(dim_size, dim_size, kernel_size=3, padding=1),
                                          nn.ReLU(),
                                          nn.Conv1d(dim_size, dim_size, kernel_size=1, padding=0))
        
        self.temporal_learner = TransformerModel(input_dim=128)
        
        self.global_net = nn.Sequential(Reshape(-1, self.num_clip * dim_size),
                                        nn.Linear(self.num_clip * dim_size, dim_size))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.cls_fc = nn.Linear(dim_size, num_class)
    
    
    def forward(self, x1, x2, embed=False):
        x1 = self.backbone(x1)
        x1 = self.bottleneck(x1)
        seq_features1 = self.get_token(x1)
        seq_features1 = seq_features1.permute(0,2,1)
        seq_features1 = self.step_encoder(seq_features1)
        seq_features1 = seq_features1.permute(0,2,1)
        
        x2 = self.backbone(x2)
        x2 = self.bottleneck(x2)
        seq_features2 = self.get_token(x2)
        seq_features2 = seq_features2.permute(0,2,1)
        seq_features2 = self.step_encoder(seq_features2)
        seq_features2 = seq_features2.permute(0,2,1)
        
        (B, T, C), device = seq_features1.shape, seq_features1.device
        # the similarity matrix: 16 * 16
        
        batched_seg_list1 = []
        batched_seg_list2 = []
        batched_step_list1 = []
        batched_step_list2 = []
        
        for batch in range(B):   
            # -----------------------------------------------------------------------------
            step_feature1 = seq_features1[batch, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15]]
            step_feature2 = seq_features2[batch, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15]]
            batched_step_list1.insert(0, step_feature1)
            batched_step_list2.insert(0, step_feature2)
            # -----------------------------------------------------------------------------
            
        frame2step_dist = frame2learnedstep_dist(seq_features1, batched_step_list2) \
                            + frame2learnedstep_dist(seq_features2, batched_step_list1)
        
        
        # return batched_step_list1, batched_step_list2, batched_seg_list1, batched_seg_list2
        global_features1 = self.global_net(seq_features1)
        global_features2 = self.global_net(seq_features2)
        
        if embed:
            return global_features1, global_features2, frame2step_dist
        
        global_features1 = self.dropout(global_features1)
        pred1 = self.cls_fc(global_features1)
        
        global_features2 = self.dropout(global_features2)
        pred2 = self.cls_fc(global_features2)
        
        return pred1, pred2, frame2step_dist
