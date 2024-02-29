import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.builder import *
from utils.dpm_decoder import seg_and_interpolate
from utils.loss import frame2learnedstep_dist
from utils.loss_origin import batched_step_comparison


FIX_LEN = 3

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


class Elastic_Matching_Model(nn.Module):
    def __init__(self,
                 num_class=20,
                 num_clip=16,
                 dim_size=2048,
                 pretrain=None,
                 dropout=0):

        super(Elastic_Matching_Model, self).__init__()

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
        # pred = (torch.einsum('bic,bjc->bij', seq_features1, seq_features2) / math.sqrt(C)).softmax(-1)
        pred = torch.cosine_similarity(seq_features1.unsqueeze(2), seq_features2.unsqueeze(1), dim=-1)
        pred = pred.cumsum(-2).cumsum(-1)
        
        D = torch.zeros((B, T, T, T), device=device)
        D_ind = torch.zeros((B, T, T, T), dtype=torch.long, device=device)
        D_block = torch.zeros((B, T, T, T), device=device)
        
        D[:, 0] = pred / torch.ones_like(pred).cumsum(-2).cumsum(-1)
        D_block[:, 0] = pred / torch.ones_like(pred).cumsum(-2).cumsum(-1)
        
        area = torch.ones_like(pred).cumsum(-2).cumsum(-1)
        area = area[:, :, :, None, None] - area[:, :, None, None, :] - \
            area.transpose(1,2)[:, None, :, :, None] + area[:, None, None, :, :]
        block_mat = pred[:, :, :, None, None] - pred[:, :, None, None, :] - \
            pred.transpose(1,2)[:, None, :, :, None] + pred[:, None, None, :, :]
        i, j, a, b = torch.meshgrid(*[torch.arange(T, device=device)]*4)
        area = area.clamp_min(1).sqrt()

        block_mat = block_mat.masked_fill(((a >= i) | (b >= j)).unsqueeze(0), float('-inf')) / area
        
        for k in range(1, T):
            # tmp = D[:, k-1, None, None, :, :] + block_mat
            tmp = ((D[:, k-1, None, None, :, :] * k) + block_mat) / (k+1)
            D[:, k] = torch.max(tmp.flatten(3), -1).values
            D_ind[:, k] = torch.max(tmp.flatten(3), -1).indices
            D_block[:, k] = block_mat.flatten(3).gather(dim=-1, index=D_ind[:, k].unsqueeze(-1)).squeeze(-1)
        
        final_result = D[:, :, T-1, T-1]
        loss_step = -(final_result.max(dim=-1).values).mean()
        # step_num = final_result.max(dim=-1).indices
        batched_step_list1 = []
        batched_step_list2 = []
        
        for batch in range(B):
            # current_step_num = step_num[batch].item()
            # current_step_thres = step_thres[batch].item()
            current_step_num = 14
            current_step_thres = math.ceil(current_step_num)
            
            step_values = torch.zeros((current_step_num), device=device)
            seg1_list = []
            seg2_list = []
            i, j, a, b = T-1, T-1, T-1, T-1
            k = current_step_num - 1
            
            step_list1 = []
            step_list2 = []
            
            video1_end = T
            video2_end = T
            
            # step_average_list = []
            while k > 0:
                ind = D_ind[batch, k, i, j].item()
                step_value = D_block[batch, k, i, j].item()
                
                step_values[k] = step_value
                # step_average_list.append(step_value)
                a = ind // T
                b = ind % T
                
                video1_start = a + 1
                video2_start = b + 1
                step_feature1 = seg_and_interpolate(seq_features1[batch], video1_start, video1_end, FIX_LEN)
                step_feature2 = seg_and_interpolate(seq_features2[batch], video2_start, video2_end, FIX_LEN)
                step_agg_feature1, step_frame_att1 = self.temporal_learner(step_feature1)
                step_agg_feature2, step_frame_att2 = self.temporal_learner(step_feature2)
                
                step_list1.insert(0, step_agg_feature1)
                step_list2.insert(0, step_agg_feature2)
                video1_end = video1_start
                video2_end = video2_start
                
                seg1_list.insert(0, a)
                seg2_list.insert(0, b)
                i, j, k = a, b, k-1
            
            step_values[0] = D_block[batch, 0, i, j].item()
            # selected_step_indices = step_values.topk(k=current_step_thres).indices.sort().values
            # step_value_tensor = torch.tensor(step_average_list).flip(dims=[0])
            
            video1_start = 0
            video2_start = 0
            step_feature1 = seg_and_interpolate(seq_features1[batch], video1_start, video1_end, FIX_LEN)
            step_feature2 = seg_and_interpolate(seq_features2[batch], video2_start, video2_end, FIX_LEN)
            step_agg_feature1, step_frame_att1 = self.temporal_learner(step_feature1)
            step_agg_feature2, step_frame_att2 = self.temporal_learner(step_feature2)
            
            step_list1.insert(0, step_agg_feature1)
            step_list2.insert(0, step_agg_feature2)
            
            step_features1 = torch.stack(step_list1, dim=1)
            step_features2 = torch.stack(step_list2, dim=1)
            
            # selected_step_features1 = step_features1[:, selected_step_indices, :]
            # selected_step_features2 = step_features2[:, selected_step_indices, :]
            
            batched_step_list1.append(step_features1[0])
            batched_step_list2.append(step_features2[0])
        
        step_level1 = torch.stack(batched_step_list1)
        step_level2 = torch.stack(batched_step_list2)
        # step_dict = frame2learnedstep_dist(seq_features1, batched_step_list2) \
        #           + frame2learnedstep_dist(seq_features2, batched_step_list1)
        # step_dict = batched_step_comparison(batched_step_list1, batched_step_list2)
        
        # B = len(step_comp_dict)
        
        # step_dict = torch.zeros((B,), device=device)
        # for batch in range(B):
        #     step_wise_dist = step_comp_dict[batch]
        #     matched_step_num = (step_wise_dist > 0.85).sum()
        #     step_dict[batch] = matched_step_num
        
        # return batched_step_list1, batched_step_list2, batched_seg_list1, batched_seg_list2
        global_features1 = self.global_net(seq_features1)
        global_features2 = self.global_net(seq_features2)
        
        if embed:
            return global_features1, global_features2, seq_features1, seq_features2
        
        global_features1 = self.dropout(global_features1)
        pred1 = self.cls_fc(global_features1)
        
        global_features2 = self.dropout(global_features2)
        pred2 = self.cls_fc(global_features2)
        
        return global_features1, global_features2, seq_features1, seq_features2, step_level1, step_level2
