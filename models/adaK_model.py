import torch
import torch.nn as nn
import math
from utils.builder import *
from utils.loss import frame2learnedstep_dist



class Align_adaK(nn.Module):
    def __init__(self,
                 num_class=20,
                 num_clip=16,
                 dim_size=2048,
                 pretrain=None,
                 dropout=0):

        super(Align_adaK, self).__init__()

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
        
        self.global_net = nn.Sequential(Reshape(-1, self.num_clip * dim_size),
                                        nn.Linear(self.num_clip * dim_size, dim_size))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.cls_fc = nn.Linear(dim_size, num_class)
    
    
    def forward(self, x1, x2, training=False):
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
        pred = (torch.einsum('bic,bjc->bij', seq_features1, seq_features2) / math.sqrt(C)).softmax(-1)
        
        # the similarity matrix: 16 * 16
        # if training:
        #     pred = (torch.einsum('bic,bjc->bij', seq_features1, seq_features2) / math.sqrt(C)).softmax(-1)
        # else:
        #     pred = torch.cosine_similarity(seq_features1.unsqueeze(2), seq_features2.unsqueeze(1), dim=-1)

        pred = pred.cumsum(-2).cumsum(-1)
        
        D = torch.zeros((B, T, T, T), device=device)
        D_ind = torch.zeros((B, T, T, T), dtype=torch.long, device=device)
        
        D[:, 0] = pred / torch.ones_like(pred).cumsum(-2).cumsum(-1)
        
        area = torch.ones_like(pred).cumsum(-2).cumsum(-1)
        area = area[:, :, :, None, None] - area[:, :, None, None, :] - \
            area.transpose(1,2)[:, None, :, :, None] + area[:, None, None, :, :]
        block_mat = pred[:, :, :, None, None] - pred[:, :, None, None, :] - \
            pred.transpose(1,2)[:, None, :, :, None] + pred[:, None, None, :, :]
        i, j, a, b = torch.meshgrid(*[torch.arange(T, device=device)]*4)
        
        # if training:
        #     area = area.clamp_min(1)
        # else:
        #     area = area.clamp_min(1).sqrt()
        area = area.clamp_min(1)
        
        
        block_mat = block_mat.masked_fill(((a >= i) | (b >= j)).unsqueeze(0), float('-inf')) / area
        
        for k in range(1, T):
            tmp = D[:, k-1, None, None, :, :] + block_mat
            # if training:
            #     tmp = D[:, k-1, None, None, :, :] + block_mat
            # else:
            #     tmp = ((D[:, k-1, None, None, :, :] * k) + block_mat) / (k+1)
            D[:, k] = torch.max(tmp.flatten(3), -1).values
            D_ind[:, k] = torch.max(tmp.flatten(3), -1).indices
        
        final_result = D[:, :, T-1, T-1]
        loss_step = -(final_result.max(dim=-1).values).mean()
        step_num = final_result.max(dim=-1).indices
        
        batched_seg_list1 = []
        batched_seg_list2 = []
        batched_step_list1 = []
        batched_step_list2 = []
        
        for batch in range(B):
            seg1_list = []
            seg2_list = []
            i, j, a, b = T-1, T-1, T-1, T-1 
            k = step_num[batch].item()
            # k = 13
            
            step_list1 = []
            step_list2 = []
            
            while k > 0:
                ind = D_ind[batch, k, i, j].item()
                a = ind // T
                b = ind % T
                
                step_feature1 = seq_features1[batch][a+1]
                step_feature2 = seq_features2[batch][b+1]
                step_list1.insert(0, step_feature1)
                step_list2.insert(0, step_feature2)
            
                seg1_list.insert(0, a)
                seg2_list.insert(0, b)
                i, j, k = a, b, k-1
            
            step_feature1 = seq_features1[batch][1]
            step_feature2 = seq_features2[batch][1]
            step_list1.insert(0, step_feature1)
            step_list2.insert(0, step_feature2)
            
            step_features1 = torch.stack(step_list1, dim=0)
            step_features2 = torch.stack(step_list2, dim=0)
            seg_tensor1 = torch.tensor(seg1_list, device=device)
            seg_tensor2 = torch.tensor(seg2_list, device=device)
            
            batched_step_list1.append(step_features1)
            batched_step_list2.append(step_features2)
            
            batched_seg_list1.append(seg_tensor1)
            batched_seg_list2.append(seg_tensor2)
        
        frame2step_dist = frame2learnedstep_dist(seq_features1, batched_step_list2) \
                            + frame2learnedstep_dist(seq_features2, batched_step_list1)
        # normalize distance
        frame2step_dist = frame2step_dist
        global_features1 = self.global_net(seq_features1)
        global_features1 = self.dropout(global_features1)
        pred1 = self.cls_fc(global_features1)
        
        global_features2 = self.global_net(seq_features2)
        global_features2 = self.dropout(global_features2)
        pred2 = self.cls_fc(global_features2)
        
        return pred1, pred2, frame2step_dist, loss_step
