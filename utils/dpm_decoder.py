import torch
import math
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F



def seg_and_interpolate(seq_features, seg_start, seg_end, out_length):
    step_feature = seq_features[None, seg_start: seg_end].transpose(2,1)
    step_feature = F.interpolate(step_feature, 
                                  size=out_length, 
                                  mode="linear", 
                                  align_corners=True)
    step_feature = step_feature.transpose(2,1)
    return step_feature


def var_length_dpm_decoder(D_ind, step_num, seq_features1, seq_features2):
    FIX_LEN = 5
    (B, T, _, _), device = D_ind.shape, seq_features1.device
    batched_seg_list1 = []
    batched_seg_list2 = []
    batched_step_list1 = []
    batched_step_list2 = []
    
    for batch in range(B):
        seg1_list = []
        seg2_list = []
        i, j, a, b = T-1, T-1, T-1, T-1 
        k = step_num[batch].item()
        
        step_list1 = []
        step_list2 = []
        
        video1_end = T
        video2_end = T
        
        while k > 0:
            ind = D_ind[batch, k, i, j].item()
            a = ind // T
            b = ind % T
            
            video1_start = a + 1
            video2_start = b + 1
            step_feature1 = seg_and_interpolate(seq_features1[batch], video1_start, video1_end, FIX_LEN)
            step_feature2 = seg_and_interpolate(seq_features2[batch], video2_start, video2_end, FIX_LEN)
            step_list1.insert(0, step_feature1)
            step_list2.insert(0, step_feature2)
            video1_end = video1_start
            video2_end = video2_start
        
            seg1_list.insert(0, a)
            seg2_list.insert(0, b)
            i, j, k = a, b, k-1
        
        video1_start = 0
        video2_start = 0
        step_feature1 = seg_and_interpolate(seq_features1[batch], video1_start, video1_end, FIX_LEN)
        step_feature2 = seg_and_interpolate(seq_features2[batch], video2_start, video2_end, FIX_LEN)
        step_list1.insert(0, step_feature1)
        step_list2.insert(0, step_feature2)
        
        step_features1 = torch.stack(step_list1, dim=1)
        step_features2 = torch.stack(step_list2, dim=1)
        seg_tensor1 = torch.tensor(seg1_list, device=device)
        seg_tensor2 = torch.tensor(seg2_list, device=device)
        
        batched_step_list1.append(step_features1[0])
        batched_step_list2.append(step_features2[0])
        
        batched_seg_list1.append(seg_tensor1)
        batched_seg_list2.append(seg_tensor2)
        
    return batched_step_list1, batched_step_list2, batched_seg_list1, batched_seg_list2


def fix_length_dpm_decoder(D_ind, length):
    B, T, _, _ = D_ind.shape
    video_seg1, video_seg2 = [torch.full((B, 1), 0, dtype=torch.long, device=D_ind.device)]*2
    k = length
    i, j, a, b = [torch.full((B, 1), T-1, dtype=torch.long, device=D_ind.device)]*4

    while k > 0:
        ind = D_ind[range(B), k, i.squeeze(), j.squeeze()][:, None]
        a = ind // T
        b = ind % T
        if k == length:
            video_seg1 = a
            video_seg2 = b
        else:
            video_seg1 = torch.cat([a, video_seg1], dim=-1)
            video_seg2 = torch.cat([b, video_seg2], dim=-1)
        i, j, k = a, b, k-1
    
    return video_seg1 + 1, video_seg2 + 1

def fix_length_dpm_strong_decoder(D_ind, D_block, length):
    B, T, _, _ = D_ind.shape
    video_seg1, video_seg2 = [torch.full((B, 1), 0, dtype=torch.long, device=D_ind.device)]*2
    k = length
    i, j, a, b = [torch.full((B, 1), T-1, dtype=torch.long, device=D_ind.device)]*4
    
    while k > 0:
        ind = D_ind[range(B), k, i.squeeze(), j.squeeze()][:, None]
        step = D_block[range(B), k, i.squeeze(), j.squeeze()][:, None]
        
        if k == length:
            step_values = step
        else:
            step_values = torch.cat([step, step_values], dim=-1)

        a = ind // T
        b = ind % T
        if k == length:
            video_seg1 = a
            video_seg2 = b
        else:
            video_seg1 = torch.cat([a, video_seg1], dim=-1)
            video_seg2 = torch.cat([b, video_seg2], dim=-1)
        i, j, k = a, b, k-1
    
    return video_seg1 + 1, video_seg2 + 1, step_values


def negative_aware_step_decoder(D_ind, length, total_step, match_step):
    allowed_jump = total_step - match_step
    B, _, T, _, _ = D_ind.shape
    video_seg1, video_seg2 = [torch.full((B, 1), 0, dtype=torch.long, device=D_ind.device)]*2
    k = length
    i, j, a, b = [torch.full((B, 1), T-1, dtype=torch.long, device=D_ind.device)]*4
    
    ind = D_ind[range(B), allowed_jump, k, i.squeeze(), j.squeeze()][:, None]
    
    while k > 0:
        e = ind // (T * T)
        a = ind % (T * T) // T
        b = ind % (T * T) % T
        
        if k == length:
            video_seg1 = a
            video_seg2 = b
        else:
            video_seg1 = torch.cat([a, video_seg1], dim=-1)
            video_seg2 = torch.cat([b, video_seg2], dim=-1)
            
        i, j, k = a, b, k-1
        ind = D_ind[range(B), e.squeeze(), k, i.squeeze(), j.squeeze()][:, None]
    
    return video_seg1, video_seg2


def negative_delete_step_decoder(D_ind, length, allowed_jump):
    B, _, T, _, _ = D_ind.shape
    video_seg1, video_seg2 = [torch.full((B, 1), 0, dtype=torch.long, device=D_ind.device)]*2
    k = length
    i, j, a, b = [torch.full((B, 1), T-1, dtype=torch.long, device=D_ind.device)]*4
    
    ind = D_ind[range(B), allowed_jump, k, i.squeeze(), j.squeeze()][:, None]
    
    while k > 0:
        e = ind // (T * T)
        a = ind % (T * T) // T
        b = ind % (T * T) % T
        
        if allowed_jump == e: # no jump
            if k == length:
                video_seg1 = a
                video_seg2 = b
            else:
                video_seg1 = torch.cat([a, video_seg1], dim=-1)
                video_seg2 = torch.cat([b, video_seg2], dim=-1)
                
        i, j, k, allowed_jump = a, b, k-1, e
        ind = D_ind[range(B), e.squeeze(), k, i.squeeze(), j.squeeze()][:, None]
    
    return video_seg1, video_seg2
