import torch
import torch.nn.functional as F
import math
import cv2
import numpy as np
import os

from utils.colormap import _COLORS


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
t_mean = torch.FloatTensor(mean).view(3,1,1).expand(3, 180, 320)
t_std = torch.FloatTensor(std).view(3,1,1).expand(3, 180, 320)


def visualize_consist_step(frames1, frames2, video_seg1, video_seg2):
    root = 'vis_results/'
    if not os.path.exists(root):
        os.makedirs(root)
        os.makedirs(root + 'video1')
        os.makedirs(root + 'video2')
        
    step_num = video_seg1.shape[0]
    T, device = frames1.shape[-1], video_seg1.device
    
    video_seg1 = torch.Tensor.tolist(video_seg1)
    video_seg2 = torch.Tensor.tolist(video_seg2)
    
    video_seg1.append(15)
    video_seg2.append(15)
    
    output_step1 = torch.zeros((T,), device=device)
    output_step2 = torch.zeros((T,), device=device)
    
    for label in range(step_num):
        step_start, step_end = video_seg1[label], video_seg1[label+1]
        output_step1[step_start: step_end+1] = label
    
    for label in range(step_num):
        step_start, step_end = video_seg2[label], video_seg2[label+1]
        output_step2[step_start: step_end+1] = label

    frame_list1 = []
    for i in range(frames1.shape[-1]):
        color = _COLORS[ int(output_step1[i].item()) ]
        frame = frames1[..., i]
        frame = frame.clone().detach().to(torch.device('cpu'))
        frame = frame * t_std + t_mean
        frame = frame.data.cpu().float().numpy().transpose(1,2,0)
        frame = cv2.cvtColor(frame * 255, cv2.COLOR_RGB2BGR)
        caption = np.full((50, 320, 3), color, np.uint8)
        cv2.putText(caption, str(i), (100, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 0))
        cv2.imwrite('vis_results/video1/{}.jpg'.format(i), frame)
        frame = np.vstack((frame, caption))
        frame_list1.append(frame)

    frame_sequence1 = np.hstack(frame_list1)

    frame_list2 = []
    for i in range(frames2.shape[-1]):
        color = _COLORS[ int(output_step2[i].item()) ]
        frame = frames2[..., i]
        frame = frame.clone().detach().to(torch.device('cpu'))
        frame = frame * t_std + t_mean
        frame = frame.data.cpu().float().numpy().transpose(1,2,0)
        frame = cv2.cvtColor(frame * 255, cv2.COLOR_RGB2BGR)
        caption = np.full((50, 320, 3), color, np.uint8)
        cv2.putText(caption, str(i), (100, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 0))
        cv2.imwrite('vis_results/video2/{}.jpg'.format(i), frame)
        frame = np.vstack((frame, caption))
        frame_list2.append(frame)

    frame_sequence2 = np.hstack(frame_list2)
    frame_sequence = np.vstack((frame_sequence1, frame_sequence2))
    cv2.imwrite('vis_results/consistent_step.jpg', frame_sequence)

    return


def frame2step_matching_loss(seq_features1, seq_features2, video_seg1, video_seg2):
    blank1 = seq_features1[:, :1]
    seq_features1 = seq_features1[:, 1:]
    blank2 = seq_features2[:, :1]
    seq_features2 = seq_features2[:, 1:]
    (B, T, C), device = seq_features1.shape, seq_features1.device

    sparse_seq_features1 = seq_features1[torch.arange(B, device=device).unsqueeze(-1), video_seg1]
    sparse_seq_features1 = torch.cat((blank1, sparse_seq_features1), dim=1)
    
    sparse_seq_features2 = seq_features2[torch.arange(B, device=device).unsqueeze(-1), video_seg2]
    sparse_seq_features2 = torch.cat((blank2, sparse_seq_features2), dim=1)
    
    S = sparse_seq_features1.shape[1] - 1

    prob1_2 = (torch.einsum('bic,bjc->bij', seq_features1, sparse_seq_features2) / math.sqrt(C)).transpose(0, 1).log_softmax(2)
    prob2_1 = (torch.einsum('bic,bjc->bij', seq_features2, sparse_seq_features1) / math.sqrt(C)).transpose(0, 1).log_softmax(2)
    
    target = (torch.arange(S)+1).unsqueeze(0).expand(B, -1)
    input_lengths = torch.full((B, ), T, dtype=torch.long, device=device)
    target_lengths = torch.full((B, ), S, dtype=torch.long, device=device)

    loss = F.ctc_loss(prob1_2, target, input_lengths, target_lengths, reduction='none') / target_lengths \
         + F.ctc_loss(prob2_1, target, input_lengths, target_lengths, reduction='none') / target_lengths
    
    return loss.mean(0)


def frame2step_distance(seq_features1, seq_features2, video_seg1, video_seg2):
    blank1 = seq_features1[:, :1]
    seq_features1 = seq_features1[:, 1:]
    blank2 = seq_features2[:, :1]
    seq_features2 = seq_features2[:, 1:]
    (B, T, C), device = seq_features1.shape, seq_features1.device

    sparse_seq_features1 = seq_features1[torch.arange(B, device=device).unsqueeze(-1), video_seg1]
    sparse_seq_features1 = torch.cat((blank1, sparse_seq_features1), dim=1)
    
    sparse_seq_features2 = seq_features2[torch.arange(B, device=device).unsqueeze(-1), video_seg2]
    sparse_seq_features2 = torch.cat((blank2, sparse_seq_features2), dim=1)
    
    S = sparse_seq_features1.shape[1] - 1

    prob1_2 = (torch.einsum('bic,bjc->bij', seq_features1, sparse_seq_features2) / math.sqrt(C)).transpose(0, 1).log_softmax(2)
    prob2_1 = (torch.einsum('bic,bjc->bij', seq_features2, sparse_seq_features1) / math.sqrt(C)).transpose(0, 1).log_softmax(2)
    
    target = (torch.arange(S)+1).unsqueeze(0).expand(B, -1)
    input_lengths = torch.full((B, ), T, dtype=torch.long, device=device)
    target_lengths = torch.full((B, ), S, dtype=torch.long, device=device)

    ctc_distances = F.ctc_loss(prob1_2, target, input_lengths, target_lengths, reduction='none') / target_lengths \
                  + F.ctc_loss(prob2_1, target, input_lengths, target_lengths, reduction='none') / target_lengths
    
    return ctc_distances

def consist_step_mining_train(seq_features1, seq_features2, step_num):
    seq_features1 = seq_features1[:, 1:]
    seq_features2 = seq_features2[:, 1:]
    (B, T, C), device = seq_features1.shape, seq_features1.device
    
    pred = (torch.einsum('bic,bjc->bij', seq_features1, seq_features2) / math.sqrt(C)).softmax(-1)
    pred = pred.cumsum(-2).cumsum(-1)
    
    D = torch.zeros((B, T, T, T), device=device)
    D_ind = torch.zeros((B, T, T, T), dtype=torch.long, device=device)
    
    D[:, 0] = pred / torch.ones_like(pred).cumsum(-2).cumsum(-1)
    
    area = torch.ones_like(pred).cumsum(-2).cumsum(-1)
    area = (area[:, :, :, None, None] - area[:, :, None, None, :] \
            - area.transpose(1,2)[:, None, :, :, None] + area[:, None, None, :, :])
    
    block_mat = (pred[:, :, :, None, None] - pred[:, :, None, None, :] \
                - pred.transpose(1,2)[:, None, :, :, None] + pred[:, None, None, :, :])
    
    top, left, bottom, right = torch.meshgrid(*[torch.arange(T, device=device)]*4)
    area = area.clamp_min(1)

    block_mat = block_mat.masked_fill(((bottom >= top) | (right >= left)).unsqueeze(0), float('-inf')) / area

    for k in range(1, T):
        tmp = D[:, k-1, None, None, :, :] + block_mat
        D[:, k] = tmp.flatten(3).max(-1).values
        D_ind[:, k] = tmp.flatten(3).max(-1).indices
    
    segment1, segment2 = [torch.full((B, 1), T, dtype=torch.long, device=device)]*2
    k = step_num
    i, j, a, b = [torch.full((B, 1), T-1, dtype=torch.long, device=device)]*4
    
    while k > 0:
        ind = D_ind[range(B), k, i.squeeze(), j.squeeze()][:, None]
        a = ind // T
        b = ind % T
        segment1 = torch.cat([a, segment1], dim=-1)
        segment2 = torch.cat([b, segment2], dim=-1)
        i, j, k = a, b, k-1
        
    final_result = D[:, :, T-1, T-1]
    return -(final_result.max(dim=-1).values).mean(), segment1[:, :-1] + 1, segment2[:, :-1] + 1


def consist_step_mining_inference(seq_features1, seq_features2, step_num):
    seq_features1 = seq_features1[:, 1:]
    seq_features2 = seq_features2[:, 1:]
    (B, T, C), device = seq_features1.shape, seq_features1.device
    
    pred = torch.cosine_similarity(seq_features1.unsqueeze(2), seq_features2.unsqueeze(1), dim=-1)
    pred = pred.cumsum(-2).cumsum(-1)
    
    D = torch.zeros((B, T, T, T), device=device)
    D_ind = torch.zeros((B, T, T, T), dtype=torch.long, device=device)
    
    D[:, 0] = pred / torch.ones_like(pred).cumsum(-2).cumsum(-1)
    
    area = torch.ones_like(pred).cumsum(-2).cumsum(-1)
    area = (area[:, :, :, None, None] - area[:, :, None, None, :] \
            - area.transpose(1,2)[:, None, :, :, None] + area[:, None, None, :, :])
    
    block_mat = (pred[:, :, :, None, None] - pred[:, :, None, None, :] \
                - pred.transpose(1,2)[:, None, :, :, None] + pred[:, None, None, :, :])
    
    top, left, bottom, right = torch.meshgrid(*[torch.arange(T, device=device)]*4)
    area = area.clamp_min(1).sqrt()

    block_mat = block_mat.masked_fill(((bottom >= top) | (right >= left)).unsqueeze(0), float('-inf')) / area

    for k in range(1, T):
        tmp = D[:, k-1, None, None, :, :] + block_mat
        D[:, k] = tmp.flatten(3).max(-1).values
        D_ind[:, k] = tmp.flatten(3).max(-1).indices
    
    segment1, segment2 = [torch.full((B, 1), T, dtype=torch.long, device=device)]*2
    k = step_num
    i, j, a, b = [torch.full((B, 1), T-1, dtype=torch.long, device=device)]*4
    
    while k > 0:
        ind = D_ind[range(B), k, i.squeeze(), j.squeeze()][:, None]
        a = ind // T
        b = ind % T
        segment1 = torch.cat([a, segment1], dim=-1)
        segment2 = torch.cat([b, segment2], dim=-1)
        i, j, k = a, b, k-1
        
    return segment1[:, :-1] + 1, segment2[:, :-1] + 1



def elastic_step_mining_2jump(seq_features1, seq_features2, max_jumps):
    seq_features1 = seq_features1[:, 1:]
    seq_features2 = seq_features2[:, 1:]
    
    (B, T, C), R, device = seq_features1.shape, max_jumps+1, seq_features1.device
    pred = torch.cosine_similarity(seq_features1.unsqueeze(2), seq_features2.unsqueeze(1), dim=-1)
    pred = pred.cumsum(-2).cumsum(-1)
    
    D = torch.zeros((B, R, T, T, T), device=device)
    D_ind = torch.zeros((B, R, T, T, T), dtype=torch.long, device=device)
    
    D[:, 0, 0] = pred / torch.ones_like(pred).cumsum(-2).cumsum(-1)
    
    area = torch.ones_like(pred).cumsum(-2).cumsum(-1)
    area = area[:, :, :, None, None] - area[:, :, None, None, :] \
            - area.transpose(1,2)[:, None, :, :, None] + area[:, None, None, :, :]
            
    block_mat = pred[:, :, :, None, None] - pred[:, :, None, None, :] \
                - pred.transpose(1,2)[:, None, :, :, None] + pred[:, None, None, :, :]
    
    i, j, a, b = torch.meshgrid(*[torch.arange(T, device=seq_features1.device)]*4)
    area = area.clamp_min(1).sqrt()

    block_mat = block_mat.masked_fill(((a >= i) | (b >= j)).unsqueeze(0), float('-inf')) / area
    empty_mat = torch.zeros_like(block_mat).masked_fill(((a >= i) | (b >= j)).unsqueeze(0), float('-inf'))
    
    for k in range(1, T):
        tmp_0 = D[:, 0, k-1, None, None, :, :]
        tmp_1 = D[:, 1, k-1, None, None, :, :]
        tmp_2 = D[:, 2, k-1, None, None, :, :]
        
        if k == 1:
            D01_candidate = (tmp_0 * k + block_mat) / (k+1)
            D[:, 0, 1] = D01_candidate.flatten(3).max(-1).values
            D_ind[:, 0, 1] = D01_candidate.flatten(3).max(-1).indices
            
            D11_candidate = tmp_0 + empty_mat
            D[:, 1, 1] = D11_candidate.flatten(3).max(-1).values
            D_ind[:, 1, 1] = D11_candidate.flatten(3).max(-1).indices
        
        if k == 2:
            D02_candidate = (tmp_0 * k + block_mat) / (k+1)
            D[:, 0, 2] = D02_candidate.flatten(3).max(-1).values
            D_ind[:, 0, 2] = D02_candidate.flatten(3).max(-1).indices
            
            D01_candidate = tmp_0 + empty_mat
            D11_candidate = (tmp_1 * (k-1) + block_mat) / k
            D12_candidate = torch.cat([D01_candidate.flatten(3), D11_candidate.flatten(3)], dim=-1)
            D[:, 1, 2] = D12_candidate.flatten(3).max(-1).values
            D_ind[:, 1, 2] = D12_candidate.flatten(3).max(-1).indices
            
            D22_candidate = tmp_1 + empty_mat
            D[:, 2, 2] = D22_candidate.flatten(3).max(-1).values
            D_ind[:, 2, 2] = D22_candidate.flatten(3).max(-1).indices + 256
        
        else:
            D0k_candidate = (tmp_0 * k + block_mat) / (k+1)
            D[:, 0, k] = D0k_candidate.flatten(3).max(-1).values
            D_ind[:, 0, k] = D0k_candidate.flatten(3).max(-1).indices
            
            D0k1_candidate = tmp_0 + empty_mat
            D1k1_candidate = (tmp_1 * (k-1) + block_mat) / k
            D1k_candidate = torch.cat([D0k1_candidate.flatten(3), D1k1_candidate.flatten(3)], dim=-1)
            D[:, 1, k] = D1k_candidate.max(-1).values
            D_ind[:, 1, k] = D1k_candidate.max(-1).indices
            
            D1k1_candidate = tmp_1 + empty_mat
            D2k1_candidate = (tmp_2 * (k-2) + block_mat) / (k-1)
            D2k_candidate = torch.cat([D1k1_candidate.flatten(3), D2k1_candidate.flatten(3)], dim=-1)
            D[:, 2, k] = D2k_candidate.max(-1).values
            D_ind[:, 2, k] = D2k_candidate.max(-1).indices + 256
    
    return D_ind


def elastic_step_mining(seq_features1, seq_features2, max_jumps):
    max_jumps = 2
    seq_features1 = seq_features1[:, 1:]
    seq_features2 = seq_features2[:, 1:]
    
    (B, T, C), R, device = seq_features1.shape, max_jumps+1, seq_features1.device
    pred = torch.cosine_similarity(seq_features1.unsqueeze(2), seq_features2.unsqueeze(1), dim=-1)
    pred = pred.cumsum(-2).cumsum(-1)
    
    D = torch.zeros((B, R, T, T, T), device=device)
    D_ind = torch.zeros((B, R, T, T, T), dtype=torch.long, device=device)
    
    D[:, 0, 0] = pred / torch.ones_like(pred).cumsum(-2).cumsum(-1)
    
    area = torch.ones_like(pred).cumsum(-2).cumsum(-1)
    area = area[:, :, :, None, None] - area[:, :, None, None, :] \
            - area.transpose(1,2)[:, None, :, :, None] + area[:, None, None, :, :]
            
    block_mat = pred[:, :, :, None, None] - pred[:, :, None, None, :] \
                - pred.transpose(1,2)[:, None, :, :, None] + pred[:, None, None, :, :]
    
    i, j, a, b = torch.meshgrid(*[torch.arange(T, device=seq_features1.device)]*4)
    area = area.clamp_min(1).sqrt()

    block_mat = block_mat.masked_fill(((a >= i) | (b >= j)).unsqueeze(0), float('-inf')) / area
    empty_mat = torch.zeros_like(block_mat).masked_fill(((a >= i) | (b >= j)).unsqueeze(0), float('-inf'))
    
    for k in range(1, T):
        tmp_0 = D[:, 0, k-1, None, None, :, :]
        tmp_1 = D[:, 1, k-1, None, None, :, :]
        tmp_2 = D[:, 2, k-1, None, None, :, :]
        # tmp_3 = D[:, 3, k-1, None, None, :, :]
        # tmp_4 = D[:, 4, k-1, None, None, :, :]
        
        if k == 1:
            D01_candidate = (tmp_0 * k + block_mat) / (k+1)
            D[:, 0, 1] = D01_candidate.flatten(3).max(-1).values
            D_ind[:, 0, 1] = D01_candidate.flatten(3).max(-1).indices
            
            D11_candidate = tmp_0 + empty_mat
            D[:, 1, 1] = D11_candidate.flatten(3).max(-1).values
            D_ind[:, 1, 1] = D11_candidate.flatten(3).max(-1).indices
        
        if k == 2:
            D02_candidate = (tmp_0 * k + block_mat) / (k+1)
            D[:, 0, 2] = D02_candidate.flatten(3).max(-1).values
            D_ind[:, 0, 2] = D02_candidate.flatten(3).max(-1).indices
            
            D01_candidate = tmp_0 + empty_mat
            D11_candidate = (tmp_1 * (k-1) + block_mat) / k
            D12_candidate = torch.cat([D01_candidate.flatten(3), D11_candidate.flatten(3)], dim=-1)
            D[:, 1, 2] = D12_candidate.flatten(3).max(-1).values
            D_ind[:, 1, 2] = D12_candidate.flatten(3).max(-1).indices
            
            D22_candidate = tmp_1 + empty_mat
            D[:, 2, 2] = D22_candidate.flatten(3).max(-1).values
            D_ind[:, 2, 2] = D22_candidate.flatten(3).max(-1).indices + 256
            
        # if k == 3:
        #     D02_candidate = (tmp_0 * k + block_mat) / (k+1)
        #     D[:, 0, 2] = D02_candidate.flatten(3).max(-1).values
        #     D_ind[:, 0, 2] = D02_candidate.flatten(3).max(-1).indices
            
        #     D01_candidate = tmp_0 + empty_mat
        #     D11_candidate = (tmp_1 * (k-1) + block_mat) / k
        #     D12_candidate = torch.cat([D01_candidate.flatten(3), D11_candidate.flatten(3)], dim=-1)
        #     D[:, 1, 2] = D12_candidate.flatten(3).max(-1).values
        #     D_ind[:, 1, 2] = D12_candidate.flatten(3).max(-1).indices
            
        #     D22_candidate = tmp_1 + empty_mat
        #     D[:, 2, 2] = D22_candidate.flatten(3).max(-1).values
        #     D_ind[:, 2, 2] = D22_candidate.flatten(3).max(-1).indices + 256
        
        # if k == 4:
        #     D02_candidate = (tmp_0 * k + block_mat) / (k+1)
        #     D[:, 0, 2] = D02_candidate.flatten(3).max(-1).values
        #     D_ind[:, 0, 2] = D02_candidate.flatten(3).max(-1).indices
            
        #     D01_candidate = tmp_0 + empty_mat
        #     D11_candidate = (tmp_1 * (k-1) + block_mat) / k
        #     D12_candidate = torch.cat([D01_candidate.flatten(3), D11_candidate.flatten(3)], dim=-1)
        #     D[:, 1, 2] = D12_candidate.flatten(3).max(-1).values
        #     D_ind[:, 1, 2] = D12_candidate.flatten(3).max(-1).indices
            
        #     D22_candidate = tmp_1 + empty_mat
        #     D[:, 2, 2] = D22_candidate.flatten(3).max(-1).values
        #     D_ind[:, 2, 2] = D22_candidate.flatten(3).max(-1).indices + 256
        
        else:
            D0k_candidate = (tmp_0 * k + block_mat) / (k+1)
            D[:, 0, k] = D0k_candidate.flatten(3).max(-1).values
            D_ind[:, 0, k] = D0k_candidate.flatten(3).max(-1).indices
            
            D0k1_candidate = tmp_0 + empty_mat
            D1k1_candidate = (tmp_1 * (k-1) + block_mat) / k
            D1k_candidate = torch.cat([D0k1_candidate.flatten(3), D1k1_candidate.flatten(3)], dim=-1)
            D[:, 1, k] = D1k_candidate.max(-1).values
            D_ind[:, 1, k] = D1k_candidate.max(-1).indices
            
            D1k1_candidate = tmp_1 + empty_mat
            D2k1_candidate = (tmp_2 * (k-2) + block_mat) / (k-1)
            D2k_candidate = torch.cat([D1k1_candidate.flatten(3), D2k1_candidate.flatten(3)], dim=-1)
            D[:, 2, k] = D2k_candidate.max(-1).values
            D_ind[:, 2, k] = D2k_candidate.max(-1).indices + 256
    
    return D_ind



def matched_step_extractor(D_ind, total_step, match_step):
    allowed_jump = total_step - match_step
    (B, _, T, _, _), device = D_ind.shape, D_ind.device
    video_seg1, video_seg2 = [torch.full((B, 1), T, dtype=torch.long, device=device)]*2
    k = total_step
    i, j, a, b = [torch.full((B, 1), T-1, dtype=torch.long, device=device)]*4
    ind = D_ind[range(B), allowed_jump, k, i.squeeze(), j.squeeze()][:, None]
    
    while k > 0:
        e = ind // (T * T)
        a = ind % (T * T) // T
        b = ind % (T * T) % T
        
        if allowed_jump == e: # no jump
            video_seg1 = torch.cat([a, video_seg1], dim=-1)
            video_seg2 = torch.cat([b, video_seg2], dim=-1)
                
        i, j, k, allowed_jump = a, b, k-1, e
        ind = D_ind[range(B), e.squeeze(), k, i.squeeze(), j.squeeze()][:, None]
    
    return video_seg1[:, :-1] + 1, video_seg2[:, :-1] + 1