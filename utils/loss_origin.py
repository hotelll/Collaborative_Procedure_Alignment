import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn import CrossEntropyLoss
import math

def compute_cls_loss(pred, labels, use_cosface=False):
    if use_cosface:
        # CosFace Loss
        s, m = 30.0, 0.4
        cos_value = torch.diagonal(pred.transpose(0, 1)[labels])
        numerator = s * (cos_value - m)
        excl = torch.cat([torch.cat((pred[i, :y], pred[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(s * excl), dim=1)
        L = numerator - torch.log(denominator)
        loss = -torch.mean(L)
    else:
        # Softmax Loss
        criterion = CrossEntropyLoss().cuda()
        loss = criterion(pred, labels)

    return loss


def frame_blank_align_loss(seq_features1, seq_features2, step_index):
    seq_features1 = seq_features1[:, 1:]
    blank2 = seq_features2[:, :1]
    seq_features2 = seq_features2[:, 1:]
    (B, T, C), device = seq_features1.shape, seq_features1.device
    
    step_num = step_index.shape[1] 
    K = 2 * step_num + 1
    sparse_seq_features2 = torch.cat((blank2, seq_features2[torch.arange(B, device=seq_features1.device).unsqueeze(-1), step_index, :]), dim=1)
    pred = (torch.einsum('bic,bjc->bij', seq_features1, sparse_seq_features2) / math.sqrt(C)).log_softmax(-1)

    D_pre = torch.full((B, K), fill_value=float('-99999999'), device=device)
    D_pre[:, 0] = pred[:, 0, 0]
    D_pre[:, 1] = pred[:, 0, 1]
    
    for t in range(1, T):
        D_cur = torch.full((B, K), fill_value=float('-99999999'), device=device)
        D_cur[:, 0] = D_pre[:, 0] + pred[:, t, 0]
        D_cur[:, 1] = torch.logsumexp(torch.stack([D_pre[:, 0], D_pre[:, 1]]), dim=0) + pred[:, t, 1]
        
        # blank term
        blank_pre_ind = torch.arange(1, K, 2)[None, :].repeat(B, 1)
        blank_pre = D_pre[torch.arange(B, device=device).unsqueeze(-1), blank_pre_ind]
        
        blank_cur_ind = torch.arange(2, K, 2)[None, :].repeat(B, 1)
        blank_cur = D_pre[torch.arange(B, device=device).unsqueeze(-1), blank_cur_ind]
        
        blank_log_prob = torch.logsumexp(torch.stack([blank_pre, blank_cur]), dim=0)
        D_cur[:, 2:][:, ::2] = blank_log_prob + pred[:, t, 0][:, None].repeat(1, blank_log_prob.shape[-1])
        
        # step term
        step_prepre_ind = torch.arange(1, K, 2)[None, :-1].repeat(B, 1)
        step_prepre = D_pre[torch.arange(B, device=device).unsqueeze(-1), step_prepre_ind]
        
        step_pre_ind = torch.arange(2, K, 2)[None, :-1].repeat(B, 1)
        step_pre = D_pre[torch.arange(B, device=device).unsqueeze(-1), step_pre_ind]
        
        step_cur_ind = torch.arange(3, K, 2)[None, :].repeat(B, 1)
        step_cur = D_pre[torch.arange(B, device=device).unsqueeze(-1), step_cur_ind]
        
        step_log_prob = torch.logsumexp(torch.stack([step_prepre, step_pre, step_cur]), dim=0)
        D_cur[:, 2:][:, 1::2] = step_log_prob + pred[:, t, 2:]
        D_pre = D_cur

    fsa_distance = -torch.logsumexp(D_cur[:, -2:], dim=-1) / step_num
    loss = fsa_distance.mean(0)
    
    return loss


def frame_blank_align_distance(seq_features1, seq_features2, step_index):
    seq_features1 = seq_features1[:, 1:]
    blank2 = seq_features2[:, :1]
    seq_features2 = seq_features2[:, 1:]
    (B, T, C), device = seq_features1.shape, seq_features1.device
    
    step_num = step_index.shape[1] 
    K = 2 * step_num + 1
    sparse_seq_features2 = torch.cat((blank2, seq_features2[torch.arange(B, device=seq_features1.device).unsqueeze(-1), step_index, :]), dim=1)
    pred = (torch.einsum('bic,bjc->bij', seq_features1, sparse_seq_features2) / math.sqrt(C)).log_softmax(-1)

    D_pre = torch.full((B, K), fill_value=float('-99999999'), device=device)
    D_pre[:, 0] = pred[:, 0, 0]
    D_pre[:, 1] = pred[:, 0, 1]
    
    for t in range(1, T):
        D_cur = torch.full((B, K), fill_value=float('-99999999'), device=device)
        D_cur[:, 0] = D_pre[:, 0] + pred[:, t, 0]
        D_cur[:, 1] = torch.logsumexp(torch.stack([D_pre[:, 0], D_pre[:, 1]]), dim=0) + pred[:, t, 1]
        
        # blank term
        blank_pre_ind = torch.arange(1, K, 2)[None, :].repeat(B, 1)
        blank_pre = D_pre[torch.arange(B, device=device).unsqueeze(-1), blank_pre_ind]
        
        blank_cur_ind = torch.arange(2, K, 2)[None, :].repeat(B, 1)
        blank_cur = D_pre[torch.arange(B, device=device).unsqueeze(-1), blank_cur_ind]
        
        blank_log_prob = torch.logsumexp(torch.stack([blank_pre, blank_cur]), dim=0)
        D_cur[:, 2:][:, ::2] = blank_log_prob + pred[:, t, 0][:, None].repeat(1, blank_log_prob.shape[-1])
        
        # step term
        step_prepre_ind = torch.arange(1, K, 2)[None, :-1].repeat(B, 1)
        step_prepre = D_pre[torch.arange(B, device=device).unsqueeze(-1), step_prepre_ind]
        
        step_pre_ind = torch.arange(2, K, 2)[None, :-1].repeat(B, 1)
        step_pre = D_pre[torch.arange(B, device=device).unsqueeze(-1), step_pre_ind]
        
        step_cur_ind = torch.arange(3, K, 2)[None, :].repeat(B, 1)
        step_cur = D_pre[torch.arange(B, device=device).unsqueeze(-1), step_cur_ind]
        
        step_log_prob = torch.logsumexp(torch.stack([step_prepre, step_pre, step_cur]), dim=0)
        D_cur[:, 2:][:, 1::2] = step_log_prob + pred[:, t, 2:]
        D_pre = D_cur

    fsa_distance = -torch.logsumexp(D_cur[:, -2:], dim=-1) / step_num
    
    return fsa_distance


def consist_step_mining(seq_features1, seq_features2, step_num):
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
        a = torch.div(ind, T, rounding_mode='trunc')
        b = ind % T
        segment1 = torch.cat([a, segment1], dim=-1)
        segment2 = torch.cat([b, segment2], dim=-1)
        i, j, k = a, b, k-1
        
    final_result = D[:, :, T-1, T-1]
    
    video_seg1 = segment1[:, :-1] + 1
    video_seg2 = segment2[:, :-1] + 1
    
    loss_step = -final_result.max(dim=-1).values.mean()
    
    return loss_step, video_seg1, video_seg2



def frame2step_wblank_dist(frame_feats1, step_feats2, pair_labels):
    B = frame_feats1.shape[0]
    dists = []
    for batch in range(B):
        frame_feat1 = frame_feats1[batch]
        step_feat2 = step_feats2[batch]
        pair_label = pair_labels[batch]

        frame_dist = single_align_loss(frame_feat1, step_feat2, pair_label)
        dists.append(frame_dist)
        
    return torch.stack(dists, dim=-1)


def batched_step_comparison(step_feats1, step_feats2):
    B = len(step_feats1)
    dists = []
    for batch in range(B):
        step_feat1 = step_feats1[batch]
        step_feat2 = step_feats2[batch]
        B, C = step_feat1.shape
        # step_dist = (torch.einsum('ic,jc->ij', step_feat1, step_feat2) / math.sqrt(C)).diagonal()
        step_dist = torch.cosine_similarity(step_feat1, step_feat2)
        dists.append(step_dist)
        
    return dists


def single_align_loss(frame_features1, step_features2):
    # frame_features1 的第一个是blank feature
    frame_features1 = frame_features1[1:]
    blank = frame_features1[:1]
    (T, C), device = frame_features1.shape, frame_features1.device
    step_num = step_features2.shape[0]
    K = 2 * step_num + 1
    step_features2_with_blank = torch.cat((blank, step_features2), dim=0)
    
    pred = (torch.einsum('ic,jc->ij', frame_features1, step_features2_with_blank) / math.sqrt(C)).log_softmax(-1)
    
    D_pre = torch.full((K,), fill_value=float('-99999999'), device=device)
    D_pre[0] = pred[0, 0]
    D_pre[1] = pred[0, 1]
    
    for t in range(1, T):
        D_cur = torch.full((K,), fill_value=float('-99999999'), device=device)
        D_cur[0] = D_pre[0] + pred[t, 0]
        D_cur[1] = torch.logsumexp(torch.stack([D_pre[0], D_pre[1]]), dim=0) + pred[t, 1]
        
        # blank term
        blank_pre_ind = torch.arange(1, K, 2)[None, :]
        blank_pre = D_pre[blank_pre_ind]
        
        blank_cur_ind = torch.arange(2, K, 2)[None, :]
        blank_cur = D_pre[blank_cur_ind]
        
        blank_log_prob = torch.logsumexp(torch.stack([blank_pre, blank_cur]), dim=0)
        D_cur[2:][::2] = blank_log_prob + pred[t, 0].repeat(1, blank_log_prob.shape[-1])
        
        # step term
        step_prepre_ind = torch.arange(1, K, 2)[None, :-1]
        step_prepre = D_pre[step_prepre_ind]
        
        step_pre_ind = torch.arange(2, K, 2)[None, :-1]
        step_pre = D_pre[step_pre_ind]
        
        step_cur_ind = torch.arange(3, K, 2)[None, :]
        step_cur = D_pre[step_cur_ind]
        
        step_log_prob = torch.logsumexp(torch.stack([step_prepre, step_pre, step_cur]), dim=0)
        D_cur[2:][1::2] = step_log_prob + pred[t, 2:]
        D_pre = D_cur

    fsa_distance = -torch.logsumexp(D_cur[-2:], dim=-1) / K
    
    return fsa_distance

