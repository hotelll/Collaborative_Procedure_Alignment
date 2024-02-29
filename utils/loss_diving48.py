import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn import CrossEntropyLoss
import math


def compute_cls_loss(pred, labels, use_cosface=False):
    if use_cosface:
        # CosFace Loss

        s = 30.0
        m = 0.4

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


def compute_interctc_withneg_loss(seq_features1, seq_features2, pair_labels, margin, segment1=None, segment2=None):
    blank1 = seq_features1[:, :1]
    seq_features1 = seq_features1[:, 1:]
    blank2 = seq_features2[:, :1]
    seq_features2 = seq_features2[:, 1:]
    (B, T, C), device = seq_features1.shape, seq_features1.device

    if segment1 != None and segment2 != None:
        sparse_seq_features1 = seq_features1[torch.arange(B, device=device).unsqueeze(-1), segment1]
        sparse_seq_features1 = torch.cat((blank1, sparse_seq_features1), dim=1)
        
        sparse_seq_features2 = seq_features2[torch.arange(B, device=device).unsqueeze(-1), segment2]
        sparse_seq_features2 = torch.cat((blank2, sparse_seq_features2), dim=1)
        
    else:
        sparse_seq_features1 = torch.cat((blank1, seq_features1[:, [2, 3, 4, 5, 6], :]), dim=1)
        sparse_seq_features2 = torch.cat((blank2, seq_features2[:, [2, 3, 4, 5, 6], :]), dim=1)
        
    # sparse_choice1 = torch.rand(B, T, device=seq_features1.device).argsort(-1)[:, :12].sort(-1).values
    # sparse_seq_features1 = seq_features1[torch.arange(B, device=seq_features1.device).unsqueeze(-1), sparse_choice1]
    # sparse_seq_features1 = torch.cat((blank1, sparse_seq_features1), dim=1)
    
    # sparse_choice2 = torch.rand(B, T, device=seq_features2.device).argsort(-1)[:, :12].sort(-1).values
    # sparse_seq_features2 = seq_features2[torch.arange(B, device=seq_features2.device).unsqueeze(-1), sparse_choice2]
    # sparse_seq_features2 = torch.cat((blank2, sparse_seq_features2), dim=1)
    
    S = sparse_seq_features1.shape[1] - 1

    prob1_2 = (torch.einsum('bic,bjc->bij', seq_features1, sparse_seq_features2) / math.sqrt(C)).transpose(0, 1).log_softmax(2)
    prob2_1 = (torch.einsum('bic,bjc->bij', seq_features2, sparse_seq_features1) / math.sqrt(C)).transpose(0, 1).log_softmax(2)
    
    target = (torch.arange(S)+1).unsqueeze(0).expand(B, -1)
    input_lengths = torch.full((B, ), T, dtype=torch.long, device=device)
    target_lengths = torch.full((B, ), S, dtype=torch.long, device=device)

    ctc_distances = F.ctc_loss(prob1_2, target, input_lengths, target_lengths, reduction='none') / target_lengths \
        + F.ctc_loss(prob2_1, target, input_lengths, target_lengths, reduction='none') / target_lengths
    
    pair_labels = pair_labels.long() * 2 - 1

    loss = (ctc_distances * pair_labels).clamp(min=-margin).mean(0)
    return loss



def compute_interctc_distance(seq_features1, seq_features2, segment1=None, segment2=None):
    blank1 = seq_features1[:, :1]
    seq_features1 = seq_features1[:, 1:]
    blank2 = seq_features2[:, :1]
    seq_features2 = seq_features2[:, 1:]
    B, T, C = seq_features1.shape

    if segment1 != None and segment2 != None:
        sparse_seq_features1 = seq_features1[torch.arange(B, device=seq_features1.device).unsqueeze(-1), segment1]
        sparse_seq_features1 = torch.cat((blank1, sparse_seq_features1), dim=1)
        
        sparse_seq_features2 = seq_features2[torch.arange(B, device=seq_features2.device).unsqueeze(-1), segment2]
        sparse_seq_features2 = torch.cat((blank2, sparse_seq_features2), dim=1)
        
    else:
        sparse_seq_features1 = torch.cat((blank1, seq_features1[:, [1, 3, 5, 7], :]), dim=1)
        sparse_seq_features2 = torch.cat((blank2, seq_features2[:, [1, 3, 5, 7], :]), dim=1)
    
    S = sparse_seq_features1.shape[1] - 1

    prob1_2 = (torch.einsum('bic,bjc->bij', seq_features1, sparse_seq_features2) / math.sqrt(C)).transpose(0, 1).log_softmax(2)
    prob2_1 = (torch.einsum('bic,bjc->bij', seq_features2, sparse_seq_features1) / math.sqrt(C)).transpose(0, 1).log_softmax(2)
    
    target = (torch.arange(S)+1).unsqueeze(0).expand(B, -1)
    input_lengths = torch.full((B, ), T, dtype=torch.long, device=seq_features1.device)
    target_lengths = torch.full((B, ), S, dtype=torch.long, device=seq_features1.device)

    ctc_distances = F.ctc_loss(prob1_2, target, input_lengths, target_lengths, reduction='none') / target_lengths \
                  + F.ctc_loss(prob2_1, target, input_lengths, target_lengths, reduction='none') / target_lengths

    return ctc_distances


def frame2step_loss(seq_features1, seq_features2, sparse_label, margin, pair_labels):
    B, T, C = seq_features1.shape
    label_weight = pair_labels * 2 - 1
    
    sparse_seq_features2 = seq_features2[torch.arange(B, device=seq_features1.device).unsqueeze(-1), sparse_label]
    pred = (torch.einsum('bic,bjc->bij', seq_features1, sparse_seq_features2) / math.sqrt(C)).log_softmax(-1)
    
    B, T, _ = pred.shape
    ZERO_PAD = torch.zeros((B, 1), device=seq_features1.device)
    ONE_PAD = torch.ones((B, 1), device=seq_features1.device)
    S = sparse_seq_features2.shape[1]

    target = (torch.arange(S, device=seq_features1.device))
    
    D_TABLE = ONE_PAD.log()
    for t in range(T):
        D_VEC_1 = torch.logsumexp(torch.stack([D_TABLE[:, 1:t+1], D_TABLE[:, :-1][:, :t]]), 0) + pred[:, t, target[:t]]
        D_VEC_2 = D_TABLE[:, t:t+1] + pred[:, t, target[t:t+1]]
        D_TABLE = torch.cat([ZERO_PAD.log(), D_VEC_1, D_VEC_2], dim=-1)

    ctc_distance = -D_TABLE[:, S] / S
    return (ctc_distance * label_weight).clamp(min=-margin).mean(0)


def frame2step_distance(seq_features1, seq_features2, sparse_label):
    B, T, C = seq_features1.shape
    sparse_seq_features2 = seq_features2[torch.arange(B, device=seq_features1.device).unsqueeze(-1), sparse_label]

    pred = (torch.einsum('bic,bjc->bij', seq_features1, sparse_seq_features2) / math.sqrt(C)).log_softmax(-1)
    
    B, T, _ = pred.shape
    ZERO_PAD = torch.zeros((B, 1), device=seq_features1.device)
    ONE_PAD = torch.ones((B, 1), device=seq_features1.device)
    S = sparse_seq_features2.shape[1]

    target = (torch.arange(S, device=seq_features1.device))
    
    D_TABLE = ONE_PAD.log()
    for t in range(T):
        D_VEC_1 = torch.logsumexp(torch.stack([D_TABLE[:, 1:t+1], D_TABLE[:, :-1][:, :t]]), 0) + pred[:, t, target[:t]]
        D_VEC_2 = D_TABLE[:, t:t+1] + pred[:, t, target[t:t+1]]
        D_TABLE = torch.cat([ZERO_PAD.log(), D_VEC_1, D_VEC_2], dim=-1)

    ctc_distance = -D_TABLE[:, S] / S
    return ctc_distance


def frame_blank_align_loss(seq_features1, seq_features2, step_num, pair_labels):
    margin = 10
    seq_features1 = seq_features1[:, 1:]
    blank2 = seq_features2[:, :1]
    seq_features2 = seq_features2[:, 1:]
    (B, T, C), device = seq_features1.shape, seq_features1.device
    
    K = 2 * step_num + 1
    
    sparse_choice2 = torch.rand(B, T, device=device).argsort(-1)[:, :step_num].sort(-1).values
    sparse_seq_features2 = seq_features2[torch.arange(B, device=device).unsqueeze(-1), sparse_choice2]
    sparse_seq_features2 = torch.cat((blank2, sparse_seq_features2), dim=1)
    # sparse_seq_features2 = torch.cat((blank2, seq_features1[:, [5, 7, 8, 9, 11, 12], :]), dim=1)

    pred = (torch.einsum('bic,bjc->bij', seq_features1, sparse_seq_features2) / math.sqrt(C)).log_softmax(-1)

    D_pre = torch.full((B, K), fill_value=float('-100000'), device=device)
    D_pre[:, 0] = pred[:, 0, 0]
    D_pre[:, 1] = pred[:, 0, 1]
    
    for t in range(1, T):
        D_cur = torch.full((B, K), fill_value=float('-100000'), device=device)
        
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
    
    pair_labels = pair_labels.long() * 2 - 1

    fsa_distance = -torch.logsumexp(D_cur[:, -2:], dim=-1) / T
    loss = (fsa_distance * pair_labels).clamp(min=-margin).mean(0) + margin
    
    return loss


def frame_blank_align_distance(seq_features1, seq_features2, step_num):
    seq_features1 = seq_features1[:, 1:]
    blank2 = seq_features2[:, :1]
    seq_features2 = seq_features2[:, 1:]
    (B, T, C), device = seq_features1.shape, seq_features1.device
    
    K = 2 * step_num + 1
    
    sparse_choice2 = torch.rand(B, T, device=device).argsort(-1)[:, :step_num].sort(-1).values
    sparse_seq_features2 = seq_features2[torch.arange(B, device=device).unsqueeze(-1), sparse_choice2]
    sparse_seq_features2 = torch.cat((blank2, sparse_seq_features2), dim=1)
    # sparse_seq_features2 = torch.cat((blank2, seq_features1[:, [5, 7, 8, 9, 11, 12], :]), dim=1)

    pred = (torch.einsum('bic,bjc->bij', seq_features1, sparse_seq_features2) / math.sqrt(C)).log_softmax(-1)

    D_pre = torch.full((B, K), fill_value=float('-100000'), device=device)
    D_pre[:, 0] = pred[:, 0, 0]
    D_pre[:, 1] = pred[:, 0, 1]
    
    for t in range(1, T):
        D_cur = torch.full((B, K), fill_value=float('-100000'), device=device)
        
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
    
    fsa_distance = -torch.logsumexp(D_cur[:, -2:], dim=-1) / T
    return fsa_distance



def consist_step_mining_train(seq_features1, seq_features2, step_num, pair_labels):
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
    return -(pair_labels * final_result.max(dim=-1).values).mean(), segment1[:, :-1] + 1, segment2[:, :-1] + 1



def consist_step_mining_inference(seq_features1, seq_features2, step_num):
    seq_features1 = seq_features1[:, 1:]
    seq_features2 = seq_features2[:, 1:]
    (B, T, C), device = seq_features1.shape, seq_features1.device
    
    # pred = (torch.einsum('bic,bjc->bij', seq_features1, seq_features2) / math.sqrt(C)).softmax(-1)
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
        tmp = ((D[:, k-1, None, None, :, :] * k) + block_mat) / (k+1)
        D[:, k] = torch.max(tmp.flatten(3), -1).values
        D_ind[:, k] = torch.max(tmp.flatten(3), -1).indices
    
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
