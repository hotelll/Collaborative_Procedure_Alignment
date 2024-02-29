import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.distributions import Categorical
import math


def compute_cls_loss(pred, labels, use_cosface=False):
    if use_cosface:
        s = 30.0
        m = 0.4

        cos_value = torch.diagonal(pred.transpose(0, 1)[labels])
        numerator = s * (cos_value - m)
        excl = torch.cat([torch.cat((pred[i, :y], pred[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(s * excl), dim=1)
        L = numerator - torch.log(denominator)
        loss = -torch.mean(L)
    else:
        criterion = CrossEntropyLoss().cuda()
        loss = criterion(pred, labels)

    return loss


def compute_interctc_withneg_loss(seq_features1, seq_features2, pair_labels, margin):
    blank1 = seq_features1[:, :1]
    seq_features1 = seq_features1[:, 1:]
    blank2 = seq_features2[:, :1]
    seq_features2 = seq_features2[:, 1:]
    B, T, C = seq_features1.shape
    
    sparse_choice1 = torch.rand(B, T, device=seq_features1.device).argsort(-1)[:, :12].sort(-1).values
    sparse_seq_features1 = seq_features1[torch.arange(B, device=seq_features1.device).unsqueeze(-1), sparse_choice1]
    sparse_seq_features1 = torch.cat((blank1, sparse_seq_features1), dim=1)
    
    sparse_choice2 = torch.rand(B, T, device=seq_features2.device).argsort(-1)[:, :12].sort(-1).values
    sparse_seq_features2 = seq_features2[torch.arange(B, device=seq_features2.device).unsqueeze(-1), sparse_choice2]
    sparse_seq_features2 = torch.cat((blank2, sparse_seq_features2), dim=1)
    
    S = sparse_seq_features1.shape[1] - 1

    prob1_2 = (torch.einsum('bic,bjc->bij', seq_features1, sparse_seq_features2) / math.sqrt(C)).transpose(0, 1).log_softmax(2)
    prob2_1 = (torch.einsum('bic,bjc->bij', seq_features2, sparse_seq_features1) / math.sqrt(C)).transpose(0, 1).log_softmax(2)
    
    target = (torch.arange(S)+1).unsqueeze(0).expand(B, -1)
    input_lengths = torch.full((B, ), T, dtype=torch.long, device=seq_features1.device)
    target_lengths = torch.full((B, ), S, dtype=torch.long, device=seq_features1.device)

    ctc_distances = F.ctc_loss(prob1_2, target, input_lengths, target_lengths, reduction='none') / target_lengths \
        + F.ctc_loss(prob2_1, target, input_lengths, target_lengths, reduction='none') / target_lengths
    
    pair_labels = pair_labels.long() * 2 - 1

    loss = (ctc_distances * pair_labels).clamp(min=-margin).mean(0) + margin
    return loss


def compute_interctc_distance(seq_features1, seq_features2):
    blank1 = seq_features1[:, :1]
    seq_features1 = seq_features1[:, 1:]
    blank2 = seq_features2[:, :1]
    seq_features2 = seq_features2[:, 1:]
    B, T, C = seq_features1.shape

    # diving48 12 frames
    sparse_seq_features1 = torch.cat((blank1, seq_features1[:, [1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15], :]), dim=1)
    sparse_seq_features2 = torch.cat((blank2, seq_features2[:, [1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15], :]), dim=1)

    S = sparse_seq_features1.shape[1] - 1

    prob1_2 = (torch.einsum('bic,bjc->bij', seq_features1, sparse_seq_features2) / math.sqrt(C)).transpose(0, 1).log_softmax(2)
    prob2_1 = (torch.einsum('bic,bjc->bij', seq_features2, sparse_seq_features1) / math.sqrt(C)).transpose(0, 1).log_softmax(2)
    
    target = (torch.arange(S)+1).unsqueeze(0).expand(B, -1)
    input_lengths = torch.full((B, ), T, dtype=torch.long, device=seq_features1.device)
    target_lengths = torch.full((B, ), S, dtype=torch.long, device=seq_features1.device)

    ctc_distances = F.ctc_loss(prob1_2, target, input_lengths, target_lengths, reduction='none') / target_lengths \
        + F.ctc_loss(prob2_1, target, input_lengths, target_lengths, reduction='none') / target_lengths

    return ctc_distances


def frame2step_loss(seq_features1, seq_features2, sparse_label):
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
    return ctc_distance.mean(0)


def consist_step_mining_train(seq_features1, seq_features2, step_num, pair_labels):
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
    
    video_seg1 = segment1[:, :-1] + 1
    video_seg2 = segment2[:, :-1] + 1
    
    # loss_step = (-(pair_labels * final_result.max(dim=-1).values)).sum()
    loss_step = -(pair_labels * final_result.max(dim=-1).values).mean()
    
    return loss_step, video_seg1, video_seg2


def consist_step_mining_inference(seq_features1, seq_features2, step_num):
    (B, T, C), device = seq_features1.shape, seq_features1.device
    pred = (torch.einsum('bic,bjc->bij', seq_features1, seq_features2) / math.sqrt(C)).softmax(-1)
    # pred = torch.cosine_similarity(seq_features1.unsqueeze(2), seq_features2.unsqueeze(1), dim=-1)
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


def compute_reinforce_ctc_loss(seq_features1, seq_features2, frame_scores1, frame_scores2):
    seq_features1 = seq_features1[:, 1:]
    B, T, C = seq_features1.shape
    
    # probs = policy_network(state)
    # Note that this is equivalent to what used to be called multinomial
    m = Categorical(logits=frame_scores2)
    
    action = m.sample()

    # sample_result = sample_results1 | sample_results2 | sample_results3
    target = torch.nonzero(action, as_tuple=False)[:, 1] + 1
    input_lengths = torch.full(size=(B, ), fill_value=T, dtype=torch.long, device=seq_features1.device)
    target_lengths = action.squeeze(-1).bool().sum(-1)
    
    prob = (torch.einsum('bic,bjc->bij', seq_features1, seq_features2) / math.sqrt(C)).transpose(0, 1).log_softmax(2)

    ctc_loss = F.ctc_loss(prob, target, input_lengths, target_lengths, reduction='none') / target_lengths

    reward = (-ctc_loss)[:, None, None]
    loss = -m.log_prob(action) * reward.detach()
    return loss.mean(), ctc_loss.mean(), m.entropy().mean()

