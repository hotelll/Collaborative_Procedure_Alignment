import os
import sys
sys.path.append('/home/hty/CFSA')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import torch
import torch.nn.functional as F
import argparse
import numpy as np
from tqdm import tqdm
import math

from configs.defaults import get_cfg_defaults
from data.dataset import load_dataset
from models.align_blank_model import Align
from utils.preprocess import frames_preprocess
from utils.tools import setup_seed
import json
import cv2
import matplotlib.pyplot as plt


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
t_mean = torch.FloatTensor(mean).view(3,1,1).expand(3, 180, 320)
t_std = torch.FloatTensor(std).view(3,1,1).expand(3, 180, 320)

vis_root = 'vis_results/'

def save_frame_img(frames1, frames2, features1, features2):
    frame_list1 = []
    for i in range(frames1.shape[-1]):
        frame = frames1[..., i]
        frame = frame.clone().detach().to(torch.device('cpu'))
        frame = frame * t_std + t_mean
        frame = frame.data.cpu().float().numpy().transpose(1,2,0)
        frame = cv2.cvtColor(frame * 255, cv2.COLOR_RGB2BGR)
        video_folder_1 = os.path.join(vis_root, 'video1')
        if not os.path.exists(video_folder_1):
            os.makedirs(video_folder_1)
            
        frame1_path = os.path.join(video_folder_1, '{}.png'.format(i))
        cv2.imwrite(frame1_path, frame)
    
    frame_list2 = []
    for i in range(frames2.shape[-1]):
        frame = frames2[..., i]
        frame = frame.clone().detach().to(torch.device('cpu'))
        frame = frame * t_std + t_mean
        frame = frame.data.cpu().float().numpy().transpose(1,2,0)
        frame = cv2.cvtColor(frame * 255, cv2.COLOR_RGB2BGR)
        video_folder_2 = os.path.join(vis_root, 'video2')
        if not os.path.exists(video_folder_2):
            os.makedirs(video_folder_2)
            
        frame2_path = os.path.join(video_folder_2, '{}.png'.format(i))
        cv2.imwrite(frame2_path, frame)

    
    return


def frame2step_wblank_dist(frame_feats1, step_feats2):
    B = frame_feats1.shape[0]
    dists = []
    for batch in range(B):
        frame_feat1 = frame_feats1[batch]
        step_feat2 = step_feats2[batch]

        frame_dist = single_align_loss(frame_feat1, step_feat2)
        dists.append(frame_dist)
        
    return torch.stack(dists, dim=-1)


def single_align_loss(frame_features1, step_features2):
    # frame_features1 的第一个是blank feature
    frame_features1 = frame_features1[1:]
    blank = frame_features1[:1]
    (T, C), device = frame_features1.shape, frame_features1.device
    step_num = step_features2.shape[0]
    K = 2 * step_num + 1
    step_features2_with_blank = torch.cat((blank, step_features2), dim=0)
    
    pred = (torch.einsum('ic,jc->ij', frame_features1, step_features2_with_blank) / math.sqrt(C)).log_softmax(-1)
    pred_origin = (torch.einsum('ic,jc->ij', frame_features1, step_features2_with_blank) / math.sqrt(C)).softmax(-1)
    # pred_origin = torch.cosine_similarity(seq_features1.unsqueeze(2), seq_features2.unsqueeze(1), dim=-1)
    show_mat = pred_origin.detach().cpu().numpy()
    plt.matshow(show_mat, cmap=plt.cm.Reds)
    save_path = os.path.join(vis_root, 'assignment_prob.png')
    plt.savefig(save_path)
    
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


def read_json(file_path):
        with open(file_path, 'r') as f:
            data = json.loads(f.read())
        return data


def eval_one_model(model):
    if torch.cuda.device_count() > 1 and torch.cuda.is_available():
        model = torch.nn.DataParallel(model)

    # auc metric
    model.eval()

    video1_list = []
    video2_list = []
    label1_list = []
    label2_list = []

    with torch.no_grad():
        for iter, sample in enumerate(test_loader):
            frames1_list = sample['clips1']
            frames2_list = sample['clips2']

            assert len(frames1_list) == len(frames2_list)

            labels1 = sample['labels1']
            labels2 = sample['labels2']
            label = torch.tensor(np.array(labels1) == np.array(labels2)).to(device)
            
            frames1 = frames_preprocess(frames1_list[0]).to(device, non_blocking=True)
            frames2 = frames_preprocess(frames2_list[0]).to(device, non_blocking=True)
            frame_emb1, frame_level1 = model(frames1)
            frame_emb2, frame_level2 = model(frames2)
            # save_frame_img(frames1_list[0][0], frames2_list[0][0], frame_level1, frame_level2)
            frames1_feat_avg = frame_level1
            frames2_feat_avg = frame_level2
            
            frame_sequence1 = frames1_feat_avg[:, 1:]
            frame_sequence2 = frames2_feat_avg[:, 1:]
            (B, T, C) = frame_sequence1.shape
            
            # pred = (torch.einsum('bic,bjc->bij', frame_sequence1, frame_sequence2) / math.sqrt(C)).softmax(-1)
            pred = torch.cosine_similarity(frame_sequence1.unsqueeze(2), frame_sequence2.unsqueeze(1), dim=-1)
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
            area = area.clamp_min(1)
            
            block_mat = block_mat.masked_fill(((a >= i) | (b >= j)).unsqueeze(0), float('-inf')) / area
        
            for k in range(1, T):
                # tmp = D[:, k-1, None, None, :, :] + block_mat
                tmp = ((D[:, k-1, None, None, :, :] * k) + block_mat) / (k+1)
                D[:, k] = torch.max(tmp.flatten(3), -1).values
                D_ind[:, k] = torch.max(tmp.flatten(3), -1).indices
                D_block[:, k] = block_mat.flatten(3).gather(dim=-1, index=D_ind[:, k].unsqueeze(-1)).squeeze(-1)
            
            final_result = D[:, :, T-1, T-1]
            
            batched_step_list1 = []
            batched_step_list2 = []
            batched_step_values_list = []
            
            for batch in range(B):
                current_step_num = 13

                step_values = torch.zeros((current_step_num), device=device)
                seg1_list = []
                seg2_list = []
                i, j, a, b = T-1, T-1, T-1, T-1
                k = current_step_num - 1
                
                step_list1 = []
                step_list2 = []

                while k > 0:
                    ind = D_ind[batch, k, i, j].item()
                    step_value = D_block[batch, k, i, j].item()
                    
                    step_values[k] = step_value
                    a = ind // T
                    b = ind % T
                    
                    step_feature1 = frame_sequence1[batch][a + 1]
                    step_feature2 = frame_sequence2[batch][b + 1]
                    
                    step_list1.insert(0, step_feature1)
                    step_list2.insert(0, step_feature2)

                    seg1_list.insert(0, a)
                    seg2_list.insert(0, b)
                    i, j, k = a, b, k-1
                
                step_values[0] = D_block[batch, 0, i, j].item()
                
                step_feature1 = frame_sequence1[batch][1]
                step_feature2 = frame_sequence2[batch][1]
                
                step_list1.insert(0, step_feature1)
                step_list2.insert(0, step_feature2)
                
                step_features1 = torch.stack(step_list1, dim=0)
                step_features2 = torch.stack(step_list2, dim=0)
                
                selected_step_features1 = step_features1
                selected_step_features2 = step_features2
                
                batched_step_list1.append(selected_step_features1)
                batched_step_list2.append(selected_step_features2)
                batched_step_values_list.append(step_values[None, :])
            
            
            pred = frame2step_wblank_dist(frames1_feat_avg, batched_step_list2) \
                            + frame2step_wblank_dist(frames2_feat_avg, batched_step_list1)
            print("Distance: ", pred.item())
            if iter == 0:
                preds = pred
            else:
                preds = torch.cat([preds, pred])
    
    return


def eval():
    model = Align(num_class=cfg.DATASET.NUM_CLASS,
                  dim_size=cfg.MODEL.DIM_EMBEDDING,
                  num_clip=cfg.DATASET.NUM_CLIP,
                  pretrain=cfg.MODEL.PRETRAIN,
                  dropout=cfg.TRAIN.DROPOUT).to(device)
    
    if args.model_path == None:
        model_path = os.path.join(args.root_path, 'save_models')
    else:
        model_path = args.model_path

    start_time = time.time()

    if os.path.isdir(model_path):
        # Evaluate models
        print('To evaluate %d models in %s' % (len(os.listdir(model_path)) - args.start_epoch + 1, model_path))
        current_epoch = args.start_epoch

        while True:
            current_model_path = os.path.join(model_path, 'epoch_' + str(current_epoch) + '.tar')
            if not os.path.exists(current_model_path):
                break
            
            checkpoint = torch.load(current_model_path)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            eval_one_model(model)

            current_epoch += 1
            
    elif os.path.isfile(model_path):
        # Evaluate one model
        print('To evaluate 1 models in %s' % (model_path))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        eval_one_model(model)

    else:
        print('Wrong model path: %s' % model_path)
        exit(-1)

    end_time = time.time()
    duration = end_time - start_time

    hour = duration // 3600
    min = (duration % 3600) // 60
    sec = duration % 60

    print('Evaluate cost %dh%dm%ds' % (hour, min, sec))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='configs/eval_resnet_config.yml', help='config file path')
    parser.add_argument('--root_path', default=None, help='path to load models and save log')
    parser.add_argument('--model_path', default=None, help='path to load one model')
    parser.add_argument('--log_name', default='eval_log', help='log name')
    parser.add_argument('--start_epoch', default=1, type=int, help='index of the first evaluated epoch while evaluating epochs')

    args = parser.parse_args([
        '--config', 'configs/visualize_config.yml',
        '--root_path', 'train_logs/csv_logs/train_align',
        '--model_path', 'train_logs/csv_logs/train_align_blank/save_models/epoch_8.tar',
        # '--start_epoch', '1'
    ])

    return args


if __name__ == "__main__":

    args = parse_args()

    cfg = get_cfg_defaults()
    if args.config:
        cfg.merge_from_file(args.config)

    setup_seed(cfg.TRAIN.SEED)
    use_cuda = cfg.TRAIN.USE_CUDA and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    test_loader = load_dataset(cfg)
    eval()
