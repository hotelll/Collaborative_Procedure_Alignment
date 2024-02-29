import os
import sys
sys.path.append('/home/hty/CFSA')
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import time
import torch
import torch.nn.functional as F
import argparse
import numpy as np
from tqdm import tqdm
import math

from configs.defaults import get_cfg_defaults
from data.elastic_matching_dataset import load_dataset
from models.align_blank_model import Align
from utils.preprocess import frames_preprocess
from utils.loss_origin import frame2step_wblank_dist, batched_step_comparison
from utils.tools import setup_seed
import json

RATIO = 0.4

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
        for iter, sample in enumerate(tqdm(test_loader)):
            frames1_list = sample['clips1']
            frames2_list = sample['clips2']

            assert len(frames1_list) == len(frames2_list)

            labels1 = sample['labels1']
            labels2 = sample['labels2']
            pair_labels = sample['pair_label']
            step_num = sample['step_num'].to(device, non_blocking=True)
            step_thres = sample['step_thres'].to(device, non_blocking=True)
            
            frames1_feat_list = []
            frames2_feat_list = []
            
            for i in range(len(frames1_list)):
                frames1 = frames_preprocess(frames1_list[i]).to(device, non_blocking=True)
                frames2 = frames_preprocess(frames2_list[i]).to(device, non_blocking=True)
                frame_emb1, frame_level1 = model(frames1)
                frame_emb2, frame_level2 = model(frames2)
                
                frames1_feat_list.append(frame_level1)
                frames2_feat_list.append(frame_level2)
            
            frames1_feat_avg = torch.stack(frames1_feat_list).mean(0)
            frames2_feat_avg = torch.stack(frames2_feat_list).mean(0)
            
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
            loss_step = -(final_result.max(dim=-1).values).mean()
            
            batched_seg_list1 = []
            batched_seg_list2 = []
            batched_step_list1 = []
            batched_step_list2 = []
            batched_step_values_list = []
            
            for batch in range(B):
                # current_step_num = step_num[batch].item()
                # current_step_thres = current_step_num
                # current_step_thres = step_thres[batch].item()
                # current_step_num = 8
                # current_step_thres = math.ceil(current_step_num * RATIO)
                # current_step_thres = 11
                current_step_num = 13
                current_step_thres = 13

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
                    # step_average_list.append(step_value)
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
                selected_step_indices = step_values.topk(k=current_step_thres).indices.sort().values
                # selected_step_indices = step_values
                
                step_feature1 = frame_sequence1[batch][1]
                step_feature2 = frame_sequence2[batch][1]
                
                step_list1.insert(0, step_feature1)
                step_list2.insert(0, step_feature2)
                
                step_features1 = torch.stack(step_list1, dim=0)
                step_features2 = torch.stack(step_list2, dim=0)
                
                # selected_step_features1 = step_features1[selected_step_indices, :]
                # selected_step_features2 = step_features2[selected_step_indices, :]
                selected_step_features1 = step_features1
                selected_step_features2 = step_features2
                
                batched_step_list1.append(selected_step_features1)
                batched_step_list2.append(selected_step_features2)
                batched_step_values_list.append(step_values[None, :])
            
            step_level1 = torch.stack(batched_step_list1)
            step_level2 = torch.stack(batched_step_list2)
            # batched_step_values = torch.cat(batched_step_values_list, dim=0)
            
            step_dist = (torch.cosine_similarity(step_level1, step_level2, dim=2) > 0.85).sum(-1)
            pred = step_dist
            
            # pred = frame2step_wblank_dist(frames1_feat_avg, batched_step_list2, pair_labels) \
            #                 + frame2step_wblank_dist(frames2_feat_avg, batched_step_list1, pair_labels)
                            
            # step_dist = batched_step_comparison(batched_step_list1, batched_step_list2)
            # B = len(step_dist)
            # pred = torch.zeros((B,), device=device)
            
            # for b in range(B):
            #     # matched_step_num = (frame2step_dist[b] > 0.8).sum()
            #     cur_step_thres = step_thres[b]
            #     # cur_step_thres = 6
            #     dist = step_dist[b].topk(cur_step_thres).values.min()
            #     pred[b] = dist
            # pred = ((batched_step_values > 0.5).sum(-1) > step_thres.to(device, non_blocking=True)).long()
            # pred = ((batched_step_values > 0.4).sum(-1)).long()
            # pred = frame2step_dist

            video_names1 = sample['video_name1']
            video_names2 = sample['video_name2']
            video_labels1 = sample['video_label1']
            video_labels2 = sample['video_label2']
            
            video1_list += video_names1
            video2_list += video_names2
            label1_list += video_labels1
            label2_list += video_labels2
            
            if iter == 0:
                preds = pred
            else:
                preds = torch.cat([preds, pred])
    
    saved_path = 'eval_logs/sampleStep_cls_blank'
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
        
    with open(os.path.join(saved_path, 'elastic_matching_{}_results.txt'.format(RATIO)), 'w') as f:
        for i in range(len(video1_list)):
            line_to_write = "{} {} {} {} {}\n".format(video1_list[i], label1_list[i], video2_list[i], label2_list[i], preds[i].item())
            f.writelines(line_to_write)
    
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
        '--config', 'configs/eval_elastic_matching_config.yml',
        '--root_path', 'train_logs/csv_logs/train_align',
        '--model_path', 'train_logs/csv_logs/train_align/save_models/epoch_23.tar'
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
