import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
sys.path.append('/home/hty/CFSA')

import torch
import torch.nn.functional as F
import argparse
import numpy as np
import time
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm

from configs.defaults import get_cfg_defaults
from data.elastic_matching_dataset import load_dataset
from models.elastic_matching_model import Elastic_Matching_Model
from utils.preprocess import frames_preprocess
from utils.loss import *
from utils.tools import setup_seed
import json
from utils.dpm_decoder import *

RATIO = 1.0


def compute_metrics(predictions, ground_truth):
    # 计算预测准确率
    accuracy = torch.sum(predictions == ground_truth).item() / len(ground_truth)
    
    # 计算真正例（True Positives）
    true_positives = torch.sum(predictions & ground_truth).item()
    
    # 计算预测正例（Predicted Positives）
    predicted_positives = torch.sum(predictions).item()
    
    # 计算真正例（Ground Truth Positives）
    ground_truth_positives = torch.sum(ground_truth).item()
    
    # 计算精确度（Precision）
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
    
    # 计算召回率（Recall）
    recall = true_positives / ground_truth_positives if ground_truth_positives > 0 else 0.0
    
    # 计算F1分数
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return accuracy, precision, recall, f1_score


def read_json(file_path):
        with open(file_path, 'r') as f:
            data = json.loads(f.read())
        return data


def eval_one_model(model):
    if torch.cuda.device_count() > 1 and torch.cuda.is_available():
        # logger.info("Let's use %d GPUs" % torch.cuda.device_count())
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
            pair_label = sample['pair_label']
            step_num = sample['step_num']
            step_thres = sample['step_thres']
            
            frame2step_dist_list = []

            frames1 = frames_preprocess(frames1_list[0]).to(device, non_blocking=True)
            frames2 = frames_preprocess(frames2_list[0]).to(device, non_blocking=True)
            video_level1, video_level2, frame_level1, frame_level2, step_level1, step_level2 = model(x1=frames1, x2=frames2, embed=False)
            
            # video-wise
            video_dist = torch.sum((F.normalize(video_level1, p=2, dim=1) - F.normalize(video_level2, p=2, dim=1)) ** 2, dim=1)
            pred = video_dist
            
            # frame-wise
            # frame_dist = (torch.cosine_similarity(frame_level1, frame_level2, dim=2) > 0.6).sum(-1)
            # pred = frame_dist
            
            # sliding-window
            # swin_step_feat1 = frame_level1[:, [[i, i+1, i+2] for i in range(14)], :].mean(dim=2)
            # swin_step_feat2 = frame_level2[:, [[i, i+1, i+2] for i in range(14)], :].mean(dim=2)
            # swin_dist = (torch.cosine_similarity(swin_step_feat1, swin_step_feat2, dim=2) > 0.6).sum(-1)
            # pred = swin_dist
            
            # CPA
            # step_dist = (torch.cosine_similarity(step_level1, step_level2, dim=2) > 0.75).sum(-1)
            # pred = step_dist
            
            
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
    
    
    with open('eval_logs/elastic_matching_{}_results.txt'.format(RATIO), 'w') as f:
        for i in range(len(video1_list)):
            line_to_write = "{} {} {} {} {}\n".format(video1_list[i], label1_list[i], video2_list[i], label2_list[i], preds[i].item())
            f.writelines(line_to_write)
    return


def eval():
    model = Elastic_Matching_Model(num_class=cfg.DATASET.NUM_CLASS,
                                   dim_size=cfg.MODEL.DIM_EMBEDDING,
                                   num_clip=cfg.DATASET.NUM_CLIP,
                                   pretrain=cfg.MODEL.PRETRAIN,
                                   dropout=cfg.TRAIN.DROPOUT).to(device)
    
    if args.model_path == None:
        model_path = os.path.join(args.root_path, 'save_models')
    else:
        model_path = args.model_path

    start_time = time.time()

    if os.path.exists(model_path):
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
        '--root_path', 'train_logs/csv_logs/learnStep_Transformer',
        # '--model_path', 'train_logs/csv_logs/align_adaK/best_model.tar',
        '--model_path', 'train_logs/csv_logs/learnStep_Transformer/save_models/epoch_37.tar',
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

    eval_result_path = "eval_logs/elastic_matching_{}_results.txt".format(RATIO)
    ground_truth_path = "Datasets/Elastic-Matching/elastic_pair_{}.json".format(RATIO)
    
    pred_dict = {}
    with open(eval_result_path, 'r') as f:
        for line in f.readlines():
            video1, label1, video2, label2, distance = line.strip().split(' ')
            distance = float(distance)
            pred_key = "{}-{}-{}-{}".format(label1, video1, label2, video2)
            pred_dict[pred_key] = distance
            
    with open(ground_truth_path, 'r') as gt_f:
        gt_dict = json.load(gt_f)
    
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []
    
    for query_key in gt_dict:
        preds = []
        labels = []
        task = gt_dict[query_key]
        for candidate_key in task:
            pred_key = "{}-{}".format(query_key, candidate_key)
            distance = pred_dict[pred_key]
            pair_label = task[candidate_key]['label']
            preds.append(distance)
            labels.append(pair_label)
        
        labels = torch.tensor(labels)
        threshold = 9
        preds = torch.tensor(preds) > threshold
        # threshold = 1.2
        # preds = torch.tensor(preds) < threshold
        
        
        accuracy, precision, recall, f1_score = compute_metrics(preds, labels)
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_score_list.append(f1_score)
    
    average_accuracy = np.array(accuracy_list).mean()
    average_precision = np.array(precision_list).mean()
    average_recall = np.array(recall_list).mean()
    average_f1_score = np.array(f1_score_list).mean()
    
    print("Threshold: ", threshold)
    print("Accuracy: ", average_accuracy)
    print("F1 Score: ", average_f1_score)