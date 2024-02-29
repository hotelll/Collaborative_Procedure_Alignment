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
from utils.logger import setup_logger
from models.elastic_baseline import Baseline
from utils.preprocess import frames_preprocess
from utils.loss_origin import frame2step_wblank_dist
from utils.tools import setup_seed
import json

RATIO = 1.0

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
            pair_label = sample['pair_label']
            step_num = sample['step_num']
            step_thres = sample['step_thres']
            
            embeds1_list = []
            embeds2_list = []
            frames1_feat_list = []
            frames2_feat_list = []
            
            for i in range(len(frames1_list)):
                frames1 = frames_preprocess(frames1_list[i]).to(device, non_blocking=True)
                frames2 = frames_preprocess(frames2_list[i]).to(device, non_blocking=True)
                frame_emb1, frame_level1 = model(frames1)
                frame_emb2, frame_level2 = model(frames2)
                
                embeds1_list.append(frame_emb1)
                embeds2_list.append(frame_emb2)
                frames1_feat_list.append(frame_level1)
                frames2_feat_list.append(frame_level2)
            
            embeds1_avg = torch.stack(embeds1_list).mean(0)
            embeds2_avg = torch.stack(embeds2_list).mean(0)
            
            frames1_feat_avg = torch.stack(frames1_feat_list).mean(0)
            frames2_feat_avg = torch.stack(frames2_feat_list).mean(0)
            
            # frame_level1 = frames1_feat_avg[:, 1:, :]
            # frame_level2 = frames2_feat_avg[:, 1:, :]
            
            # torch.cosine_similarity(frame_level1, frame_level2, dim=2)
            
            pred = torch.sum((F.normalize(embeds1_avg, p=2, dim=1) - F.normalize(embeds2_avg, p=2, dim=1)) ** 2, dim=1)

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
    
    saved_path = 'eval_logs/baseline'
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
        
    with open(os.path.join(saved_path, 'elastic_matching_{}_results.txt'.format(RATIO)), 'w') as f:
        for i in range(len(video1_list)):
            line_to_write = "{} {} {} {} {}\n".format(video1_list[i], label1_list[i], video2_list[i], label2_list[i], preds[i].item())
            f.writelines(line_to_write)
    
    return


def eval():
    model = Baseline(num_class=cfg.DATASET.NUM_CLASS,
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
        logger.info('To evaluate %d models in %s' % (len(os.listdir(model_path)) - args.start_epoch + 1, model_path))
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
        logger.info('To evaluate 1 models in %s' % (model_path))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        eval_one_model(model)

    else:
        logger.info('Wrong model path: %s' % model_path)
        exit(-1)

    end_time = time.time()
    duration = end_time - start_time

    hour = duration // 3600
    min = (duration % 3600) // 60
    sec = duration % 60

    logger.info('Evaluate cost %dh%dm%ds' % (hour, min, sec))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='configs/eval_resnet_config.yml', help='config file path')
    parser.add_argument('--root_path', default=None, help='path to load models and save log')
    parser.add_argument('--model_path', default=None, help='path to load one model')
    parser.add_argument('--log_name', default='eval_log', help='log name')
    parser.add_argument('--start_epoch', default=1, type=int, help='index of the first evaluated epoch while evaluating epochs')

    args = parser.parse_args([
        '--config', 'configs/eval_elastic_matching_config.yml',
        '--root_path', 'train_logs/csv_logs/baseline',
        '--model_path', 'train_logs/csv_logs/baseline/save_models/epoch_5.tar'
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

    logger_path = os.path.join(args.root_path, 'logs')
    logger = setup_logger('Sequence Verification', logger_path, args.log_name, 0)
    logger.info('Running with config:\n{}\n'.format(cfg))

    test_loader = load_dataset(cfg)
    eval()
