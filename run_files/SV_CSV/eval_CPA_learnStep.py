import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
sys.path.append('/home/hty/CFSA')

import torch
import torch.nn.functional as F
import argparse
import numpy as np
import time


from tqdm import tqdm
from sklearn.metrics import auc, roc_curve

from configs.defaults import get_cfg_defaults
from data.dataset import load_dataset
from utils.logger import setup_logger
from models.CPA_sampleStep_model import CPA_sampleStep
from utils.preprocess import frames_preprocess
from utils.loss import *
from utils.tools import setup_seed
import json
from utils.dpm_decoder import *


def read_json(file_path):
        with open(file_path, 'r') as f:
            data = json.loads(f.read())
        return data


def eval_one_model(model, dist='NormL2'):
    if torch.cuda.device_count() > 1 and torch.cuda.is_available():
        # logger.info("Let's use %d GPUs" % torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    # auc metric
    model.eval()

    with torch.no_grad():
        for iter, sample in enumerate(tqdm(test_loader)):
            frames1_list = sample['clips1']
            frames2_list = sample['clips2']

            assert len(frames1_list) == len(frames2_list)

            labels1 = sample['labels1']
            labels2 = sample['labels2']
            label = torch.tensor(np.array(labels1) == np.array(labels2)).to(device)
            
            trend = 1 - label.float()
            frame2step_dist_list = []
            embeds1_list = []
            embeds2_list = []
            
            for i in range(len(frames1_list)):
                frames1 = frames_preprocess(frames1_list[i]).to(device, non_blocking=True)
                frames2 = frames_preprocess(frames2_list[i]).to(device, non_blocking=True)
                video_emb1, video_emb2, frame2step_dist, loss_step = model(frames1, frames2, embed=True)

                frame2step_dist_list.append(frame2step_dist)
                embeds1_list.append(video_emb1)
                embeds2_list.append(video_emb2)
            
            embeds1_avg = torch.stack(embeds1_list).mean(0)
            embeds2_avg = torch.stack(embeds2_list).mean(0)
            frame2step_dist_avg = torch.stack(frame2step_dist_list).mean(0)
            
            if dist == 'adaK':
                pred = frame2step_dist_avg
            elif dist == 'NormL2':
                pred = torch.sum((F.normalize(embeds1_avg, p=2, dim=1) - F.normalize(embeds2_avg, p=2, dim=1)) ** 2, dim=1)
                
            if iter == 0:
                preds = pred
                labels = label
                labels1_all = labels1
                labels2_all = labels2
            else:
                preds = torch.cat([preds, pred])
                labels = torch.cat([labels, label])
                labels1_all += labels1
                labels2_all += labels2

    fpr, tpr, thresholds = roc_curve(labels.cpu().detach().numpy(), preds.cpu().detach().numpy(), pos_label=0)
    auc_value = auc(fpr, tpr)
    wdr_value = compute_WDR(preds, labels1_all, labels2_all)
    
    return auc_value, wdr_value


def compute_WDR(preds, labels1, labels2):
    import json
    def read_json(file_path):
        with open(file_path, 'r') as f:
            data = json.loads(f.read())
        return data

    def compute_edit_dist(seq1, seq2):
        matrix = [[i + j for j in range(len(seq2) + 1)] for i in range(len(seq1) + 1)]
        for i in range(1, len(seq1) + 1):
            for j in range(1, len(seq2) + 1):
                if (seq1[i - 1] == seq2[j - 1]):
                    d = 0
                else:
                    d = 2
                matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)
        return matrix[len(seq1)][len(seq2)]


    # Load steps info for the corresponding dataset
    label_bank_path = os.path.join('Datasets', cfg.DATASET.NAME, 'label_bank.json')
    label_bank = read_json(label_bank_path)

    # Calcualte wdr
    labels = torch.tensor(np.array(labels1) == np.array(labels2))
    m_dists = preds[labels]
    um_dists = []
    for i in range(labels.size(0)):
        label = labels[i]
        if not label: # only compute negative pairs
            # unmatched pair
            # NormL2 dist / edit distance
            um_dists.append(preds[i] / compute_edit_dist(label_bank[labels1[i]], label_bank[labels2[i]]))

    return torch.tensor(um_dists).mean() / m_dists.mean()


def eval():
    model = CPA_sampleStep(num_class=cfg.DATASET.NUM_CLASS,
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

        best_auc = 0
        best_wdr = 0    # wdr of the model with best auc
        best_model_path = ''

        current_epoch = args.start_epoch

        while True:
            current_model_path = os.path.join(model_path, 'epoch_' + str(current_epoch) + '.tar')
            if not os.path.exists(current_model_path):
                break
            
            checkpoint = torch.load(current_model_path)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            auc, wdr = eval_one_model(model, args.dist)
            logger.info('Model is %s, AUC is %.4f, wdr is %.4f' % ('Epoch ' + str(current_epoch), auc, wdr))

            if auc > best_auc:
                best_auc = auc
                best_wdr = wdr
                best_model_path = current_model_path

            current_epoch += 1

        logger.info('*** Best models is %s, Best AUC is %.4f, Best wdr is %.4f ***' % (best_model_path, best_auc, best_wdr))
        logger.info('----------------------------------------------------------------')

    elif os.path.isfile(model_path):
        # Evaluate one model
        logger.info('To evaluate 1 models in %s' % (model_path))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        auc, wdr = eval_one_model(model, args.dist)
        logger.info('Model is %s, AUC is %.4f' % (model_path, auc))

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
    parser.add_argument('--dist', default='NormL2')

    args = parser.parse_args([
        '--config', 'configs/eval_CPA_learnStep_config.yml',
        '--root_path', 'train_logs/csv_logs/CPA_learnStep',
        # '--model_path', 'train_logs/csv_logs/align_adaK/best_model.tar',
        # '--model_path', 'train_logs/csv_logs/learnStep_Transformer/save_models/epoch_37.tar',
        '--dist', 'adaK',
        '--start_epoch', '1'
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
