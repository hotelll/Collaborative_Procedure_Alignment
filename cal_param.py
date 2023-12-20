import torch
import argparse
import os
import sys
import torchvision.models as models

sys.path.append('/home/hty/svip')

from configs.defaults import get_cfg_defaults
from models.baseline import Baseline
from models.CPA_sampleStep_model import Align_adaK_learnGaussStep
from utils.loss import *
from utils.dpm_decoder import *
from utils.tools import setup_seed

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='configs/train_resnet_config.yml', help='config file path')
    parser.add_argument('--save_path', default=None, help='path to save models and log')
    parser.add_argument('--load_path', default=None, help='path to load the model')
    parser.add_argument('--log_name', default='train_log', help='log name')


    args = parser.parse_args([
        '--config', 'configs/train_adaK_learnGaussStep_config.yml'
    ])
    return args


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total: {}\n Trainable: {}'.format(total_num, trainable_num))


if __name__ == "__main__":
    args = parse_args()

    cfg = get_cfg_defaults()
    if args.config:
        cfg.merge_from_file(args.config)

    setup_seed(cfg.TRAIN.SEED)
    use_cuda = cfg.TRAIN.USE_CUDA and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    # model = Baseline(num_class=cfg.DATASET.NUM_CLASS,
    #                  dim_size=cfg.MODEL.DIM_EMBEDDING,
    #                  num_clip=cfg.DATASET.NUM_CLIP,
    #                  pretrain=cfg.MODEL.PRETRAIN,
    #                  dropout=cfg.TRAIN.DROPOUT)
    
    model = Align_adaK_learnGaussStep(num_class=cfg.DATASET.NUM_CLASS,
                                      dim_size=cfg.MODEL.DIM_EMBEDDING,
                                      num_clip=cfg.DATASET.NUM_CLIP,
                                      pretrain=cfg.MODEL.PRETRAIN,
                                      dropout=cfg.TRAIN.DROPOUT).to(device)

    get_parameter_number(model)
