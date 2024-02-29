import torch
from torch.utils import data
from torchvision import transforms as tf
import logging
import os
import numpy as np
from PIL import Image
import cv2
import random
from decord import VideoReader
from data.label import LABELS
import json

logger = logging.getLogger('Sequence Verification')


class ElasticMatchingDataset(data.Dataset):

    def __init__(self,
                 mode='train',
                 dataset_name='CSV',
                 dataset_root=None,
                 txt_path=None,
                 subclass_json=None,
                 normalization=None,
                 num_clip=16,
                 augment=True,
                 num_sample=600,
                 load_video=False):

        assert mode in ['train', 'test', 'val'], 'Dataset mode is expected to be train, test, or val. But get %s instead.' % mode
        
        self.mode = mode
        self.dataset_name = dataset_name
        self.dataset_root = dataset_root
        self.normalization = normalization
        self.num_clip = num_clip
        self.augment = augment
        self.load_video = load_video
        if augment:
            self.aug_flip = True
            self.aug_crop = True
            self.aug_color = True
            self.aug_rot = True
        self.num_sample = num_sample  # num of pairs randomly selected from all training pairs
        self.data_list = [line.strip() for line in open(txt_path, 'r').readlines()]
        # with open(subclass_json, 'r') as f:
        #     subclass_dict = json.load(f)
        # self.subclass_dict = subclass_dict

        logger.info('Successfully construct dataset with [%s] mode and [%d] samples randomly selected from [%d] samples' % (mode, len(self), len(self.data_list)))


    def __getitem__(self, index):
        data_path = self.data_list[index]
        data_path_split = data_path.strip().split(' ')
        query_video, query_label, candidate_video, candidate_label, pair_label, match_ratio = data_path.split(' ')
        step_num = int(match_ratio.split('/')[0])
        step_thres = int(match_ratio.split('/')[1])
        
        query_name = os.path.join(query_label, query_video)
        candidate_name = os.path.join(candidate_label, candidate_video)
        
        sample = {
            'video_name1': query_video.split('.')[0],
            'video_label1': query_label,
            'video_name2': candidate_video.split('.')[0],
            'video_label2': candidate_label,
            'clips1': self.sample_clips_from_video(os.path.join(self.dataset_root, query_name)),
            'clips2': self.sample_clips_from_video(os.path.join(self.dataset_root, candidate_name)),
            'labels1': LABELS[self.dataset_name][self.mode].index(query_label) if self.mode == 'train' else query_label,
            'labels2': LABELS[self.dataset_name][self.mode].index(candidate_label) if self.mode == 'train' else candidate_label,
            'pair_label': pair_label,
            'step_num': step_num,
            'step_thres': step_thres
        }

        return sample


    def __len__(self):
        if self.mode == 'train':
            return self.num_sample
        else:
            return len(self.data_list)

    def sample_clips_from_video(self, video_path):
        vr = VideoReader(video_path)
        segments = np.linspace(0, vr._num_frame - 2, self.num_clip + 1, dtype=int)
        sampled_clips = []
        num_sampled_per_segment = 1

        for i in range(num_sampled_per_segment):
            sampled_frames = []
            for j in range(self.num_clip):
                if self.mode == 'train':
                    frame_index = np.random.randint(segments[j], segments[j + 1])
                else:
                    frame_index = segments[j] + int((segments[j + 1] - segments[j]) / 2)
                sampled_frames.append(frame_index)
            sampled_frames = [Image.fromarray(i) for i in vr.get_batch(sampled_frames).asnumpy()]
            sampled_clips.append(self.preprocess(sampled_frames))

        return sampled_clips


    def preprocess(self, frames, apply_normalization=True):
        # Apply augmentation and normalization on a clip of frames

        # Data augmentation on the frames
        transforms = []

        if self.augment:
            # Flip
            if np.random.random() > 0.5 and self.aug_flip:
                transforms.append(tf.RandomHorizontalFlip(1))

            # Random crop
            if np.random.random() > 0.5 and self.aug_crop:
                transforms.append(tf.RandomResizedCrop((180, 320), (0.7, 1.0)))

            # Color augmentation
            if np.random.random() > 0.5 and self.aug_color:
                transforms.append(tf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5))

            # # Rotation
            # if np.random.random() > 0.5 and self.aug_rot:
            #     transforms.append(tf.RandomRotation(30))
        transforms.append(tf.Resize((180,320)))
        # PIL image to tensor
        transforms.append(tf.ToTensor())

        # Normalization
        if self.normalization is not None and apply_normalization:
            transforms.append(tf.Normalize(self.normalization[0], self.normalization[1]))

        transforms = tf.Compose(transforms)

        frames = torch.cat([transforms(frame).unsqueeze(-1) for frame in frames], dim=-1)

        return frames



class RandomSampler(data.Sampler):
    # randomly sample [len(self.dataset)] items from [len(self.data_list))] items

    def __init__(self, dataset, txt_path, shuffle=False):
        self.dataset = dataset
        self.data_list = [line.strip() for line in open(txt_path, 'r').readlines()]
        self.shuffle = shuffle

    def __iter__(self):

        tmp = random.sample(range(len(self.data_list)), len(self.dataset))
        if not self.shuffle:
            tmp.sort()

        # print(tmp)
        return iter(tmp)

    def __len__(self):
        return len(self.dataset)


def load_dataset(cfg):

    ImageNet_normalization = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # ImageNet_normalization = None
    dataset = ElasticMatchingDataset(mode=cfg.DATASET.MODE,
                                     dataset_name=cfg.DATASET.NAME,
                                     dataset_root=cfg.DATASET.ROOT,
                                     txt_path=cfg.DATASET.TXT_PATH,
                                     subclass_json=cfg.DATASET.SUBCLASS_JSON,
                                     normalization=ImageNet_normalization,
                                     num_clip=cfg.DATASET.NUM_CLIP,
                                     augment=cfg.DATASET.AUGMENT,
                                     num_sample=cfg.DATASET.NUM_SAMPLE,
                                     load_video=cfg.DATASET.LOAD_VIDEO)

    sampler = RandomSampler(dataset, cfg.DATASET.TXT_PATH, cfg.DATASET.SHUFFLE)

    loaders = data.DataLoader(dataset=dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              shuffle=False,
                              sampler=sampler,
                              drop_last=False,
                              num_workers=cfg.DATASET.NUM_WORKERS,
                              pin_memory=True)

    return loaders


if __name__ == "__main__":

    import sys
    # sys.path.append('/public/home/qianych/code/SVIP-Sequence-VerIfication-for-Procedures-in-Videos')
    from configs.defaults import get_cfg_defaults

    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/train_resnet_config.yml')

    train_loader = load_dataset(cfg)

    for iter, sample in enumerate(train_loader):
        print(sample.keys())
        print(sample['clips1'][0].size())
        print(sample['labels1'])
        break
