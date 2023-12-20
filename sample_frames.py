import torch
from torch.utils import data
from torchvision import transforms as tf
from PIL import Image
import logging
import os
import numpy as np
from PIL import Image
import cv2
import random
from decord import VideoReader
from data.label import LABELS
import json

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
t_mean = torch.FloatTensor(mean).view(3,1,1).expand(3, 180, 320)
t_std = torch.FloatTensor(std).view(3,1,1).expand(3, 180, 320)


def save_frame_img(frames1, save_path):
    for i in range(len(frames1)):
        frame = frames1[i]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        frame1_path = os.path.join(save_path, '{}.jpg'.format(i))
        frame.save(frame1_path)
    
    return


def sample_clips_from_video(video_dir_path, save_path):
    input_path = video_dir_path
    
    vr = VideoReader(input_path)
    segments = np.linspace(0, vr._num_frame - 2, 16 + 1, dtype=int)
    sampled_clips = []
    num_sampled_per_segment = 1

    # segments_id = [25, 75, 125, 175, 225, 275, 325, 375, 425, 475, 525, 575, 625, 675, 725, 775]
    # segments_id = [25, 75, 125, 175, 225, 275, 305, 325, 525, 545, 565, 585, 625, 675, 725, 775]
    segments_id = [19, 58, 97, 136, 175, 214, 254, 293, 332, 410, 430, 450, 489, 528, 567, 606]
    # segments_id = [19, 58, 97, 136, 175, 190, 214, 410, 430, 450, 460, 470, 489, 528, 567, 606]
    
    for i in range(num_sampled_per_segment):
        sampled_frames = []
        for j in range(16):
            frame_index = segments_id[j]
            sampled_frames.append(frame_index)
        sampled_frames = [Image.fromarray(i) for i in vr.get_batch(sampled_frames).asnumpy()]
        save_frame_img(sampled_frames, save_path)
        sampled_clips.append(sampled_frames)

    return sampled_clips


if __name__ == "__main__":
    # 1.2/zengziyun 1.2 
    # 1.2/luoweixin 1.2
    # video_path = 'Datasets/CSV/video_compressed/1.2/zengziyun.MP4'
    video_path = 'Datasets/CSV/video_compressed/1.2/luoweixin.MP4'
    save_path = 'vis_results/video2'
    sample_clips_from_video(video_path, save_path)