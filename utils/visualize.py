import torch
from typing import List
import torch.nn.functional as F
import json
import cv2
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import cv2
import matplotlib as mpl

from utils.twfinch import FINCH
from utils.colormap import _COLORS


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
t_mean = torch.FloatTensor(mean).view(3,1,1).expand(3, 180, 320)
t_std = torch.FloatTensor(std).view(3,1,1).expand(3, 180, 320)

def vis_ctc(subcls_logits1, subcls_logits2, step_labels1, step_labels2, frames1, frames2):
    
    id2step = {}
    with open('Datasets/CSV/id2step.json', 'r') as f:
        id2step = json.load(f)

    if not os.path.exists('csv_ctc_visualize/'):
        os.mkdir('csv_ctc_visualize/')

    subcls_logits1 = subcls_logits1.transpose(0, 1).log_softmax(2)
    subcls_logits2 = subcls_logits2.transpose(0, 1).log_softmax(2)

    for b in range(frames1.shape[0]):
        frames_list1 = []
        step_name_list1 = []

        frames_list2 = []
        step_name_list2 = []

        single_batch_frames1 = frames1[b]
        single_batch_frames2 = frames2[b]
        step_labels1 = subcls_logits1[:, b].argmax(-1)
        step_labels2 = subcls_logits2[:, b].argmax(-1)


        for i in step_labels1:
            id = i.item()
            if id != 0:
                step_name = id2step[str(id)]
            else: # this is a blank
                step_name = 'blank'
            step_name_list1.append(step_name)
        
        for i in step_labels2:
            id = i.item()
            if id != 0:
                step_name = id2step[str(id)]
            else: # this is a blank
                step_name = 'blank'
            step_name_list2.append(step_name)

        for i in range(single_batch_frames1.shape[-1]):
            frame = single_batch_frames1[..., i]
            frame = frame.clone().detach().to(torch.device('cpu'))
            frame = frame * t_std + t_mean
            frame = frame.data.cpu().float().numpy().transpose(1,2,0)
            frame = cv2.cvtColor(frame * 255, cv2.COLOR_RGB2BGR)
            caption = np.full((50, 320, 3), 255, np.uint8)
            cv2.putText(caption, step_name_list1[i], (20, 20), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 0))
            frame = np.vstack((frame, caption))

            frames_list1.append(frame)
        
        for i in range(single_batch_frames2.shape[-1]):
            frame = single_batch_frames2[..., i]
            frame = frame.clone().detach().to(torch.device('cpu'))
            frame = frame * t_std + t_mean
            frame = frame.data.cpu().float().numpy().transpose(1,2,0)
            frame = cv2.cvtColor(frame * 255, cv2.COLOR_RGB2BGR)
            caption = np.full((50, 320, 3), 255, np.uint8)
            cv2.putText(caption, step_name_list2[i], (20, 20), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 0))
            frame = np.vstack((frame, caption))

            frames_list2.append(frame)
        
        
        frame_sequence1 = np.hstack(frames_list1)
        frame_sequence2 = np.hstack(frames_list2)
        frame_sequence = np.vstack((frame_sequence1, frame_sequence2))

        cv2.imwrite(os.path.join('csv_ctc_visualize', str(b)+'.png'), frame_sequence)
    
    return



def Similar(embed1, embed2):
    return torch.sum((F.normalize(embed1, p=2, dim=0) - F.normalize(embed2, p=2, dim=0)) ** 2)
    # return torch.cosine_similarity(embed1, embed2, dim=0)

def vis_similarity(frames1, frames2, embeds1, embeds2):
    
    vis_folder = 'visualize_results/csv_x3d_32_sim_visualize/'
    if not os.path.exists(vis_folder):
        os.mkdir(vis_folder)

    for b in range(frames1.shape[0]):
        frames_list1 = []
        frames_list2 = []

        single_batch_frames1 = frames1[b]
        single_batch_frames2 = frames2[b]
        single_batch_embeds1 = embeds1[b]
        single_batch_embeds2 = embeds2[b]

        similarity_list1 = ['0.00',]
        similarity_list2 = ['0.00',]

        for i in range(single_batch_embeds1.shape[0]-1):
            neighbor_distance = Similar(single_batch_embeds1[i], single_batch_embeds1[i+1]).item()
            similarity_list1.append(format(neighbor_distance, '.2f'))
        
        for i in range(single_batch_embeds2.shape[0]-1):
            neighbor_distance = Similar(single_batch_embeds2[i], single_batch_embeds2[i+1]).item()
            similarity_list2.append(format(neighbor_distance, '.2f'))


        for i in range(single_batch_frames1.shape[-1]):
            frame = single_batch_frames1[..., i]
            frame = frame.clone().detach().to(torch.device('cpu'))
            frame = frame * t_std + t_mean
            frame = frame.data.cpu().float().numpy().transpose(1,2,0)
            frame = cv2.cvtColor(frame * 255, cv2.COLOR_RGB2BGR)
            caption = np.full((50, 320, 3), 255, np.uint8)
            cv2.putText(caption, similarity_list1[i], (100, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 0))
            frame = np.vstack((frame, caption))

            frames_list1.append(frame)
        
        for i in range(single_batch_frames2.shape[-1]):
            frame = single_batch_frames2[..., i]
            frame = frame.clone().detach().to(torch.device('cpu'))
            frame = frame * t_std + t_mean
            frame = frame.data.cpu().float().numpy().transpose(1,2,0)
            frame = cv2.cvtColor(frame * 255, cv2.COLOR_RGB2BGR)
            caption = np.full((50, 320, 3), 255, np.uint8)
            cv2.putText(caption, similarity_list2[i], (100, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 0))
            frame = np.vstack((frame, caption))

            frames_list2.append(frame)
        
        frame_sequence1 = np.hstack(frames_list1)
        frame_sequence2 = np.hstack(frames_list2)
        frame_sequence = np.vstack((frame_sequence1, frame_sequence2))
        
        cv2.imwrite(os.path.join(vis_folder, str(b)+'.png'), frame_sequence)

    return

def vis_cross_similarity(frames1, frames2, embeds1, embeds2):
    
    vis_folder = 'visualize_results/csv_x3d_32_crosssim_visualize/'
    if not os.path.exists(vis_folder):
        os.mkdir(vis_folder)

    for b in range(frames1.shape[0]):
        frames_list1 = []
        frames_list2 = []

        single_batch_frames1 = frames1[b]
        single_batch_frames2 = frames2[b]
        single_batch_embeds1 = embeds1[b]
        single_batch_embeds2 = embeds2[b]

        most_similar_list = []
        similar_score_list = []
        index_list = []

        for i in range(single_batch_embeds1.shape[0]):
            most_similar_index = 0
            most_similar_distance = 100.0
            for j in range(single_batch_embeds2.shape[0]):
                neighbor_distance = Similar(single_batch_embeds1[i], single_batch_embeds2[j]).item()
                if neighbor_distance < most_similar_distance:
                    most_similar_distance = neighbor_distance
                    most_similar_index = j

            most_similar_list.append(str(most_similar_index))
            similar_score_list.append(format(most_similar_distance, '.2f'))
            index_list.append(str(i))


        for i in range(single_batch_frames1.shape[-1]):
            frame = single_batch_frames1[..., i]
            frame = frame.clone().detach().to(torch.device('cpu'))
            frame = frame * t_std + t_mean
            frame = frame.data.cpu().float().numpy().transpose(1,2,0)
            frame = cv2.cvtColor(frame * 255, cv2.COLOR_RGB2BGR)
            caption = np.full((50, 320, 3), 255, np.uint8)
            cv2.putText(caption, most_similar_list[i], (30, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 0))
            cv2.putText(caption, similar_score_list[i], (200, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 0))
            frame = np.vstack((frame, caption))

            frames_list1.append(frame)
        
        for i in range(single_batch_frames2.shape[-1]):
            frame = single_batch_frames2[..., i]
            frame = frame.clone().detach().to(torch.device('cpu'))
            frame = frame * t_std + t_mean
            frame = frame.data.cpu().float().numpy().transpose(1,2,0)
            frame = cv2.cvtColor(frame * 255, cv2.COLOR_RGB2BGR)
            caption = np.full((50, 320, 3), 255, np.uint8)
            cv2.putText(caption, index_list[i], (100, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 0))
            frame = np.vstack((frame, caption))

            frames_list2.append(frame)
        
        frame_sequence1 = np.hstack(frames_list1)
        frame_sequence2 = np.hstack(frames_list2)
        frame_sequence = np.vstack((frame_sequence1, frame_sequence2))
        
        cv2.imwrite(os.path.join(vis_folder, str(b)+'.png'), frame_sequence)

    return

def vis_cross_similarity(frames1, frames2, embeds1, embeds2):
    
    vis_folder = 'visualize_results/csv_x3d_32_crosssim_visualize/'
    if not os.path.exists(vis_folder):
        os.mkdir(vis_folder)

    for b in range(frames1.shape[0]):
        frames_list1 = []
        frames_list2 = []

        single_batch_frames1 = frames1[b]
        single_batch_frames2 = frames2[b]
        single_batch_embeds1 = embeds1[b]
        single_batch_embeds2 = embeds2[b]

        most_similar_list = []
        similar_score_list = []
        index_list = []

        for i in range(single_batch_embeds1.shape[0]):
            most_similar_index = 0
            most_similar_distance = 100.0
            for j in range(single_batch_embeds2.shape[0]):
                neighbor_distance = Similar(single_batch_embeds1[i], single_batch_embeds2[j]).item()
                if neighbor_distance < most_similar_distance:
                    most_similar_distance = neighbor_distance
                    most_similar_index = j

            most_similar_list.append(str(most_similar_index))
            similar_score_list.append(format(most_similar_distance, '.2f'))
            index_list.append(str(i))


        for i in range(single_batch_frames1.shape[-1]):
            frame = single_batch_frames1[..., i]
            frame = frame.clone().detach().to(torch.device('cpu'))
            frame = frame * t_std + t_mean
            frame = frame.data.cpu().float().numpy().transpose(1,2,0)
            frame = cv2.cvtColor(frame * 255, cv2.COLOR_RGB2BGR)
            caption = np.full((50, 320, 3), 255, np.uint8)
            cv2.putText(caption, most_similar_list[i], (30, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 0))
            cv2.putText(caption, similar_score_list[i], (200, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 0))
            frame = np.vstack((frame, caption))

            frames_list1.append(frame)
        
        for i in range(single_batch_frames2.shape[-1]):
            frame = single_batch_frames2[..., i]
            frame = frame.clone().detach().to(torch.device('cpu'))
            frame = frame * t_std + t_mean
            frame = frame.data.cpu().float().numpy().transpose(1,2,0)
            frame = cv2.cvtColor(frame * 255, cv2.COLOR_RGB2BGR)
            caption = np.full((50, 320, 3), 255, np.uint8)
            cv2.putText(caption, index_list[i], (100, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 0))
            frame = np.vstack((frame, caption))

            frames_list2.append(frame)
        
        frame_sequence1 = np.hstack(frames_list1)
        frame_sequence2 = np.hstack(frames_list2)
        frame_sequence = np.vstack((frame_sequence1, frame_sequence2))
        
        cv2.imwrite(os.path.join(vis_folder, str(b)+'.png'), frame_sequence)

    return

def vis_inter_ctc_matrix(frames1, frames2, seq_features1, seq_features2):
    frames1 = frames1[0]
    frames2 = frames2[0]
    vis_folder = 'visualize_results/csv_x3d_32_interctc_visualize/'
    if not os.path.exists(vis_folder):
        os.mkdir(vis_folder)
    

    sparse_frame2 = frames2[:, :, :, ::2]

    frames_list1 = []
    frames_list2 = []
    blank1 = seq_features1[:, :1]
    seq_features1 = seq_features1[:, 1:]
    blank2 = seq_features2[:, :1]
    seq_features2 = seq_features2[:, 1:]
    B, T, C = seq_features1.shape

    sparse_seq_features1 = torch.cat((blank1, seq_features1[:, ::2]), dim=1)
    sparse_seq_features2 = torch.cat((blank2, seq_features2[:, ::2]), dim=1)

    S = sparse_seq_features1.shape[1] - 1

    prob1_2 = (torch.einsum('bic,bjc->bij', seq_features1, sparse_seq_features2) / math.sqrt(C)).transpose(0, 1).softmax(2)
    
    
    for i in range(frames1.shape[-1]):
        frame = frames1[..., i]
        frame = frame.clone().detach().to(torch.device('cpu'))
        frame = frame * t_std + t_mean
        frame = frame.data.cpu().float().numpy().transpose(1,2,0)
        frame = cv2.cvtColor(frame * 255, cv2.COLOR_RGB2BGR)

        frames_list1.append(frame)

    for i in range(sparse_frame2.shape[-1]):
        frame = sparse_frame2[..., i]
        frame = frame.clone().detach().to(torch.device('cpu'))
        frame = frame * t_std + t_mean
        frame = frame.data.cpu().float().numpy().transpose(1,2,0)
        frame = cv2.cvtColor(frame * 255, cv2.COLOR_RGB2BGR)

        frames_list2.append(frame)
    
    frame_sequence1 = np.vstack(frames_list1)
    frame_sequence2 = np.hstack(frames_list2)
    blank = np.zeros_like(frame)

    plt.imsave(os.path.join(vis_folder, 'mat1_2.png'), prob1_2[:, :, 1:].squeeze(1).cpu())
    prob_mat1_2 = cv2.imread(os.path.join(vis_folder, 'mat1_2.png'))
    prob_mat1_2 = cv2.resize(prob_mat1_2, (frame_sequence2.shape[1], frame_sequence1.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    final1_2 = np.hstack([np.vstack([blank, frame_sequence1]), np.vstack([frame_sequence2, prob_mat1_2])])
    
    cv2.imwrite(os.path.join(vis_folder, 'final1_2.png'), final1_2)

    # prob2_1 = (torch.einsum('bic,bjc->bij', seq_features2, sparse_seq_features1) / math.sqrt(C)).transpose(0, 1).softmax(2)
    # plt.imsave(os.path.join(vis_folder, 'mat2_1.png'), prob2_1.squeeze(1).cpu())
    
    return

def vis_inter_ctc(frames1, frames2, seq_features1, seq_features2):
    frames1 = frames1[0]
    frames2 = frames2[0]
    vis_folder = 'visualize_results/csv_r50_32_interctc_visualize/'
    if not os.path.exists(vis_folder):
        os.mkdir(vis_folder)
    
    frames_list1 = []
    frames_list2 = []
    blank1 = seq_features1[:, :1]
    seq_features1 = seq_features1[:, 1:]
    blank2 = seq_features2[:, :1]
    seq_features2 = seq_features2[:, 1:]
    B, T, C = seq_features1.shape

    # sparse_seq_features2 = torch.cat((blank2, seq_features2), dim=1)

    prob1_2 = (torch.einsum('bic,bjc->bij', seq_features1, seq_features2) / math.sqrt(C)).transpose(0, 1).softmax(2)
    
    for i in range(frames1.shape[-1]):
        frame = frames1[..., i]
        frame = frame.clone().detach().to(torch.device('cpu'))
        frame = frame * t_std + t_mean
        frame = frame.data.cpu().float().numpy().transpose(1,2,0)
        frame = cv2.cvtColor(frame * 255, cv2.COLOR_RGB2BGR)

        frames_list1.append(frame)

    for i in range(frames2.shape[-1]):
        frame = frames2[..., i]
        frame = frame.clone().detach().to(torch.device('cpu'))
        frame = frame * t_std + t_mean
        frame = frame.data.cpu().float().numpy().transpose(1,2,0)
        frame = cv2.cvtColor(frame * 255, cv2.COLOR_RGB2BGR)

        frames_list2.append(frame)
    
    h, w, _ = frame.shape
    frame_sequence1 = np.hstack(frames_list1)
    frame_sequence2 = np.hstack(frames_list2)

    canvas = np.full_like(frame_sequence1, 255)

    prob1_2 = prob1_2.squeeze(1).cpu()

    for i, row in enumerate(prob1_2.tolist()):
        for j, v in enumerate(row):
            thickness = int(v * 6)
            if thickness > 0:
                canvas = cv2.line(canvas, (i * w + w // 2, 0), (j * w + w // 2, h), (0, 0, 0), thickness)
    final1_2 = np.vstack([frame_sequence1, canvas, frame_sequence2])
    cv2.imwrite(os.path.join(vis_folder, 'final1_2.png'), final1_2)
    return


def vis_block(frames1, frames2, seq_features1, seq_features2, label, indice):
    if not os.path.exists('visualize_results/similarity/'):
        os.mkdir('visualize_results/similarity/')
    frames1 = frames1[0]
    frames2 = frames2[0]
    frames_list1 = []
    frames_list2 = []
    for i in range(frames1.shape[-1]):
        frame = frames1[..., i]
        frame = frame.clone().detach().to(torch.device('cpu'))
        frame = frame * t_std + t_mean
        frame = frame.data.cpu().float().numpy().transpose(1,2,0)
        frame = cv2.cvtColor(frame * 255, cv2.COLOR_RGB2BGR)

        frames_list1.append(frame)

    for i in range(frames2.shape[-1]):
        frame = frames2[..., i]
        frame = frame.clone().detach().to(torch.device('cpu'))
        frame = frame * t_std + t_mean
        frame = frame.data.cpu().float().numpy().transpose(1,2,0)
        frame = cv2.cvtColor(frame * 255, cv2.COLOR_RGB2BGR)

        frames_list2.append(frame)
    
    frame_sequence1 = np.hstack(frames_list1)
    frame_sequence2 = np.hstack(frames_list2)


    B, T, C = seq_features1.shape
    # pred = 0.5 * ((torch.einsum('bic,bjc->bij', seq_features1, seq_features2) / math.sqrt(C)).softmax(-1) + \
    #               (torch.einsum('bic,bjc->bij', seq_features1, seq_features2) / math.sqrt(C)).softmax(-2))
    pred_origin = (torch.einsum('bic,bjc->bij', seq_features1, seq_features2) / math.sqrt(C))
    # pred_origin = torch.cosine_similarity(seq_features1.unsqueeze(2), seq_features2.unsqueeze(1), dim=-1)
    show_mat = pred_origin[0].detach().cpu().numpy()
    plt.matshow(show_mat)
    plt.title(str(label) + ' | ' +str(indice))
    plt.savefig('visualize_results/similarity/block_similarity_1.png')
    pred = pred_origin.cumsum(-2).cumsum(-1)
    
    D = torch.zeros((B, T, T, T), device=seq_features1.device)
    PATH_AREA = torch.zeros((B, T, T, T), device=seq_features1.device)
    D_ind = torch.zeros((B, T, T, T), dtype=torch.long, device=seq_features1.device)

    D[:, 0] = pred / torch.ones_like(pred).cumsum(-2).cumsum(-1)
    PATH_AREA[:, 0] = torch.ones_like(pred).cumsum(-2).cumsum(-1)

    area = torch.ones_like(pred).cumsum(-2).cumsum(-1)
    area = area[:, :, :, None, None] - area[:, :, None, None, :] - area.transpose(1,2)[:, None, :, :, None] + area[:, None, None, :, :]
    block_mat = pred[:, :, :, None, None] - pred[:, :, None, None, :] - pred.transpose(1,2)[:, None, :, :, None] + pred[:, None, None, :, :]
    
    i, j, a, b = torch.meshgrid(*[torch.arange(T, device=seq_features1.device)]*4)

    block_mat = block_mat.masked_fill(((a >= i) | (b >= j)).unsqueeze(0), float('-inf'))
    area = area.masked_fill(((a >= i) | (b >= j)).unsqueeze(0), float('1'))

    for k in range(1, T):
        tmp_pred = D[:, k-1, None, None, :, :] + block_mat
        tmp_area = PATH_AREA[:, k-1, None, None, :, :] + area
        
        max_indices = torch.argmax(tmp_pred.flatten(3) / tmp_area.flatten(3), -1, keepdim=True)
        D_ind[:, k] = max_indices.squeeze(-1)
        D[:, k] = torch.gather(input=tmp_pred.flatten(3), dim=-1, index=max_indices).squeeze(-1)
        PATH_AREA[:, k] = torch.gather(input=tmp_area.flatten(3), dim=-1, index=max_indices).squeeze(-1)

    weight = (D[:, :, T-1, T-1] / PATH_AREA[:, :, T-1, T-1]).softmax(-1)

    block_list = []
    block_scores = []

    for step_num in range(T):
        i, j = torch.full((B,), fill_value=T-1, device=seq_features1.device), torch.full((B,), fill_value=T-1, device=seq_features1.device)
        k = step_num
        
        block_choice = torch.full((B, T, T), False, device=seq_features1.device)

        while k >= 0:
            ind = D_ind[range(B), k-1, i, j]
            a = ind // T
            b = ind % T

            grid_x, grid_y = torch.meshgrid(*[torch.arange(T, device=seq_features1.device)]*2)

            if k != 0:
                block_choice = block_choice | \
                                ( (grid_x >  a[:, None, None]) \
                                & (grid_x <= i[:, None, None]) \
                                & (grid_y >  b[:, None, None]) \
                                & (grid_y <= j[:, None, None]))
            else:
                block_choice = block_choice | \
                    ( (grid_x <= i[:, None, None]) \
                    & (grid_y <= j[:, None, None]))

            i, j, k = a, b, k-1
            
        block_list.append(block_choice)
    # block_supervision = (torch.stack(block_list, dim=1) * weight[:, :, None, None]).sum(1)
    # visualize_mat = np.array((block_supervision[0]>0.3).detach().cpu())
    # plt.imshow(visualize_mat, cmap=plt.cm.Blues)
    # plt.title(str(label) + ' | ' + 'block_supervision')
    # plt.savefig('block_supervision.png')

    block_supervision = block_list[weight.argmax()]
    visualize_mat = np.array(block_supervision[0].detach().cpu())
    plt.imshow(visualize_mat, cmap=plt.cm.Blues)
    plt.title(str(label) + ' | ' + str(indice))
    plt.savefig(os.path.join('visualize_results/similarity/', 'matrix.png'))
    # plt.imsave(os.path.join('visualize_results/similarity/', 'mat.png'), visualize_mat)
    # prob_mat = cv2.imread(os.path.join('visualize_results/similarity/', 'mat.png'))
    # prob_mat = cv2.resize(prob_mat, (frame_sequence2.shape[1], frame_sequence1.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # final = np.hstack([np.vstack([blank, frame_sequence1]), np.vstack([frame_sequence2, prob_mat])])

    # cv2.imwrite(os.path.join('visualize_results/similarity/', 'final.png'), final)
    
    h, w, _ = frame.shape

    canvas = np.full_like(frame_sequence1, 255)
    
    prob1_2 = visualize_mat

    for i, row in enumerate(prob1_2.tolist()):
        for j, v in enumerate(row):
            thickness = int(v * 6)
            if thickness > 0:
                canvas = cv2.line(canvas, (i * w + w // 2, 0), (j * w + w // 2, h), (0, 0, 0), thickness)
    final1_2 = np.vstack([frame_sequence1, canvas, frame_sequence2])

    label_canvas1 = np.full_like(frame, 255)
    cv2.putText(label_canvas1, str(label), (100, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0, 0, 0))
    middle_blank = np.full_like(frame, 255)
    label_canvas2 = np.full_like(frame, 255)
    cv2.putText(label_canvas2, str(label), (100, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0, 0, 0))

    complete_label = np.vstack([label_canvas1, middle_blank, label_canvas2])

    final_image = np.hstack([complete_label, final1_2])

    cv2.imwrite(os.path.join('visualize_results/similarity/', 'match.png'), final_image)

    return


def visualize_finch(frames1, frames2, features1, features2):
    input_data1 = features1[0].cpu().detach().numpy()
    cluster_label1 = FINCH(data=input_data1, distance='euclidean')
    
    input_data2 = features2[0].cpu().detach().numpy()
    cluster_label2 = FINCH(data=input_data2, distance='euclidean')
    
    frame_list1 = []
    for i in range(frames1.shape[-1]):
        frame = frames1[..., i]
        frame = frame.clone().detach().to(torch.device('cpu'))
        frame = frame * t_std + t_mean
        frame = frame.data.cpu().float().numpy().transpose(1,2,0)
        frame = cv2.cvtColor(frame * 255, cv2.COLOR_RGB2BGR)
        caption = np.full((50, 320, 3), 255, np.uint8)
        cv2.putText(caption, str(cluster_label1[i].item()), (100, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 0))
        frame = np.vstack((frame, caption))
        frame_list1.append(frame)

    frame_sequence1 = np.hstack(frame_list1)
    
    frame_list2 = []
    for i in range(frames2.shape[-1]):
        frame = frames2[..., i]
        frame = frame.clone().detach().to(torch.device('cpu'))
        frame = frame * t_std + t_mean
        frame = frame.data.cpu().float().numpy().transpose(1,2,0)
        frame = cv2.cvtColor(frame * 255, cv2.COLOR_RGB2BGR)
        caption = np.full((50, 320, 3), 255, np.uint8)
        cv2.putText(caption, str(cluster_label2[i].item()), (100, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 0))
        frame = np.vstack((frame, caption))
        frame_list2.append(frame)

    frame_sequence2 = np.hstack(frame_list2)
    
    frame_sequence = np.vstack((frame_sequence1, frame_sequence2))

    cv2.imwrite('finch_result.png', frame_sequence)
    
    return cluster_label1, cluster_label2


def vis_block_diagonal(seq_features1, seq_features2):
    vis_path = 'vis_results/block_diagonal.png'

    B, T, C = seq_features1.shape

    
    prob1_2 = (torch.einsum('bic,bjc->bij', seq_features1, seq_features2) / math.sqrt(C)).softmax(-1)
    # prob1_2 = torch.cosine_similarity(seq_features1.unsqueeze(2), seq_features2.unsqueeze(1), dim=-1).softmax(-1)
    rel_mat = prob1_2[0].detach().cpu().numpy()
    
    
    plt.matshow(rel_mat, cmap=plt.cm.Blues)
    plt.savefig(vis_path)

    return
