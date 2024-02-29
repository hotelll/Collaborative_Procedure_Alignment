import json
import os
from random import sample
import math


ELASTIC_RATIOS = [1.0, 0.8, 0.6, 0.4, 0.2]
QUERY_NUM = 20
SAMPLE_NUM = 100
ROOT_PATH = 'Datasets/Elastic-Matching'




def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.loads(f.read())
    return data


def length_LIS(sequence):
    if sequence == []:
        return 0
    seq_length = len(sequence)
    dp = [1] * seq_length
    for i in range(seq_length):
        for j in range(i):
            if sequence[j] < sequence[i]:
                dp[i] = max(dp[i], dp[j]+1)
    return max(dp)


def consist_number_of_pair():
    label_bank_path = 'Datasets/CSV/label_bank.json'
    label_bank = read_json(label_bank_path)

    test_label_bank = {}
    for key in label_bank:
        if float(key) < 7:
            test_label_bank[key] = label_bank[key]

    pair_match_step = {}
    for q_key in test_label_bank:
        query_procedure = test_label_bank[q_key]

        for c_key in test_label_bank:
            query_procedure_copy = query_procedure.copy()
            candidate_procedure =test_label_bank[c_key]
            consist_step_id = []
            for step in candidate_procedure:
                if step in query_procedure_copy:
                    step_id = query_procedure_copy.index(step)
                    query_procedure_copy[step_id] = 'MASK'
                    consist_step_id.append(step_id)

            consist_step_num = length_LIS(consist_step_id)
            key = '{}-{}'.format(q_key, c_key)
            pair_match_step[key] = consist_step_num
            
    return pair_match_step


def get_candidate_videos():
    test_split_path = 'Datasets/CSV/test_split.txt'    
    candidate_videos = []
    with open(test_split_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            video, label = line.strip().split(' ')
            candidate_videos.append([label, video])
    return candidate_videos


def choose_queries():
    test_video = {}
    test_split_path = 'Datasets/CSV/test_split.txt'
    with open(test_split_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            video, label = line.strip().split(' ')

            video_name = "{}/{}".format(label, video)
            if label not in test_video:
                test_video[label] = [video_name]
            else:
                test_video[label].append(video_name)

    procedure_list = ['1.5', '2.5', '3.5', '4.5', '5.5']
    chosen_queries = []
    for procedure in procedure_list:
        video_list = test_video[procedure]
        chosen_queries += sample(video_list, QUERY_NUM)

    return chosen_queries


def get_elastic_pair(pair_match_step, candidate_videos, chosen_queries, elastic_ratio):
    label_bank_path = 'Datasets/CSV/label_bank.json'
    label_bank = read_json(label_bank_path)
    
    elastic_dir = os.path.join(ROOT_PATH, 'elastic_pair_{}/'.format(elastic_ratio))
    if not os.path.exists(elastic_dir):
        os.makedirs(elastic_dir)
        
    for query in chosen_queries:
        pos_num, neg_num, pos_list, neg_list = 0, 0, [], []
        query_label, query_video = query.split('/')
        query_name = '{}-{}.txt'.format(query_label, query_video.split('.')[0])
        query_step_num = len(label_bank[query_label])
        elastic_step_thresh = math.ceil(query_step_num * elastic_ratio)
        elastic_pair_path = os.path.join(elastic_dir, query_name)
        
        with open(elastic_pair_path, 'w') as f:
            for candidate in candidate_videos:
                candidate_label, candidate_video = candidate
                consist_step_num = pair_match_step['{}-{}'.format(query_label, candidate_label)]
                matched_label = (consist_step_num >= elastic_step_thresh)
                
                if matched_label: # this is a positive pair
                    sample_info = '{} {} {} {} {} {}/{}\n'.format(query_video, query_label, 
                                                                  candidate_video, candidate_label, 
                                                                  matched_label, query_step_num, elastic_step_thresh)
                    pos_list.append(sample_info)
                else: # this is a negative pair
                    sample_info = '{} {} {} {} {} {}/{}\n'.format(query_video, query_label, 
                                                                  candidate_video, candidate_label, 
                                                                  matched_label, query_step_num, elastic_step_thresh)
                    neg_list.append(sample_info)
            
            if len(pos_list) < SAMPLE_NUM // 2:
                sampled_pos_list = pos_list
                pos_num = len(pos_list)
            else:
                sampled_pos_list = sample(pos_list, SAMPLE_NUM // 2)
                pos_num = SAMPLE_NUM // 2
            
            sampled_neg_list = sample(neg_list, SAMPLE_NUM - pos_num)
            sampled_list = sampled_pos_list + sampled_neg_list
            for sample_info in sampled_list:
                f.writelines(sample_info)
    return


def get_all_elastic_pairs(elastic_ratio):
    pair_match_step = consist_number_of_pair()
    candidate_videos = get_candidate_videos()
    chosen_queries = choose_queries()
    get_elastic_pair(pair_match_step, candidate_videos, chosen_queries, elastic_ratio)

    return


if __name__ == '__main__':
    
    for elastic_ratio in ELASTIC_RATIOS:
        get_all_elastic_pairs(elastic_ratio)
    