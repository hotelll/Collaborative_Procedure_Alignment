import torch
import os
import numpy as np
import json


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


if __name__ == "__main__":
    eval_result_path = "eval_logs/baseline/elastic_matching_1.0_results.txt"
    # eval_result_path = "eval_logs/elastic_matching_1.0_results.txt"
    # eval_result_path = "eval_logs/sampleStep_cls_blank/elastic_matching_0.8_results.txt"
    ground_truth_path = "Datasets/Elastic-Matching/elastic_pair_1.0.json"
    
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
        # threshold = 6
        # preds = torch.tensor(preds) > threshold
        threshold = 1.3
        
        preds = torch.tensor(preds) < threshold
        
        
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
    # print("Precision: ", average_precision)
    # print("Recall: ", average_recall)
    print("F1 Score: ", average_f1_score)

    # fpr, tpr, thresholds = roc_curve(labels.cpu().detach().numpy(), preds.cpu().detach().numpy(), pos_label=0)
    # auc_value = auc(fpr, tpr)
    # wdr_value = compute_WDR(preds, labels1_all, labels2_all)