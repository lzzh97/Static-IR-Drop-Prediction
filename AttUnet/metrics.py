# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 18:49:01 2023

@author: Lizi Zhang
"""

import numpy as np
import torch

from sklearn.metrics import f1_score

def F1_Score(x, y):
    x_ = x
    y_ = y
    F_list = []
    for i in range(x_.shape[0]):
        thresh_o = x_[i,:].max() * .9
        x_[i,:][x_[i,:]<=thresh_o] = 0
        x_[i,:][x_[i,:]>thresh_o] = 1
        
        thresh_y = y_[i,0].max() * .9
        y_[i,:][y_[i,:]<=thresh_y] = 0
        y_[i,:][y_[i,:]>thresh_y] = 1
        
        F_list.append(f1_score(x_[i,:].flatten(), y_[i,:].flatten(), average='binary'))
    return F_list

def compute_f1_score(output, ground_truth):
    # Flatten the tensors
    output_flat = output.view(-1)
    ground_truth_flat = ground_truth.view(-1)
    
    # Determine the threshold for ground truth
    threshold = 0.9 * torch.max(ground_truth_flat)
    
    # Apply the threshold to ground truth to get binary labels
    ground_truth_binary = (ground_truth_flat >= threshold).float()
    
    # Binarize the output tensor (using 0.5 as threshold for this example)
    output_binary = (output_flat >= threshold).float()
    
    # Calculate true positives, false positives, and false negatives
    true_positives = (output_binary * ground_truth_binary).sum()
    false_positives = (output_binary * (1 - ground_truth_binary)).sum()
    false_negatives = ((1 - output_binary) * ground_truth_binary).sum()
    
    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    return f1_score.item()


def compute_masked_mae(output, ir, threshold = 0.95):
    threshold = torch.quantile(ir, 0.95)
    high_value_mask = ir > threshold

    masked_output = output[high_value_mask]
    masked_ir = ir[high_value_mask]
    
    return torch.mean(torch.abs(masked_output - masked_ir)).item()
    