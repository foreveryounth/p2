#!/usr/bin/env python
# coding: utf-8
"""
评估指标工具 - 用于Kaggle Plant Pathology 2021-FGVC8比赛

比赛官方评估指标：多标签分类准确率（Multi-label Accuracy）
- 只有当所有标签都预测正确时，才算正确
- 公式：准确率 = 完全正确的样本数 / 总样本数
"""

import numpy as np
import torch
from typing import Union, Tuple
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import hamming_loss, jaccard_score


def multilabel_accuracy(y_true: Union[np.ndarray, torch.Tensor], 
                        y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """
    计算多标签分类准确率（Kaggle比赛官方指标）
    
    对于多标签分类，准确率定义为：
    - 只有当所有标签都预测正确时，该样本才算正确
    - 准确率 = 完全正确的样本数 / 总样本数
    
    Args:
        y_true: 真实标签，形状为 (n_samples, n_classes) 的二进制数组
        y_pred: 预测标签，形状为 (n_samples, n_classes) 的二进制数组
    
    Returns:
        float: 多标签准确率 (0-1之间)
    """
    # 转换为numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # 确保是二进制数组
    y_true = (y_true > 0.5).astype(int)
    y_pred = (y_pred > 0.5).astype(int)
    
    # 计算每个样本是否完全正确
    exact_matches = np.all(y_true == y_pred, axis=1)
    
    # 准确率 = 完全正确的样本数 / 总样本数
    accuracy = np.mean(exact_matches)
    
    return float(accuracy)


def calculate_all_metrics(y_true: Union[np.ndarray, torch.Tensor],
                          y_pred: Union[np.ndarray, torch.Tensor],
                          y_pred_proba: Union[np.ndarray, torch.Tensor] = None,
                          threshold: float = 0.4) -> dict:
    """
    计算所有相关评估指标
    
    Args:
        y_true: 真实标签，形状为 (n_samples, n_classes)
        y_pred: 预测标签（二进制），形状为 (n_samples, n_classes)
        y_pred_proba: 预测概率（可选），形状为 (n_samples, n_classes)
        threshold: 用于将概率转换为二进制的阈值
    
    Returns:
        dict: 包含所有指标的字典
    """
    # 转换为numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if y_pred_proba is not None and isinstance(y_pred_proba, torch.Tensor):
        y_pred_proba = y_pred_proba.cpu().numpy()
    
    # 如果提供了概率，使用阈值转换为二进制
    if y_pred_proba is not None:
        y_pred = (y_pred_proba > threshold).astype(int)
    
    # 确保是二进制数组
    y_true = (y_true > 0.5).astype(int)
    y_pred = (y_pred > 0.5).astype(int)
    
    metrics = {}
    
    # 1. 多标签准确率（Kaggle官方指标）
    metrics['multilabel_accuracy'] = multilabel_accuracy(y_true, y_pred)
    
    # 2. Hamming Loss（越小越好，0-1之间）
    metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
    
    # 3. Jaccard Score（IoU，交集/并集）
    # 可以计算macro、micro、samples平均
    metrics['jaccard_macro'] = jaccard_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['jaccard_micro'] = jaccard_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['jaccard_samples'] = jaccard_score(y_true, y_pred, average='samples', zero_division=0)
    
    # 4. F1 Score（各种平均方式）
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['f1_samples'] = f1_score(y_true, y_pred, average='samples', zero_division=0)
    
    # 5. Precision（精确率）
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    
    # 6. Recall（召回率）
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
    
    # 7. 子集准确率（与multilabel_accuracy相同，但使用sklearn实现）
    metrics['subset_accuracy'] = accuracy_score(y_true, y_pred)
    
    # 8. 计算每个类别的指标
    for i, class_name in enumerate(['rust', 'complex', 'healthy', 'powdery_mildew', 'scab', 'frog_eye_leaf_spot']):
        if y_true.shape[1] > i:
            class_precision = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
            class_recall = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
            class_f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
            
            metrics[f'precision_{class_name}'] = class_precision
            metrics[f'recall_{class_name}'] = class_recall
            metrics[f'f1_{class_name}'] = class_f1
    
    return metrics


def find_optimal_threshold(y_true: Union[np.ndarray, torch.Tensor],
                          y_pred_proba: Union[np.ndarray, torch.Tensor],
                          metric: str = 'multilabel_accuracy',
                          threshold_range: Tuple[float, float] = (0.1, 0.9),
                          step: float = 0.05) -> Tuple[float, float]:
    """
    寻找最优阈值（基于指定指标）
    
    Args:
        y_true: 真实标签
        y_pred_proba: 预测概率
        metric: 要优化的指标名称（'multilabel_accuracy', 'f1_macro', 'jaccard_macro'等）
        threshold_range: 阈值搜索范围
        step: 搜索步长
    
    Returns:
        Tuple[最优阈值, 最优指标值]
    """
    # 转换为numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred_proba, torch.Tensor):
        y_pred_proba = y_pred_proba.cpu().numpy()
    
    best_threshold = threshold_range[0]
    best_score = 0.0
    
    thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
    
    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        metrics = calculate_all_metrics(y_true, y_pred)
        
        score = metrics.get(metric, 0.0)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


def print_metrics_summary(metrics: dict, title: str = "评估指标"):
    """
    打印指标摘要（格式化输出）
    
    Args:
        metrics: 指标字典
        title: 标题
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    # Kaggle官方指标（最重要）
    print(f"\n【Kaggle官方指标】")
    print(f"  多标签准确率 (Multilabel Accuracy): {metrics['multilabel_accuracy']:.4f}")
    
    # 其他重要指标
    print(f"\n【其他重要指标】")
    print(f"  Hamming Loss: {metrics['hamming_loss']:.4f} (越小越好)")
    print(f"  Jaccard Score (Macro): {metrics['jaccard_macro']:.4f}")
    print(f"  Jaccard Score (Micro): {metrics['jaccard_micro']:.4f}")
    print(f"  F1 Score (Macro): {metrics['f1_macro']:.4f}")
    print(f"  F1 Score (Micro): {metrics['f1_micro']:.4f}")
    print(f"  Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (Macro): {metrics['recall_macro']:.4f}")
    
    # 各类别指标
    print(f"\n【各类别F1分数】")
    classes = ['rust', 'complex', 'healthy', 'powdery_mildew', 'scab', 'frog_eye_leaf_spot']
    for cls in classes:
        f1_key = f'f1_{cls}'
        if f1_key in metrics:
            print(f"  {cls:20s}: {metrics[f1_key]:.4f}")
    
    print(f"{'='*60}\n")

