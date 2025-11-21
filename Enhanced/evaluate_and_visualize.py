#!/usr/bin/env python
# coding: utf-8
"""
模型评估和可视化脚本 - 用于课程项目展示
生成完整的评估报告和可视化图表
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import os
import sys
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import classification_report
import logging
from datetime import datetime

# 导入项目模块
from metrics import multilabel_accuracy, calculate_all_metrics, print_metrics_summary, find_optimal_threshold
from utils import get_enhanced_augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==================== 配置 ====================
script_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE_DIR = os.path.join(script_dir, 'out')
EVALUATION_DIR = os.path.join(OUTPUT_BASE_DIR, 'evaluation')
VISUALIZATIONS_DIR = os.path.join(EVALUATION_DIR, 'visualizations')
REPORTS_DIR = os.path.join(EVALUATION_DIR, 'reports')

os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== 路径配置 ====================
parent_dir = os.path.dirname(script_dir)
data_dir = os.path.join(parent_dir, 'plant-pathology-2021-fgvc8')
if not os.path.exists(data_dir):
    data_dir = './plant-pathology-2021-fgvc8'

path = data_dir + '/' if not data_dir.endswith('/') else data_dir
TRAIN_DIR = os.path.join(path, 'train_images')
TEST_DIR = os.path.join(path, 'test_images')
TRAIN_DATA_FILE = os.path.join(path, 'train.csv')

CLASSES = ['rust', 'complex', 'healthy', 'powdery_mildew', 'scab', 'frog_eye_leaf_spot']
IM_SIZE = 299
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ==================== 数据集类 ====================
def get_image(image_id, kind='train'):
    from PIL import Image
    folders = {
        'train': TRAIN_DIR,
        'val': TRAIN_DIR,
        'test': TEST_DIR
    }
    fname = os.path.join(folders[kind], image_id)
    return Image.open(fname)

class PlantDataset(Dataset):
    def __init__(self, image_ids, targets, transform=None, kind='train'):
        self.image_ids = image_ids
        self.targets = targets
        self.transform = transform
        self.kind = kind
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img = np.array(get_image(self.image_ids.iloc[idx], kind=self.kind))
        if self.transform:
            img = self.transform(image=img)['image']
        target = self.targets[idx]
        return img, target

# ==================== 加载模型 ====================
def load_model(model_path):
    """加载训练好的模型"""
    model = torchvision.models.inception_v3(pretrained=False)
    model.aux_logits = False
    model.fc = nn.Sequential(
        nn.Linear(2048, 2048),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(2048, 6),
        nn.Sigmoid()
    )
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model'])
    model = model.to(DEVICE)
    model.eval()
    
    logger.info(f"成功加载模型: {model_path}")
    if 'best_accuracy' in checkpoint:
        logger.info(f"模型最佳准确率: {checkpoint['best_accuracy']:.4f}")
    if 'best_f1' in checkpoint:
        logger.info(f"模型最佳F1分数: {checkpoint['best_f1']:.4f}")
    
    return model

# ==================== 可视化函数 ====================

def plot_metrics_comparison(metrics_dict, save_path):
    """绘制多个指标的对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 主要指标对比
    ax = axes[0, 0]
    main_metrics = {
        'Multilabel\nAccuracy': metrics_dict['multilabel_accuracy'],
        'F1 Score\n(Macro)': metrics_dict['f1_macro'],
        'Jaccard\n(Macro)': metrics_dict['jaccard_macro']
    }
    bars = ax.bar(main_metrics.keys(), main_metrics.values(), color=['#2ecc71', '#3498db', '#9b59b6'])
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('主要评估指标对比', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    # 2. 精确率和召回率对比
    ax = axes[0, 1]
    x = np.arange(len(CLASSES))
    width = 0.35
    precision_scores = [metrics_dict.get(f'precision_class_{i}', 0) for i in range(len(CLASSES))]
    recall_scores = [metrics_dict.get(f'recall_class_{i}', 0) for i in range(len(CLASSES))]
    
    ax.bar(x - width/2, precision_scores, width, label='Precision', color='#e74c3c')
    ax.bar(x + width/2, recall_scores, width, label='Recall', color='#f39c12')
    ax.set_xlabel('类别', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('各类别精确率和召回率', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    # 3. 多标签混淆矩阵热力图（简化版）
    ax = axes[1, 0]
    # 计算每个类别的准确率
    class_accuracies = []
    for i, cls in enumerate(CLASSES):
        # 这里简化处理，实际应该计算每个类别的TP, FP, TN, FN
        class_accuracies.append(metrics_dict.get(f'class_{i}_accuracy', 0))
    
    im = ax.imshow(np.array(class_accuracies).reshape(2, 3), cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(3))
    ax.set_yticks(range(2))
    ax.set_xticklabels(CLASSES[:3], rotation=45, ha='right')
    ax.set_yticklabels(CLASSES[3:])
    ax.set_title('各类别准确率热力图', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax)
    
    # 4. 指标雷达图
    ax = axes[1, 1]
    categories = ['Accuracy', 'F1\nMacro', 'F1\nMicro', 'Jaccard\nMacro', 'Precision\nMacro', 'Recall\nMacro']
    values = [
        metrics_dict['multilabel_accuracy'],
        metrics_dict['f1_macro'],
        metrics_dict['f1_micro'],
        metrics_dict['jaccard_macro'],
        metrics_dict['precision_macro'],
        metrics_dict['recall_macro']
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]  # 闭合
    angles += angles[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, color='#3498db')
    ax.fill(angles, values, alpha=0.25, color='#3498db')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('综合性能雷达图', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"指标对比图已保存: {save_path}")

def plot_confusion_matrices(y_true, y_pred, save_path):
    """绘制每个类别的混淆矩阵"""
    cm_matrices = multilabel_confusion_matrix(y_true, y_pred)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('各类别混淆矩阵', fontsize=16, fontweight='bold')
    
    for idx, (cm, class_name) in enumerate(zip(cm_matrices, CLASSES)):
        ax = axes[idx // 3, idx % 3]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   cbar_kws={'label': 'Count'})
        ax.set_xlabel('预测标签', fontsize=10)
        ax.set_ylabel('真实标签', fontsize=10)
        ax.set_title(f'{class_name}', fontsize=12, fontweight='bold')
        ax.set_xticklabels(['Negative', 'Positive'])
        ax.set_yticklabels(['Negative', 'Positive'])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"混淆矩阵图已保存: {save_path}")

def plot_class_performance(y_true, y_pred, save_path):
    """绘制每个类别的详细性能指标"""
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # 计算每个类别的准确率（简化版）
    class_accuracies = []
    for i in range(len(CLASSES)):
        true_pos = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
        true_neg = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 0))
        total = len(y_true)
        class_accuracies.append((true_pos + true_neg) / total)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 精确率
    ax = axes[0, 0]
    bars = ax.barh(CLASSES, precision, color='#e74c3c')
    ax.set_xlabel('Precision', fontsize=12)
    ax.set_title('各类别精确率', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    for i, (bar, val) in enumerate(zip(bars, precision)):
        ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    
    # 2. 召回率
    ax = axes[0, 1]
    bars = ax.barh(CLASSES, recall, color='#f39c12')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_title('各类别召回率', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    for i, (bar, val) in enumerate(zip(bars, recall)):
        ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    
    # 3. F1分数
    ax = axes[1, 0]
    bars = ax.barh(CLASSES, f1, color='#3498db')
    ax.set_xlabel('F1 Score', fontsize=12)
    ax.set_title('各类别F1分数', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    for i, (bar, val) in enumerate(zip(bars, f1)):
        ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    
    # 4. 样本数量和支持度
    ax = axes[1, 1]
    bars = ax.barh(CLASSES, support, color='#9b59b6')
    ax.set_xlabel('样本数量', fontsize=12)
    ax.set_title('各类别样本数量', fontsize=14, fontweight='bold')
    for i, (bar, val) in enumerate(zip(bars, support)):
        ax.text(val + 50, i, f'{int(val)}', va='center', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"类别性能图已保存: {save_path}")

def plot_error_analysis(y_true, y_pred, y_pred_proba, save_path):
    """错误分析图"""
    # 计算每个样本的预测是否正确
    exact_matches = np.all(y_true == y_pred, axis=1)
    error_rate = 1 - np.mean(exact_matches)
    
    # 计算每个样本预测的标签数量
    pred_label_counts = np.sum(y_pred, axis=1)
    true_label_counts = np.sum(y_true, axis=1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 正确/错误分布
    ax = axes[0, 0]
    correct_count = np.sum(exact_matches)
    error_count = len(exact_matches) - correct_count
    ax.pie([correct_count, error_count], 
           labels=[f'正确 ({correct_count})', f'错误 ({error_count})'],
           autopct='%1.1f%%', startangle=90,
           colors=['#2ecc71', '#e74c3c'])
    ax.set_title(f'预测准确率分布\n(总体准确率: {1-error_rate:.2%})', 
                 fontsize=14, fontweight='bold')
    
    # 2. 预测标签数量分布
    ax = axes[0, 1]
    unique_pred, counts_pred = np.unique(pred_label_counts, return_counts=True)
    unique_true, counts_true = np.unique(true_label_counts, return_counts=True)
    x = np.arange(max(len(unique_pred), len(unique_true)))
    width = 0.35
    
    pred_counts_arr = np.zeros(len(x))
    true_counts_arr = np.zeros(len(x))
    for i, (u, c) in enumerate(zip(unique_pred, counts_pred)):
        if u < len(x):
            pred_counts_arr[u] = c
    for i, (u, c) in enumerate(zip(unique_true, counts_true)):
        if u < len(x):
            true_counts_arr[u] = c
    
    ax.bar(x - width/2, pred_counts_arr, width, label='预测', color='#3498db', alpha=0.7)
    ax.bar(x + width/2, true_counts_arr, width, label='真实', color='#2ecc71', alpha=0.7)
    ax.set_xlabel('标签数量', fontsize=12)
    ax.set_ylabel('样本数量', fontsize=12)
    ax.set_title('每个样本的标签数量分布', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 3. 错误类型分析（多预测 vs 少预测）
    ax = axes[1, 0]
    false_positives = np.sum((y_pred == 1) & (y_true == 0))
    false_negatives = np.sum((y_pred == 0) & (y_true == 1))
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    true_negatives = np.sum((y_pred == 0) & (y_true == 0))
    
    error_types = ['假阳性\n(多预测)', '假阴性\n(少预测)', '真阳性', '真阴性']
    error_counts = [false_positives, false_negatives, true_positives, true_negatives]
    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#95a5a6']
    
    bars = ax.bar(error_types, error_counts, color=colors)
    ax.set_ylabel('标签数量', fontsize=12)
    ax.set_title('错误类型分析', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, error_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}',
                ha='center', va='bottom', fontsize=10)
    
    # 4. 置信度分布（预测概率）
    ax = axes[1, 1]
    # 只分析预测为正类的概率
    positive_probs = y_pred_proba[y_pred == 1]
    negative_probs = y_pred_proba[y_pred == 0]
    
    ax.hist(positive_probs.flatten(), bins=20, alpha=0.7, label='预测为正类', color='#3498db')
    ax.hist(negative_probs.flatten(), bins=20, alpha=0.7, label='预测为负类', color='#95a5a6')
    ax.set_xlabel('预测概率', fontsize=12)
    ax.set_ylabel('频数', fontsize=12)
    ax.set_title('预测概率分布', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"错误分析图已保存: {save_path}")

# ==================== 主评估函数 ====================
def evaluate_model(model_path, threshold=0.4):
    """完整评估模型"""
    logger.info("="*60)
    logger.info("开始模型评估")
    logger.info("="*60)
    
    # 加载模型
    model = load_model(model_path)
    
    # 加载数据
    train_df = pd.read_csv(TRAIN_DATA_FILE).set_index('image')
    
    # 标签处理
    def get_single_labels(unique_labels):
        single_labels = []
        for label in unique_labels:
            single_labels += label.split()
        return list(set(single_labels))
    
    def get_one_hot_encoded_labels(dataset_df):
        df = dataset_df.copy()
        unique_labels = df.labels.unique()
        column_names = get_single_labels(unique_labels)
        df[column_names] = 0
        for label in unique_labels:
            label_indices = df[df['labels'] == label].index
            splited_labels = label.split()
            df.loc[label_indices, splited_labels] = 1
        return df
    
    tr_df = get_one_hot_encoded_labels(train_df)
    
    # 数据分割（使用与训练时相同的随机种子）
    from sklearn.model_selection import train_test_split
    X_Train, X_Valid, Y_Train, Y_Valid = train_test_split(
        pd.Series(train_df.index),
        np.array(tr_df[CLASSES]),
        test_size=0.2,
        random_state=42
    )
    
    # 创建数据加载器
    _, val_transform = get_enhanced_augmentations(IM_SIZE, use_advanced=False)
    validset = PlantDataset(X_Valid, Y_Valid, transform=val_transform, kind='val')
    validloader = DataLoader(validset, batch_size=16, shuffle=False)
    
    # 预测
    logger.info("正在进行预测...")
    all_preds_proba = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in validloader:
            images = images.to(DEVICE)
            pred_proba = model(images.float()).cpu().numpy()
            all_preds_proba.append(pred_proba)
            all_labels.append(labels.numpy())
    
    y_pred_proba = np.vstack(all_preds_proba)
    y_true = np.vstack(all_labels)
    y_pred = (y_pred_proba > threshold).astype(int)
    
    # 计算所有指标
    logger.info("计算评估指标...")
    metrics = calculate_all_metrics(y_true, y_pred, y_pred_proba, threshold)
    
    # 打印指标摘要
    print_metrics_summary(metrics, "模型评估结果")
    
    # 寻找最优阈值
    logger.info("寻找最优阈值...")
    best_threshold, best_score = find_optimal_threshold(
        y_true, y_pred_proba, 
        metric='multilabel_accuracy',
        threshold_range=(0.1, 0.9),
        step=0.05
    )
    logger.info(f"最优阈值: {best_threshold:.3f}, 准确率: {best_score:.4f}")
    
    # 使用最优阈值重新计算
    y_pred_optimal = (y_pred_proba > best_threshold).astype(int)
    metrics_optimal = calculate_all_metrics(y_true, y_pred_optimal, y_pred_proba, best_threshold)
    
    # 生成可视化
    logger.info("生成可视化图表...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plot_metrics_comparison(metrics, 
                           os.path.join(VISUALIZATIONS_DIR, f'metrics_comparison_{timestamp}.png'))
    plot_confusion_matrices(y_true, y_pred,
                           os.path.join(VISUALIZATIONS_DIR, f'confusion_matrices_{timestamp}.png'))
    plot_class_performance(y_true, y_pred,
                          os.path.join(VISUALIZATIONS_DIR, f'class_performance_{timestamp}.png'))
    plot_error_analysis(y_true, y_pred, y_pred_proba,
                       os.path.join(VISUALIZATIONS_DIR, f'error_analysis_{timestamp}.png'))
    
    # 保存详细报告
    report_path = os.path.join(REPORTS_DIR, f'evaluation_report_{timestamp}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("模型评估报告\n")
        f.write("="*60 + "\n\n")
        f.write(f"模型路径: {model_path}\n")
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"使用阈值: {threshold}\n")
        f.write(f"最优阈值: {best_threshold:.3f}\n\n")
        
        f.write("【Kaggle官方指标】\n")
        f.write(f"多标签准确率: {metrics['multilabel_accuracy']:.4f}\n")
        f.write(f"最优阈值下的准确率: {metrics_optimal['multilabel_accuracy']:.4f}\n\n")
        
        f.write("【其他指标】\n")
        for key, value in metrics.items():
            if key != 'multilabel_accuracy':
                f.write(f"{key}: {value:.4f}\n")
        
        f.write("\n【分类报告】\n")
        from sklearn.metrics import classification_report
        report = classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0)
        f.write(report)
    
    logger.info(f"详细报告已保存: {report_path}")
    logger.info("评估完成！")
    
    return metrics, metrics_optimal, best_threshold

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='模型评估和可视化')
    parser.add_argument('--model', type=str, 
                       help='模型路径（如果不指定，自动查找最新的最佳模型）')
    parser.add_argument('--threshold', type=float, default=0.4,
                       help='预测阈值（默认0.4）')
    
    args = parser.parse_args()
    
    # 如果没有指定模型，自动查找
    if args.model is None:
        model_dir = os.path.join(OUTPUT_BASE_DIR, 'models', 'best')
        model_files = glob.glob(os.path.join(model_dir, "inception_v3_enhanced_best_epoch*.pth"))
        if model_files:
            model_files.sort(key=lambda x: int(x.split('epoch')[-1].split('.')[0]))
            args.model = model_files[-1]
            logger.info(f"自动选择模型: {args.model}")
        else:
            logger.error("找不到模型文件，请使用 --model 参数指定模型路径")
            sys.exit(1)
    
    evaluate_model(args.model, args.threshold)

