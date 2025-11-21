#!/usr/bin/env python
# coding: utf-8
"""
ConvNeXt模型评估脚本
用于全面评估训练好的ConvNeXt模型性能，包括多种指标和可视化
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import sys
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime
import PIL
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 导入评估指标
from metrics import multilabel_accuracy, calculate_all_metrics, print_metrics_summary, find_optimal_threshold

# ==================== 配置 ====================
script_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE_DIR = os.path.join(script_dir, 'out')
VISUALIZATIONS_DIR = os.path.join(OUTPUT_BASE_DIR, 'models', 'visualizations')
REPORTS_DIR = os.path.join(OUTPUT_BASE_DIR, 'models', 'reports')

os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# 设置字体（使用默认字体，避免中文字体问题）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== 路径和参数配置 ====================
BATCH = 16
IM_SIZE = 224  # ConvNeXt使用224x224输入
CONVNEXT_VARIANT = 'convnext_base'  # 默认模型变体，可通过命令行参数覆盖
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 自动查找数据目录（从当前脚本目录向上查找）
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
path_candidates = [
    os.path.join(script_dir, 'plant-pathology-2021-fgvc8'),
    os.path.join(parent_dir, 'plant-pathology-2021-fgvc8'),
    './plant-pathology-2021-fgvc8',
    '../plant-pathology-2021-fgvc8'
]

path = None
for candidate in path_candidates:
    if os.path.exists(os.path.join(candidate, 'train.csv')):
        path = candidate + '/' if not candidate.endswith('/') else candidate
        break

if path is None:
    raise FileNotFoundError(
        "找不到数据集目录！请确保plant-pathology-2021-fgvc8目录存在，"
        "并且包含train.csv文件。"
    )

TRAIN_DIR = os.path.join(path, 'train_images')
TEST_DIR = os.path.join(path, 'test_images')
TRAIN_DATA_FILE = os.path.join(path, 'train.csv')

CLASSES = ['rust', 'complex', 'healthy', 'powdery_mildew', 'scab', 'frog_eye_leaf_spot']

folders = dict({
    'data': path,
    'train': TRAIN_DIR,
    'val': TRAIN_DIR,
    'test': TEST_DIR
})

# ==================== 数据加载函数 ====================
def get_image(image_id, kind='train'):
    """加载图像"""
    fname = os.path.join(folders[kind], image_id)
    return Image.open(fname)

def get_single_labels(unique_labels):
    """获取所有单个标签"""
    single_labels = []
    for label in unique_labels:
        single_labels += label.split()
    return list(set(single_labels))

def get_one_hot_encoded_labels(dataset_df):
    """将标签转换为one-hot编码"""
    df = dataset_df.copy()
    unique_labels = df.labels.unique()
    column_names = get_single_labels(unique_labels)
    df[column_names] = 0
    
    for label in unique_labels:
        label_indices = df[df['labels'] == label].index
        splited_labels = label.split()
        df.loc[label_indices, splited_labels] = 1
    
    return df

# ==================== 数据集类 ====================
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

# ==================== ConvNeXt模型加载 ====================
def create_convnext_model(variant='convnext_base', num_classes=6):
    """
    创建ConvNeXt模型
    
    Args:
        variant: 模型变体 ('convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large')
        num_classes: 输出类别数
    """
    # 兼容新旧版本的PyTorch/torchvision
    try:
        # 新版本API (torchvision >= 0.13)
        if variant == 'convnext_tiny':
            model = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            feature_dim = 768
        elif variant == 'convnext_small':
            model = torchvision.models.convnext_small(weights=torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
            feature_dim = 768
        elif variant == 'convnext_base':
            model = torchvision.models.convnext_base(weights=torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
            feature_dim = 1024
        elif variant == 'convnext_large':
            model = torchvision.models.convnext_large(weights=torchvision.models.ConvNeXt_Large_Weights.IMAGENET1K_V1)
            feature_dim = 1536
        else:
            raise ValueError(f"未知的ConvNeXt变体: {variant}")
    except (AttributeError, TypeError):
        # 旧版本API (torchvision < 0.13)
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, message='.*pretrained.*')
        if variant == 'convnext_tiny':
            model = torchvision.models.convnext_tiny(pretrained=True)
            feature_dim = 768
        elif variant == 'convnext_small':
            model = torchvision.models.convnext_small(pretrained=True)
            feature_dim = 768
        elif variant == 'convnext_base':
            model = torchvision.models.convnext_base(pretrained=True)
            feature_dim = 1024
        elif variant == 'convnext_large':
            model = torchvision.models.convnext_large(pretrained=True)
            feature_dim = 1536
        else:
            raise ValueError(f"未知的ConvNeXt变体: {variant}")
    
    # 替换分类头用于多标签分类
    model.classifier = nn.Sequential(
        nn.LayerNorm((feature_dim,), eps=1e-6, elementwise_affine=True),
        nn.Flatten(start_dim=1),
        nn.Linear(feature_dim, feature_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(feature_dim, num_classes),
        nn.Sigmoid()
    )
    
    return model

def load_model(model_path, variant=None):
    """加载训练好的ConvNeXt模型"""
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # 从checkpoint中获取变体信息（如果存在）
    if variant is None:
        variant = checkpoint.get('variant', CONVNEXT_VARIANT)
    
    logger.info(f"加载模型变体: {variant}")
    
    model = create_convnext_model(variant, num_classes=len(CLASSES))
    model.load_state_dict(checkpoint['model'])
    model = model.to(DEVICE)
    model.eval()
    
    logger.info(f"成功加载模型: {model_path}")
    if 'best_accuracy' in checkpoint:
        logger.info(f"模型最佳准确率: {checkpoint['best_accuracy']:.4f}")
    
    return model, variant

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
    ax.set_title('Main Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    # 2. 精确率和召回率对比
    ax = axes[0, 1]
    x = np.arange(len(CLASSES))
    width = 0.35
    precision_scores = [metrics_dict.get(f'precision_{cls}', 0) for cls in CLASSES]
    recall_scores = [metrics_dict.get(f'recall_{cls}', 0) for cls in CLASSES]
    
    ax.bar(x - width/2, precision_scores, width, label='Precision', color='#e74c3c')
    ax.bar(x + width/2, recall_scores, width, label='Recall', color='#f39c12')
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Precision and Recall by Class', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    # 3. 各类别F1分数
    ax = axes[1, 0]
    f1_scores = [metrics_dict.get(f'f1_{cls}', 0) for cls in CLASSES]
    bars = ax.barh(CLASSES, f1_scores, color='#3498db')
    ax.set_xlabel('F1 Score', fontsize=12)
    ax.set_title('F1 Score by Class', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    for i, (bar, val) in enumerate(zip(bars, f1_scores)):
        ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    
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
    values += values[:1]
    angles += angles[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, color='#3498db')
    ax.fill(angles, values, alpha=0.25, color='#3498db')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Overall Performance Radar', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"指标对比图已保存: {save_path}")

def plot_confusion_matrices(y_true, y_pred, save_path):
    """绘制每个类别的混淆矩阵"""
    cm_matrices = multilabel_confusion_matrix(y_true, y_pred)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Confusion Matrices by Class', fontsize=16, fontweight='bold')
    
    for idx, (cm, class_name) in enumerate(zip(cm_matrices, CLASSES)):
        ax = axes[idx // 3, idx % 3]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)
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
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 精确率
    ax = axes[0, 0]
    bars = ax.barh(CLASSES, precision, color='#e74c3c')
    ax.set_xlabel('Precision', fontsize=12)
    ax.set_title('Precision by Class', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    for i, (bar, val) in enumerate(zip(bars, precision)):
        ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    
    # 2. 召回率
    ax = axes[0, 1]
    bars = ax.barh(CLASSES, recall, color='#f39c12')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_title('Recall by Class', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    for i, (bar, val) in enumerate(zip(bars, recall)):
        ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    
    # 3. F1分数
    ax = axes[1, 0]
    bars = ax.barh(CLASSES, f1, color='#3498db')
    ax.set_xlabel('F1 Score', fontsize=12)
    ax.set_title('F1 Score by Class', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    for i, (bar, val) in enumerate(zip(bars, f1)):
        ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    
    # 4. 样本数量和支持度
    ax = axes[1, 1]
    bars = ax.barh(CLASSES, support, color='#9b59b6')
    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_title('Sample Count by Class', fontsize=14, fontweight='bold')
    for i, (bar, val) in enumerate(zip(bars, support)):
        ax.text(val + 50, i, f'{int(val)}', va='center', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"类别性能图已保存: {save_path}")

# ==================== 主评估函数 ====================
def evaluate_model(model_path, threshold=0.4, variant=None):
    """完整评估ConvNeXt模型"""
    logger.info("="*60)
    logger.info("开始ConvNeXt模型评估")
    logger.info("="*60)
    
    # 加载模型
    model, model_variant = load_model(model_path, variant)
    
    # 加载数据
    if not os.path.exists(TRAIN_DATA_FILE):
        raise FileNotFoundError(f"训练数据文件不存在: {TRAIN_DATA_FILE}")
    
    train_df = pd.read_csv(TRAIN_DATA_FILE).set_index('image')
    tr_df = get_one_hot_encoded_labels(train_df)
    
    # 数据分割（使用与训练时相同的随机种子）
    X_Train, X_Valid, Y_Train, Y_Valid = train_test_split(
        pd.Series(train_df.index),
        np.array(tr_df[CLASSES]),
        test_size=0.2,
        random_state=42
    )
    
    # 创建数据加载器
    val_transform = A.Compose([
        A.Resize(height=IM_SIZE, width=IM_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    validset = PlantDataset(X_Valid, Y_Valid, transform=val_transform, kind='val')
    validloader = DataLoader(validset, batch_size=BATCH, shuffle=False)
    
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
    print_metrics_summary(metrics, "ConvNeXt模型评估结果")
    
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
    
    logger.info(f"\n使用最优阈值 {best_threshold:.3f} 的结果:")
    print_metrics_summary(metrics_optimal, "最优阈值下的评估结果")
    
    # 生成可视化
    logger.info("生成可视化图表...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plot_metrics_comparison(metrics, 
                           os.path.join(VISUALIZATIONS_DIR, f'metrics_comparison_{timestamp}.png'))
    plot_confusion_matrices(y_true, y_pred,
                           os.path.join(VISUALIZATIONS_DIR, f'confusion_matrices_{timestamp}.png'))
    plot_class_performance(y_true, y_pred,
                          os.path.join(VISUALIZATIONS_DIR, f'class_performance_{timestamp}.png'))
    
    # 保存详细报告
    report_path = os.path.join(REPORTS_DIR, f'evaluation_report_{timestamp}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(f"ConvNeXt ({model_variant}) 模型评估报告\n")
        f.write("="*60 + "\n\n")
        f.write(f"模型路径: {model_path}\n")
        f.write(f"模型变体: {model_variant}\n")
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"使用阈值: {threshold}\n")
        f.write(f"最优阈值: {best_threshold:.3f}\n\n")
        
        f.write("【Kaggle官方指标】\n")
        f.write(f"多标签准确率 (阈值={threshold}): {metrics['multilabel_accuracy']:.4f}\n")
        f.write(f"多标签准确率 (最优阈值={best_threshold:.3f}): {metrics_optimal['multilabel_accuracy']:.4f}\n\n")
        
        f.write("【其他指标 (阈值={})】\n".format(threshold))
        for key, value in metrics.items():
            if key != 'multilabel_accuracy' and not key.startswith('precision_') and not key.startswith('recall_') and not key.startswith('f1_'):
                f.write(f"{key}: {value:.4f}\n")
        
        f.write("\n【各类别指标 (阈值={})】\n".format(threshold))
        for cls in CLASSES:
            f.write(f"\n{cls}:\n")
            f.write(f"  Precision: {metrics.get(f'precision_{cls}', 0):.4f}\n")
            f.write(f"  Recall: {metrics.get(f'recall_{cls}', 0):.4f}\n")
            f.write(f"  F1 Score: {metrics.get(f'f1_{cls}', 0):.4f}\n")
        
        f.write("\n【分类报告】\n")
        report = classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0)
        f.write(report)
    
    logger.info(f"详细报告已保存: {report_path}")
    logger.info("评估完成！")
    
    return metrics, metrics_optimal, best_threshold

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='ConvNeXt模型评估和可视化')
    parser.add_argument('--model', type=str, 
                       help='模型路径（如果不指定，自动查找最新的最佳模型）')
    parser.add_argument('--threshold', type=float, default=0.4,
                       help='预测阈值（默认0.4）')
    parser.add_argument('--variant', type=str, default=None,
                       help='模型变体 (convnext_tiny/small/base/large)，如果不指定则从checkpoint中读取')
    
    args = parser.parse_args()
    
    # 如果没有指定模型，自动查找
    if args.model is None:
        model_dir = os.path.join(OUTPUT_BASE_DIR, 'models', 'best')
        # 尝试查找任何convnext模型
        model_files = glob.glob(os.path.join(model_dir, "convnext_*_best_epoch*.pth"))
        if model_files:
            # 按epoch编号排序，选择最新的
            model_files.sort(key=lambda x: int(x.split('epoch')[-1].split('.')[0]))
            args.model = model_files[-1]
            logger.info(f"自动选择模型: {args.model}")
        else:
            logger.error("找不到模型文件，请使用 --model 参数指定模型路径")
            sys.exit(1)
    
    evaluate_model(args.model, args.threshold, args.variant)

