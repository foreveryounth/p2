#!/usr/bin/env python
# coding: utf-8
"""
ConvNeXt模型推理脚本 - Plant Pathology 2021-FGVC8 Kaggle比赛
用于生成测试集的预测结果并提交到Kaggle
"""

import numpy as np 
import pandas as pd
from PIL import Image
import os
import glob
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils import get_enhanced_augmentations

# ==================== 配置参数 ====================
BATCH = 16
IM_SIZE = 224  # ConvNeXt通常使用224x224输入
CONVNEXT_VARIANT = 'convnext_base'  # 应与训练时使用的变体一致
THRESHOLD = 0.4  # 多标签分类阈值

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ==================== 路径配置 ====================
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# 输出目录配置 - 统一输出到 out/ 目录，与Inception_v3相同的结构
OUTPUT_BASE_DIR = os.path.join(script_dir, 'out')
MODELS_BEST_DIR = os.path.join(OUTPUT_BASE_DIR, 'models', 'best')
MODELS_REPORTS_DIR = os.path.join(OUTPUT_BASE_DIR, 'models', 'reports')
MODELS_VISUALIZATIONS_DIR = os.path.join(OUTPUT_BASE_DIR, 'models', 'visualizations')
VISUALIZATIONS_DIR = os.path.join(OUTPUT_BASE_DIR, 'visualizations')
PREDICTIONS_DIR = os.path.join(OUTPUT_BASE_DIR, 'predictions')

# 创建输出目录
os.makedirs(MODELS_REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_VISUALIZATIONS_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# 尝试多个可能的数据路径
possible_paths = [
    os.path.join(parent_dir, 'plant-pathology-2021-fgvc8'),  # 父目录（优先）
    './plant-pathology-2021-fgvc8',  # 当前目录
]

path = None
for p in possible_paths:
    if os.path.exists(p):
        path = p + '/' if not p.endswith('/') else p
        break

if path is None:
    path = './plant-pathology-2021-fgvc8/'  # 默认路径

TRAIN_DIR = path + 'train_images/'
TEST_DIR = path + 'test_images/'

print(f"数据目录: {path}")

# ==================== 读取训练数据（用于获取类别信息） ====================
TRAIN_DATA_FILE = os.path.join(path, 'train.csv')
def read_image_labels():
    """读取训练数据标签"""
    if not os.path.exists(TRAIN_DATA_FILE):
        raise FileNotFoundError(
            f"训练数据文件不存在: {TRAIN_DATA_FILE}\n"
            "请从Kaggle下载Plant Pathology 2021-FGVC8数据集并解压到项目目录"
        )
    df = pd.read_csv(TRAIN_DATA_FILE).set_index('image')
    return df

train_df = read_image_labels()

# ==================== 标签处理 ====================
from typing import List
def get_single_labels(unique_labels) -> List[str]:
    """Splitting multi-labels and returning a list of classes"""
    single_labels = []
    for label in unique_labels:
        single_labels += label.split()
    single_labels = set(single_labels)
    return list(single_labels)

CLASSES = ['rust', 'complex', 'healthy', 'powdery_mildew', 'scab', 'frog_eye_leaf_spot']
print(f"类别: {CLASSES}")

# ==================== 数据变换 ====================
val_transform = A.Compose([
    A.Resize(height=IM_SIZE, width=IM_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

folders = dict({
    'data': path.rstrip('/'),
    'train': os.path.join(path, 'train_images'),
    'val': os.path.join(path, 'train_images'),
    'test': os.path.join(path, 'test_images')
})

# ==================== 数据集类 ====================
def get_image(image_id, kind='train'):
    """Loads an image from file"""
    fname = os.path.join(folders[kind], image_id)
    return Image.open(fname)

class PlantDataset(Dataset):
    """植物病理学数据集"""
    def __init__(self, 
                 image_ids, 
                 targets,
                 transform=None, 
                 target_transform=None, 
                 kind='train'):
        self.image_ids = image_ids
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.kind = kind
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        # load and transform image
        img = np.array(get_image(self.image_ids.iloc[idx], kind=self.kind))
        
        if self.transform:
            img = self.transform(image=img)['image']
        
        # get image target 
        target = self.targets[idx]
        if self.target_transform:
            target = self.target_transform(target)
        
        return img, target

# ==================== 创建ConvNeXt模型 ====================
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

model = create_convnext_model(CONVNEXT_VARIANT, num_classes=len(CLASSES))
model = model.to(DEVICE)

# ==================== 加载模型 ====================
# 模型保存在统一的输出目录 out/models/best/
MODEL_NAME = os.environ.get('MODEL_NAME', None)  # 可以通过环境变量指定模型名

if MODEL_NAME:
    # 使用指定的模型名（支持完整路径或相对路径）
    if os.path.isabs(MODEL_NAME) or os.path.dirname(MODEL_NAME):
        model_path = MODEL_NAME
    else:
        model_path = os.path.join(MODELS_BEST_DIR, MODEL_NAME)
else:
    # 自动查找最新的 best 模型文件
    model_files = glob.glob(os.path.join(MODELS_BEST_DIR, f"convnext_{CONVNEXT_VARIANT}_best_epoch*.pth"))
    if model_files:
        # 按文件名排序，获取最新的
        model_files.sort(key=lambda x: int(x.split('epoch')[-1].split('.')[0]))
        model_path = model_files[-1]
        print(f"自动选择模型: {os.path.basename(model_path)}")
    else:
        # 如果找不到，尝试通用模式
        model_files = glob.glob(os.path.join(MODELS_BEST_DIR, f"convnext_*_best_epoch*.pth"))
        if model_files:
            model_files.sort(key=lambda x: int(x.split('epoch')[-1].split('.')[0]))
            model_path = model_files[-1]
            # 从模型路径提取变体信息
            variant_name = os.path.basename(model_path).split('_')[1]
            print(f"找到模型: {os.path.basename(model_path)}")
            print(f"检测到变体: {variant_name}")
        else:
            raise FileNotFoundError(
                f"找不到模型文件。请确保已完成训练，或通过环境变量 MODEL_NAME 指定模型路径。\n"
                f"查找路径: {MODELS_BEST_DIR}\n"
                f"期望模式: convnext_{CONVNEXT_VARIANT}_best_epoch*.pth"
            )

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"模型文件不存在: {model_path}\n"
        f"请确保已完成训练，或通过环境变量 MODEL_NAME 指定模型文件名"
    )

checkpoint = torch.load(model_path, map_location=DEVICE)
model.load_state_dict(checkpoint['model'])

# 从checkpoint中获取变体信息（如果存在）
if 'variant' in checkpoint:
    print(f"模型变体: {checkpoint['variant']}")
if 'best_accuracy' in checkpoint:
    print(f"模型最佳准确率: {checkpoint['best_accuracy']:.4f}")

print(f"成功加载模型: {model_path}")
model.eval()

# ==================== 生成预测结果 ====================
def save_submission(model, threshold=0.4):
    """
    生成测试集的预测结果并保存为CSV文件
    
    Args:
        model: 训练好的模型
        threshold: 多标签分类阈值
    """
    # 读取测试集图像ID
    sample_submission_file = os.path.join(path, 'sample_submission.csv')
    if not os.path.exists(sample_submission_file):
        raise FileNotFoundError(
            f"样本提交文件不存在: {sample_submission_file}\n"
            "请确保数据集完整"
        )
    
    image_ids = pd.read_csv(sample_submission_file)
    print(f"测试集图像数量: {len(image_ids)}")
    
    # 创建虚拟标签（多标签分类不需要真实标签）
    dummy_labels = np.zeros((len(image_ids), len(CLASSES)))
    
    dataset = PlantDataset(
        image_ids['image'], 
        dummy_labels, 
        transform=val_transform, 
        kind='test'
    )
    
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=False, num_workers=2)
    
    model.eval()
    predictions = []
    all_pred_proba = []
    
    print("开始预测...")
    with torch.no_grad():
        for idx, (X, _) in enumerate(loader):
            X = X.float().to(DEVICE)
            y_pred_proba = model(X).detach().cpu().numpy()
            all_pred_proba.append(y_pred_proba)
            
            # 使用阈值进行多标签预测
            y_pred = (y_pred_proba > threshold).astype(int)
            
            # 将多标签转换为字符串格式
            for pred in y_pred:
                labels = [CLASSES[i] for i in range(len(CLASSES)) if pred[i] == 1]
                pred_labels = ' '.join(labels) if labels else 'healthy'
                predictions.append(pred_labels)
            
            if (idx + 1) % 10 == 0:
                print(f"已处理 {idx + 1} / {len(loader)} 批次")
    
    # 更新标签
    image_ids['labels'] = predictions
    
    # 保存数据框为CSV
    image_ids.set_index('image', inplace=True)
    submission_path = os.path.join(PREDICTIONS_DIR, f'submission_convnext_{CONVNEXT_VARIANT}.csv')
    image_ids.to_csv(submission_path)
    print(f"\n预测结果已保存到: {submission_path}")
    print(f"使用阈值: {threshold}")
    print(f"预测样本数: {len(predictions)}")
    
    # 统计预测分布
    label_counts = {}
    for pred in predictions:
        labels = pred.split()
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
    
    print("\n预测标签分布:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {count}")
    
    return image_ids

# ==================== 生成提交文件 ====================
if __name__ == "__main__":
    print("\n开始生成预测结果...")
    submission_df = save_submission(model, threshold=THRESHOLD)
    print("\n完成！可以上传到Kaggle提交。")

