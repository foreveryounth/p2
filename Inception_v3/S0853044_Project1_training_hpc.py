#!/usr/bin/env python
# coding: utf-8
# HPC适配版本 - 适用于CWRU HPC集群

import numpy as np 
import pandas as pd
import PIL
from PIL import Image
import os
import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Dict
from sklearn.model_selection import train_test_split
from torchmetrics.classification import F1Score
import logging
from datetime import datetime

# ==================== 配置日志 ====================
# 创建日志文件，文件名包含时间戳
# 日志文件保存在统一的输出目录 out/logs/
script_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE_DIR = os.path.join(script_dir, 'out')
LOGS_DIR = os.path.join(OUTPUT_BASE_DIR, 'logs')
MODELS_BEST_DIR = os.path.join(OUTPUT_BASE_DIR, 'models', 'best')
MODELS_CHECKPOINT_DIR = os.path.join(OUTPUT_BASE_DIR, 'models', 'checkpoints')

# 创建输出目录
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_BEST_DIR, exist_ok=True)
os.makedirs(MODELS_CHECKPOINT_DIR, exist_ok=True)

log_filename = os.path.join(LOGS_DIR, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==================== 配置参数 ====================
BATCH = 16
LR = 0.0001
IM_SIZE = 299
first_epochs = 0
last_epochs = 20

# 数据加载器配置 - 根据HPC分配的CPU核心数调整
# SLURM会自动设置 SLURM_CPUS_PER_TASK 环境变量
num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
if num_workers > 8:
    num_workers = 8  # 限制最大workers数量，避免过多进程

# 设备配置
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {DEVICE}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA版本: {torch.version.cuda}")
logger.info(f"PyTorch版本: {torch.__version__}")
logger.info(f"数据加载器 workers: {num_workers}")

# ==================== 路径配置 ====================
# 优先使用环境变量，否则使用默认路径
# 在HPC上，建议使用绝对路径
base_dir = os.environ.get('PROJECT_DIR', os.path.expanduser('~'))
if not os.path.isabs(base_dir):
    base_dir = os.path.abspath(base_dir)

# 项目目录 - 尝试多个可能的位置
# 首先尝试父目录（因为脚本可能在子文件夹中）
parent_dir = os.path.dirname(script_dir)

possible_dirs = [
    parent_dir,  # 父目录（优先）
    os.path.join(base_dir, 'projects', 'plant-pathology'),
    os.path.join(base_dir, 'plant-pathology'),
    os.path.abspath('.'),  # 当前目录
]

project_dir = None
for dir_path in possible_dirs:
    if os.path.exists(dir_path):
        project_dir = dir_path
        break

if project_dir is None:
    project_dir = os.path.abspath('.')  # 默认使用当前目录

# 数据路径
data_dir = os.path.join(project_dir, 'plant-pathology-2021-fgvc8')
if not os.path.exists(data_dir):
    # 尝试相对路径（相对于脚本所在目录）
    data_dir = os.path.join(parent_dir, 'plant-pathology-2021-fgvc8')
    if not os.path.exists(data_dir):
        data_dir = './plant-pathology-2021-fgvc8'

path = data_dir + '/' if not data_dir.endswith('/') else data_dir
TRAIN_DIR = os.path.join(path, 'train_images')
TEST_DIR = os.path.join(path, 'test_images')

logger.info(f"项目目录: {project_dir}")
logger.info(f"数据目录: {path}")
logger.info(f"输出基础目录: {OUTPUT_BASE_DIR}")
logger.info(f"模型最佳目录: {MODELS_BEST_DIR}")
logger.info(f"模型检查点目录: {MODELS_CHECKPOINT_DIR}")
logger.info(f"日志文件: {log_filename}")

# ==================== 读取训练数据 ====================
TRAIN_DATA_FILE = os.path.join(path, 'train.csv')

def read_image_labels():
    """读取训练数据标签"""
    if not os.path.exists(TRAIN_DATA_FILE):
        raise FileNotFoundError(
            f"训练数据文件不存在: {TRAIN_DATA_FILE}\n"
            "请确保数据集已上传到HPC并解压到正确位置"
        )
    df = pd.read_csv(TRAIN_DATA_FILE).set_index('image')
    return df

logger.info("读取训练数据...")
train_df = read_image_labels().sample(frac=1.0, random_state=42)
logger.info(f"训练数据量: {len(train_df)}")

# ==================== 标签处理 ====================
def get_single_labels(unique_labels) -> List[str]:
    """Splitting multi-labels and returning a list of classes"""
    single_labels = []
    for label in unique_labels:
        single_labels += label.split()
    single_labels = set(single_labels)
    return list(single_labels)

def get_one_hot_encoded_labels(dataset_df) -> pd.DataFrame:
    """将多标签转换为one-hot编码"""
    df = dataset_df.copy()
    unique_labels = df.labels.unique()
    column_names = get_single_labels(unique_labels)
    df[column_names] = 0
    
    # one-hot-encoding
    for label in unique_labels:
        label_indices = df[df['labels'] == label].index
        splited_labels = label.split()
        df.loc[label_indices, splited_labels] = 1
    
    return df

tr_df = get_one_hot_encoded_labels(train_df)
CLASSES = ['rust', 'complex', 'healthy', 'powdery_mildew', 'scab', 'frog_eye_leaf_spot']
logger.info(f"类别: {CLASSES}")

# ==================== 数据路径配置 ====================
folders = dict({
    'data': path,
    'train': TRAIN_DIR,
    'val': TRAIN_DIR,
    'test': TEST_DIR
})

# ==================== 数据分割 ====================
X_Train, X_Valid, Y_Train, Y_Valid = train_test_split(
    pd.Series(train_df.index), 
    np.array(tr_df[CLASSES]),  
    test_size=0.2, 
    random_state=42
)

logger.info(f"训练集大小: {len(X_Train)}")
logger.info(f"验证集大小: {len(X_Valid)}")

# ==================== 数据增强和变换 ====================
# 新版本albumentations (>=2.0) 使用size参数
train_transform = A.Compose([
    A.RandomResizedCrop(size=(IM_SIZE, IM_SIZE)),
    A.HorizontalFlip(p=0.5),
    A.Affine(translate_percent=0.1, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Normalize(),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(height=IM_SIZE, width=IM_SIZE),
    A.Normalize(),
    ToTensorV2(),
])

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

# ==================== 创建数据加载器 ====================
logger.info(f"创建数据加载器 (num_workers={num_workers})...")
trainset = PlantDataset(X_Train, Y_Train, transform=train_transform, kind='train')
trainloader = DataLoader(
    trainset, 
    batch_size=BATCH, 
    shuffle=True, 
    num_workers=num_workers, 
    pin_memory=True if torch.cuda.is_available() else False
)

validset = PlantDataset(X_Valid, Y_Valid, transform=val_transform, kind='val')
validloader = DataLoader(
    validset, 
    batch_size=BATCH, 
    shuffle=False, 
    num_workers=num_workers, 
    pin_memory=True if torch.cuda.is_available() else False
)

# ==================== 创建模型 ====================
logger.info("创建模型...")
# 使用新版本API (torchvision >= 0.13) - 支持weights参数
model = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
model.aux_logits = False
model.fc = nn.Sequential(
    nn.Linear(2048, 2048),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(2048, 6),
    nn.Sigmoid()
)
model = model.to(DEVICE)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

logger.info("模型创建完成")

# ==================== 训练监控类 ====================
class MetricMonitor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.losses = []
        self.scores = []
        self.metrics = dict({
            'loss': self.losses,
            'f1': self.scores
        })

    def update(self, metric_name, value):
        self.metrics[metric_name] += [value]

monitor = MetricMonitor()

# ==================== 训练循环 ====================
best_f1score = 0
logger.info(f"开始训练 (Epochs: {first_epochs} to {last_epochs})...")
logger.info(f"批次大小: {BATCH}, 学习率: {LR}, 图像尺寸: {IM_SIZE}")

for epoch in range(first_epochs, last_epochs):
    # 训练阶段
    tr_loss = 0.0
    # 新版本torchmetrics需要task参数，多标签分类使用'multilabel'
    f1 = F1Score(task='multilabel', num_labels=6, threshold=0.4, average='macro').to(DEVICE)
    f1score = 0
    model = model.train()

    num_train_batches = 0
    for i, (images, labels) in enumerate(trainloader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        pred = model(images.float())
        loss = criterion(pred.float(), labels.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tr_loss += loss.detach().item()
        f1score += f1(pred, labels.long())
        num_train_batches = i + 1
    
    model.eval()
    train_loss = tr_loss / num_train_batches
    train_f1 = f1score / num_train_batches
    logger.info('Train - Epoch: %d | Loss: %.4f | F1: %.4f' % (epoch+1, train_loss, train_f1))
    monitor.update('loss', train_loss)
    monitor.update('f1', train_f1)

    # 验证阶段
    tr_loss = 0.0
    # 新版本torchmetrics需要task参数，多标签分类使用'multilabel'
    f1 = F1Score(task='multilabel', num_labels=6, threshold=0.4, average='macro').to(DEVICE)
    f1score = 0
    num_valid_batches = 0
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(validloader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            pred = model(images.float())
            loss = criterion(pred.float(), labels.float())
            
            tr_loss += loss.detach().item()
            f1score += f1(pred, labels.long())
            num_valid_batches = i + 1
    
    model.eval()
    valid_loss = tr_loss / num_valid_batches
    valid_f1 = f1score / num_valid_batches
    logger.info('Valid - Epoch: %d | Loss: %.4f | F1: %.4f' % (epoch+1, valid_loss, valid_f1))
    monitor.update('loss', valid_loss)
    monitor.update('f1', valid_f1)
    
    # 保存最佳模型
    if valid_f1 > best_f1score:
        checkpoint = {
            "model": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'best_f1': valid_f1.item() if hasattr(valid_f1, 'item') else float(valid_f1)
        }
        checkpoint_path = os.path.join(MODELS_BEST_DIR, f"inception_v3_best_epoch{epoch+1}.pth")
        torch.save(checkpoint, checkpoint_path)
        best_f1score = valid_f1.item() if hasattr(valid_f1, 'item') else float(valid_f1)
        logger.info(f"保存最佳模型: {checkpoint_path} (F1: {best_f1score:.4f})")
    
    # 每20个epoch保存一次检查点
    if (epoch+1) % 20 == 0:
        checkpoint = {
            "model": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1
        }
        checkpoint_path = os.path.join(MODELS_CHECKPOINT_DIR, f"inception_v3_checkpoint_epoch{epoch+1}.pth")
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"保存检查点: {checkpoint_path}")

logger.info(f"\n训练完成！最佳F1分数: {best_f1score:.4f}")
logger.info(f"模型保存在: {MODELS_BEST_DIR} 和 {MODELS_CHECKPOINT_DIR}")


