#!/usr/bin/env python
# coding: utf-8
# 改进的训练脚本 - 使用数据预处理和增强策略

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
from typing import List, Dict
from sklearn.model_selection import train_test_split
from torchmetrics.classification import F1Score
import logging
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# 导入工具函数
from utils import get_enhanced_augmentations
from metrics import multilabel_accuracy, calculate_all_metrics, print_metrics_summary

# ==================== 配置日志 ====================
script_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE_DIR = os.path.join(script_dir, 'out')
LOGS_DIR = os.path.join(OUTPUT_BASE_DIR, 'logs')
MODELS_BEST_DIR = os.path.join(OUTPUT_BASE_DIR, 'models', 'best')
MODELS_CHECKPOINT_DIR = os.path.join(OUTPUT_BASE_DIR, 'models', 'checkpoints')
PREPROCESSING_DIR = os.path.join(OUTPUT_BASE_DIR, 'preprocessing')

# 创建输出目录
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_BEST_DIR, exist_ok=True)
os.makedirs(MODELS_CHECKPOINT_DIR, exist_ok=True)

log_filename = os.path.join(LOGS_DIR, f'training_enhanced_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
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
BATCH = 8  # 减少批次大小以降低内存使用
LR = 0.0001
IM_SIZE = 299
first_epochs = 0
last_epochs = 30  # 增加训练轮数
USE_PREPROCESSED_DATA = True  # 是否使用预处理后的数据
USE_ENHANCED_AUG = True  # 是否使用增强的数据增强策略
USE_LR_SCHEDULER = True  # 是否使用学习率调度器

# 数据加载器配置
num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
if num_workers > 4:
    num_workers = 4  # 减少workers以降低内存使用

# 设备配置
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {DEVICE}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA版本: {torch.version.cuda}")
logger.info(f"PyTorch版本: {torch.__version__}")
logger.info(f"数据加载器 workers: {num_workers}")

# ==================== 路径配置 ====================
base_dir = os.environ.get('PROJECT_DIR', os.path.expanduser('~'))
if not os.path.isabs(base_dir):
    base_dir = os.path.abspath(base_dir)

parent_dir = os.path.dirname(script_dir)
possible_dirs = [
    parent_dir,
    os.path.join(base_dir, 'projects', 'plant-pathology'),
    os.path.join(base_dir, 'plant-pathology'),
    os.path.abspath('.'),
]

project_dir = None
for dir_path in possible_dirs:
    if os.path.exists(dir_path):
        project_dir = dir_path
        break

if project_dir is None:
    project_dir = os.path.abspath('.')

data_dir = os.path.join(project_dir, 'plant-pathology-2021-fgvc8')
if not os.path.exists(data_dir):
    data_dir = os.path.join(parent_dir, 'plant-pathology-2021-fgvc8')
    if not os.path.exists(data_dir):
        data_dir = './plant-pathology-2021-fgvc8'

path = data_dir + '/' if not data_dir.endswith('/') else data_dir
TRAIN_DIR = os.path.join(path, 'train_images')
TEST_DIR = os.path.join(path, 'test_images')

logger.info(f"项目目录: {project_dir}")
logger.info(f"数据目录: {path}")
logger.info(f"输出基础目录: {OUTPUT_BASE_DIR}")

# ==================== 读取训练数据 ====================
def read_image_labels():
    """读取训练数据标签"""
    # 优先使用预处理后的数据
    if USE_PREPROCESSED_DATA:
        preprocessed_file = os.path.join(PREPROCESSING_DIR, 'train_cleaned.csv')
        if os.path.exists(preprocessed_file):
            logger.info(f"使用预处理后的数据: {preprocessed_file}")
            df = pd.read_csv(preprocessed_file).set_index('image')
            return df
        else:
            logger.warning(f"预处理后的数据不存在: {preprocessed_file}，使用原始数据")
    
    # 使用原始数据
    TRAIN_DATA_FILE = os.path.join(path, 'train.csv')
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
logger.info(f"使用{'增强' if USE_ENHANCED_AUG else '基础'}数据增强策略")
train_transform, val_transform = get_enhanced_augmentations(
    im_size=IM_SIZE, 
    use_advanced=USE_ENHANCED_AUG
)

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
    pin_memory=True if torch.cuda.is_available() else False,
    drop_last=True  # 丢弃最后一个不完整的批次，避免BatchNorm问题
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
model = torchvision.models.inception_v3(pretrained=True)
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

# 学习率调度器
if USE_LR_SCHEDULER:
    scheduler = CosineAnnealingLR(optimizer, T_max=last_epochs, eta_min=LR * 0.01)
    logger.info("使用余弦退火学习率调度器")
else:
    scheduler = None

logger.info("模型创建完成")

# ==================== 训练监控类 ====================
class MetricMonitor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.losses = []
        self.scores = []
        self.accuracies = []
        self.metrics = dict({
            'loss': self.losses,
            'f1': self.scores,
            'accuracy': self.accuracies
        })

    def update(self, metric_name, value):
        self.metrics[metric_name] += [value]

monitor = MetricMonitor()

# ==================== 训练循环 ====================
# 使用多标签准确率作为主要指标（Kaggle官方指标）
best_accuracy = 0
best_f1score = 0  # 保留F1作为辅助指标
patience = 5  # 早停耐心值
patience_counter = 0

logger.info(f"开始训练 (Epochs: {first_epochs} to {last_epochs})...")
logger.info(f"批次大小: {BATCH}, 学习率: {LR}, 图像尺寸: {IM_SIZE}")
logger.info(f"使用预处理数据: {USE_PREPROCESSED_DATA}")
logger.info(f"使用增强数据增强: {USE_ENHANCED_AUG}")
logger.info(f"使用学习率调度器: {USE_LR_SCHEDULER}")
logger.info(f"主要评估指标: 多标签准确率 (Multilabel Accuracy) - Kaggle官方指标")

for epoch in range(first_epochs, last_epochs):
    # 训练阶段
    tr_loss = 0.0
    f1 = F1Score(num_classes=6, threshold=0.4, average='macro', multiclass=False).to(DEVICE)
    f1score = 0
    model.train()
    model.aux_logits = False  # 确保训练时禁用辅助分类器

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
    
    # 更新学习率
    if scheduler is not None:
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
    else:
        current_lr = LR
    
    logger.info('Train - Epoch: %d | Loss: %.4f | F1: %.4f | LR: %.6f' % 
                (epoch+1, train_loss, train_f1, current_lr))
    monitor.update('loss', train_loss)
    monitor.update('f1', train_f1)
    # 注意：训练阶段不计算准确率（太耗时），只在验证阶段计算

    # 验证阶段
    tr_loss = 0.0
    f1 = F1Score(num_classes=6, threshold=0.4, average='macro', multiclass=False).to(DEVICE)
    f1score = 0
    num_valid_batches = 0
    
    # 收集所有预测和标签用于计算多标签准确率
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(validloader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            pred = model(images.float())
            loss = criterion(pred.float(), labels.float())
            
            tr_loss += loss.detach().item()
            f1score += f1(pred, labels.long())
            num_valid_batches = i + 1
            
            # 收集预测和标签
            all_preds.append(pred.cpu())
            all_labels.append(labels.cpu())
    
    model.eval()
    valid_loss = tr_loss / num_valid_batches
    valid_f1 = f1score / num_valid_batches
    
    # 计算多标签准确率（Kaggle官方指标）
    all_preds_tensor = torch.cat(all_preds, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)
    valid_accuracy = multilabel_accuracy(all_labels_tensor, all_preds_tensor)
    
    logger.info('Valid - Epoch: %d | Loss: %.4f | Accuracy: %.4f | F1: %.4f' % 
                (epoch+1, valid_loss, valid_accuracy, valid_f1))
    monitor.update('loss', valid_loss)
    monitor.update('f1', valid_f1)
    monitor.update('accuracy', valid_accuracy)
    
    # 保存最佳模型（基于多标签准确率）
    if valid_accuracy > best_accuracy:
        checkpoint = {
            "model": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'best_accuracy': valid_accuracy,
            'best_f1': valid_f1.item() if hasattr(valid_f1, 'item') else float(valid_f1),
            'scheduler': scheduler.state_dict() if scheduler else None
        }
        checkpoint_path = os.path.join(MODELS_BEST_DIR, f"inception_v3_enhanced_best_epoch{epoch+1}.pth")
        torch.save(checkpoint, checkpoint_path)
        best_accuracy = valid_accuracy
        best_f1score = valid_f1.item() if hasattr(valid_f1, 'item') else float(valid_f1)
        logger.info(f"保存最佳模型: {checkpoint_path} (Accuracy: {best_accuracy:.4f}, F1: {best_f1score:.4f})")
        patience_counter = 0
    else:
        patience_counter += 1
    
    # 早停检查
    if patience_counter >= patience:
        logger.info(f"验证F1分数在{patience}个epoch内未提升，提前停止训练")
        break
    
    # 每10个epoch保存一次检查点
    if (epoch+1) % 10 == 0:
        checkpoint = {
            "model": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'scheduler': scheduler.state_dict() if scheduler else None
        }
        checkpoint_path = os.path.join(MODELS_CHECKPOINT_DIR, f"inception_v3_enhanced_checkpoint_epoch{epoch+1}.pth")
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"保存检查点: {checkpoint_path}")

logger.info(f"\n训练完成！")
logger.info(f"最佳多标签准确率 (Kaggle官方指标): {best_accuracy:.4f}")
logger.info(f"最佳F1分数 (辅助指标): {best_f1score:.4f}")
logger.info(f"模型保存在: {MODELS_BEST_DIR} 和 {MODELS_CHECKPOINT_DIR}")

