#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


BATCH = 16
LR = 0.0001
IM_SIZE = 299

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 路径配置 - 支持从子文件夹运行
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# 输出目录配置 - 统一输出到 out/ 目录
OUTPUT_BASE_DIR = os.path.join(script_dir, 'out')
MODELS_BEST_DIR = os.path.join(OUTPUT_BASE_DIR, 'models', 'best')
VISUALIZATIONS_DIR = os.path.join(OUTPUT_BASE_DIR, 'visualizations')
PREDICTIONS_DIR = os.path.join(OUTPUT_BASE_DIR, 'predictions')

# 创建输出目录
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


# In[3]:


TRAIN_DATA_FILE = os.path.join(path, 'train.csv')
def read_image_labels():
    """
    """
    if not os.path.exists(TRAIN_DATA_FILE):
        raise FileNotFoundError(
            f"训练数据文件不存在: {TRAIN_DATA_FILE}\n"
            "请从Kaggle下载Plant Pathology 2021-FGVC8数据集并解压到项目目录"
        )
    df = pd.read_csv(TRAIN_DATA_FILE).set_index('image')
    return df


# In[4]:


train_df = read_image_labels().sample(
    frac=1.0, 
    random_state=42
)

train_df.head()


# In[5]:


from typing import List, Dict
def get_single_labels(unique_labels) -> List[str]:
    """Splitting multi-labels and returning a list of classes"""
    single_labels = []
    
    for label in unique_labels:
        single_labels += label.split()
        
    single_labels = set(single_labels)
    return list(single_labels)


# In[6]:


def get_one_hot_encoded_labels(dataset_df) -> pd.DataFrame:
    """
    """
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


# In[7]:


tr_df = get_one_hot_encoded_labels(train_df)
tr_df.head()


# In[8]:


from sklearn.model_selection import train_test_split
CLASSES = [
        'rust', 
        'complex', 
        'healthy', 
        'powdery_mildew', 
        'scab', 
        'frog_eye_leaf_spot'
    ]
X_Train, X_Valid, Y_Train, Y_Valid = train_test_split(
    pd.Series(train_df.index), 
    np.array(tr_df[CLASSES]),  
    test_size=0.2, 
    random_state=42
)
X_Train.head()


# In[9]:


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


# In[10]:


folders = dict({
        'data': path.rstrip('/'),
        'train': os.path.join(path, 'train_images'),
        'val': os.path.join(path, 'train_images'),
        'test': os.path.join(path, 'test_images')
    })


# In[11]:


def get_image(image_id, kind='train'):
    """Loads an image from file
    """
    fname = os.path.join(folders[kind], image_id)
    return Image.open(fname)


# In[12]:


from scipy.stats import bernoulli
from torch.utils.data import Dataset

class PlantDataset(Dataset):
    """
    """
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


# In[13]:


validset = PlantDataset(X_Valid, Y_Valid, transform=val_transform, kind='val')
validloader = DataLoader(validset, batch_size=BATCH, shuffle=False)


# In[14]:


# 兼容新旧版本的PyTorch/torchvision
try:
    # 新版本API (torchvision >= 0.13)
    model = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
except (AttributeError, TypeError):
    # 旧版本API (torchvision < 0.13)
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, message='.*pretrained.*')
    model = torchvision.models.inception_v3(pretrained=True)
model.aux_logits=False
model.fc = nn.Sequential(
            nn.Linear(2048, 6),
            nn.Sigmoid()
            
        )
model = model.to(DEVICE)


# In[15]:


# 加载模型 - 支持自动查找最新模型或指定模型路径
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
    model_files = glob.glob(os.path.join(MODELS_BEST_DIR, "inception_v3_best_epoch*.pth"))
    if model_files:
        # 按文件名排序，获取最新的
        model_files.sort(key=lambda x: int(x.split('epoch')[-1].split('.')[0]))
        model_path = model_files[-1]
        print(f"自动选择模型: {os.path.basename(model_path)}")
    else:
        # 如果找不到，尝试旧路径（向后兼容）
        old_model_dir = os.path.join(script_dir, 'inception_v3_bestmodel')
        old_model_files = glob.glob(os.path.join(old_model_dir, "inception_v3_bestmodel_epoch*.pth"))
        if old_model_files:
            old_model_files.sort(key=lambda x: int(x.split('epoch')[-1].split('.')[0]))
            model_path = old_model_files[-1]
            print(f"从旧路径找到模型: {os.path.basename(model_path)}")
        else:
            raise FileNotFoundError(
                f"找不到模型文件。请确保已完成训练，或通过环境变量 MODEL_NAME 指定模型路径。\n"
                f"查找路径: {MODELS_BEST_DIR}"
            )

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"模型文件不存在: {model_path}\n"
        f"请确保已完成训练，或通过环境变量 MODEL_NAME 指定模型文件名"
    )

checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model'])
print(f"成功加载模型: {model_path}")
### now you can evaluate it
model.eval()


# In[16]:


import matplotlib.pyplot as plt
y_true = np.empty(shape=(0, 6), dtype=np.int)
y_pred_proba = np.empty(shape=(0, 6), dtype=np.int)
model.eval()
for BATCH, (X, y) in enumerate(validloader):
    X = X.to(DEVICE)
    y = y.to(DEVICE).detach().cpu().numpy()
    pred = model(X).detach().cpu().numpy()
    
    y_true = np.vstack((y_true, y))
    y_pred_proba = np.vstack((y_pred_proba, pred))


# In[17]:


from sklearn.metrics import multilabel_confusion_matrix

def plot_confusion_matrix(
    y_test, 
    y_pred_proba, 
    threshold=0.4, 
    label_names=CLASSES,
    save_path=None
)-> None:
    """
    绘制混淆矩阵并保存到文件（非交互式模式）
    """
    y_pred = np.where(y_pred_proba > threshold, 1, 0)
    c_matrices = multilabel_confusion_matrix(y_test, y_pred)
    
    cmap = plt.get_cmap('Blues')
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))

    for cm, label, ax in zip(c_matrices, label_names, axes.flatten()):
        sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap=cmap);

        ax.set_xlabel('Predicted labels');
        ax.set_ylabel('True labels'); 
        ax.set_title(f'{label}');

    plt.tight_layout()
    
    # 保存图片而不是显示
    if save_path is None:
        save_path = os.path.join(VISUALIZATIONS_DIR, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存
    print(f"混淆矩阵已保存到: {save_path}")


# In[18]:


import seaborn as sns


# In[19]:


plot_confusion_matrix(y_true, y_pred_proba)


# In[20]:


from torchmetrics.classification import BinaryF1Score
f1 = BinaryF1Score(threshold=0.4)
y_pred = torch.as_tensor(np.where(y_pred_proba > 0.4, 1, 0))
y_true = torch.as_tensor(y_true)
f1 = f1(y_pred ,y_true).numpy()

pd.DataFrame({
    'name': ['F1'],
    'sorce': [f1]
}).set_index('name')


# In[21]:


def save_submission(model):
    """
    """
    image_ids = pd.read_csv(os.path.join(path, 'sample_submission.csv'))
    
    # 创建虚拟标签（多标签分类不需要真实标签）
    dummy_labels = np.zeros((len(image_ids), 6))
    
    dataset = PlantDataset(
        image_ids['image'], 
        dummy_labels, 
        transform=val_transform, 
        kind='test'
    )
    
    loader = DataLoader(dataset)
    
    model.eval()
    predictions = []
    with torch.no_grad():
        for idx, (X, _) in enumerate(loader):
            X = X.float().to(DEVICE)
            y_pred_proba = model(X).detach().cpu().numpy()
            
            # 使用阈值0.4进行多标签预测
            y_pred = (y_pred_proba > 0.4).astype(int)
            
            # 将多标签转换为字符串格式
            pred_labels = []
            for pred in y_pred:
                labels = [CLASSES[i] for i in range(len(CLASSES)) if pred[i] == 1]
                pred_labels.append(' '.join(labels) if labels else 'healthy')
            
            predictions.extend(pred_labels)
    
    # 更新标签
    image_ids['labels'] = predictions
    
    # save data frame as csv
    image_ids.set_index('image', inplace=True)
    submission_path = os.path.join(PREDICTIONS_DIR, 'submission.csv')
    image_ids.to_csv(submission_path)
    print(f"预测结果已保存到: {submission_path}")
    
    return image_ids


# In[22]:


save_submission(model) 


# In[ ]:




