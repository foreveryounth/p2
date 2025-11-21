#!/usr/bin/env python
# coding: utf-8
"""
数据预处理脚本 - 用于植物病理学数据集
功能：
1. 数据清洗（检查损坏的图像）
2. 数据统计分析
3. 图像质量增强（可选）
4. 数据平衡分析
"""

import numpy as np
import pandas as pd
import PIL
from PIL import Image
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import logging
from datetime import datetime
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ==================== 配置日志 ====================
script_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE_DIR = os.path.join(script_dir, 'out')
PREPROCESSING_DIR = os.path.join(OUTPUT_BASE_DIR, 'preprocessing')
os.makedirs(PREPROCESSING_DIR, exist_ok=True)

log_filename = os.path.join(PREPROCESSING_DIR, f'preprocessing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

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
TRAIN_DATA_FILE = os.path.join(path, 'train.csv')

logger.info(f"数据目录: {path}")
logger.info(f"预处理输出目录: {PREPROCESSING_DIR}")

# ==================== 数据清洗 ====================
def check_image_integrity(image_path: str) -> Tuple[bool, str]:
    """
    检查图像是否损坏
    返回: (是否有效, 错误信息)
    """
    try:
        img = Image.open(image_path)
        img.verify()  # 验证图像完整性
        img = Image.open(image_path)  # 重新打开（verify后需要重新打开）
        img.load()  # 加载图像数据
        
        # 检查图像尺寸
        if img.size[0] == 0 or img.size[1] == 0:
            return False, "图像尺寸为0"
        
        # 检查图像模式
        if img.mode not in ['RGB', 'RGBA', 'L', 'P']:
            return False, f"不支持的图像模式: {img.mode}"
        
        return True, ""
    except Exception as e:
        return False, str(e)

def clean_dataset(df: pd.DataFrame, image_dir: str) -> pd.DataFrame:
    """
    清洗数据集，移除损坏的图像
    """
    logger.info("开始检查图像完整性...")
    valid_images = []
    invalid_images = []
    
    for idx, image_id in enumerate(tqdm(df.index, desc="检查图像")):
        image_path = os.path.join(image_dir, image_id)
        if not os.path.exists(image_path):
            invalid_images.append((image_id, "文件不存在"))
            continue
        
        is_valid, error_msg = check_image_integrity(image_path)
        if is_valid:
            valid_images.append(image_id)
        else:
            invalid_images.append((image_id, error_msg))
    
    logger.info(f"有效图像: {len(valid_images)}/{len(df)}")
    logger.info(f"无效图像: {len(invalid_images)}")
    
    if invalid_images:
        logger.warning("发现以下无效图像:")
        for img_id, error in invalid_images[:10]:  # 只显示前10个
            logger.warning(f"  {img_id}: {error}")
        if len(invalid_images) > 10:
            logger.warning(f"  ... 还有 {len(invalid_images) - 10} 个无效图像")
        
        # 保存无效图像列表
        invalid_df = pd.DataFrame(invalid_images, columns=['image', 'error'])
        invalid_df.to_csv(os.path.join(PREPROCESSING_DIR, 'invalid_images.csv'), index=False)
    
    # 返回清洗后的数据框
    cleaned_df = df.loc[valid_images]
    return cleaned_df

# ==================== 数据统计分析 ====================
def analyze_dataset(df: pd.DataFrame, image_dir: str, output_dir: str):
    """
    分析数据集统计信息
    """
    logger.info("开始数据统计分析...")
    
    # 1. 标签分布分析
    logger.info("分析标签分布...")
    label_counts = Counter(df['labels'])
    label_df = pd.DataFrame(list(label_counts.items()), columns=['label', 'count'])
    label_df = label_df.sort_values('count', ascending=False)
    
    # 保存标签分布
    label_df.to_csv(os.path.join(output_dir, 'label_distribution.csv'), index=False)
    
    # 绘制标签分布图
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(label_df)), label_df['count'])
    plt.xticks(range(len(label_df)), label_df['label'], rotation=45, ha='right')
    plt.ylabel('样本数量')
    plt.xlabel('标签')
    plt.title('标签分布')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'label_distribution.png'), dpi=150)
    plt.close()
    
    # 2. 多标签分析
    logger.info("分析多标签组合...")
    df['num_labels'] = df['labels'].apply(lambda x: len(x.split()))
    multi_label_stats = df['num_labels'].value_counts().sort_index()
    logger.info(f"多标签统计:\n{multi_label_stats}")
    
    # 3. 图像尺寸分析（采样分析）
    logger.info("分析图像尺寸...")
    sample_size = min(1000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    
    image_sizes = []
    for image_id in tqdm(sample_df.index, desc="分析图像尺寸"):
        image_path = os.path.join(image_dir, image_id)
        try:
            img = Image.open(image_path)
            image_sizes.append(img.size)
        except:
            continue
    
    if image_sizes:
        sizes_df = pd.DataFrame(image_sizes, columns=['width', 'height'])
        sizes_df.to_csv(os.path.join(output_dir, 'image_sizes_sample.csv'), index=False)
        
        logger.info(f"图像尺寸统计 (采样 {len(image_sizes)} 张):")
        logger.info(f"  宽度: min={sizes_df['width'].min()}, max={sizes_df['width'].max()}, mean={sizes_df['width'].mean():.1f}")
        logger.info(f"  高度: min={sizes_df['height'].min()}, max={sizes_df['height'].max()}, mean={sizes_df['height'].mean():.1f}")
        
        # 绘制尺寸分布
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(sizes_df['width'], bins=50, edgecolor='black')
        plt.xlabel('宽度 (像素)')
        plt.ylabel('频数')
        plt.title('图像宽度分布')
        
        plt.subplot(1, 2, 2)
        plt.hist(sizes_df['height'], bins=50, edgecolor='black')
        plt.xlabel('高度 (像素)')
        plt.ylabel('频数')
        plt.title('图像高度分布')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'image_size_distribution.png'), dpi=150)
        plt.close()
    
    # 4. 类别不平衡分析
    logger.info("分析类别不平衡...")
    
    def get_single_labels(unique_labels) -> List[str]:
        single_labels = []
        for label in unique_labels:
            single_labels += label.split()
        single_labels = set(single_labels)
        return list(single_labels)
    
    unique_labels = df.labels.unique()
    class_names = get_single_labels(unique_labels)
    
    class_counts = {cls: 0 for cls in class_names}
    for label in df.labels:
        for cls in label.split():
            class_counts[cls] += 1
    
    class_df = pd.DataFrame(list(class_counts.items()), columns=['class', 'count'])
    class_df = class_df.sort_values('count', ascending=False)
    class_df.to_csv(os.path.join(output_dir, 'class_distribution.csv'), index=False)
    
    # 绘制类别分布
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(class_df)), class_df['count'])
    plt.xticks(range(len(class_df)), class_df['class'], rotation=45, ha='right')
    plt.ylabel('样本数量')
    plt.xlabel('类别')
    plt.title('类别分布（多标签统计）')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=150)
    plt.close()
    
    logger.info("数据统计分析完成！")

# ==================== 主函数 ====================
def main():
    """
    主预处理流程
    """
    logger.info("=" * 50)
    logger.info("开始数据预处理")
    logger.info("=" * 50)
    
    # 1. 读取数据
    if not os.path.exists(TRAIN_DATA_FILE):
        raise FileNotFoundError(f"训练数据文件不存在: {TRAIN_DATA_FILE}")
    
    logger.info("读取训练数据...")
    train_df = pd.read_csv(TRAIN_DATA_FILE).set_index('image')
    logger.info(f"原始数据量: {len(train_df)}")
    
    # 2. 数据清洗
    logger.info("\n" + "=" * 50)
    logger.info("步骤 1: 数据清洗")
    logger.info("=" * 50)
    cleaned_df = clean_dataset(train_df, TRAIN_DIR)
    logger.info(f"清洗后数据量: {len(cleaned_df)}")
    
    # 保存清洗后的数据
    cleaned_df.to_csv(os.path.join(PREPROCESSING_DIR, 'train_cleaned.csv'))
    logger.info(f"清洗后的数据已保存到: {os.path.join(PREPROCESSING_DIR, 'train_cleaned.csv')}")
    
    # 3. 数据统计分析
    logger.info("\n" + "=" * 50)
    logger.info("步骤 2: 数据统计分析")
    logger.info("=" * 50)
    analyze_dataset(cleaned_df, TRAIN_DIR, PREPROCESSING_DIR)
    
    # 4. 生成预处理报告
    logger.info("\n" + "=" * 50)
    logger.info("预处理完成！")
    logger.info("=" * 50)
    logger.info(f"预处理结果保存在: {PREPROCESSING_DIR}")
    logger.info("生成的文件:")
    logger.info("  - train_cleaned.csv: 清洗后的训练数据")
    logger.info("  - invalid_images.csv: 无效图像列表（如果存在）")
    logger.info("  - label_distribution.csv: 标签分布统计")
    logger.info("  - class_distribution.csv: 类别分布统计")
    logger.info("  - image_sizes_sample.csv: 图像尺寸统计（采样）")
    logger.info("  - *.png: 各种统计图表")
    
    return cleaned_df

if __name__ == "__main__":
    cleaned_df = main()

