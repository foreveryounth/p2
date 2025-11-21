#!/usr/bin/env python
# coding: utf-8
"""
工具函数 - 包含改进的数据增强策略等
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_enhanced_augmentations(im_size: int = 224, use_advanced: bool = True):
    """
    获取改进的数据增强策略
    
    Args:
        im_size: 图像尺寸
        use_advanced: 是否使用高级增强策略
    
    Returns:
        train_transform: 训练时的数据增强
        val_transform: 验证时的数据变换
    """
    if use_advanced:
        # 高级数据增强策略 - 针对植物病理学任务优化
        train_transform = A.Compose([
            # 几何变换
            A.RandomResizedCrop(
                height=im_size, 
                width=im_size, 
                scale=(0.7, 1.0), 
                ratio=(0.75, 1.33),
                interpolation=1  # 双线性插值
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),  # 垂直翻转（叶子可能上下颠倒）
            A.Rotate(limit=30, p=0.5, border_mode=0),  # 旋转
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.2, 
                rotate_limit=15, 
                p=0.5,
                border_mode=0
            ),
            
            # 颜色和亮度变换 - 模拟不同光照条件
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.5
            ),
            A.ColorJitter(
                brightness=0.2, 
                contrast=0.2, 
                saturation=0.2, 
                hue=0.1, 
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20, 
                sat_shift_limit=30, 
                val_shift_limit=20, 
                p=0.5
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),  # 对比度增强
            
            # 模糊和噪声 - 模拟拍摄条件
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.MotionBlur(blur_limit=3, p=0.3),
                A.MedianBlur(blur_limit=3, p=0.3),
            ], p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            
            # 遮挡和裁剪 - 提高模型鲁棒性
            A.CoarseDropout(
                max_holes=8, 
                max_height=32, 
                max_width=32, 
                fill_value=0,
                p=0.3
            ),
            A.Cutout(
                num_holes=8, 
                max_h_size=32, 
                max_w_size=32, 
                fill_value=0,
                p=0.3
            ),
            
            # 归一化和转换 - ImageNet标准化
            A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
    else:
        # 基础数据增强（保守策略）
        train_transform = A.Compose([
            A.RandomResizedCrop(height=im_size, width=im_size),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent=0.1, 
                scale=(0.9, 1.1), 
                rotate=(-15, 15), 
                p=0.5
            ),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
    
    # 验证集变换（不使用增强）
    val_transform = A.Compose([
        A.Resize(height=im_size, width=im_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform

def get_test_time_augmentation(im_size: int = 224):
    """
    获取测试时增强（TTA）策略
    用于推理时提高预测准确性
    """
    tta_transforms = [
        # 原始图像
        A.Compose([
            A.Resize(height=im_size, width=im_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # 水平翻转
        A.Compose([
            A.Resize(height=im_size, width=im_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # 垂直翻转
        A.Compose([
            A.Resize(height=im_size, width=im_size),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        # 旋转90度
        A.Compose([
            A.Resize(height=im_size, width=im_size),
            A.Rotate(limit=90, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
    ]
    return tta_transforms

