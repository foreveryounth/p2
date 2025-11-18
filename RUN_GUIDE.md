# 项目运行指南

## 1. 环境要求

- Python 3.7+
- CUDA（可选，用于GPU加速训练）

## 2. 安装依赖

首先安装所需的Python包：

```bash
pip install torch torchvision
pip install numpy pandas pillow matplotlib
pip install scikit-learn scikit-image
pip install albumentations
pip install torchmetrics
pip install seaborn
```

或者使用requirements.txt（如果存在）：

```bash
pip install -r requirements.txt
```

## 3. 数据准备

### 3.1 下载数据集

从Kaggle下载 **Plant Pathology 2021-FGVC8** 数据集：
- 访问：https://www.kaggle.com/c/plant-pathology-2021-fgvc8
- 下载数据集并解压

### 3.2 目录结构

确保项目目录结构如下：

```
p2/
├── S0853044_Project1_training.py
├── S0853044_Project1_inference.py
├── README.md
└── plant-pathology-2021-fgvc8/
    ├── train.csv                    # 训练标签文件
    ├── sample_submission.csv        # 提交样本文件
    ├── train_images/                # 训练图像文件夹
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── test_images/                 # 测试图像文件夹
    │   ├── test1.jpg
    │   └── ...
    └── inception_v3_bestmodel/      # 模型保存目录（训练时自动创建）
        └── ...
```

### 3.3 创建必要的目录

在运行训练脚本前，需要创建模型保存目录：

```bash
mkdir -p plant-pathology-2021-fgvc8/inception_v3_bestmodel
```

## 4. 运行步骤

### 4.1 训练模型

运行训练脚本：

```bash
python S0853044_Project1_training.py
```

**训练参数说明：**
- `BATCH = 16`：批次大小
- `LR = 0.0001`：学习率
- `IM_SIZE = 299`：图像尺寸
- `first_epochs = 0`：起始轮次
- `last_epochs = 20`：总训练轮次

**训练过程：**
- 模型会自动保存最佳F1-Score的检查点
- 每20个epoch保存一次检查点
- 训练和验证的损失和F1分数会打印到控制台

**输出文件：**
- `plant-pathology-2021-fgvc8/inception_v3_bestmodel/inception_v3_bestmodel_epoch{N}.pth`：最佳模型
- `plant-pathology-2021-fgvc8/inception_v3_bestmodel/inception_v3_epoch{N}.pth`：定期保存的模型

### 4.2 推理和评估

运行推理脚本：

```bash
python S0853044_Project1_inference.py
```

**推理脚本功能：**
1. 加载训练好的模型（默认：`inception_v3_bestmodel_epoch20.pth`）
2. 在验证集上评估模型性能
3. 生成混淆矩阵可视化
4. 计算F1-Score
5. 生成测试集的预测结果（`submission.csv`）

**注意：** 如果模型文件名不同，需要修改推理脚本中的模型路径：

```python
# 第224行
checkpoint = torch.load("./plant-pathology-2021-fgvc8/inception_v3_bestmodel/inception_v3_bestmodel_epoch20.pth")
```

## 5. 常见问题

### 5.1 内存不足

如果遇到内存不足，可以：
- 减小 `BATCH` 大小（例如改为8或4）
- 使用更小的图像尺寸

### 5.2 CUDA错误

如果没有GPU或CUDA不可用：
- 代码会自动使用CPU（`DEVICE`会自动检测）
- CPU训练会非常慢，建议使用GPU

### 5.3 数据路径错误

确保：
- `plant-pathology-2021-fgvc8/` 目录在项目根目录下
- `train.csv` 文件存在
- `train_images/` 和 `test_images/` 文件夹存在

### 5.4 模型文件不存在

如果推理时找不到模型文件：
- 确保已经完成训练
- 检查模型保存路径是否正确
- 修改推理脚本中的模型文件名

## 6. 使用Jupyter Notebook

这两个脚本原本是为Jupyter Notebook设计的（包含 `# In[N]:` 标记）。如果需要：

1. 将代码转换为 `.ipynb` 格式
2. 或者直接运行 `.py` 文件（Python会忽略这些注释）

## 7. 预期结果

- **训练时间**：根据硬件配置，每个epoch可能需要几分钟到几十分钟
- **验证F1-Score**：应该达到0.9+（根据README，最佳验证F1为0.9136）
- **测试F1-Score**：在Kaggle上约为0.76387

