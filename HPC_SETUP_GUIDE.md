# CWRU HPC 运行指南

本指南将帮助你在CWRU大学的HPC（高性能计算）集群上运行Plant Pathology项目。

## 📋 前置准备

### 1. HPC账号和访问
- 确保你拥有CWRU HPC的账号和访问权限
- 了解如何通过SSH连接到HPC集群
- 熟悉基本的Linux命令和HPC环境

### 2. 数据传输
- 将项目代码和数据上传到HPC集群
- 推荐使用 `scp` 或 `rsync` 传输文件

## 🚀 步骤详解

### 步骤 1: 连接到HPC集群

```bash
# 通过SSH连接到CWRU HPC（示例，实际地址可能不同）
ssh your_username@hpc.cwru.edu
# 或者
ssh your_username@login.hpc.cwru.edu
```

### 步骤 2: 准备项目目录

在HPC上创建项目目录并上传文件：

```bash
# 在HPC上创建项目目录
mkdir -p ~/projects/plant-pathology
cd ~/projects/plant-pathology

# 从本地传输文件（在本地机器上执行）
# 传输代码文件
scp -r /Volumes/Harlen/code/p2/*.py /Volumes/Harlen/code/p2/*.txt /Volumes/Harlen/code/p2/*.sh \
      your_username@hpc.cwru.edu:~/projects/plant-pathology/

# 或者使用rsync（推荐，支持断点续传）
rsync -avz --progress \
      --include="*.py" --include="*.txt" --include="*.sh" --include="*.md" \
      --exclude="*" \
      /Volumes/Harlen/code/p2/ your_username@hpc.cwru.edu:~/projects/plant-pathology/
```

**注意：** 数据集文件较大，可能需要单独传输或从Kaggle在HPC上下载。

### 步骤 3: 检查HPC环境

```bash
# 检查可用的模块
module avail

# 检查Python版本
python3 --version

# 检查是否有GPU可用
nvidia-smi  # 如果可用
```

### 步骤 4: 设置Python环境

#### 方法A: 使用Conda（推荐）

```bash
# 加载Conda模块（如果HPC提供）
module load conda
# 或者
module load anaconda3

# 创建新的conda环境
conda create -n plant-pathology python=3.9 -y
conda activate plant-pathology

# 安装依赖
pip install -r requirements.txt
```

#### 方法B: 使用虚拟环境

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 步骤 5: 准备数据集

确保数据集已上传到HPC：

```bash
# 检查数据集是否存在
ls -lh plant-pathology-2021-fgvc8/
# 应该看到：
# - train.csv
# - train_images/
# - test_images/
```

如果数据集还没上传，可以：
1. 从本地传输（如果数据集在本地）
2. 在HPC上从Kaggle下载（需要配置Kaggle API）

### 步骤 6: 配置SLURM作业脚本

编辑 `submit_job.sh`，根据CWRU HPC的实际配置调整：

1. **分区名称**：修改 `#SBATCH --partition=gpu` 为实际的GPU分区名称
   ```bash
   # 查看可用分区
   sinfo
   ```

2. **模块加载**：取消注释并修改相应的模块加载命令
   ```bash
   # 查看可用模块
   module avail
   ```

3. **Python环境**：取消注释并选择你的环境激活方式
   ```bash
   # conda环境
   conda activate plant-pathology
   
   # 或虚拟环境
   source venv/bin/activate
   ```

### 步骤 7: 提交作业

```bash
# 确保脚本有执行权限（如果还没有）
chmod +x submit_job.sh

# 提交作业到SLURM队列
sbatch submit_job.sh

# 查看作业状态
squeue -u $USER

# 查看作业输出（实时）
tail -f slurm-<job_id>.out

# 查看错误日志
tail -f slurm-<job_id>.err
```

### 步骤 8: 监控训练进度

```bash
# 查看作业状态
squeue -u $USER

# 查看实时输出
tail -f slurm-<job_id>.out

# 或者使用watch命令（每5秒刷新）
watch -n 5 tail -20 slurm-<job_id>.out

# 查看训练日志文件（如果创建了）
tail -f training_*.log
```

### 步骤 9: 检查结果

```bash
# 训练完成后，检查输出目录
ls -lh plant-pathology-2021-fgvc8/inception_v3_bestmodel/

# 查看训练日志
cat training_*.log
```

## ⚙️ 配置调整

### 根据HPC资源调整参数

#### 1. SLURM资源请求（在 `submit_job.sh` 中）

```bash
# 如果训练时间较长，增加时间限制
#SBATCH --time=48:00:00

# 如果需要更多内存
#SBATCH --mem=64G

# 如果需要更多CPU核心（用于数据加载）
#SBATCH --cpus-per-task=16
```

#### 2. 训练参数（在 `S0853044_Project1_training_hpc.py` 中）

```python
BATCH = 16        # 根据GPU显存调整（HPC GPU通常更强大，可以增大到32或64）
LR = 0.0001       # 学习率
IM_SIZE = 299     # 图像尺寸
last_epochs = 20  # 训练轮数
```

### 查找CWRU HPC的实际配置

```bash
# 查看可用分区
sinfo

# 查看可用模块
module avail

# 查看GPU信息
nvidia-smi

# 查看作业队列
squeue

# 查看账户信息
sacctmgr show user $USER
```

## 🔍 常见问题

### 1. 找不到GPU
- **问题**: `CUDA not available` 或 `No GPU found`
- **解决**: 
  - 检查是否请求了GPU资源：`#SBATCH --gres=gpu:1`
  - 检查GPU分区名称是否正确（使用 `sinfo` 查看）
  - 运行 `nvidia-smi` 检查GPU状态
  - 确认CUDA模块已加载

### 2. 模块未找到
- **问题**: `module: command not found`
- **解决**: 
  - 检查HPC是否使用module系统
  - 某些HPC可能需要先加载module系统：`source /etc/profile.d/modules.sh`
  - 或者直接使用系统Python和pip（不加载模块）

### 3. 数据路径错误
- **问题**: `FileNotFoundError: 训练数据文件不存在`
- **解决**: 
  - 检查数据集是否已上传到HPC
  - 确认路径是否正确（使用绝对路径更安全）
  - 在脚本中使用 `os.path.expanduser('~')` 获取用户目录
  - 检查 `S0853044_Project1_training_hpc.py` 中的路径配置

### 4. 内存不足
- **问题**: `Out of Memory` 或作业被杀死
- **解决**: 
  - 增加SLURM内存请求：`#SBATCH --mem=64G`
  - 减小批次大小：`BATCH = 8`
  - 减少num_workers：在训练脚本中调整

### 5. 作业超时
- **问题**: 作业在完成前被终止
- **解决**: 
  - 增加时间限制：`#SBATCH --time=48:00:00`
  - 使用检查点恢复训练（修改 `first_epochs`）
  - 减少训练轮数进行测试

### 6. 依赖包安装失败
- **问题**: `pip install` 失败或权限错误
- **解决**: 
  - 使用 `--user` 标志：`pip install --user -r requirements.txt`
  - 在虚拟环境或conda环境中安装
  - 检查网络连接和HPC的pip源配置

### 7. 作业一直在排队
- **问题**: 作业状态一直是 `PD` (Pending)
- **解决**: 
  - 检查分区名称是否正确
  - 检查资源请求是否合理（时间、内存等）
  - 查看队列信息：`squeue`
  - 联系HPC管理员

## 📊 监控训练进度

### 查看实时输出

```bash
# 查看作业输出文件
tail -f slurm-<job_id>.out

# 或者使用watch命令
watch -n 5 tail -20 slurm-<job_id>.out
```

### 检查GPU使用情况

```bash
# 在另一个终端SSH连接到HPC，然后运行
watch -n 1 nvidia-smi
```

### 检查作业状态

```bash
# 查看所有作业
squeue -u $USER

# 查看作业详细信息
scontrol show job <job_id>

# 取消作业
scancel <job_id>

# 查看作业历史
sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
```

## 💾 保存和下载结果

### 在HPC上保存

模型检查点会自动保存到配置的输出目录（通常是 `plant-pathology-2021-fgvc8/inception_v3_bestmodel/`）

### 下载结果到本地

```bash
# 在本地机器上执行
# 下载模型文件
scp your_username@hpc.cwru.edu:~/projects/plant-pathology/plant-pathology-2021-fgvc8/inception_v3_bestmodel/*.pth ./

# 或者使用rsync下载整个目录
rsync -avz your_username@hpc.cwru.edu:~/projects/plant-pathology/plant-pathology-2021-fgvc8/inception_v3_bestmodel/ ./

# 下载训练日志
scp your_username@hpc.cwru.edu:~/projects/plant-pathology/training_*.log ./
```

## 🎯 优化建议

### 1. 提高训练效率
- 使用HPC的快速存储（如SSD）存储数据集
- 增加批次大小（如果GPU显存允许）
- 使用混合精度训练（需要修改代码）
- 优化数据加载（使用更多workers）

### 2. 节省资源
- 先进行小规模测试（减少epochs）
- 使用较小的图像尺寸进行测试
- 合理设置作业时间限制

### 3. 提高稳定性
- 定期保存检查点（代码已包含）
- 使用作业数组进行多次实验
- 记录训练日志到文件（已实现）

## 📞 获取帮助

如果遇到问题：
1. 查看HPC文档和用户指南
2. 联系CWRU HPC支持团队
3. 检查SLURM日志文件：`slurm-<job_id>.out` 和 `slurm-<job_id>.err`
4. 查看训练日志文件：`training_*.log`
5. 查看HPC系统状态和公告

## 🔗 有用链接

- CWRU HPC文档（请查找实际链接）
- SLURM文档: https://slurm.schedmd.com/documentation.html
- PyTorch文档: https://pytorch.org/docs/

## 📝 快速参考命令

```bash
# 提交作业
sbatch submit_job.sh

# 查看作业
squeue -u $USER

# 查看输出
tail -f slurm-<job_id>.out

# 取消作业
scancel <job_id>

# 查看GPU
nvidia-smi

# 查看分区
sinfo
```

---

**祝训练顺利！** 🚀


