#!/bin/bash
#SBATCH --job-name=plant_pathology
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=24:00:00          # 作业运行时间限制（24小时）
#SBATCH --nodes=1                # 节点数
#SBATCH --ntasks=1               # 任务数
#SBATCH --cpus-per-task=8        # 每个任务的CPU核心数（用于数据加载）
#SBATCH --mem=32G                # 内存需求
#SBATCH --gres=gpu:1             # 请求1个GPU
#SBATCH --partition=gpu          # GPU分区（根据CWRU HPC实际分区名称调整）
                                 # 常见分区名称: gpu, gpu-partition, gpu-queue 等

# 打印作业信息
echo "=========================================="
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $SLURM_NODELIST"
echo "开始时间: $(date)"
echo "工作目录: $(pwd)"
echo "=========================================="

# 加载必要的模块
# 注意：根据CWRU HPC实际可用的模块调整
# 查看可用模块: module avail

# 示例：加载CUDA模块（根据HPC实际版本调整）
# module load cuda/11.8
# module load cudnn/8.6.0

# 示例：加载Python模块（如果HPC提供）
# module load python/3.9
# module load anaconda3

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 激活Python环境
# 方法1: 如果使用conda环境
# source activate plant-pathology
# 或者
# conda activate plant-pathology

# 方法2: 如果使用虚拟环境
# source venv/bin/activate

# 方法3: 如果使用系统Python（确保已安装依赖）
# 不需要激活

# 设置项目目录（可选，如果使用环境变量）
# export PROJECT_DIR=$HOME/projects/plant-pathology

# 进入项目目录
cd $SLURM_SUBMIT_DIR
# 或者使用绝对路径
# cd ~/projects/plant-pathology

# 打印环境信息
echo "=========================================="
echo "环境信息:"
echo "Python版本: $(python --version 2>&1)"
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)' 2>&1)"
echo "CUDA可用: $(python -c 'import torch; print(torch.cuda.is_available())' 2>&1)"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU信息:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
fi
echo "分配的CPU核心数: $SLURM_CPUS_PER_TASK"
echo "=========================================="
echo "开始训练..."
echo "=========================================="

# 运行训练脚本
python S0853044_Project1_training_hpc.py

# 检查退出状态
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "训练成功完成！"
    echo "结束时间: $(date)"
    echo "=========================================="
    exit 0
else
    echo "=========================================="
    echo "训练失败，退出代码: $EXIT_CODE"
    echo "请检查错误日志: slurm-${SLURM_JOB_ID}.err"
    echo "结束时间: $(date)"
    echo "=========================================="
    exit $EXIT_CODE
fi


