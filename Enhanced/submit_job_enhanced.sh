#!/bin/bash
#SBATCH --job-name=plant_pathology_enhanced
#SBATCH --partition=markov_gpu
#SBATCH --output=out/logs/slurm-%j.out
#SBATCH --error=out/logs/slurm-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=classt23  # 排除H100节点（PyTorch 1.10.2不支持）

# 获取脚本目录（优先使用SLURM提交目录，否则使用脚本所在目录）
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi

# 切换到脚本目录
cd "$SCRIPT_DIR" || exit 1

PARENT_DIR=$(dirname "$SCRIPT_DIR")
VENV_PATH="$PARENT_DIR/Inception_v3/v1"

if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "已激活虚拟环境: $VENV_PATH"
else
    echo "警告: 虚拟环境不存在: $VENV_PATH"
    echo "尝试使用系统Python..."
fi

# 检查Python是否可用，如果python不存在则使用python3
if command -v python &> /dev/null; then
    PYTHON_CMD=python
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    echo "错误: 找不到Python命令"
    exit 1
fi

# 显示Python路径
echo "Python路径: $(which $PYTHON_CMD)"
echo "Python版本: $($PYTHON_CMD --version)"

# 创建必要的输出目录（确保SLURM日志可以正常写入）
mkdir -p out/logs
mkdir -p out/preprocessing
mkdir -p out/models/best
mkdir -p out/models/checkpoints

echo "=========================================="
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $SLURM_NODELIST"
echo "开始时间: $(date)"
echo "工作目录: $(pwd)"
echo "脚本目录: $SCRIPT_DIR"
echo "=========================================="

# 步骤1: 数据预处理（如果还没有运行过）
PREPROCESSED_FILE="out/preprocessing/train_cleaned.csv"
if [ ! -f "$PREPROCESSED_FILE" ]; then
    echo "=========================================="
    echo "步骤1: 运行数据预处理"
    echo "=========================================="
    $PYTHON_CMD data_preprocessing.py
    if [ $? -ne 0 ]; then
        echo "错误: 数据预处理失败"
        exit 1
    fi
else
    echo "预处理后的数据已存在，跳过预处理步骤"
fi

# 步骤2: 训练模型
echo "=========================================="
echo "步骤2: 开始训练"
echo "=========================================="
$PYTHON_CMD training_enhanced.py

if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "训练成功完成！"
    echo "结束时间: $(date)"
    echo "=========================================="
else
    echo "=========================================="
    echo "训练失败！"
    echo "结束时间: $(date)"
    echo "=========================================="
    exit 1
fi

