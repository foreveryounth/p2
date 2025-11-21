#!/bin/bash
#SBATCH --job-name=plant_pathology
#SBATCH --output=out/logs/slurm-%j.out
#SBATCH --error=out/logs/slurm-%j.err
#SBATCH --time=24:00:00          # 作业运行时间限制（24小时）
#SBATCH --nodes=1                # 节点数
#SBATCH --ntasks=1               # 任务数
#SBATCH --cpus-per-task=8        # 每个任务的CPU核心数（用于数据加载）
#SBATCH --mem=32G                # 内存需求
#SBATCH --gres=gpu:1             # 请求1个GPU
#SBATCH --partition=markov_gpu   # GPU分区（Markov HPC）

# 打印作业信息
echo "=========================================="
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $SLURM_NODELIST"
echo "开始时间: $(date)"
echo "工作目录: $(pwd)"
echo "=========================================="

# 加载必要的模块
# 加载CUDA模块（根据HPC实际版本调整）
module load CUDA/11.7.0

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 进入脚本所在目录（Inception_v3文件夹）
cd $SLURM_SUBMIT_DIR

# 确保输出目录存在
mkdir -p out/logs

# 激活v2虚拟环境（Python 3.11，支持新API）
# 如果v2不存在，回退到v1（向后兼容）
if [ -f "v2/bin/activate" ]; then
    source v2/bin/activate
    echo "已激活v2虚拟环境（Python 3.11）"
elif [ -f "$SLURM_SUBMIT_DIR/v2/bin/activate" ]; then
    source "$SLURM_SUBMIT_DIR/v2/bin/activate"
    echo "已激活v2虚拟环境（Python 3.11）"
elif [ -f "v1/bin/activate" ]; then
    source v1/bin/activate
    echo "已激活v1虚拟环境（Python 3.6，旧版本）"
elif [ -f "$SLURM_SUBMIT_DIR/v1/bin/activate" ]; then
    source "$SLURM_SUBMIT_DIR/v1/bin/activate"
    echo "已激活v1虚拟环境（Python 3.6，旧版本）"
else
    echo "错误: 找不到虚拟环境（v1或v2）"
    exit 1
fi

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

# 运行训练脚本（脚本在Inception_v3文件夹中，就在当前目录）
if [ -f "S0853044_Project1_training_hpc.py" ]; then
    python S0853044_Project1_training_hpc.py
else
    echo "错误: 找不到训练脚本 S0853044_Project1_training_hpc.py"
    exit 1
fi

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
    echo "请检查错误日志: out/logs/slurm-${SLURM_JOB_ID}.err"
    echo "结束时间: $(date)"
    echo "=========================================="
    exit $EXIT_CODE
fi


