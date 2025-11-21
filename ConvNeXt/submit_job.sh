#!/bin/bash
#SBATCH --job-name=convnext_training
#SBATCH --partition=markov_gpu
#SBATCH --output=out/logs/slurm-%j.out
#SBATCH --error=out/logs/slurm-%j.err
#SBATCH --time=24:00:00          # 作业运行时间限制（24小时）
#SBATCH --nodes=1                # 节点数
#SBATCH --ntasks=1               # 任务数
#SBATCH --cpus-per-task=8        # 每个任务的CPU核心数（用于数据加载）
#SBATCH --mem=32G                # 内存需求
#SBATCH --gres=gpu:1             # 请求1个GPU

# 打印作业信息
echo "=========================================="
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $SLURM_NODELIST"
echo "开始时间: $(date)"
echo "工作目录: $(pwd)"
echo "=========================================="

# 获取脚本目录（优先使用SLURM提交目录，否则使用脚本所在目录）
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi

# 切换到脚本目录
cd "$SCRIPT_DIR" || exit 1

# 加载必要的模块
# 加载CUDA模块（根据HPC实际版本调整）
if command -v module &> /dev/null; then
    module load CUDA/11.7.0 2>/dev/null || echo "警告: CUDA模块加载失败，使用系统CUDA"
fi

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 激活虚拟环境
# 优先查找Inception_v3的虚拟环境（如果有共享环境）
PARENT_DIR=$(dirname "$SCRIPT_DIR")
VENV_PATHS=(
    "$PARENT_DIR/Inception_v3/v2/bin/activate"
    "$PARENT_DIR/Inception_v3/v1/bin/activate"
    "$SCRIPT_DIR/v2/bin/activate"
    "$SCRIPT_DIR/v1/bin/activate"
)

VENV_ACTIVATED=false
for venv_path in "${VENV_PATHS[@]}"; do
    if [ -f "$venv_path" ]; then
        source "$venv_path"
        echo "已激活虚拟环境: $venv_path"
        VENV_ACTIVATED=true
        break
    fi
done

if [ "$VENV_ACTIVATED" = false ]; then
    echo "警告: 找不到虚拟环境，尝试使用系统Python..."
fi

# 检查Python是否可用
if command -v python &> /dev/null; then
    PYTHON_CMD=python
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    echo "错误: 找不到Python命令"
    exit 1
fi

# 显示环境信息
echo "=========================================="
echo "环境信息:"
echo "Python路径: $(which $PYTHON_CMD)"
echo "Python版本: $($PYTHON_CMD --version 2>&1)"
if $PYTHON_CMD -c "import torch" 2>/dev/null; then
    echo "PyTorch版本: $($PYTHON_CMD -c 'import torch; print(torch.__version__)' 2>&1)"
    echo "CUDA可用: $($PYTHON_CMD -c 'import torch; print(torch.cuda.is_available())' 2>&1)"
    if $PYTHON_CMD -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q True; then
        echo "GPU数量: $($PYTHON_CMD -c 'import torch; print(torch.cuda.device_count())' 2>&1)"
        echo "GPU名称: $($PYTHON_CMD -c 'import torch; print(torch.cuda.get_device_name(0))' 2>&1)"
    fi
else
    echo "警告: PyTorch未安装或无法导入"
fi
if command -v nvidia-smi &> /dev/null; then
    echo "GPU信息 (nvidia-smi):"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "无法获取GPU信息"
fi
echo "分配的CPU核心数: $SLURM_CPUS_PER_TASK"
echo "工作目录: $(pwd)"
echo "脚本目录: $SCRIPT_DIR"
echo "=========================================="

# 创建必要的输出目录（确保SLURM日志可以正常写入）
mkdir -p out/logs
mkdir -p out/models/best
mkdir -p out/models/checkpoints
mkdir -p out/models/reports
mkdir -p out/models/visualizations

echo "=========================================="
echo "开始训练ConvNeXt模型..."
echo "=========================================="

# 检查训练脚本是否存在
if [ ! -f "$SCRIPT_DIR/training.py" ]; then
    echo "错误: 找不到训练脚本 training.py"
    echo "当前目录: $(pwd)"
    echo "脚本目录: $SCRIPT_DIR"
    exit 1
fi

# 运行训练脚本
$PYTHON_CMD training.py

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
    echo "或查看训练日志: out/logs/training_convnext_*.log"
    echo "结束时间: $(date)"
    echo "=========================================="
    exit $EXIT_CODE
fi


