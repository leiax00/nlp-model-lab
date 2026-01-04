#!/bin/bash

# NLP Model Lab 安装脚本
# 此脚本会帮助你根据系统配置安装合适的依赖

set -e

echo "================================"
echo "NLP Model Lab 安装向导"
echo "================================"
echo ""

# 检测 Python 版本
echo "检查 Python 版本..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "当前 Python 版本: $python_version"

# 检测是否有 GPU
echo ""
echo "检测 GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ 检测到 NVIDIA GPU"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    has_gpu=true
else
    echo "✗ 未检测到 NVIDIA GPU，将安装 CPU 版本"
    has_gpu=false
fi

# 询问安装方式
echo ""
echo "请选择安装方式:"
echo "1. Conda (推荐)"
echo "2. pip"
read -p "请输入选项 (1/2): " install_method

if [ "$install_method" = "1" ]; then
    # Conda 安装
    echo ""
    echo "使用 Conda 安装..."

    # 检查 conda 是否存在
    if ! command -v conda &> /dev/null; then
        echo "错误: 未找到 conda，请先安装 Miniconda 或 Anaconda"
        echo "下载地址: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi

    # 创建环境
    env_name="nlp-lab"
    read -p "环境名称 (默认: $env_name): " user_env_name
    env_name=${user_env_name:-$env_name}

    echo "创建 Conda 环境: $env_name"
    conda create -n $env_name python=3.10 -y

    echo "激活环境..."
    eval "$(conda shell.bash hook)"
    conda activate $env_name

    # 安装 PyTorch
    if [ "$has_gpu" = true ]; then
        echo ""
        echo "选择 PyTorch CUDA 版本:"
        echo "1. CUDA 11.8"
        echo "2. CUDA 12.1"
        echo "3. CUDA 12.8"
        read -p "请输入选项 (1/2, 默认: 1): " cuda_version
        cuda_version=${cuda_version:-1}

        if [ "$cuda_version" = "2" ]; then
            echo "安装 PyTorch (CUDA 12.1)..."
            conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
        elif [ "$cuda_version" = "3" ]; then
            echo "安装 PyTorch (CUDA 12.8)..."
            conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia -y
        else
            echo "安装 PyTorch (CUDA 11.8)..."
            conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
        fi
    else
        echo "安装 PyTorch (CPU only)..."
        conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
    fi

    # 安装其他依赖
    echo "安装其他依赖..."
    pip install -r requirements-core.txt

elif [ "$install_method" = "2" ]; then
    # pip 安装
    echo ""
    echo "使用 pip 安装..."

    # 创建虚拟环境
    read -p "是否创建新的虚拟环境? (y/n, 默认: y): " create_venv
    create_venv=${create_venv:-y}

    if [ "$create_venv" = "y" ]; then
        venv_name="venv"
        read -p "虚拟环境目录 (默认: $venv_name): " user_venv
        venv_name=${user_venv:-$venv_name}

        echo "创建虚拟环境: $venv_name"
        python -m venv $venv_name

        echo "激活虚拟环境..."
        source $venv_name/bin/activate
    fi

    # 安装 PyTorch
    if [ "$has_gpu" = true ]; then
        echo ""
        echo "选择 PyTorch CUDA 版本:"
        echo "1. CUDA 11.8"
        echo "2. CUDA 12.1"
        echo "3. CUDA 12.8"
        read -p "请输入选项 (1/2, 默认: 1): " cuda_version
        cuda_version=${cuda_version:-1}

        if [ "$cuda_version" = "2" ]; then
            echo "安装 PyTorch (CUDA 12.1)..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        elif [ "$cuda_version" = "3" ]; then
            echo "安装 PyTorch (CUDA 12.8)..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
        else
            echo "安装 PyTorch (CUDA 11.8)..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        fi
    else
        echo "安装 PyTorch (CPU only)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi

    # 安装其他依赖
    echo "安装其他依赖..."
    pip install -r requirements-core.txt
else
    echo "无效的选项"
    exit 1
fi

# 询问是否安装额外依赖
echo ""
read -p "是否安装可视化工具 (matplotlib, seaborn, wandb)? (y/n, 默认: n): " install_viz
install_viz=${install_viz:-n}

if [ "$install_viz" = "y" ]; then
    echo "安装可视化工具..."
    pip install -r requirements-viz.txt
fi

echo ""
read -p "是否安装 PEFT 工具 (LoRA, QLoRA)? (y/n, 默认: n): " install_peft
install_peft=${install_peft:-n}

if [ "$install_peft" = "y" ]; then
    echo "安装 PEFT 工具..."
    pip install -r requirements-peft.txt
fi

# 验证安装
echo ""
echo "================================"
echo "验证安装..."
echo "================================"

python -c "
import torch
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ CUDA version: {torch.version.cuda}')
    print(f'✓ GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# 测试其他依赖
python -c "
import transformers
import datasets
import sklearn
import pandas
import yaml
print('✓ 所有依赖安装成功！')
" || {
    echo "✗ 部分依赖安装失败，请检查错误信息"
    exit 1
}

echo ""
echo "================================"
echo "安装完成！"
echo "================================"
echo ""
echo "下一步:"
if [ "$install_method" = "1" ]; then
    echo "  激活环境: conda activate $env_name"
else
    if [ "$create_venv" = "y" ]; then
        echo "  激活环境: source $venv_name/bin/activate"
    fi
fi
echo "  运行示例: python examples/quickstart.py"
echo "  查看文档: cat INSTALLATION.md"
echo ""