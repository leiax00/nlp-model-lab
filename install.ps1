# NLP Model Lab Windows 安装脚本

Write-Host "================================" -ForegroundColor Cyan
Write-Host "NLP Model Lab 安装向导" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# 检测 Python 版本
Write-Host "检查 Python 版本..." -ForegroundColor Yellow
try {
    $python_version = python --version 2>&1
    Write-Host "当前 Python 版本: $python_version" -ForegroundColor Green
} catch {
    Write-Host "错误: 未找到 Python，请先安装 Python 3.8+" -ForegroundColor Red
    exit 1
}

# 检测是否有 GPU
Write-Host ""
Write-Host "检测 GPU..." -ForegroundColor Yellow
try {
    $gpu_info = nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ 检测到 NVIDIA GPU:" -ForegroundColor Green
        Write-Host $gpu_info
        $has_gpu = $true
    } else {
        throw
    }
} catch {
    Write-Host "✗ 未检测到 NVIDIA GPU，将安装 CPU 版本" -ForegroundColor Yellow
    $has_gpu = $false
}

# 询问安装方式
Write-Host ""
Write-Host "请选择安装方式:" -ForegroundColor Cyan
Write-Host "1. Conda (推荐)"
Write-Host "2. pip"
$install_method = Read-Host "请输入选项 (1/2)"

if ($install_method -eq "1") {
    # Conda 安装
    Write-Host ""
    Write-Host "使用 Conda 安装..." -ForegroundColor Yellow

    # 检查 conda 是否存在
    try {
        $conda_version = conda --version 2>&1
        Write-Host "✓ Conda 版本: $conda_version" -ForegroundColor Green
    } catch {
        Write-Host "错误: 未找到 conda，请先安装 Miniconda 或 Anaconda" -ForegroundColor Red
        Write-Host "下载地址: https://docs.conda.io/en/latest/miniconda.html" -ForegroundColor Yellow
        exit 1
    }

    # 创建环境
    $env_name = "nlp-lab"
    $user_env = Read-Host "环境名称 (默认: $env_name)"
    if ($user_env -ne "") {
        $env_name = $user_env
    }

    Write-Host "创建 Conda 环境: $env_name" -ForegroundColor Green
    conda create -n $env_name python=3.10 -y

    Write-Host "激活环境..." -ForegroundColor Yellow
    conda activate $env_name

    # 安装 PyTorch
    if ($has_gpu) {
        Write-Host ""
        Write-Host "选择 PyTorch CUDA 版本:" -ForegroundColor Cyan
        Write-Host "1. CUDA 11.8"
        Write-Host "2. CUDA 12.1"
        Write-Host "3. CUDA 12.8"
        $cuda_version = Read-Host "请输入选项 (1/2/3, 默认: 1)"

        if ($cuda_version -eq "2") {
            Write-Host "安装 PyTorch (CUDA 12.1)..." -ForegroundColor Yellow
            conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
        } elseif ($cuda_version -eq "3") {
            Write-Host "安装 PyTorch (CUDA 12.8)..." -ForegroundColor Yellow
            conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia -y
        } else {
            Write-Host "安装 PyTorch (CUDA 11.8)..." -ForegroundColor Yellow
            conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
        }
    } else {
        Write-Host "安装 PyTorch (CPU only)..." -ForegroundColor Yellow
        conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
    }

    # 安装其他依赖
    Write-Host "安装其他依赖..." -ForegroundColor Yellow
    pip install -r requirements-core.txt

} elseif ($install_method -eq "2") {
    # pip 安装
    Write-Host ""
    Write-Host "使用 pip 安装..." -ForegroundColor Yellow

    # 创建虚拟环境
    $create_venv = Read-Host "是否创建新的虚拟环境? (y/n, 默认: y)"
    if ($create_venv -eq "") {
        $create_venv = "y"
    }

    if ($create_venv -eq "y") {
        $venv_name = "venv"
        $user_venv = Read-Host "虚拟环境目录 (默认: $venv_name)"
        if ($user_venv -ne "") {
            $venv_name = $user_venv
        }

        Write-Host "创建虚拟环境: $venv_name" -ForegroundColor Green
        python -m venv $venv_name

        Write-Host "激活虚拟环境..." -ForegroundColor Yellow
        & "$venv_name\Scripts\Activate.ps1"
    }

    # 安装 PyTorch
    if ($has_gpu) {
        Write-Host ""
        Write-Host "选择 PyTorch CUDA 版本:" -ForegroundColor Cyan
        Write-Host "1. CUDA 11.8"
        Write-Host "2. CUDA 12.1"
        Write-Host "3. CUDA 12.8"
        $cuda_version = Read-Host "请输入选项 (1/2/3, 默认: 1)"

        if ($cuda_version -eq "2") {
            Write-Host "安装 PyTorch (CUDA 12.1)..." -ForegroundColor Yellow
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        } elseif ($cuda_version -eq "3") {
            Write-Host "安装 PyTorch (CUDA 12.8)..." -ForegroundColor Yellow
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
        } else {
            Write-Host "安装 PyTorch (CUDA 11.8)..." -ForegroundColor Yellow
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        }
    } else {
        Write-Host "安装 PyTorch (CPU only)..." -ForegroundColor Yellow
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    }

    # 安装其他依赖
    Write-Host "安装其他依赖..." -ForegroundColor Yellow
    pip install -r requirements-core.txt
} else {
    Write-Host "无效的选项" -ForegroundColor Red
    exit 1
}

# 询问是否安装额外依赖
Write-Host ""
$install_viz = Read-Host "是否安装可视化工具 (matplotlib, seaborn, wandb)? (y/n, 默认: n)"
if ($install_viz -eq "") {
    $install_viz = "n"
}

if ($install_viz -eq "y") {
    Write-Host "安装可视化工具..." -ForegroundColor Yellow
    pip install -r requirements-viz.txt
}

Write-Host ""
$install_peft = Read-Host "是否安装 PEFT 工具 (LoRA, QLoRA)? (y/n, 默认: n)"
if ($install_peft -eq "") {
    $install_peft = "n"
}

if ($install_peft -eq "y") {
    Write-Host "安装 PEFT 工具..." -ForegroundColor Yellow
    pip install -r requirements-peft.txt
}

# 验证安装
Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "验证安装..." -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

python -c "import torch; print(f'✓ PyTorch version: {torch.__version__}'); print(f'✓ CUDA available: {torch.cuda.is_available()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None"

# 测试其他依赖
python -c "import transformers, datasets, sklearn, pandas, yaml; print('✓ 所有依赖安装成功!')"

Write-Host ""
Write-Host "================================" -ForegroundColor Green
Write-Host "安装完成!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host ""
Write-Host "下一步:" -ForegroundColor Cyan
if ($install_method -eq "1") {
    Write-Host "  激活环境: conda activate $env_name" -ForegroundColor Yellow
} else {
    if ($create_venv -eq "y") {
        Write-Host "  激活环境: $venv_name\Scripts\Activate.ps1" -ForegroundColor Yellow
    }
}
Write-Host "  运行示例: python examples\quickstart.py" -ForegroundColor Yellow
Write-Host "  查看文档: cat INSTALLATION.md" -ForegroundColor Yellow
Write-Host ""