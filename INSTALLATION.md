# 安装指南

## 环境要求

- Python >= 3.8
- 操作系统：Linux / macOS / Windows

## 🔍 第一步：检测你的 CUDA 环境（推荐）

在安装之前，建议先运行我们的检测工具来了解你的系统配置：

```bash
python scripts/utils/check_cuda.py
```

这个工具会：
- 检测你的 NVIDIA 驱动版本
- 检测 GPU 型号和内存
- 检查当前 PyTorch 安装情况
- **推荐最适合你的 PyTorch 版本**

### CUDA 版本兼容性说明

**重要**：CUDA 驱动版本向下兼容！

- ✅ 如果你的驱动是 **CUDA 13.0**，可以安装 **CUDA 12.8**、**12.1** 或 **11.8** 的 PyTorch
- ✅ 如果你的驱动是 **CUDA 12.x**，可以安装 **CUDA 12.1**、**12.8** 或 **11.8** 的 PyTorch
- ✅ 如果你的驱动是 **CUDA 11.x**，可以安装 **CUDA 11.8** 的 PyTorch

**示例**：用户有 CUDA 13.0 驱动，推荐安装：
```bash
# 方案1：最新版本（推荐）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 方案2：稳定版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

两者都能在 CUDA 13.0 驱动上正常运行，无需担心兼容性问题。

## 方式一：使用 Conda（推荐）

Conda 是管理 PyTorch 和深度学习环境的最简单方式，它可以自动处理 CUDA 依赖。

### 1. 安装 Miniconda 或 Anaconda

从 [Conda 官网](https://docs.conda.io/en/latest/miniconda.html) 下载并安装。

### 2. 创建环境

```bash
# 创建环境
conda create -n nlp-lab python=3.10 -y
conda activate nlp-lab
```

### 3. 安装 PyTorch

#### CPU 版本

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

#### GPU 版本（CUDA 11.8）

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

#### GPU 版本（CUDA 12.1）

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

#### GPU 版本（CUDA 12.8）

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia -y
```

### 4. 安装其他依赖

```bash
pip install -r requirements-core.txt
```

## 方式二：使用 pip

### 1. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 2. 安装 PyTorch

#### CPU 版本

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### GPU 版本（CUDA 11.8）

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### GPU 版本（CUDA 12.1）

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### GPU 版本（CUDA 12.8）

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

#### GPU 版本（ROCm - AMD GPU）

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

### 3. 安装其他依赖

```bash
pip install -r requirements-core.txt
```

## 方式三：使用 pip + requirements 文件

我们提供了针对不同场景的 requirements 文件：

### CPU 版本

```bash
pip install -r requirements-cpu.txt
```

### GPU 版本（需要手动安装 PyTorch）

```bash
# 先安装 PyTorch（参考上面的命令）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 再安装其他依赖
pip install -r requirements-core.txt
```

## 验证安装

运行以下命令验证安装是否成功：

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

预期输出（GPU 版本）：
```
PyTorch version: 2.1.0+cu118
CUDA available: True
```

预期输出（CPU 版本）：
```
PyTorch version: 2.1.0+cpu
CUDA available: False
```

## 额外依赖

### 开发工具

```bash
pip install -r requirements-dev.txt
```

包含：pytest、black、ruff、mypy 等

### 可视化和实验追踪

```bash
pip install -r requirements-viz.txt
```

包含：matplotlib、seaborn、tensorboard、wandb

### 参数高效微调（PEFT）

```bash
pip install -r requirements-peft.txt
```

包含：peft（LoRA、QLoRA 等）

## Docker 安装（可选）

我们提供了 Dockerfile 用于容器化部署：

```bash
# 构建镜像
docker build -t nlp-lab:latest .

# 运行容器（GPU）
docker run --gpus all -it nlp-lab:latest

# 运行容器（CPU）
docker run -it nlp-lab:latest
```

## 常见问题

### Q1: 我的 CUDA 驱动是 13.0，应该安装哪个版本的 PyTorch？

**答**：CUDA 驱动向下兼容！你可以：
- ✅ **推荐**：安装 CUDA 12.8 的 PyTorch（最新版本）
- ✅ **备选**：安装 CUDA 12.1 的 PyTorch（稳定版本）
- ✅ **兼容**：安装 CUDA 11.8 的 PyTorch（兼容性最好）

```bash
# 推荐方案（CUDA 12.8 - 最新）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 备选方案（CUDA 12.1 - 稳定）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

所有版本都能在 CUDA 13.0 驱动上正常运行，无需担心兼容性问题。

### Q2: 如何查看 CUDA 版本？

```bash
# 查看 NVIDIA 驱动版本
nvidia-smi

# 查看 CUDA 运行时版本
nvcc --version

# 或使用我们的检测工具
python scripts/utils/check_cuda.py
```

### Q3: 如何选择合适的 PyTorch 版本？

**简单方法**：运行检测工具
```bash
python scripts/utils/check_cuda.py
```

**手动选择**：访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取针对你系统的安装命令。

**推荐规则**：
- CUDA 13.x/12.x 驱动 → 使用 PyTorch CUDA 12.8（最新）或 12.1（稳定）
- CUDA 11.x 驱动 → 使用 PyTorch CUDA 11.8
- 无 NVIDIA GPU → 使用 CPU 版本

### Q4: CPU 版本可以在有 GPU 的机器上运行吗？

可以，但无法使用 GPU 加速。建议安装对应 CUDA 版本的 PyTorch 以充分利用硬件。

### Q4: 安装后运行报错 "No module named 'torch'"

检查是否正确激活了虚拟环境，以及 pip 安装位置是否正确。

```bash
which pip  # 检查 pip 路径
pip list | grep torch  # 检查是否安装了 torch
```

### Q5: Windows 上安装失败？

1. 确保 Visual Studio C++ Build Tools 已安装
2. 使用 conda 而不是 pip（更简单）
3. 检查 Python 版本是否为 64 位

## 硬件要求

### 最低配置（CPU 训练）
- CPU: 4核心以上
- 内存: 16GB
- 硬盘: 20GB 可用空间

### 推荐配置（GPU 训练）
- GPU: NVIDIA RTX 3060 或更高（8GB+ VRAM）
- 内存: 32GB
- 硬盘: 50GB SSD
- CUDA: 11.8、12.1 或 12.8

### 大规模训练
- GPU: NVIDIA A100 / RTX 4090（24GB+ VRAM）
- 内存: 64GB+
- 多 GPU 支持更快的训练

## 更新依赖

定期更新依赖以获得最新功能和 bug 修复：

```bash
pip install --upgrade -r requirements-core.txt
```

或使用 conda：

```bash
conda update --all
```