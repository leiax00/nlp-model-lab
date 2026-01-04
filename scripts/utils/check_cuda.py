#!/usr/bin/env python3
"""
CUDA 环境检测工具
帮助用户选择合适的 PyTorch CUDA 版本
"""
import subprocess
import sys
from typing import Optional, Tuple


def run_command(cmd: str) -> Optional[str]:
    """执行 shell 命令并返回输出"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception:
        return None


def get_nvidia_driver_version() -> Optional[str]:
    """获取 NVIDIA 驱动版本"""
    output = run_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
    return output


def get_cuda_version() -> Optional[str]:
    """获取 CUDA 运行时版本"""
    output = run_command("nvcc --version")
    if output:
        for line in output.split('\n'):
            if 'release' in line.lower():
                # 提取版本号
                import re
                match = re.search(r'(\d+\.\d+)', line)
                if match:
                    return match.group(1)
    return None


def get_gpu_info() -> list:
    """获取 GPU 信息"""
    gpu_info = []
    output = run_command("nvidia-smi --query-gpu,name,memory.total --format=csv,noheader")
    if output:
        for line in output.split('\n'):
            if line.strip():
                parts = line.split(', ')
                if len(parts) >= 2:
                    gpu_info.append({
                        'name': parts[0],
                        'memory': parts[1] if len(parts) > 1 else 'Unknown'
                    })
    return gpu_info


def check_pytorch_installation() -> Tuple[bool, str]:
    """检查当前 PyTorch 安装情况"""
    try:
        import torch
        version = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else None

        if cuda_available:
            return True, f"PyTorch {version} (CUDA {cuda_version})"
        else:
            return True, f"PyTorch {version} (CPU only)"
    except ImportError:
        return False, "PyTorch 未安装"


def recommend_pytorch_version(driver_version: str) -> dict:
    """
    根据 NVIDIA 驱动版本推荐 PyTorch 版本

    驱动版本与 CUDA 运行时版本的兼容性表：
    https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html
    """
    # 简化的版本推荐逻辑
    major, minor = map(int, driver_version.split('.')[:2])

    recommendations = {
        'cuda_12.8': {
            'supported': True,
            'pip_command': 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128',
            'conda_command': 'conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia',
            'reason': '最新版本，性能最佳'
        },
        'cuda_12.1': {
            'supported': True,
            'pip_command': 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121',
            'conda_command': 'conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia',
            'reason': '稳定版本，兼容性好'
        },
        'cuda_11.8': {
            'supported': True,
            'pip_command': 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118',
            'conda_command': 'conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia',
            'reason': '广泛兼容，最稳定'
        }
    }

    # 根据驱动版本调整推荐
    if major >= 545:  # CUDA 12.8+ 驱动
        recommendations['cuda_12.8']['reason'] += ' - 与您的驱动版本最匹配，强烈推荐'
        recommendations['cuda_12.1']['reason'] += ' - 也可以使用'
        recommendations['cuda_11.8']['reason'] += ' - 兼容版本'
    elif major >= 530:  # CUDA 12.x 驱动
        recommendations['cuda_12.8']['reason'] += ' - 最新版本，推荐使用'
        recommendations['cuda_12.1']['reason'] += ' - 与您的驱动版本最匹配'
        recommendations['cuda_11.8']['reason'] += ' - 也可以使用'
    elif major >= 515:  # CUDA 11.x 驱动
        recommendations['cuda_11.8']['reason'] += ' - 与您的驱动版本最匹配，强烈推荐'
        recommendations['cuda_12.1']['reason'] += ' - 需要确认驱动兼容性'
        recommendations['cuda_12.8']['reason'] += ' - 需要确认驱动兼容性'

    return recommendations


def main():
    print("=" * 70)
    print("CUDA 环境检测工具")
    print("=" * 70)
    print()

    # 检测 NVIDIA 驱动
    print("1. 检测 NVIDIA 驱动和 CUDA")
    print("-" * 70)

    driver_version = get_nvidia_driver_version()
    if driver_version:
        print(f"✓ NVIDIA 驱动版本: {driver_version}")
    else:
        print("✗ 未检测到 NVIDIA 驱动")
        print("  请确认：")
        print("  1. 是否安装了 NVIDIA GPU")
        print("  2. 是否安装了 NVIDIA 驱动")
        print("  3. nvidia-smi 是否在 PATH 中")
        print()
        print("建议安装 CPU 版本的 PyTorch：")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
        return

    cuda_version = get_cuda_version()
    if cuda_version:
        print(f"✓ CUDA 运行时版本: {cuda_version}")
    else:
        print("  未检测到 CUDA 运行时（可能未安装 CUDA Toolkit）")

    # 检测 GPU 信息
    print()
    print("2. 检测 GPU 信息")
    print("-" * 70)

    gpu_info = get_gpu_info()
    if gpu_info:
        for i, gpu in enumerate(gpu_info):
            print(f"  GPU {i}: {gpu['name']} ({gpu['memory']})")
    else:
        print("  未检测到 GPU 信息")

    # 检查 PyTorch 安装
    print()
    print("3. 检查 PyTorch 安装情况")
    print("-" * 70)

    installed, info = check_pytorch_installation()
    if installed:
        print(f"✓ {info}")
    else:
        print("✗ PyTorch 未安装")

    # 推荐版本
    print()
    print("4. PyTorch 安装推荐")
    print("-" * 70)

    recommendations = recommend_pytorch_version(driver_version)

    print(f"\n根据您的驱动版本 {driver_version}，推荐以下安装方案：\n")

    for version, info in recommendations.items():
        status = "✓" if info['supported'] else "✗"
        print(f"{status} {version.replace('_', ' ').upper()}")
        print(f"  说明: {info['reason']}")
        print(f"  pip:")
        print(f"    {info['pip_command']}")
        print(f"  conda:")
        print(f"    {info['conda_command']}")
        print()

    # 兼容性说明
    print("5. 版本兼容性说明")
    print("-" * 70)
    print(f"✓ 您的 CUDA {driver_version.split('.')[0]}.x 驱动向下兼容所有 CUDA 11.x、12.x")
    print("  这意味着您可以安装 CUDA 11.8、12.1 或 12.8 的 PyTorch，都能正常运行")
    print()
    print("  建议: 选择 CUDA 12.8 以获得最佳性能和最新特性")
    print()

    print("=" * 70)
    print("检测完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
