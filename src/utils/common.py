"""
通用工具函数
"""
import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int, deterministic: bool = False):
    """
    设置随机种子以保证可重复性

    Args:
        seed: 随机种子
        deterministic: 是否使用确定性算法（可能会影响性能）
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> torch.device:
    """
    获取设备

    Args:
        device: 指定设备（cuda/cpu/auto）

    Returns:
        torch.device对象
    """
    if device:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    计算模型参数量

    Args:
        model: PyTorch模型
        trainable_only: 是否只统计可训练参数

    Returns:
        参数数量
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def format_number(num: int) -> str:
    """
    格式化数字（添加千分位）

    Args:
        num: 数字

    Returns:
        格式化后的字符串
    """
    return f"{num:,}"


def print_model_info(model: torch.nn.Module):
    """
    打印模型信息

    Args:
        model: PyTorch模型
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    print("\n模型信息:")
    print(f"  总参数量: {format_number(total_params)}")
    print(f"  可训练参数量: {format_number(trainable_params)}")

    if trainable_params < total_params:
        percentage = trainable_params / total_params * 100
        print(f"  冻结参数量: {format_number(total_params - trainable_params)}")
        print(f"  可训练比例: {percentage:.2f}%")