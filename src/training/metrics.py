"""
评估指标计算
"""
from typing import Dict, List
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


def compute_metrics(pred) -> Dict[str, float]:
    """
    计算分类指标

    Args:
        pred: EvalPrediction对象，包含predictions和label_ids

    Returns:
        指标字典
    """
    # 获取预测结果
    logits = pred.predictions
    labels = pred.label_ids

    # 获取预测类别
    if len(logits.shape) > 2:
        # 处理序列模型的输出
        logits = logits.mean(axis=1)

    predictions = np.argmax(logits, axis=1)

    # 计算指标
    accuracy = accuracy_score(labels, predictions)

    # 计算precision, recall, f1（平均方式）
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average='weighted',
        zero_division=0
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def compute_metrics_per_class(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_names: List[str]
) -> Dict[str, float]:
    """
    计算每个类别的指标

    Args:
        predictions: 预测结果
        labels: 真实标签
        label_names: 标签名称列表

    Returns:
        每个类别的指标字典
    """
    report = classification_report(
        labels,
        predictions,
        target_names=label_names,
        output_dict=True,
        zero_division=0
    )

    return report


def print_evaluation_report(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_names: List[str]
):
    """
    打印详细的评估报告

    Args:
        predictions: 预测结果
        labels: 真实标签
        label_names: 标签名称列表
    """
    print("\n" + "="*60)
    print("评估报告")
    print("="*60)

    report = classification_report(
        labels,
        predictions,
        target_names=label_names,
        digits=4
    )

    print(report)

    # 混淆矩阵
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(labels, predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names
    )
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()

    return plt.gcf()