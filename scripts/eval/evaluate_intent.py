"""
意图识别模型评估脚本

使用示例:
    python scripts/eval/evaluate_intent.py \\
        --checkpoint outputs/exp_001_bert_intent \\
        --test_file data/processed/intent_test.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data import IntentClassificationDataset
from src.training.metrics import compute_metrics_per_class, print_evaluation_report
from src.inference import IntentPredictor


def main():
    parser = argparse.ArgumentParser(description="评估意图识别模型")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--test_file", type=str, required=True, help="测试数据文件")
    parser.add_argument("--output_dir", type=str, default=None, help="结果输出目录")
    parser.add_argument("--device", type=str, default=None, help="设备（cuda/cpu）")
    args = parser.parse_args()

    # 加载预测器
    print(f"加载模型: {args.checkpoint}")
    predictor = IntentPredictor(
        model_path=args.checkpoint,
        device=args.device
    )

    # 加载测试数据
    print(f"\n加载测试数据: {args.test_file}")
    test_dataset = IntentClassificationDataset(
        data_path=args.test_file,
        tokenizer=predictor.tokenizer,
        max_length=128
    )

    print(f"测试集大小: {len(test_dataset)}")

    # 获取标签名称
    label_names = test_dataset.get_labels()
    print(f"标签数量: {len(label_names)}")

    # 更新预测器的标签名称
    predictor.label_names = {i: name for i, name in enumerate(label_names)}

    # 预测
    print("\n开始预测...")
    predictions = []
    labels = []

    for idx in tqdm(range(len(test_dataset)), desc="预测中"):
        item = test_dataset[idx]
        text = item["text"]
        true_label = item["labels"]

        # 预测
        result = predictor.predict(text)
        pred_label = result["intent_id"]

        predictions.append(pred_label)
        labels.append(true_label)

    predictions = np.array(predictions)
    labels = np.array(labels)

    # 计算指标
    print("\n" + "="*60)
    print("评估结果")
    print("="*60)

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )

    print(f"\n总体指标:")
    print(f"  准确率 (Accuracy): {accuracy:.4f}")
    print(f"  精确率 (Precision): {precision:.4f}")
    print(f"  召回率 (Recall): {recall:.4f}")
    print(f"  F1分数: {f1:.4f}")

    # 每个类别的指标
    print("\n各类别指标:")
    metrics_per_class = compute_metrics_per_class(predictions, labels, label_names)

    for label in label_names:
        if label in metrics_per_class:
            metrics = metrics_per_class[label]
            print(f"\n  {label}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1-Score: {metrics['f1-score']:.4f}")
            print(f"    Support: {metrics['support']}")

    # 生成混淆矩阵图
    try:
        import matplotlib.pyplot as plt
        fig = print_evaluation_report(predictions, labels, label_names)

        # 保存混淆矩阵
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            cm_path = output_dir / "confusion_matrix.png"
            plt.savefig(cm_path, dpi=150, bbox_inches='tight')
            print(f"\n混淆矩阵已保存到: {cm_path}")
            plt.close()
    except ImportError:
        print("\n注意：需要安装 matplotlib 才能生成混淆矩阵")
        print("安装命令: pip install matplotlib")

    # 保存结果
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "metrics_per_class": metrics_per_class
        }

        results_path = output_dir / "eval_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n评估结果已保存到: {results_path}")


if __name__ == "__main__":
    main()