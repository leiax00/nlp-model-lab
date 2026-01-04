"""
BERT意图识别训练脚本

使用示例:
    python scripts/train/train_intent_classification.py --config configs/experiments/intent_classification_customer_service.yaml
"""
import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed

# 添加项目根目录到Python路径
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data import IntentClassificationDataset
from src.training import IntentClassificationTrainer, compute_metrics


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f) if config_path.endswith('.json') else json.loads(f.read())

    # 如果是YAML，需要使用yaml库
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

    return config


def main():
    parser = argparse.ArgumentParser(description="训练意图识别模型")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--device", type=str, default=None, help="设备（cuda/cpu）")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    args = parser.parse_args()

    # 加载配置
    print(f"加载配置文件: {args.config}")
    config = load_config(args.config)

    # 设置随机种子
    seed = config.get("seed", 42)
    set_seed(seed)
    torch.manual_seed(seed)

    # 设置设备
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 调试模式：减少数据和epoch
    if args.debug:
        print("调试模式：使用少量数据和epoch")
        config["training"]["num_epochs"] = 1
        config["training"]["batch_size"] = 2

    # 1. 加载tokenizer和模型
    print("\n" + "="*60)
    print("加载模型和tokenizer")
    print("="*60)

    model_name = config["model"]["name"]
    print(f"模型: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=config["model"].get("num_labels", 10),
        hidden_dropout_prob=config["model"].get("dropout", 0.1),
        attention_probs_dropout_prob=config["model"].get("attention_dropout", 0.1)
    )

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 2. 准备数据集
    print("\n" + "="*60)
    print("加载数据集")
    print("="*60)

    data_config = config["data"]

    train_dataset = IntentClassificationDataset(
        data_path=data_config["train_file"],
        tokenizer=tokenizer,
        max_length=data_config.get("max_length", 128),
        text_column=data_config.get("text_column", "text"),
        label_column=data_config.get("label_column", "intent")
    )

    val_dataset = IntentClassificationDataset(
        data_path=data_config["val_file"],
        tokenizer=tokenizer,
        max_length=data_config.get("max_length", 128),
        text_column=data_config.get("text_column", "text"),
        label_column=data_config.get("label_column", "intent")
    )

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"标签数量: {len(train_dataset.get_labels())}")

    # 3. 创建训练器
    print("\n" + "="*60)
    print("设置训练器")
    print("="*60)

    trainer = IntentClassificationTrainer(config)
    trainer.setup(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics_fn=compute_metrics
    )

    # 4. 开始训练
    print("\n" + "="*60)
    print("开始训练")
    print("="*60)

    metrics = trainer.train()

    print("\n训练完成！最终指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # 5. 最终评估
    print("\n" + "="*60)
    print("最终评估")
    print("="*60)

    eval_metrics = trainer.evaluate()

    print("\n验证集指标:")
    for key, value in eval_metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\n训练完成！模型已保存。")


if __name__ == "__main__":
    main()