"""
BERT意图识别快速入门示例

这个脚本演示了如何使用本项目进行完整的训练-评估-推理流程
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import create_sample_data, IntentDataPreprocessor
from src.training import IntentClassificationTrainer, compute_metrics
from src.inference import IntentPredictor
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main():
    print("="*60)
    print("BERT意图识别快速入门示例")
    print("="*60)

    # 第1步：创建示例数据
    print("\n第1步：创建示例数据")
    print("-" * 60)
    create_sample_data("./data/raw/intent_samples.json", num_samples=1000)

    # 第2步：预处理数据
    print("\n第2步：预处理数据")
    print("-" * 60)
    preprocessor = IntentDataPreprocessor(
        input_file="./data/raw/intent_samples.json",
        output_dir="./data/processed",
        test_size=0.2,
        val_size=0.1
    )
    train_path, val_path, test_path = preprocessor.process()

    # 第3步：准备模型和tokenizer
    print("\n第3步：加载模型和tokenizer")
    print("-" * 60)
    model_name = "bert-base-chinese"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=10
    )

    print(f"模型: {model_name}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 第4步：准备数据集
    print("\n第4步：准备数据集")
    print("-" * 60)

    from src.data import IntentClassificationDataset

    train_dataset = IntentClassificationDataset(
        data_path=train_path,
        tokenizer=tokenizer,
        max_length=128
    )

    val_dataset = IntentClassificationDataset(
        data_path=val_path,
        tokenizer=tokenizer,
        max_length=128
    )

    print(f"训练集: {len(train_dataset)} 条")
    print(f"验证集: {len(val_dataset)} 条")
    print(f"类别数: {len(train_dataset.get_labels())}")

    # 第5步：配置训练器
    print("\n第5步：配置训练器")
    print("-" * 60)

    config = {
        "model": {
            "name": model_name,
            "num_labels": 10
        },
        "data": {
            "max_length": 128
        },
        "training": {
            "num_epochs": 3,
            "batch_size": 32,
            "learning_rate": 3e-5,
            "warmup_ratio": 0.1,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "save_total_limit": 2,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_f1",
            "logging_steps": 20
        },
        "output": {
            "output_dir": "./outputs/quickstart_demo",
            "logging_dir": "./outputs/quickstart_demo/logs"
        },
        "seed": 42
    }

    trainer = IntentClassificationTrainer(config)
    trainer.setup(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics_fn=compute_metrics
    )

    # 第6步：训练模型
    print("\n第6步：开始训练")
    print("-" * 60)
    print("提示：首次运行会下载BERT模型，可能需要几分钟...")

    # 取消注释以下行开始训练
    # metrics = trainer.train()

    # 第7步：模型推理
    print("\n第7步：模型推理")
    print("-" * 60)
    print("训练完成后，可以使用以下代码进行推理：")

    print("""
    # 加载训练好的模型
    predictor = IntentPredictor(
        model_path="./outputs/quickstart_demo"
    )

    # 预测单个文本
    result = predictor.predict("查询银行卡余额")
    print(f"预测意图: {result['intent']}")
    print(f"置信度: {result['confidence']:.4f}")

    # 交互式预测
    predictor.interactive_mode()
    """)

    print("\n" + "="*60)
    print("快速入门示例完成！")
    print("="*60)
    print("\n提示：取消注释第6步的代码以开始实际训练")
    print("或者运行：")
    print("  python scripts/train/train_intent_classification.py --config configs/experiments/intent_classification_customer_service.yaml")


if __name__ == "__main__":
    main()