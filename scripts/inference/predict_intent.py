"""
意图识别推理脚本

使用示例:
    # 单个文本预测
    python scripts/inference/predict_intent.py \\
        --checkpoint outputs/exp_001_bert_intent \\
        --text "查询银行卡余额"

    # 交互式模式
    python scripts/inference/predict_intent.py \\
        --checkpoint outputs/exp_001_bert_intent \\
        --interactive

    # 批量预测
    python scripts/inference/predict_intent.py \\
        --checkpoint outputs/exp_001_bert_intent \\
        --input data/raw/test_data.json \\
        --output outputs/predictions.json
"""
import argparse
import json
from pathlib import Path

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.inference import IntentPredictor


def main():
    parser = argparse.ArgumentParser(description="意图识别推理")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--text", type=str, help="要预测的文本")
    parser.add_argument("--interactive", action="store_true", help="交互式模式")
    parser.add_argument("--input", type=str, help="输入文件路径")
    parser.add_argument("--output", type=str, help="输出文件路径")
    parser.add_argument("--device", type=str, default=None, help="设备（cuda/cpu）")
    args = parser.parse_args()

    # 加载预测器
    predictor = IntentPredictor(
        model_path=args.checkpoint,
        device=args.device
    )

    # 根据模式执行预测
    if args.interactive:
        # 交互式模式
        predictor.interactive_mode()

    elif args.text:
        # 单个文本预测
        result = predictor.predict(args.text, return_prob=True)

        print("\n预测结果:")
        print(f"  文本: {result['text']}")
        print(f"  意图: {result['intent']}")
        print(f"  置信度: {result['confidence']:.4f}")
        print(f"  所有类别概率:")
        for label, prob in sorted(
            result['probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]:
            print(f"    {label}: {prob:.4f}")

    elif args.input:
        # 批量预测
        if not args.output:
            raise ValueError("批量预测需要指定 --output 参数")

        predictor.predict_from_file(
            input_file=args.input,
            output_file=args.output
        )

    else:
        # 默认进入交互模式
        predictor.interactive_mode()


if __name__ == "__main__":
    main()