"""
意图识别预测器
"""
from typing import Dict, List, Union, Optional
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class IntentPredictor:
    """意图识别预测器"""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        label_names: Optional[List[str]] = None
    ):
        """
        Args:
            model_path: 模型路径
            device: 设备（cuda/cpu）
            label_names: 标签名称列表
        """
        self.model_path = Path(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"加载模型: {self.model_path}")
        print(f"使用设备: {self.device}")

        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

        # 加载标签名称
        if label_names:
            self.label_names = label_names
        else:
            # 尝试从配置文件加载
            config_path = self.model_path / "config.json"
            if config_path.exists():
                import json
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.label_names = config.get("data", {}).get("label_map", {})
            else:
                self.label_names = {}

        print(f"标签数量: {len(self.label_names)}")

    def predict(
        self,
        text: str,
        return_prob: bool = False
    ) -> Union[str, Dict[str, any]]:
        """
        预测单个文本的意图

        Args:
            text: 输入文本
            return_prob: 是否返回概率

        Returns:
            预测结果（标签或包含概率的字典）
        """
        # 分词
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 预测
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # 获取预测结果
        probs = torch.softmax(logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_id].item()

        result = {
            "text": text,
            "intent_id": pred_id,
            "intent": self.label_names.get(pred_id, str(pred_id)),
            "confidence": confidence
        }

        if return_prob:
            # 返回所有类别的概率
            prob_dict = {}
            for idx, prob in enumerate(probs[0].cpu().numpy()):
                label = self.label_names.get(idx, str(idx))
                prob_dict[label] = float(prob)
            result["probabilities"] = prob_dict

        return result

    def predict_batch(
        self,
        texts: List[str],
        return_prob: bool = False
    ) -> List[Dict]:
        """
        批量预测

        Args:
            texts: 文本列表
            return_prob: 是否返回概率

        Returns:
            预测结果列表
        """
        results = []

        # 分批处理
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # 分词
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 预测
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            probs = torch.softmax(logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1).cpu().numpy()
            confidences = torch.max(probs, dim=-1).values.cpu().numpy()

            for text, pred_id, confidence in zip(batch_texts, pred_ids, confidences):
                result = {
                    "text": text,
                    "intent_id": int(pred_id),
                    "intent": self.label_names.get(int(pred_id), str(pred_id)),
                    "confidence": float(confidence)
                }
                results.append(result)

        return results

    def predict_from_file(
        self,
        input_file: str,
        output_file: str,
        text_column: str = "text"
    ):
        """
        从文件预测并保存结果

        Args:
            input_file: 输入文件路径（JSON/JSONL）
            output_file: 输出文件路径
            text_column: 文本列名
        """
        import json

        # 读取数据
        with open(input_file, 'r', encoding='utf-8') as f:
            if input_file.endswith('.jsonl'):
                data = [json.loads(line) for line in f]
            else:
                data = json.load(f)

        print(f"加载 {len(data)} 条数据进行预测")

        # 提取文本
        texts = [item[text_column] for item in data]

        # 预测
        results = self.predict_batch(texts, return_prob=True)

        # 合并结果
        for item, result in zip(data, results):
            item.update(result)

        # 保存结果
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"结果已保存到: {output_file}")

    def interactive_mode(self):
        """交互式预测模式"""
        print("\n" + "="*60)
        print("进入交互式预测模式（输入 'quit' 退出）")
        print("="*60 + "\n")

        while True:
            text = input("请输入文本: ").strip()

            if text.lower() in ['quit', 'exit', 'q', '退出']:
                print("退出交互模式")
                break

            if not text:
                continue

            result = self.predict(text, return_prob=True)

            print(f"\n预测结果:")
            print(f"  意图: {result['intent']}")
            print(f"  置信度: {result['confidence']:.4f}")
            print(f"  所有类别概率:")
            for label, prob in sorted(
                result['probabilities'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]:
                print(f"    {label}: {prob:.4f}")
            print()