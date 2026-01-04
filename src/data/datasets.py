"""
数据集类定义
"""
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

import json
from pathlib import Path

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


@dataclass
class IntentExample:
    """意图识别样本"""
    text: str
    label: int
    label_name: str = ""


class IntentClassificationDataset(Dataset):
    """意图分类数据集"""

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
        text_column: str = "text",
        label_column: str = "intent"
    ):
        """
        Args:
            data_path: 数据文件路径（JSON格式）
            tokenizer: 分词器
            max_length: 最大序列长度
            text_column: 文本列名
            label_column: 标签列名
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column

        # 加载数据
        self.examples = self._load_data()
        self.label2id = self._build_label_map()
        self.id2label = {v: k for k, v in self.label2id.items()}

    def _load_data(self) -> List[IntentExample]:
        """加载数据文件"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")

        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        examples = []
        for item in data:
            example = IntentExample(
                text=item[self.text_column],
                label=item[self.label_column],
                label_name=str(item.get("label_name", item[self.label_column]))
            )
            examples.append(example)

        return examples

    def _build_label_map(self) -> Dict[str, int]:
        """构建标签映射"""
        labels = sorted(list(set(ex.label for ex in self.examples)))
        return {label: idx for idx, label in enumerate(labels)}

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Union[int, str]]:
        example = self.examples[idx]

        # 分词
        encoding = self.tokenizer(
            example.text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None
        )

        # 返回模型输入
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": example.label,
            "text": example.text
        }

    def get_labels(self) -> List[str]:
        """获取所有标签"""
        return list(self.label2id.keys())


class IntentClassificationDatasetCSV(IntentClassificationDataset):
    """从CSV文件加载的意图分类数据集"""

    def _load_data(self) -> List[IntentExample]:
        """从CSV加载数据"""
        import pandas as pd

        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")

        df = pd.read_csv(self.data_path)

        examples = []
        for _, row in df.iterrows():
            example = IntentExample(
                text=row[self.text_column],
                label=row[self.label_column],
                label_name=str(row.get("label_name", row[self.label_column]))
            )
            examples.append(example)

        return examples