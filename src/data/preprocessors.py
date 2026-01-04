"""
数据预处理工具
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

from sklearn.model_selection import train_test_split


class IntentDataPreprocessor:
    """意图识别数据预处理器"""

    def __init__(
        self,
        input_file: str,
        output_dir: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_seed: int = 42
    ):
        """
        Args:
            input_file: 输入数据文件（JSON/JSONL/CSV）
            output_dir: 输出目录
            test_size: 测试集比例
            val_size: 验证集比例（从训练集中划分）
            random_seed: 随机种子
        """
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.test_size = test_size
        self.val_size = val_size
        self.random_seed = random_seed

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process(self) -> Tuple[str, str, str]:
        """
        处理数据并划分训练/验证/测试集

        Returns:
            (train_path, val_path, test_path)
        """
        # 加载数据
        data = self._load_data()

        # 数据清洗
        data = self._clean_data(data)

        # 划分数据集
        train_data, test_data = train_test_split(
            data,
            test_size=self.test_size,
            random_state=self.random_seed,
            stratify=[item["intent"] for item in data]
        )

        train_data, val_data = train_test_split(
            train_data,
            test_size=self.val_size,
            random_state=self.random_seed,
            stratify=[item["intent"] for item in train_data]
        )

        # 保存数据
        train_path = self.output_dir / "intent_train.json"
        val_path = self.output_dir / "intent_val.json"
        test_path = self.output_dir / "intent_test.json"

        self._save_data(train_data, train_path)
        self._save_data(val_data, val_path)
        self._save_data(test_data, test_path)

        # 打印统计信息
        self._print_statistics(train_data, val_data, test_data)

        return str(train_path), str(val_path), str(test_path)

    def _load_data(self) -> List[Dict]:
        """加载数据文件"""
        suffix = self.input_file.suffix.lower()

        if suffix == ".json":
            with open(self.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

        elif suffix == ".jsonl":
            data = []
            with open(self.input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))

        elif suffix == ".csv":
            import pandas as pd
            df = pd.read_csv(self.input_file)
            data = df.to_dict('records')

        else:
            raise ValueError(f"不支持的数据格式: {suffix}")

        return data

    def _clean_data(self, data: List[Dict]) -> List[Dict]:
        """数据清洗"""
        cleaned_data = []

        for item in data:
            # 移除空文本
            text = item.get("text", "").strip()
            if not text:
                continue

            # 确保有标签
            if "intent" not in item:
                continue

            # 文本长度过滤
            if len(text) < 2 or len(text) > 500:
                continue

            cleaned_data.append(item)

        print(f"原始数据: {len(data)} 条, 清洗后: {len(cleaned_data)} 条")

        return cleaned_data

    def _save_data(self, data: List[Dict], path: Path):
        """保存数据"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"保存数据到: {path} ({len(data)} 条)")

    def _print_statistics(self, train_data, val_data, test_data):
        """打印数据统计信息"""
        print("\n" + "="*50)
        print("数据集统计")
        print("="*50)

        for name, data in [("训练集", train_data), ("验证集", val_data), ("测试集", test_data)]:
            labels = [item["intent"] for item in data]
            label_counts = Counter(labels)

            print(f"\n{name} ({len(data)} 条)")
            print("-" * 40)
            for label, count in label_counts.most_common():
                percentage = count / len(data) * 100
                print(f"  {label}: {count} ({percentage:.1f}%)")


def create_sample_data(output_file: str, num_samples: int = 1000):
    """
    创建示例数据用于测试

    Args:
        output_file: 输出文件路径
        num_samples: 样本数量
    """
    intents = [
        "query_balance",
        "transfer_money",
        "bill_payment",
        "card_loss",
        "password_reset",
        "account_opening",
        "investment_consultation",
        "complaint",
        "human_service",
        "greeting"
    ]

    templates = {
        "query_balance": [
            "查询余额",
            "我的账户还有多少钱",
            "帮我查一下银行卡余额",
            "还剩多少钱",
            "账户余额查询"
        ],
        "transfer_money": [
            "转账给{amount}元",
            "我要转{amount}块钱",
            "汇款{amount}元",
            "给他人转账{amount}元",
            "转出{amount}元"
        ],
        "bill_payment": [
            "缴纳话费",
            "支付电费",
            "交水费",
            "充值话费{amount}元",
            "支付燃气费"
        ],
        "card_loss": [
            "我的银行卡丢了",
            "卡片遗失怎么办",
            "挂失银行卡",
            "卡不见了",
            "信用卡丢失"
        ],
        "password_reset": [
            "修改密码",
            "重置登录密码",
            "忘记密码了",
            "设置新密码",
            "密码找回"
        ],
        "account_opening": [
            "我要开户",
            "怎么办理银行卡",
            "开立新账户",
            "申请银行卡",
            "注册账户"
        ],
        "investment_consultation": [
            "有什么理财产品",
            "投资建议",
            "理财咨询",
            "基金推荐",
            "如何投资"
        ],
        "complaint": [
            "我要投诉",
            "服务态度不好",
            "申请投诉",
            "对服务不满意",
            "问题反馈"
        ],
        "human_service": [
            "转人工客服",
            "人工服务",
            "我要找客服",
            "转人工",
            "客服接听"
        ],
        "greeting": [
            "你好",
            "嗨",
            "早上好",
            "在吗",
            "您好"
        ]
    }

    data = []
    amounts = ["100", "200", "500", "1000", "50", "300"]

    for _ in range(num_samples):
        intent = random.choice(intents)
        template = random.choice(templates[intent])

        # 填充模板
        if "{amount}" in template:
            text = template.format(amount=random.choice(amounts))
        else:
            text = template

        # 添加一些噪声
        if random.random() < 0.1:
            text = text + "哈"

        data.append({
            "text": text,
            "intent": intent
        })

    # 保存数据
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"已创建示例数据: {output_file} ({num_samples} 条)")


if __name__ == "__main__":
    # 创建示例数据
    create_sample_data("./data/raw/intent_samples.json", num_samples=1000)

    # 预处理数据
    preprocessor = IntentDataPreprocessor(
        input_file="./data/raw/intent_samples.json",
        output_dir="./data/processed",
        test_size=0.2,
        val_size=0.1
    )

    train_path, val_path, test_path = preprocessor.process()
    print(f"\n数据处理完成！")