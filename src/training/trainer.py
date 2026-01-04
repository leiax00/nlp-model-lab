"""
训练器基类
"""
from typing import Dict, Any, Optional
from pathlib import Path
import json

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
from transformers import DataCollatorWithPadding


class BaseTrainer:
    """训练器基类"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 配置字典
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset,
        eval_dataset,
        compute_metrics_fn
    ):
        """设置训练环境"""
        raise NotImplementedError

    def train(self):
        """开始训练"""
        raise NotImplementedError

    def save_model(self, output_dir: str):
        """保存模型"""
        raise NotImplementedError


class IntentClassificationTrainer(BaseTrainer):
    """意图分类训练器"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # 配置
        self.model_config = config.get("model", {})
        self.data_config = config.get("data", {})
        self.training_config = config.get("training", {})
        self.output_config = config.get("output", {})

    def setup(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset,
        eval_dataset,
        compute_metrics_fn
    ):
        """设置训练器
        :param model:
        :param tokenizer:
        :param train_dataset:
        :param eval_dataset:
        :param compute_metrics_fn:
        :return:
        """
        self.model = model
        self.tokenizer = tokenizer

        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.output_config.get("output_dir", "./outputs"),
            logging_dir=self.output_config.get("logging_dir", "./outputs/logs"),

            # 训练参数
            num_train_epochs=self.training_config.get("num_epochs", 3),
            per_device_train_batch_size=self.training_config.get("batch_size", 32),
            per_device_eval_batch_size=self.training_config.get("batch_size", 64),
            gradient_accumulation_steps=self.training_config.get("gradient_accumulation_steps", 1),

            # 优化器
            learning_rate=self.training_config.get("learning_rate", 2e-5),
            weight_decay=self.training_config.get("weight_decay", 0.01),
            warmup_ratio=self.training_config.get("warmup_ratio", 0.1),

            # 调度器
            lr_scheduler_type=self.training_config.get("scheduler", "linear"),

            # 梯度裁剪
            max_grad_norm=self.training_config.get("max_grad_norm", 1.0),

            # 评估和保存
            evaluation_strategy=self.training_config.get("eval_strategy", "epoch"),
            save_strategy=self.training_config.get("save_strategy", "epoch"),
            save_total_limit=self.training_config.get("save_total_limit", 3),
            load_best_model_at_end=self.training_config.get("load_best_model_at_end", True),
            metric_for_best_model=self.training_config.get("metric_for_best_model", "eval_f1"),

            # 早停
            greater_is_better=True,

            # 日志
            logging_steps=self.training_config.get("logging_steps", 10),
            overwrite_output_dir=self.output_config.get("overwrite_output_dir", True),

            # 性能优化
            fp16=self.training_config.get("fp16", False),
            gradient_checkpointing=self.training_config.get("gradient_checkpointing", False),

            # 其他
            seed=self.config.get("seed", 42),
            report_to=["tensorboard"],
        )

        # 数据整理器
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True
        )

        # 创建Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn,
        )

        return self.trainer

    def train(self):
        """开始训练"""
        if self.trainer is None:
            raise ValueError("请先调用 setup() 方法设置训练器")

        print("开始训练...")
        train_result = self.trainer.train()

        # 保存训练结果
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)

        # 保存模型
        self.save_model(self.trainer.args.output_dir)

        print("训练完成！")
        return metrics

    def save_model(self, output_dir: str):
        """保存模型和tokenizer"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存模型
        self.trainer.save_model(output_dir)

        # 保存tokenizer
        self.tokenizer.save_pretrained(output_dir)

        # 保存配置
        config_path = output_path / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)

        print(f"模型已保存到: {output_dir}")

    def evaluate(self, test_dataset=None):
        """评估模型"""
        if self.trainer is None:
            raise ValueError("请先调用 setup() 方法设置训练器")

        print("开始评估...")
        metrics = self.trainer.evaluate(test_dataset)

        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)

        return metrics