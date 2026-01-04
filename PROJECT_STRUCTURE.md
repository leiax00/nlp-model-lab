# NLP Model Lab - 项目结构设计

## 目录结构

```
nlp-model-lab/
├── configs/                    # 配置文件
│   ├── base/                  # 基础配置
│   │   ├── llm/              # 大语言模型配置
│   │   ├── vision/           # 视觉模型配置
│   │   └── audio/            # 音频模型配置
│   └── experiments/          # 实验特定配置
│       └── exp_001_*.yaml
│
├── data/                      # 数据目录
│   ├── raw/                  # 原始数据（不提交到git）
│   ├── processed/            # 处理后的数据（不提交到git）
│   └── cache/                # 缓存数据（不提交到git）
│
├── scripts/                   # 脚本目录
│   ├── train/                # 训练脚本
│   │   ├── train_llm.py
│   │   ├── finetune_lora.py
│   │   └── train_vision.py
│   ├── eval/                 # 评估脚本
│   │   ├── evaluate.py
│   │   └── benchmark.py
│   ├── inference/            # 推理脚本
│   │   ├── generate.py
│   │   └── serve.py
│   └── data/                 # 数据处理脚本
│       ├── prepare_dataset.py
│       └── convert_format.py
│
├── src/                       # 源代码
│   ├── models/               # 模型定义
│   │   ├── llm/             # 语言模型
│   │   ├── vision/          # 视觉模型
│   │   └── multimodal/      # 多模态模型
│   ├── training/            # 训练相关
│   │   ├── trainer.py      # 训练器基类
│   │   ├── strategies/     # 训练策略（LoRA, QLoRA, 全量等）
│   │   └── callbacks.py    # 训练回调
│   ├── data/                # 数据处理
│   │   ├── datasets.py     # 数据集类
│   │   ├── preprocessors.py # 预处理器
│   │   └── collators.py    # 数据整理器
│   ├── utils/               # 工具函数
│   │   ├── logger.py       # 日志工具
│   │   ├── metrics.py      # 评估指标
│   │   └── distributed.py  # 分布式训练工具
│   └── inference/           # 推理相关
│       ├── generator.py     # 生成器
│       └── api.py          # API接口
│
├── outputs/                   # 输出目录（不提交到git）
│   ├── checkpoints/         # 模型检查点
│   ├── logs/                # 训练日志
│   └── results/             # 实验结果
│
├── notebooks/                 # Jupyter notebooks
│   ├── exploratory/         # 探索性分析
│   └── tutorials/           # 教程
│
├── tests/                     # 测试
│   ├── test_models.py
│   ├── test_data.py
│   └── test_training.py
│
├── docs/                      # 文档
│   ├── experiments/         # 实验记录
│   ├── guides/              # 使用指南
│   └── api/                 # API文档
│
├── .env.example               # 环境变量示例
├── requirements.txt           # 依赖列表
├── setup.py                   # 包安装配置
├── pyproject.toml            # 项目配置
└── README.md                  # 项目说明
```

## 核心设计原则

### 1. 模块化设计
- **分离关注点**：训练、推理、评估各司其职
- **可复用组件**：模型、数据、训练策略独立开发
- **插件化**：方便添加新的模型类型和训练方法

### 2. 配置管理
```yaml
# configs/base/llm/qwen2.5.yaml
model:
  name: "Qwen/Qwen2.5-7B"
  type: "causal_lm"
  trust_remote_code: true

training:
  strategy: "lora"  # full, lora, qlora
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2e-4
  epochs: 3

lora:
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "v_proj"]
  dropout: 0.05
```

### 3. 实验追踪
```
outputs/
├── exp_001_qwen_lora_20250104/
│   ├── checkpoints/
│   │   ├── checkpoint-100/
│   │   └── final/
│   ├── logs/
│   │   ├── train.log
│   │   └── wandb/
│   ├── config.yaml
│   └── results.json
```

### 4. 代码组织示例

#### 训练器基类
```python
# src/training/trainer.py
class BaseTrainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.dataset = None

    def setup(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def save_checkpoint(self):
        pass

    def load_checkpoint(self):
        pass
```

#### 具体训练策略
```python
# src/training/strategies/lora.py
class LoRATrainer(BaseTrainer):
    def setup(self):
        from peft import LoraConfig, get_peft_model
        # LoRA配置逻辑
```

### 5. 统一入口
```bash
# 训练
python scripts/train/train_llm.py --config configs/experiments/exp_001.yaml

# 评估
python scripts/eval/evaluate.py --checkpoint outputs/exp_001/checkpoints/final

# 推理
python scripts/inference/generate.py --checkpoint outputs/exp_001/checkpoints/final
```

## 技术栈建议

### 核心依赖
- **模型框架**: transformers, peft, trl
- **训练引擎**: transformers.Trainer, PyTorch Lightning
- **数据处理**: datasets, tokenizers
- **实验追踪**: wandb, mlflow, tensorboard
- **分布式**: deepspeed, accelerate, fsdp
- **推理**: vllm, text-generation-inference

### 开发工具
- **代码质量**: ruff, black, isort, mypy
- **测试**: pytest
- **文档**: mkdocs

## 下一步行动

1. 创建基础目录结构
2. 设置配置文件模板
3. 实现核心训练器基类
4. 添加第一个训练示例
5. 设置环境变量和依赖管理

是否需要我为你创建这些目录和初始文件？