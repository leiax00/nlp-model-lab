# AGENTS.md

本文件为 AI 助手（Claude Code、GitHub Copilot 等）提供在此代码仓库中工作时所需的指导。

## 项目概述

NLP Model Lab 是一个用于训练和微调 NLP 模型的模块化框架，专注于基于 BERT 的意图识别任务。项目采用配置驱动的方式，使用 YAML 配置文件，将数据处理、训练、评估和推理的关注点分离。

**当前实现**：面向客服领域的 BERT 意图识别。

## 核心命令

### 环境配置
```bash
# 检测 CUDA 并获取 PyTorch 版本推荐
python scripts/utils/check_cuda.py

# 安装核心依赖（在安装 PyTorch 之后）
pip install -r requirements-core.txt

# 可选：可视化工具
pip install -r requirements-viz.txt

# 可选：PEFT 工具（LoRA、QLoRA）
pip install -r requirements-peft.txt
```

### 数据处理
```bash
# 生成示例数据并预处理
python -m src.data.preprocessors
```

### 训练
```bash
# 使用配置文件进行完整训练
python scripts/train/train_intent_classification.py --config configs/experiments/intent_classification_customer_service.yaml

# 调试模式（使用少量数据快速测试）
python scripts/train/train_intent_classification.py --config configs/experiments/intent_classification_customer_service.yaml --debug
```

### 评估
```bash
# 在测试集上评估
python scripts/eval/evaluate_intent.py --checkpoint outputs/exp_001_bert_intent --test_file data/processed/intent_test.json --output_dir outputs/eval_results
```

### 推理
```bash
# 交互式模式
python scripts/inference/predict_intent.py --checkpoint outputs/exp_001_bert_intent --interactive

# 单条文本预测
python scripts/inference/predict_intent.py --checkpoint outputs/exp_001_bert_intent --text "查询银行卡余额"

# 批量预测
python scripts/inference/predict_intent.py --checkpoint outputs/exp_001_bert_intent --input data/raw/test_data.json --output outputs/predictions.json
```

### 代码质量
```bash
# 格式化代码
black src/ scripts/

# 代码检查
ruff check src/ scripts/

# 类型检查
mypy src/
```

## 架构设计

### 配置驱动设计
框架使用分层 YAML 配置：
- **基础配置** (`configs/base/`)：模型特定的默认参数
- **实验配置** (`configs/experiments/`)：针对特定运行覆盖基础配置

配置结构：
```yaml
model: {name, num_labels, dropout}
data: {max_length, train_file, val_file, text_column, label_column}
training: {num_epochs, batch_size, learning_rate, strategy}
output: {output_dir, logging_dir}
seed: 42
```

### 核心抽象

**训练器模式**：所有训练器继承自 `BaseTrainer` (src/training/trainer.py:19)
- `BaseTrainer`：抽象接口，定义 `setup()`、`train()`、`save_model()` 方法
- `IntentClassificationTrainer`：使用 HuggingFace `Trainer` 的具体实现
- 为不同任务创建新训练器时，继承 `BaseTrainer`

**数据集模式**：所有数据集继承自 `IntentClassificationDataset` (src/data/datasets.py:15)
- 处理 JSON/CSV 格式
- 自动标签映射
- 集成分词功能
- 通过重写 `_load_data()` 方法扩展自定义数据格式

**预测器模式**：推理逻辑位于 `IntentPredictor` (src/inference/predictor.py:11)
- 单条/批量预测
- 交互模式
- 文件推理

### 关键设计决策

1. **以 HuggingFace Transformers 为核心**：使用 `transformers.Trainer` 进行训练循环，而非自定义训练循环
2. **配置优于代码**：实验参数在 YAML 中，而非硬编码
3. **关注点分离**：
   - `src/training/`：仅包含训练逻辑
   - `src/data/`：数据加载和预处理
   - `src/inference/`：预测逻辑
   - `scripts/`：串联各个组件的入口点
4. **无需模型类定义**：直接使用 HuggingFace 模型（AutoModelForSequenceClassification），无需自定义模型架构文件

### 实验输出结构
```
outputs/
├── exp_001_xxx/
│   ├── checkpoints/
│   │   ├── checkpoint-100/
│   │   └── final/
│   ├── logs/
│   │   └── events.out.tfevents.*
│   ├── config.yaml
│   └── results.json
```

训练脚本会将实验配置保存到输出目录以确保可复现性。

## 添加新功能

### 新的模型/任务类型
1. 在 `configs/base/<task>/<model>.yaml` 创建基础配置
2. 在 `configs/experiments/<task>_<experiment>.yaml` 创建实验配置
3. 如果数据格式不同，在 `src/data/datasets.py` 添加数据集类
4. 在 `src/training/trainer.py` 添加训练器或创建新文件
5. 在 `scripts/train/train_<task>.py` 创建训练脚本

### 新的训练策略（如 LoRA）
配置文件已包含 `training.strategy` 字段。实现时需要：
1. 在训练器的 `setup()` 方法中添加策略特定逻辑
2. 导入必要的库（LoRA 使用 peft）
3. 使用策略包装基础模型

### 添加评估指标
编辑 `src/training/metrics.py:9`（`compute_metrics` 函数）。该函数接收包含 `predictions`（logits）和 `label_ids`（int 标签）的 `EvalPrediction` 对象。返回指标名称到浮点数值的字典。

## 重要说明

### PyTorch 安装
PyTorch 版本区分 CPU/GPU。使用 `python scripts/utils/check_cuda.py` 检测并推荐正确版本。CUDA 驱动向下兼容（例如 CUDA 13.0 驱动可运行 PyTorch CUDA 12.8、12.1 或 11.8）。

### 数据格式
期望的 JSON 格式：
```json
[
  {"text": "示例文本", "intent": "标签名称"},
  ...
]
```
预处理器会自动将其分割为训练/验证/测试集。

### 训练调试
在训练脚本中使用 `--debug` 标志。此模式：
- 将训练轮次减少到 1
- 将 batch_size 设置为 2
- 使用最少的数据子集
- 用于代码修改时的快速迭代

### 配置文件加载
训练脚本手动加载 YAML 配置（不使用 omegaconf/hydra 解析），但配置遵循结构化格式。参考 `configs/base/intent_classification/bert-base-chinese.yaml`。

## 项目目录结构说明

```
nlp-model-lab/
├── configs/                    # 配置文件
│   ├── base/                  # 基础配置（模型默认参数）
│   └── experiments/           # 实验配置（特定运行的参数）
├── data/                      # 数据目录
│   ├── raw/                  # 原始数据（不提交到 git）
│   ├── processed/            # 处理后的数据（不提交到 git）
│   └── cache/                # 缓存数据（不提交到 git）
├── scripts/                   # 可执行脚本
│   ├── train/                # 训练脚本入口
│   ├── eval/                 # 评估脚本
│   ├── inference/            # 推理脚本
│   └── utils/                # 工具脚本（如 CUDA 检测）
├── src/                       # 源代码
│   ├── data/                 # 数据处理（数据集类、预处理器）
│   ├── training/             # 训练逻辑（训练器、指标）
│   ├── inference/            # 推理逻辑（预测器）
│   └── utils/                # 工具函数（日志、通用工具）
└── outputs/                   # 输出目录（不提交到 git）
    ├── checkpoints/          # 模型检查点
    ├── logs/                 # 训练日志
    └── results/              # 实验结果
```

## 开发工作流

### 典型的训练流程
1. 准备数据（JSON 格式）→ `data/raw/`
2. 运行预处理器 → `python -m src.data.preprocessors`
3. 创建/修改实验配置 → `configs/experiments/xxx.yaml`
4. 训练模型 → `python scripts/train/train_intent_classification.py --config xxx.yaml`
5. 评估模型 → `python scripts/eval/evaluate_intent.py --checkpoint outputs/xxx`
6. 模型推理 → `python scripts/inference/predict_intent.py --checkpoint outputs/xxx`

### 调试流程
1. 使用 `--debug` 标志快速验证代码
2. 检查 `outputs/` 中的日志
3. 使用 TensorBoard 查看训练曲线：`tensorboard outputs/logs`
4. 在 `scripts/inference/predict_intent.py --interactive` 中测试模型

## 常见问题

### CUDA 版本选择
- CUDA 13.x/12.x 驱动：推荐 PyTorch CUDA 12.8（最新）或 12.1（稳定）
- CUDA 11.x 驱动：使用 PyTorch CUDA 11.8
- 运行 `python scripts/utils/check_cuda.py` 获取个性化推荐

### 内存不足
- 减少 `batch_size`
- 启用 `gradient_accumulation_steps`
- 使用 `--debug` 模式测试
- 考虑使用 LoRA 等参数高效微调方法

### 扩展到新任务
1. 查看现有实现（意图识别）
2. 创建新的配置文件
3. 添加数据集类（如需要）
4. 创建训练器（如需要）
5. 保持相同的代码结构和命名约定