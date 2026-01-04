# NLP Model Lab

> 一个专业的NLP模型训练和微调框架，专注于BERT等预训练模型的实际应用

## 📋 项目简介

本项目提供了一个结构化、模块化的框架，用于训练和微调各种NLP模型。当前实现了基于BERT的意图识别任务，后续将扩展到更多任务和模型类型。

### 特性

- 🏗️ **模块化设计**：清晰的代码组织，易于扩展和维护
- ⚙️ **配置驱动**：YAML配置文件，灵活管理实验参数
- 📊 **完整流程**：数据处理、训练、评估、推理一站式解决方案
- 🔧 **即用型工具**：开箱即用的训练脚本和工具类
- 📈 **实验追踪**：支持TensorBoard和WandB日志记录

## 📁 项目结构

```
nlp-model-lab/
├── configs/              # 配置文件
│   ├── base/            # 基础配置
│   └── experiments/     # 实验配置
├── data/                # 数据目录
│   ├── raw/            # 原始数据
│   ├── processed/      # 处理后的数据
│   └── cache/          # 缓存数据
├── scripts/             # 脚本目录
│   ├── train/          # 训练脚本
│   ├── eval/           # 评估脚本
│   └── inference/      # 推理脚本
├── src/                 # 源代码
│   ├── models/         # 模型定义
│   ├── training/       # 训练相关
│   ├── data/           # 数据处理
│   ├── utils/          # 工具函数
│   └── inference/      # 推理相关
└── outputs/            # 输出目录
    ├── checkpoints/    # 模型检查点
    ├── logs/           # 训练日志
    └── results/        # 实验结果
```

## 🚀 快速开始

### 1. 环境安装

#### 🔍 第一步：检测 CUDA 环境（推荐）

```bash
# 检测你的 GPU 和 CUDA 驱动版本
python scripts/utils/check_cuda.py
```

这个工具会推荐最适合你系统的 PyTorch 版本。**特别说明**：CUDA 驱动向下兼容，例如 CUDA 13.0 驱动可以运行 CUDA 12.1 或 11.8 的 PyTorch。

#### 方式二：自动安装

**Linux/Mac:**
```bash
bash install.sh
```

**Windows:**
```powershell
.\install.ps1
```

安装脚本会自动检测你的系统并安装合适的版本。

#### 方式三：手动安装

详细的安装说明请查看 [INSTALLATION.md](INSTALLATION.md)

**快速安装（使用 Conda）:**
```bash
# 1. 创建环境
conda create -n nlp-lab python=3.10 -y
conda activate nlp-lab

# 2. 安装 PyTorch（根据你的系统选择）
# CPU 版本：
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# GPU 版本（CUDA 12.8，推荐）：
conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia -y

# GPU 版本（CUDA 12.1，备选）：
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# GPU 版本（CUDA 11.8，兼容）：
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 3. 安装其他依赖
pip install -r requirements-core.txt
```

**快速安装（使用 pip）:**
```bash
# CPU 版本：
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-core.txt

# GPU 版本（CUDA 12.8，推荐）：
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-core.txt

# GPU 版本（CUDA 12.1，备选）：
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-core.txt

# GPU 版本（CUDA 11.8，兼容）：
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-core.txt
```

### 2. 准备数据

创建示例数据用于测试：

```bash
python -m src.data.preprocessors
```

这将在 `data/processed/` 目录下生成训练、验证和测试数据集。

### 3. 训练模型

```bash
# 使用默认配置训练
python scripts/train/train_intent_classification.py --config configs/experiments/intent_classification_customer_service.yaml

# 调试模式（使用少量数据快速测试）
python scripts/train/train_intent_classification.py \
    --config configs/experiments/intent_classification_customer_service.yaml \
    --debug
```

### 4. 评估模型

```bash
python scripts/eval/evaluate_intent.py \
    --checkpoint outputs/exp_001_bert_intent \
    --test_file data/processed/intent_test.json \
    --output_dir outputs/eval_results
```

### 5. 模型推理

```bash
# 交互式模式
python scripts/inference/predict_intent.py \
    --checkpoint outputs/exp_001_bert_intent \
    --interactive

# 预测单个文本
python scripts/inference/predict_intent.py \
    --checkpoint outputs/exp_001_bert_intent \
    --text "查询银行卡余额"

# 批量预测
python scripts/inference/predict_intent.py \
    --checkpoint outputs/exp_001_bert_intent \
    --input data/raw/test_data.json \
    --output outputs/predictions.json
```

## 📊 使用示例

### 自定义数据集

1. 准备JSON格式数据：

```json
[
  {
    "text": "查询银行卡余额",
    "intent": "query_balance"
  },
  {
    "text": "转账给朋友",
    "intent": "transfer_money"
  }
]
```

2. 使用预处理器处理数据：

```python
from src.data import IntentDataPreprocessor

preprocessor = IntentDataPreprocessor(
    input_file="./data/raw/my_data.json",
    output_dir="./data/processed",
    test_size=0.2,
    val_size=0.1
)

train_path, val_path, test_path = preprocessor.process()
```

3. 创建新的配置文件，修改数据路径和标签数量

4. 开始训练！

### 自定义训练配置

编辑 `configs/experiments/` 中的配置文件：

```yaml
model:
  num_labels: 10  # 你的类别数量

training:
  num_epochs: 5
  batch_size: 32
  learning_rate: 3.0e-5
  warmup_ratio: 0.1
```

## 🎯 支持的任务

- [x] **意图识别**（Intent Classification）
  - [x] BERT-base-chinese
  - [ ] RoBERTa
  - [ ] ERNIE
- [ ] 文本分类
- [ ] 命名实体识别
- [ ] 关系抽取
- [ ] 问答系统

## 🔧 高级功能

### 使用LoRA进行参数高效微调

```python
# 在配置文件中启用LoRA
training:
  strategy: "lora"

lora:
  r: 16
  lora_alpha: 32
  target_modules: ["query", "value"]
  dropout: 0.05
```

### 分布式训练

```bash
# 使用Accelerate
accelerate config
accelerate launch scripts/train/train_intent_classification.py --config xxx.yaml

# 使用DeepSpeed
deepspeed --num_gpus=2 scripts/train/train_intent_classification.py --config xxx.yaml
```

### 实验追踪

```bash
# 启用WandB
export WANDB_API_KEY=your_key
python scripts/train/train_intent_classification.py --config xxx.yaml
```

## 📚 开发指南

### 添加新的模型类型

1. 在 `src/models/` 下创建新模块
2. 继承 `BaseTrainer` 类
3. 实现相关方法
4. 添加对应的训练脚本

### 添加新的数据集类

1. 在 `src/data/datasets.py` 中继承 `IntentClassificationDataset`
2. 实现 `_load_data` 方法
3. 更新文档

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://github.com/pytorch/pytorch)

## 📧 联系方式

如有问题或建议，请提交 [Issue](https://github.com/yourusername/nlp-model-lab/issues)

---

**注意**：本项目正在积极开发中，API可能会发生变化。建议在生产环境使用前进行充分测试。
