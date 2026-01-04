# nlp-model-lab

A unified repository for training, fine-tuning, and deploying NLP models
(BERT, RoBERTa, Embedding Models, etc.)

## Features
- Multi-task training (classification, NER, similarity)
- Config-driven experiments
- Reproducible training & evaluation
- Easy deployment (FastAPI / WebSocket)

## Tasks
- Text Classification
- Intent Detection
- Named Entity Recognition


## 目录结构
```
nlp-model-lab/
├── README.md
├── pyproject.toml / requirements.txt
├── .gitignore
│
├── data/                     # 所有数据（不进 git 或部分进）
│   ├── raw/                  # 原始数据
│   ├── processed/            # 处理后的数据
│   └── examples/             # 示例小数据（可进 git）
│
├── models/                   # 模型定义（结构）
│   ├── bert/
│   │   ├── modeling.py
│   │   ├── tokenizer.py
│   │   └── __init__.py
│   ├── roberta/
│   └── embeddings/
│
├── tasks/                    # 任务级（最重要）
│   ├── text_classification/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   └── config.yaml
│   │
│   ├── intent_detection/
│   ├── ner/
│   ├── sentence_similarity/
│   └── qa/
│
├── experiments/              # 实验记录（可追溯）
│   ├── 2026-01-bert-intent-v1/
│   │   ├── config.yaml
│   │   ├── metrics.json
│   │   └── notes.md
│   └── ...
│
├── checkpoints/              # 训练产物（不进 git）
│   ├── bert-intent/
│   └── roberta-cls/
│
├── scripts/                  # 通用脚本
│   ├── prepare_data.py
│   ├── export_onnx.py
│   └── convert_labels.py
│
├── inference/                # 推理 & 服务化
│   ├── local_test.py
│   ├── api_fastapi.py
│   └── ws_server.py
│
├── utils/                    # 通用工具（你会非常常用）
│   ├── logger.py
│   ├── metrics.py
│   ├── seed.py
│   └── trainer_utils.py
│
└── configs/                  # 通用配置模板
    ├── bert_base.yaml
    ├── training_default.yaml
    └── optimizer.yaml
```

## Data Format (JSONL)

Each line is a JSON object:

{
  "text": string,
  "label": int
}

- One sample per line
- UTF-8 encoding
- Label must be integer (see label_maps/)
