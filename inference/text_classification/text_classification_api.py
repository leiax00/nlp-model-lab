import yaml
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification

from utils.io import load_label_map


# ---------- 请求 / 响应模型 ----------

class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label_id: int
    label_name: str
    score: float


# ---------- 服务初始化 ----------

CONFIG_PATH = "tasks/text_classification/config.yaml"
MODEL_DIR = "checkpoints/text_classification"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

id2label, _ = load_label_map(config["labels"]["label_map"])

tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# ---------- FastAPI ----------

app = FastAPI(
    title="Text Classification API",
    version="1.0.0"
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device)
    }


@app.post("/predict", response_model=PredictResponse)
@torch.no_grad()
def predict(req: PredictRequest):
    inputs = tokenizer(
        req.text,
        truncation=True,
        padding=True,
        max_length=config["model"]["max_length"],
        return_tensors="pt"
    ).to(device)

    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)[0]

    score, label_id = torch.max(probs, dim=-1)
    score = float(score)
    label_id = int(label_id)

    threshold = config["inference"]["confidence_threshold"]

    if score < threshold:
        return {
            "label_id": config["inference"]["unknown_label_id"],
            "label_name": config["inference"]["unknown_label_name"],
            "score": score
        }

    return {
        "label_id": label_id,
        "label_name": id2label[label_id],
        "score": score
    }
