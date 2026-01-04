import json

def load_label_map(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # key 一律转 int
    id2label = {int(k): v for k, v in raw.items()}
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id
