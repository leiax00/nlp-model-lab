import argparse
import yaml
import json
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

from utils.io import load_label_map


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(model_dir):
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def predict_texts(
    texts,
    tokenizer,
    model,
    id2label,
    max_length,
    threshold,
    unknown_label_id,
    unknown_label_name
):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)

    results = []
    for text, prob in zip(texts, probs):
        score, label_id = torch.max(prob, dim=-1)
        score = float(score)
        label_id = int(label_id)

        if score < threshold:
            results.append({
                "text": text,
                "label_id": unknown_label_id,
                "label_name": unknown_label_name,
                "score": score
            })
        else:
            results.append({
                "text": text,
                "label_id": label_id,
                "label_name": id2label[label_id],
                "score": score
            })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="tasks/text_classification/config.yaml")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--text", help="single text input")
    parser.add_argument("--input_jsonl", help="batch input jsonl file")
    parser.add_argument("--output", help="output json file")
    args = parser.parse_args()

    config = load_config(args.config)

    id2label, _ = load_label_map(config["labels"]["label_map"])

    tokenizer, model = load_model(args.model_dir)

    if args.text:
        texts = [args.text]
    elif args.input_jsonl:
        texts = []
        with open(args.input_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                texts.append(json.loads(line)["text"])
    else:
        raise ValueError("Either --text or --input_jsonl must be provided")

    threshold = config["inference"]["confidence_threshold"]
    unknown_label_name = config["inference"]["unknown_label_name"]
    unknown_label_id = config["inference"]["unknown_label_id"]

    results = predict_texts(
        texts,
        tokenizer,
        model,
        id2label,
        max_length=config["model"]["max_length"],
        threshold=threshold,
        unknown_label_id=unknown_label_id,
        unknown_label_name=unknown_label_name
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
