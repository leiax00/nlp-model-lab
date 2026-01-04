from pathlib import Path

import yaml
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from utils.experiment import (
    create_experiment_dir,
    save_config,
    save_metrics,
    save_trainer_state, link_experiment_and_checkpoint
)


from utils.seed import set_seed
from utils.metrics import compute_classification_metrics


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    exp_dir = create_experiment_dir(
        task="text-cls",
        model="bert",
        version="v1"
    )

    config = load_config("tasks/text_classification/config.yaml")

    checkpoint_root = Path(config["training"]["output_dir"])
    checkpoint_dir = checkpoint_root / exp_dir.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 1. seed
    set_seed(config["training"]["seed"])

    # 2. model & tokenizer
    model_name = config["model"]["pretrained_model"]
    tokenizer = BertTokenizer.from_pretrained(model_name)

    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=config["model"]["num_labels"]
    )

    # 3. dataset
    dataset = load_dataset(
        "json",
        data_files={
            "train": config["data"]["train_file"],
            "validation": config["data"]["dev_file"]
        }
    )

    def preprocess(examples):
        return tokenizer(
            examples[config["data"]["text_field"]],
            truncation=True,
            padding="max_length",
            max_length=config["model"]["max_length"]
        )

    dataset = dataset.map(preprocess, batched=True)
    dataset = dataset.rename_column(
        config["data"]["label_field"], "labels"
    )
    dataset.set_format("torch")

    # 4. training args
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        learning_rate=config["training"]["learning_rate"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        num_train_epochs=config["training"]["num_epochs"],
        logging_steps=config["training"]["logging_steps"],
        evaluation_strategy=config["training"]["eval_strategy"],
        save_strategy=config["training"]["save_strategy"],
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    # 5. trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_classification_metrics
    )

    # 6. train
    trainer.train()

    # 7. save final model
    trainer.save_model(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    # -------- 实验记录 --------

    # 1. 保存 config 快照
    save_config(config, exp_dir)

    # 2. 评估并保存指标
    metrics = trainer.evaluate()
    save_metrics(metrics, exp_dir)

    # 3. 保存 trainer 状态（可选但很有用）
    save_trainer_state(trainer, exp_dir)

    # 4. 建立关联
    link_experiment_and_checkpoint(
        exp_dir=exp_dir,
        checkpoint_dir=checkpoint_dir,
        best_metric="eval_accuracy"
    )

    print(f"[Experiment saved to] {exp_dir}")

if __name__ == "__main__":
    main()
