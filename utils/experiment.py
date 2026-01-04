import os
import json
import yaml
from datetime import datetime
from pathlib import Path
import shutil


def create_experiment_dir(
    base_dir="experiments",
    task="text-cls",
    model="bert",
    version="v1"
):
    date = datetime.now().strftime("%Y-%m-%d")
    exp_name = f"{date}-{task}-{model}-{version}"
    exp_dir = Path(base_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=False)
    return exp_dir


def save_config(config: dict, exp_dir: Path):
    with open(exp_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True)


def save_metrics(metrics: dict, exp_dir: Path):
    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def save_trainer_state(trainer, exp_dir: Path):
    state_path = Path(trainer.args.output_dir) / "trainer_state.json"
    if state_path.exists():
        shutil.copy(state_path, exp_dir / "trainer_state.json")

def link_experiment_and_checkpoint(
    exp_dir: Path,
    checkpoint_dir: Path,
    best_metric: str = None
):
    # experiment -> checkpoint
    with open(exp_dir / "checkpoint.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint_dir": str(checkpoint_dir),
                "best_metric": best_metric
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    # checkpoint -> experiment
    with open(checkpoint_dir / "experiment.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_dir": str(exp_dir),
                "created_at": datetime.now().isoformat()
            },
            f,
            ensure_ascii=False,
            indent=2
        )