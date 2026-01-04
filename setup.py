"""
项目安装配置
"""
from pathlib import Path
from setuptools import setup, find_packages

# 读取README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="nlp-model-lab",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="NLP模型训练和微调实验室",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nlp-model-lab",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "ruff>=0.0.280",
            "mypy>=1.5.0",
        ],
        "accelerate": [
            "accelerate>=0.24.0",
            "deepspeed>=0.11.0",
        ],
        "peft": [
            "peft>=0.6.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "tensorboard>=2.14.0",
        ],
        "wandb": [
            "wandb>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nlp-train=scripts.train.train_intent_classification:main",
            "nlp-predict=scripts.inference.predict_intent:main",
            "nlp-eval=scripts.eval.evaluate_intent:main",
        ],
    },
)