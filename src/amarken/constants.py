"""
Constants across codebases are defined here.
This includes model names, dataset names, and other constants that are
can change over time. Rather than hardcoding these or creating unnecessary flags,
they are defined here to be imported and used across the codebase.
"""

import tomllib
from pathlib import Path

with Path("pyproject.toml").open("rb") as f:
    project = tomllib.load(f)["project"]
    name = project["name"]
    version = project["version"]

PROJECT_NAME = name if name else "new_project"
PROJECT_VERSION = version if version else "0.0.0"
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# hf models to use for the base models
BASES = ("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-0.5B")

REVISIONS = {
    BASES[0]: "2fc06364715b967f1860aea9cf38778875588b17",
    BASES[1]: "c1899de289a04d12100db370d81485cdf75e47ca",
    BASES[2]: "060db6499f32faf8b98477b0a26969ef7d8b9987",
}

HUB_CACHE = ".cache/huggingface/hub"
DATA_CACHE = ".cache/huggingface/datasets"
WANDB_CACHE = ".cache/wandb"

# Stable across working directories; keep training state outside SDK caches.
CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "checkpoints"

# Optional Hugging Face backbone settings. Keep these consistent with checkpoints.
BASE_MODEL_ID = BASES[0]
BASE_MAX_LENGTH = 2048
BASE_LOCAL_FILES_ONLY = False

# Numeric example architecture.
EXAMPLE_INPUT_SIZE = 8
EXAMPLE_HIDDEN_SIZE = 16
EXAMPLE_NUM_CLASSES = 2

# Optional classification module: class order must match training checkpoints.
CLASSIFICATION_LABELS = ("negative", "positive")
CLASSIFICATION_FREEZE_BACKBONE = True


# Hugging Face dataset example. SST-2 labels: 0=negative, 1=positive.
DATASET_REVISIONS = {
    "stanfordnlp/sst2": "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb",
}
HF_DATASET_TRAIN_SPLIT = "train"
HF_DATASET_VALIDATION_SPLIT = "validation"
HF_DATASET_TEXT_COLUMN = "sentence"
HF_DATASET_LABEL_COLUMN = "label"
