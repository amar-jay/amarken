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
CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "checkpoints"

HUB_CACHE = ".cache/huggingface/hub"
DATA_CACHE = ".cache/huggingface/datasets"
WANDB_CACHE = ".cache/wandb"

REVISIONS = {
    "Qwen/Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen/Qwen2.5-0.5B": "060db6499f32faf8b98477b0a26969ef7d8b9987",
    "inclusionAI/Ming-3.0-tts-0.5B": "8d66b70a00f05ad183fa8a863b4ce609fc83cbc0",
}

DATASET_REVISIONS = {
    "trl-lib/ultrafeedback_binarized": "47124cb5778f5d50de1c7676a412828f3ea7c555",
    "tatsu-lab/alpaca": "dce01c9b08f87459cf36a430d809084718273017",
    "stanfordnlp/sst2": "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb",
    "bookbot/ljspeech_phonemes": "5379ba66d6b869d8ba74c8341f3100d645246d5d",
}


# Optional Hugging Face backbone settings. Keep these consistent with checkpoints.
BASE_MAX_LENGTH = 2048
BASE_LOCAL_FILES_ONLY = False

# Causal language modeling: reject overlong training text rather than silently truncate.
CAUSAL_MAX_LENGTH = 512
CAUSAL_MAX_NEW_TOKENS = 64

# Single-turn SFT: optional fixed system instruction. Qwen thinking stays disabled.
SFT_SYSTEM_PROMPT = ""

# Arithmetic GRPO task; canonical operand pairs prevent reversed-sum leakage.
ARITHMETIC_MAX_OPERAND = 99
ARITHMETIC_HARD_OPERANDS = (10_000, 99_999)
ARITHMETIC_HARD_POOL_SIZE = 8192
ARITHMETIC_MAX_NEW_TOKENS = 16

# Numeric example architecture.
EXAMPLE_INPUT_SIZE = 8
EXAMPLE_HIDDEN_SIZE = 16
EXAMPLE_NUM_CLASSES = 2

# Optional classification module: class order must match training checkpoints.
CLASSIFICATION_LABELS = ("negative", "positive")
CLASSIFICATION_FREEZE_BACKBONE = True

# Ming needs custom codec, flow-head and generation code that Transformers
# does not provide. MING_RUNTIME points to that upstream source checkout;
# MING_RUNTIME_REVISION pins its Git commit for compatibility and reproducibility.
# This code pin is separate from REVISIONS[model_id], which pins the HF weights
# and model configuration. The upstream runtime is not an installable package.
MING_RUNTIME = Path(__file__).resolve().parents[2] / ".cache/Ming-3.0-tts"
MING_RUNTIME_REVISION = "028c1eace193e7acdb256453da89b6e9207e052b"
MING_PROMPT = "Please generate speech based on the following description.\nTarget language: English.\n"
