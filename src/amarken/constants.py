"""
Constants across codebases are defined here.
This includes model names, dataset names, and other constants that are 
can change over time. Rather than hardcoding these or creating unnecessary flags,
they are defined here to be imported and used across the codebase.
"""

import tomllib
from pathlib import Path


with Path("pyproject.toml").open("rb") as f:
    name = tomllib.load(f)["project"]["name"]
    version = tomllib.load(f)["project"]["version"]

PROJECT_NAME = name if name else "new_project"
PROJECT_VERSION = version if version else "0.0.0"
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# hf models to use for the base models
BASES = ("Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-0.5B")

REVISIONS = {
    BASES[0]: "c1899de289a04d12100db370d81485cdf75e47ca",
    BASES[1]: "060db6499f32faf8b98477b0a26969ef7d8b9987",
}

HUB_CACHE = ".cache/huggingface/hub"
DATA_CACHE = ".cache/huggingface/datasets"
WANDB_CACHE = ".cache/wandb"

WANDB_MODES = ("disabled", "offline", "online")