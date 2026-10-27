"""Select datasets independently of the model implementation.

Use datasets.example with modules.example, or datasets.huggingface with
modules.classification. Plain causal language modeling uses datasets.causal; instruction/response
SFT uses datasets.sft, selected below. Ming TTS uses datasets.tts.
All expose the same dataset/dataloader functions.
"""

from amarken.datasets.sft import (
    build_dataloaders,
    build_dataset,
    build_test_dataset,
)

__all__ = ["build_dataloaders", "build_dataset", "build_test_dataset"]
