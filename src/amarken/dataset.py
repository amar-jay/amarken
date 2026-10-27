"""Select datasets independently of the model implementation.

Use datasets.example with modules.example, or datasets.huggingface with
modules.classification. Causal language modeling uses datasets.causal, selected
below. Ming TTS uses datasets.tts.
All expose the same dataset/dataloader functions.
"""

from amarken.datasets.causal import (
    build_dataloaders,
    build_dataset,
    build_test_dataset,
)

__all__ = ["build_dataloaders", "build_dataset", "build_test_dataset"]
