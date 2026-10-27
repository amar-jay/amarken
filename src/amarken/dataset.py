"""Select datasets independently of the model implementation.

Use datasets.example with modules.example, or datasets.huggingface with
modules.classification. Ming TTS uses datasets.tts, selected below.
All expose the same dataset/dataloader functions.
"""

from amarken.datasets.tts import (
    build_dataloaders,
    build_dataset,
    build_test_dataset,
)

__all__ = ["build_dataloaders", "build_dataset", "build_test_dataset"]
