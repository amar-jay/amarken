"""Select datasets independently of the model implementation.

Use datasets.example with modules.example, or datasets.huggingface with
modules.classification. Both expose the same dataset/dataloader functions.
"""

from amarken.datasets.huggingface import build_dataloaders, build_dataset

__all__ = ["build_dataloaders", "build_dataset"]
