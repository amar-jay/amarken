"""Select datasets independently of the model implementation.

Use datasets.example with modules.example, or datasets.classification with
modules.classification. Plain causal language modeling uses datasets.causal; instruction/response
SFT uses datasets.sft; the numeric example is selected below. Arithmetic SFT/RL uses datasets.arithmetic.
Ming TTS uses datasets.tts.
All expose the same dataset/dataloader functions.
"""

from amarken.datasets import example as module

build_dataloaders = module.build_dataloaders
build_dataset = module.build_dataset
build_test_dataset = module.build_test_dataset


def build_rl_dataset(config, **kwargs):
    """Delegate optional RL preparation to the selected dataset implementation."""
    builder = getattr(module, "build_rl_dataset", None)
    if not callable(builder):
        raise TypeError(
            f"Selected dataset module {module.__name__} does not support RL datasets"
        )
    return builder(config, **kwargs)


def build_preference_dataset(config, **kwargs):
    """Prepare chosen/rejected records only when the selected adapter needs them."""
    builder = getattr(module, "build_preference_dataset", None)
    if not callable(builder):
        raise TypeError(
            f"Selected dataset module {module.__name__} does not support preference datasets"
        )
    return builder(config, **kwargs)


__all__ = [
    "build_dataloaders",
    "build_dataset",
    "build_preference_dataset",
    "build_rl_dataset",
    "build_test_dataset",
]
