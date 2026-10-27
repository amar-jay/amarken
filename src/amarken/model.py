"""Select your implementation here; training, inference and validation share this import.

Causal modeling is selected below for supervised fine-tuning. Other implementations include;
Use modules.example for numeric,
use modules.arithmetic for the optional arithmetic SFT/RL demo,
use modules.classification for classification, or
use modules.ming for TTS.
Each module exports build/load, prediction, loss, validation and fsdp_modules hooks.
"""

import amarken.modules.example as module

build_metrics = module.build_metrics
build_model = module.build_model
fsdp_modules = module.fsdp_modules
load_model = module.load_model
loss_fn = module.loss_fn
predict = module.predict
validation_step = module.validation_step


def get_rl_hooks():
    """Check optional task capabilities only when an RL run is requested."""
    if not getattr(module, "SUPPORTS_RL", False):
        raise ValueError(
            f"Selected model module {module.__name__} does not support RL training"
        )

    names = ("prepare_prompts", "reward_fn")
    missing = [name for name in names if not callable(getattr(module, name, None))]
    if missing:
        raise ValueError(
            f"Selected model module {module.__name__} does not support RL: "
            f"missing {', '.join(missing)}"
        )
    return {name: getattr(module, name) for name in names}


__all__ = [
    "build_metrics",
    "build_model",
    "fsdp_modules",
    "get_rl_hooks",
    "load_model",
    "loss_fn",
    "predict",
    "validation_step",
]
