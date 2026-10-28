"""Select the RL adapter here; each algorithm owns its inputs and training state."""

from amarken.trainers.rl.grpo import GRPOTrainer as Trainer

# For preference optimization, replace the import above with:
# from amarken.trainers.rl.dpo import DPOTrainer as Trainer

__all__ = ["Trainer"]
