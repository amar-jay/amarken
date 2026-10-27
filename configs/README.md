# Configuration guide

**A config does not select Python implementations.** Before using a different
row below, change the model import in `src/amarken/model.py` and the dataset
import in `src/amarken/dataset.py`. Changing only `model_id` or `dataset.id`
changes the source used by those implementations, not their training objective.
The current default imports are `modules.causal` and `datasets.sft`.

| Config | Model module (`amarken.modules.*`) | Dataset module (`amarken.datasets.*`) | Objective / distinction |
| --- | --- | --- | --- |
| `example.json` | `example` | `example` | Small MLP on random numeric features; CPU smoke test, no downloads. |
| `classification.json` | `classification` | `classification` | SST-2 sentiment prediction with a trainable head and currently frozen backbone. |
| `causal.json` | `causal` | `causal` | Full fine-tuning on plain SST-2 text; predict every next token, ignoring sentiment labels. |
| `sft.json` | `causal` / `arithmetic` | `sft` / `arithmetic` | Response-only LoRA SFT: Alpaca by default; commented alternatives provide arithmetic warm-up. |
| `rl.json` | `arithmetic` | `arithmetic` | TRL GRPO on five-digit addition, using generated answers and exact-match rewards. |
| `tts.json` | `ming` | `tts` | Paired text/audio Ming adaptation; selected conditioning/stop layers are trained. |

For example, the SFT selection is:

```python
# src/amarken/model.py
import amarken.modules.causal as module

# src/amarken/dataset.py: keep the existing exported function names.
from amarken.datasets.sft import build_dataloaders, build_dataset, build_test_dataset
```

Then run:

```bash
uv run amarken train --config configs/sft.json
```

For arithmetic, switch both imports to `arithmetic`. In `sft.json`, comment out
the active Alpaca `dataset` and `run_name`, then uncomment the arithmetic
alternatives. Keep exactly one active definition of each. Then run:

```bash
uv run amarken train --config configs/sft.json
uv run amarken validate --config configs/sft.json --checkpoint checkpoints/qwen-arithmetic-sft/best.pt
uv run amarken train --config configs/rl.json
uv run amarken test --config configs/rl.json --checkpoint checkpoints/qwen-arithmetic-grpo/best.pt
```

The two arithmetic stages share model ID and LoRA settings. The SFT stage teaches
format/stopping on operands 0–99; RL uses operands 10,000–99,999. Standalone
validation/test use the dataset from the supplied config. Run the RL stage fresh,
without `--resume`, when moving from SFT to RL; `rl.initial_checkpoint` supplies
the starting weights. `--resume` continues an existing RL training state.

For TTS, select `ming` and `tts`, prepare the pinned runtime described in the
root README, and use `uv run --extra tts amarken train --config configs/tts.json`.

Common settings:

- `device` and `precision` apply to training and standalone inference/evaluation.
- `strategy: "auto"` permits normal single-device runs and DDP under `torchrun`.
  The example explicitly selects FSDP2 for testing; use `"auto"` for an ordinary run.
- `checkpoints.policy: "default"` retains `best.pt` and `latest.pt`; `"best"`
  retains only `best.pt`. `overwrite: false` prompts for a separate run directory.
- `wandb.enabled: false` disables W&B even when `mode` is `"online"`.
- `dataset.config` is an optional Hugging Face dataset configuration name, not an
  arithmetic difficulty setting. Synthetic datasets define their own dataset IDs.
- Supervised training monitors `val/loss` by default; RL here maximizes
  `val/accuracy`.

These are starter configurations, not hardware-independent training guarantees.
Full causal fine-tuning trains all weights, while classification freezes the
backbone and the SFT/RL presets use LoRA. Those choices have different memory
requirements despite similarly sized model downloads.
