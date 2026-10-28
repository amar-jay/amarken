# Configuration examples

These JSON files are boilerplate run settings. They do not select the Python
implementation; choose the matching imports in `src/amarken/model.py` and
`src/amarken/dataset.py`.

| Example | Objective | Model import | Dataset import |
| --- | --- | --- | --- |
| `causal.json` | Causal language modeling: predict the next token; SST-2 labels are ignored | `amarken.modules.causal` | `amarken.datasets.causal` |
| `classification.json` | Text classification: predict SST-2 sentiment labels with a classification head | `amarken.modules.classification` | `amarken.datasets.classification` |
| `dpo.json` | DPO preference optimization on chosen/rejected responses | `amarken.modules.causal` | `amarken.datasets.preference` |

Copy an example and adapt its model, dataset, hyperparameters, and checkpoint
settings. Changing only `model_id` or `dataset.id` does not change the
training objective.

For DPO, also select `DPOTrainer as Trainer` in `trainers/rl/__init__.py`.
The selected adapter interprets the `"rl"` settings. GRPO and DPO configs are
algorithm-specific; see the root README for selection and checkpoint commands.
