# Configuration examples

These JSON files are boilerplate run settings. They do not select the Python
implementation; choose the matching imports in `src/amarken/model.py` and
`src/amarken/dataset.py`.

| Example | Objective | Model import | Dataset import |
| --- | --- | --- | --- |
| `causal.json` | Causal language modeling: predict the next token; SST-2 labels are ignored | `amarken.modules.causal` | `amarken.datasets.causal` |
| `classification.json` | Text classification: predict SST-2 sentiment labels with a classification head | `amarken.modules.classification` | `amarken.datasets.classification` |

Copy an example and adapt its model, dataset, hyperparameters, and checkpoint
settings. Changing only `model_id` or `dataset.id` does not change the
training objective.
