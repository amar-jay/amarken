# amarken

Reusable model-training boilerplate.

## Configuration

Use one file, `configs/training-example.json`, for the entire run: model,
training settings, W&B tracking and checkpoint storage. The CLI loads it once
into `TrainingConfig`; the trainer passes its checkpoint settings to the logger.

```json
{
  "model_id": "synthetic-mlp",
  "seed": 42,
  "learning_rate": 0.01,
  "batch_size": 16,
  "epochs": 3,
  "device": "cpu",
  "precision": "float32",
  "run_name": "synthetic-example",
  "wandb": {
    "enabled": false,
    "mode": "offline"
  },
  "checkpoints": {
    "backend": "local",
    "policy": "default",
    "async_save": true,
    "max_pending": 2,
    "overwrite": true,
    "monitor": "val/loss",
    "mode": "min",
    "bucket": null,
    "prefix": "checkpoints",
    "endpoint_url": null,
    "region_name": null
  }
}
```

Edit the `checkpoints` section in that same file to select storage:

| Backend | Settings |
| --- | --- |
| `local` | Keep checkpoints locally; no remote storage. |
| `wandb` | Set `wandb.enabled` to `true`; save local files and W&B model artifacts. |
| `s3` | Set `bucket`, optionally `prefix`, `region_name` and `endpoint_url` for S3-compatible storage. |

W&B metric tracking is controlled by `wandb.enabled` and `wandb.mode` in the
same config, independently of checkpoint storage. Bucket storage works with
W&B disabled. Credentials come from AWS environment variables, an AWS profile
or an IAM role; keep them out of the config.

Every backend first saves under the fixed `checkpoints/` directory, outside
SDK caches. Only global rank zero writes. Failed uploads retain the local file
and raise an error. Bucket keys use `prefix` plus the checkpoint's relative path;
reusing a key overwrites it.

## Asynchronous saving

Set `checkpoints.async_save` to `true` in the same config (enabled in the example).
The code default is `false` for synchronous saves. `max_pending` limits the total
number of queued/running checkpoint snapshots; the default is 2.

With async saving, the training thread first copies tensors to independent CPU
storage and deep-copies the remaining state. This snapshot blocks briefly and
uses additional host memory. Disk serialization and local/S3/W&B storage then
run on a single background worker, preserving save order. W&B uploads complete
before a later job can replace the same file. If the queue is full, the next
save waits before allocating another snapshot. File replacement remains atomic;
serialization failures leave an older checkpoint intact.

The trainer polls worker errors between batches and waits for pending saves at
the end of training, including on exceptions. Upload failures raise and retain
the local file. A hard process kill can lose pending work. Resume only from a
completed checkpoint.

When using the logger directly, use its context manager or call `finish()` to
flush and close the worker. Async `save_checkpoint()` returns a `Future`; call
`future.result()` to wait for that save or `flush_checkpoints()` to wait for all
saves without closing the logger. Externally supplied trainer loggers are
flushed by the trainer but must be closed by their owner. Keep logger/checkpoint
calls on the training thread; snapshots support ordinary nested tensor state
containers, including model/optimizer state dictionaries.

## Training example

Run the small CPU classification task (no downloads or W&B account needed):

```bash
uv run amarken example --config configs/training-example.json
uv run amarken example --config configs/training-example.json \
  --resume "checkpoints/synthetic-example/latest.pt"
```

`epochs` is the total target epoch count, including completed epochs.

| Policy | Saved files |
| --- | --- |
| `best` | `best.pt`, updated only on strict metric improvement. |
| `default` | `best.pt` plus `latest.pt`, updated every completed epoch. |

`default` is the default policy. `monitor` selects `val/loss` or `train/loss`;
`mode` is `min` or `max`. Validation loss requires a validation loader.

`checkpoints.overwrite` controls the directory independently of policy:

- `true` (default): reuse `checkpoints/<run_name>/`, replacing the same files.
- `false`: save under `checkpoints/<run_name>/<entered-name-or-uuid>/`.
  At the beginning of `fit()`, the helper in `amarken.utils` asks for a name
  using Questionary. This works for CLI and direct trainer use. Press Enter to use
  a generated UUID. Noninteractive runs use UUIDs without prompting. Ctrl-C
  cancels training. Existing named directories are rejected for fresh runs.

Best/latest still update within an isolated run's directory. Resume restores
its name/UUID without prompting and retains the best metric when monitor and
mode match. A fresh run starts a new best comparison. Use `Trainer(...,
run_label="my-run")` to name isolated runs programmatically. Avoid concurrent
jobs sharing a directory when overwrite is enabled.

Directory rules also apply to bucket keys. W&B run names use the entered name
or UUID; artifact version history is retained independently of local files.

The synthetic example constructs an MLP; `model_id` records the task's model name.
The generic trainer accepts an already constructed model, rather than loading it.

## Inference

Run the synthetic classifier using a saved training checkpoint:

```bash
uv run amarken infer --checkpoint checkpoints/synthetic-example/best.pt \
  --input '[1, 1, 0, 0, 0, 0, 0, 0]'
```

Supply one sample with 8 numeric features or a batch shaped `[N, 8]`.
For a JSON data file, use `--input-file inputs.json` instead of `--input`.
The command prints JSON with `predictions` (class indices) and `probabilities`
(one probability per class). The device comes from `device` in the same config
file used for training. Pass `--config configs/training-example.json` to choose
that file; the example config is the default for both commands.
For isolated runs, include the entered run name/UUID in the checkpoint path.

Inference restores the selected implementation from checkpoint weights, uses
evaluation/inference mode and does not initialize W&B, prompt for a run name or
restore optimizer/RNG state. The selected implementation controls model construction, checkpoint loading and
prediction. Both `src/amarken/modules/example.py` and `src/amarken/modules/base.py`
expose the same model functions and a module-specific loss:

```python
# Change only this module path to switch implementations.
from amarken.modules.example import build_model, load_model, loss_fn, predict
# from amarken.modules.base import build_model, load_model, loss_fn, predict

model = build_model(device="cpu")  # New model / pretrained backbone
model = load_model("checkpoints/my-task/best.pt", device="cpu")
result = predict(model, inputs)  # Reuses the loaded model
```

`src/amarken/model.py` contains the import selection used by both training and
inference. Change `modules.example` to `modules.base` there to select the
Hugging Face implementation. There is no registry or model factory to configure.
`inference.py` only loads the selected model and calls its `predict()` function.

The numeric example accepts 8-feature samples and returns class predictions and
probabilities. The Hugging Face backbone accepts a string or list of strings
and returns pooled `features`; add a head and task-specific interpretation there
if needed. For text inference, pass JSON such as `--input '["hello", "world"]'`.
Inputs and outputs remain specific to each implementation, behind matching calls.

Hugging Face settings live in `constants.py`: `BASE_MODEL_ID`, `REVISIONS`,
`BASE_MAX_LENGTH`, `BASE_LOCAL_FILES_ONLY`, and `HUB_CACHE`.
The example dimensions are defined there too. Keep architecture settings the
same when restoring trained checkpoints. Backbone restore loads the pinned
config and tokenizer from the Hub/cache, constructs the architecture, then
loads checkpoint weights without reloading the original pretrained weights.
Set `BASE_LOCAL_FILES_ONLY=True` to require cached config/tokenizer assets.

Dataset selection lives in `dataset.py`; loss handling lives in the selected
model module. Choose a compatible model/dataset pair as described below.
Either implementation module can be deleted once the other is selected and
its unused imports are removed. The synthetic defaults require no Hub downloads.

## Custom training

For your own task, import `TrainingConfig`, `Trainer`, and `seed_everything` from
`amarken.trainer`. Seed before constructing the model and loaders, and supply
`loss_fn(model, batch, device)` returning a scalar loss. The callback moves its
batch to the supplied device. Optionally pass a validation loader, optimizer,
per-epoch scheduler, or an externally managed logger. The default optimizer is
AdamW. Logged epoch losses are averages of batch losses.

The trainer supports a single process/device, float32 or bfloat16 autocast, and
CUDA float16 with gradient scaling. Checkpoints include model, optimizer,
scheduler, scaler, completed epochs, batch steps, and Python/PyTorch RNG state.
Resume supports epoch boundaries; use global RNG shuffling, deterministic
transforms and `num_workers=0` for reproducibility. Custom loader generators,
worker RNG, NumPy RNG, custom samplers and distributed state need task-specific
handling. Schedulers are stepped once per epoch without arguments; metric-driven
schedulers need a custom loop. Download remote checkpoints locally before resume.

Training and storage implementations live in `src/amarken`.

`amarken/cli.py` owns argument definitions, the single parsing call, command
dispatch and terminal output. `amarken/main.py` contains the training function;
`amarken/inference.py` contains model loading and prediction functions. Neither
workflow parses arguments or depends on argparse.

Both implementations are called as `model(inputs)`. For the numeric example,
`inputs` is a feature tensor. For the Hugging Face backbone, it is a tokenized
dictionary, including `input_ids` and `attention_mask`:

```python
inputs = model.tokenize(["hello", "another text"])
features = model(inputs)
```

The backbone unpacks that dictionary internally. Each implementation owns its
`loss_fn(model, batch, device)`, selected alongside the model imports in
`model.py`. The numeric example keeps its existing cross-entropy behavior.
The base implementation is an explicit task-loss placeholder: pooled features
do not define a classification or language-modeling objective. Implement that
function when your task/targets are defined. Choose the matching dataset module; numeric data cannot be passed directly
to the text backbone.

## Optional text classification

`modules/classification.py` derives from `base` and adds a linear classification
head with a cross-entropy objective. It exposes the same `build_model`,
`load_model`, `predict`, and `loss_fn` functions. To select it, change the import
in `src/amarken/model.py` to:

```python
from amarken.modules.classification import build_model, load_model, loss_fn, predict
```

`CLASSIFICATION_LABELS` in `constants.py` defaults to `("negative", "positive")`.
Targets are integer class indices in that order. `CLASSIFICATION_FREEZE_BACKBONE`
defaults to `True`: only the head trains, and the frozen backbone stays in eval
mode. Set it to `False` to fine-tune the backbone too. Prediction returns class
indices, label names and probabilities. Restore loads both backbone and head;
keep labels and backbone settings consistent with the saved checkpoint.

```python
model = build_model(device="cpu")
inputs = model.tokenize(["I loved this", "I disliked this"])
targets = torch.tensor([1, 0])
loss = loss_fn(model, (inputs, targets), "cpu")
loss.backward()
```

Use the Hugging Face dataset module below for tokenized classification batches. Existing feature-backbone checkpoints do not include the new head;
train and save a classifier checkpoint before loading it with this module.

## Dataset modules

Datasets follow the same import-selection pattern as models:

```python
# src/amarken/dataset.py — choose one implementation:
from amarken.datasets.huggingface import build_dataset, build_dataloaders
# from amarken.datasets.example import build_dataset, build_dataloaders
```

Both expose the same interface:

```python
train_data, validation_data = build_dataset(seed=42, train_size=128, val_size=32)
train_loader, validation_loader = build_dataloaders(
    train_data,
    validation_data,
    batch_size=16,
    tokenize=getattr(model, "tokenize", None),
)
```

| Dataset module | Model module | Batches |
| --- | --- | --- |
| `datasets.example` | `modules.example` | Random feature tensors and integer labels. |
| `datasets.huggingface` | `modules.classification` | Tokenized text dictionaries and integer labels. |

The random module preserves the original 128/32 numeric demo, generating data
with a private seeded generator. Set `train_size` and `val_size` in the training
JSON config; both must be positive integers. Defaults are 128 and 32.

The Hugging Face module defaults to pinned `stanfordnlp/sst2` train/validation
splits, with 0=negative and 1=positive, matching the classifier label order.
Dataset ID, revision, split names and column names live in `constants.py`.
The same JSON `train_size` and `val_size` settings control sample counts, capped
at the available split sizes. Sampling limits
reduce training work, but the initial dataset download may include full splits.

Text is tokenized per batch through the selected model's `tokenize()`, preserving
its padding/context-length handling. The first run downloads/cache dataset files;
subsequent runs reuse the cache. Both loaders use zero workers and global
PyTorch RNG shuffling, consistent with the trainer's epoch-boundary resume.

`dataset.py` currently selects Hugging Face data to match the selected
classification model. `main.py` no longer contains hardcoded numeric data.
Select the random dataset and numeric model together for a download-free run.
To train and infer with the current text setup:

```bash
uv run amarken example --config configs/training-base.json
uv run amarken infer --config configs/training-base.json \
  --checkpoint checkpoints/qwen-backbone/best.pt \
  --input-file samples/input-base.json
```

The tests exercise both datasets and loaders, and train the text classifier
with a tiny local Qwen model and in-memory HF data. Hub downloads are mocked.

### Precision

JSON `precision` is the single setting for model weight storage and training
autocast. Training and inference pass it to the selected module's `build_model`
and `load_model`, with the same signatures across all implementations.
For bfloat16 weights and execution, set `"precision": "bfloat16"` in your config.
No separate dtype constant is needed. Direct Python calls can pass
`precision="bfloat16"` to either function; their default is `float32`.
