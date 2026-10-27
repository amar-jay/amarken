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
uv run amarken train --config configs/training-example.json
uv run amarken train --config configs/training-example.json \
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
from amarken.modules.example import (
    build_model,
    build_metrics,
    load_model,
    loss_fn,
    predict,
    validation_step,
)
# from amarken.modules.base import build_model, load_model, loss_fn, predict, validation_step, fsdp_modules, build_metrics

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
from amarken.modules.classification import (
    build_model,
    build_metrics,
    load_model,
    loss_fn,
    predict,
    validation_step,
)
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
from amarken.datasets.huggingface import (
    build_dataset,
    build_dataloaders,
    build_test_dataset,
)
# from amarken.datasets.example import build_dataset, build_dataloaders, build_test_dataset
```

Both expose the same interface:

```python
train_data, validation_data = build_dataset(
    dataset_id="stanfordnlp/sst2", seed=42, train_size=128, val_size=32
)
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
Set `"dataset_id": "stanfordnlp/sst2"` in the training JSON to choose the Hub
dataset. Optional `"dataset_config": "subset-name"` selects a dataset subset;
omit it or use `null` for the default. Revisions are looked up by ID in `DATASET_REVISIONS` in `constants.py`;
unlisted IDs use the Hub default revision. Split names and column mappings stay
in constants. The numeric dataset accepts the same argument and ignores it.
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
uv run amarken train --config configs/training-base.json
uv run amarken infer --config configs/training-base.json \
  --checkpoint checkpoints/qwen-backbone/best.pt \
  --input-file samples/input-base.json
```

The tests exercise both datasets and loaders, and train the text classifier
with a tiny local Qwen model and in-memory HF data. Hub downloads are mocked.

### Precision

JSON `precision` controls training autocast (`float32` disables it). Trainable
weights and optimizer moments stay in float32; frozen subtrees can use the
configured dtype to save memory. Training initializes models in float32 on CPU,
prepares their dtypes, then moves or shards them onto the target device. Allow
CPU RAM for that full-precision initialization.

CUDA float16 uses gradient scaling; bfloat16 does not require it. FSDP2 uses its
mixed-precision policy with FP32 trainable shards and FP32 gradient reductions,
and still supports only float32/bfloat16. No additional JSON dtype setting is needed.

Checkpoints retain the actual training dtypes. Standalone inference/validation
load weights in the configured `precision`; direct `build_model`/`load_model`
calls also retain their existing storage-dtype argument and default to float32.
Existing reduced-precision checkpoints can resume: trainable weights and optimizer
moments are promoted to float32, but precision lost in earlier saves cannot be recovered.

### Final validation

Evaluate a saved checkpoint on the configured validation split:

```bash
uv run amarken validate \
  --config configs/classification.json \
  --checkpoint checkpoints/qwen-backbone/best.pt
```

The command reports `samples`, sample-weighted `val/loss`, and
TorchMetrics scores accumulated over the complete split: `accuracy`,
`precision_macro`, `recall_macro`, `f1_macro`, `f1_weighted`, per-class precision,
recall and F1 arrays, `confusion_matrix`, and `calibration_error` (all prefixed
with `val/`). Scalar scores are fractions from 0 to 1. Macro scores give classes
equal weight; weighted F1 weights them by their true sample counts. Confusion
matrix rows are true classes and columns are predicted classes. Per-class arrays
and matrix indices follow `CLASSIFICATION_LABELS` (numeric examples use class IDs).
Calibration error uses 15 confidence bins and measures the gap between predicted
confidence and accuracy; lower is better. Undefined precision/recall/F1 divisions
use zero. These metrics are computed once from full-split state, not averaged
from batch-level scores. Evaluation uses the config's `val_size`, seed, batch size,
device and precision.
This evaluates the validation data used during training, not an independent test set.
It does not train or start a W&B run. Both classification and numeric model modules
provide `build_metrics()` for a fresh TorchMetrics collection and
`validation_step(model, batch, device, metrics)` to update it and return batch-mean
loss from one forward pass. Replace these hooks alongside `loss_fn` for another
objective. The feature-only base module leaves these hooks for your task, just
like its loss function. Include `build_metrics` and
`validation_step` when switching the imports in `model.py`.

### Held-out test evaluation

```bash
uv run amarken test \
  --config configs/classification.json \
  --checkpoint checkpoints/qwen-backbone/best.pt
```

Set `test_size` in JSON (default 32). Output contains `samples`, `test/loss` and
`test/accuracy`, plus the same TorchMetrics scores under the `test/` prefix.
This command loads a checkpoint without training or selecting it.
For Hugging Face data, the labeled validation source is shuffled with a fixed
partition seed and divided in half: one half for validation and checkpoint
selection, the other exclusively for testing. With SST-2 this allows up to 436
examples per half. The run seed samples within these fixed halves; changing sizes
or run seeds cannot move a row between them. Training still uses the train source.
The numeric example uses a separate random stream for its test samples.

**Existing checkpoints:** earlier runs used the entire validation source for
checkpoint selection. Their new test subset may therefore have been seen during
validation. Retrain with this partitioning before treating test results as held out.
Keep the test set for final evaluation; use validation to tune your training.
When switching `dataset.py`, also import `build_test_dataset` from the chosen module.

### Multi-GPU training (DDP)

Use one process per GPU, with `"device": "cuda"` in JSON:

```bash
uv run torchrun --standalone --nproc-per-node=2 --no-python \
  amarken train --config configs/classification.json
```

DDP replicates the complete model on each GPU; it does not combine GPU memory.
The launch environment enables it automatically. Use plain `amarken train` for
single-device training. Fixed GPU indices such as `cuda:0` are unsuitable for a
multi-GPU config: each process gets its own GPU from `LOCAL_RANK`.

`batch_size` is per GPU: with two GPUs and batch size 4, the global batch size is
8. Training samples are distributed and reshuffled each epoch. When the sample
count is not divisible by the GPU count, the sampler repeats some rows so every
rank takes the same number of steps. Training loss includes these repeated rows.
Batch-mean loss callbacks are aggregated using sample counts.

Rank zero alone handles the run-name prompt, directory creation, tqdm, W&B, and
checkpoint writes/uploads. The full validation split runs on rank zero with no
padded duplicate samples; its loss and checkpoint decisions are shared with all
ranks. Async checkpoint serialization remains on rank zero, with communication
on the training thread. Keep task-specific collectives out of validation callbacks.

Resume with the same number of processes and the same training settings:

```bash
uv run torchrun --standalone --nproc-per-node=2 --no-python \
  amarken train --config configs/classification.json \
  --resume checkpoints/qwen-backbone/latest.pt
```

Checkpoints retain ordinary model weights for single-process inference/evaluation,
and additionally save world size and each rank's RNG state. Training resume
requires the same world size; older checkpoints count as single-process runs.
Exact epoch-boundary reproducibility assumes unchanged data, seed, batch size,
optimizer/scheduler settings and deterministic operations, with zero loader workers.
`infer`, `validate`, and `test` remain single-process commands.

The existing test suite includes a bounded two-process CPU/Gloo integration test
for synchronization, sampling, weighted metrics, resume, checkpoint compatibility,
and rank-zero failure propagation. The command above is the CUDA smoke check;
run it on a host with at least two GPUs and confirm one checkpoint directory,
one logging run, and successful resume.

### FSDP2 model sharding

Select the strategy in JSON:

```json
{
  "strategy": "fsdp2",
  "device": "cuda",
  "precision": "bfloat16"
}
```

These are fields in the existing training config; retain its other settings.
`strategy` defaults to `auto`, preserving single-device training and automatic DDP
under multi-process launches. The configured strategy applies to `torchrun` launches, detected by
`int(os.environ.get("RANK", "-1")) != -1`. Plain launches always train on a
single device, even when the config selects `ddp` or `fsdp2`. A lone
`WORLD_SIZE` value does not activate distributed training.
FSDP2 requires PyTorch 2.14 or newer and supports `float32` and `bfloat16` initially;
`float16` remains available for the other training modes.

```bash
uv run torchrun --standalone --nproc-per-node=2 --no-python \
  amarken train --config configs/classification.json
```

Use one worker per visible GPU. On a one-GPU machine, use `--nproc-per-node=1`
for a functional smoke test; there is no cross-GPU memory saving with one worker.
CPU/Gloo FSDP2 is also supported for small integration exercises using
`"device": "cpu"`. Batch size remains per process.

Unlike DDP, FSDP2 shards parameters, gradients, and optimizer state. Models start
on CPU and are moved/sharded group by group; each process still needs CPU RAM for
the complete initial model. Each GPU must fit the active group and activations.
This implementation does not add meta initialization or CPU offloading.

Model implementations export `fsdp_modules(model)` through `model.py`, alongside
the existing builder/loss/validation functions. It returns unique communication
groups in bottom-up order, ending with the root model. Numeric modules group
parameter-bearing layers; base uses the backbone's declared transformer block
classes; classification reuses those groups and adds its head and root. Adapt
this hook when replacing the architecture. Optimizers are built after sharding;
external optimizers and schedulers are currently rejected for FSDP2.

Every rank participates in FSDP2 validation, running the same complete loader so
weight-gather collectives line up. Validation loss averages those replicated
results; it does not count them as additional validation examples. Rank zero
alone displays progress, logs results, prompts for names and writes/uploads files.
DDP retains its rank-zero validation behavior.

`best.pt` and `latest.pt` remain ordinary checkpoint files usable by standalone
`infer`, `validate`, and `test`. When saving, every rank collectively gathers full
model/optimizer state into rank-zero CPU RAM; serialization/uploads can then run
asynchronously. Allow CPU RAM for the full snapshot and pending async writes.

Resume using the same strategy, precision, world size and training settings:

```bash
uv run torchrun --standalone --nproc-per-node=2 --no-python \
  amarken train --config configs/classification.json \
  --resume checkpoints/qwen-backbone/latest.pt
```

Changing between DDP and FSDP2 during training resume is rejected. Legacy
checkpoints retain their existing inference support and count as single-process
or DDP training checkpoints according to their saved world size.

Verification includes two-process CPU FSDP2 and DDP regression tests, full-state
export/loading, deterministic resume, uneven validation batches, frozen tiny
transformer backbones, and rank-zero read/write failures. A one-GPU CUDA smoke
test also passed; multi-GPU CUDA still needs verification on a multi-GPU host.

### Console evaluation tables

`validate` and `test` display metric tables by default, including per-class arrays
and separate grids for matrix metrics. Matrix indices follow the model's class
order; confusion matrix rows are true labels and columns are predictions.
Use `--format json` to preserve full-precision machine-readable output:

```bash
uv run amarken validate --config configs/classification.json \
  --checkpoint checkpoints/qwen-backbone/best.pt --format json
```

For Python callers, `run_validation()` still returns a dictionary. Render either
that dictionary or its JSON string with the shared utility:

```python
from amarken.utils import format_validation_table

print(format_validation_table(result))
```

Tables show floats with six significant digits; underlying results are unchanged.
