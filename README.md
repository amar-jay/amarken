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

`src/example` contains compatibility imports and a thin entry point; reusable
training and storage implementations live in `src/amarken`.
