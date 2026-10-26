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
  "wandb_enabled": false,
  "wandb_mode": "offline",
  "checkpoints": {
    "backend": "local",
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
| `wandb` | Set `wandb_enabled` to `true`; save local files and W&B model artifacts. |
| `s3` | Set `bucket`, optionally `prefix`, `region_name` and `endpoint_url` for S3-compatible storage. |

W&B metric tracking is controlled by `wandb_enabled` and `wandb_mode` in the
same config, independently of checkpoint storage. Bucket storage works with
W&B disabled. Credentials come from AWS environment variables, an AWS profile
or an IAM role; keep them out of the config.

Every backend first saves under the fixed `checkpoints/` directory, outside
SDK caches. Only global rank zero writes. Failed uploads retain the local file
and raise an error. Bucket keys use `prefix` plus the checkpoint's relative path;
reusing a key overwrites it.

## Training example

Run the small CPU classification task (no downloads or W&B account needed):

```bash
uv run amarken example --config configs/training-example.json
uv run amarken example --config configs/training-example.json \
  --resume checkpoints/synthetic-example/epoch-0001.pt
```

`epochs` is the total target epoch count, including completed epochs. Each epoch
writes a separate checkpoint beneath `checkpoints/<run_name>/`. Choose a new
`run_name` for independent experiments; repeated names overwrite matching files.
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
