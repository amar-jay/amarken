# Cloud training

This setup runs on one Linux x86_64 NVIDIA GPU host. The default job prepares
CLINC, trains independent head-only and LoRA models, fits calibration, and writes
held-out reports. It never provisions, stops or deletes a paid instance.

Use a single Ampere-or-newer GPU with at least 24 GB VRAM, at least 32 GB host RAM,
and a 100–150 GB persistent volume with at least 50 GB free. A 48 GB A6000 is a
reasonable starting profile; a 24 GB GPU also supports the profile but may require
a smaller candidate chunk. The default is 32 candidates per chunk, versus the
local laptop's one-candidate safety default. Cloud throughput is not benchmarked.
The lockfile currently installs CUDA 13 PyTorch wheels: the host needs a compatible
NVIDIA driver. Setup verifies a real BF16 operation and does not replace drivers.

## Build the source bundle locally

```bash
python3 cloud/bundle.py
python3 cloud/run.py --dry-run
```

The bundle contains only explicitly selected source/config/documentation files and
`uv.lock`. It excludes `.env`, credentials, local datasets, weights, `.git`, caches,
and virtual environments. Files are fingerprinted in `bundle-manifest.json`.
The archive is ignored by Git and refuses to overwrite an existing bundle.

## Use an existing GPU host

Copy `artifacts/cloud/amarken-source.tar.gz` to the host and extract it into a
fresh checkout directory on persistent storage. For example, if the mounted volume
is `/workspace`, run on the remote machine:

```bash
mkdir -p /workspace/amarken
cd /workspace/amarken
tar -xzf /path/to/amarken-source.tar.gz
AMARKEN_STORAGE=/workspace/amarken-state bash cloud/setup.sh
```

Setup pins uv 0.12.5, installs Python 3.13 and the locked production dependencies,
and checks the GPU. Cache, prepared data, runs and outputs are linked to
`AMARKEN_STORAGE`. Point this at the provider's persistent volume; an instance's
local disk is not a backup and may disappear when the instance is deleted.
No external model API key is needed for these public model/dataset downloads.

Review/edit `cloud/clinc.json` before the first run. A new configuration requires
a new job name. The initial profile uses three epochs, effective batch size 16,
32 candidates per chunk, and a checkpoint every ten optimizer steps. Resume uses
the same source bundle, lockfile, configuration and prepared data.

Start the job in a terminal multiplexer so SSH disconnects do not stop training:

```bash
tmux new -s amarken
cd /workspace/amarken
.venv/bin/python -u cloud/run.py --config cloud/clinc.json
# Detach with Ctrl-b, then d. Reattach with: tmux attach -t amarken
```

For a different host directory, change only `cd`; scripts discover the repository
root. The runner writes `runs/clinc-qwen3-v1/status.json` and per-stage logs.
`cloud/run.py --preflight` checks hardware without starting a job.
`cloud/run.py --dry-run` prints stages without GPU access, downloads or training.
Set `CUDA_VISIBLE_DEVICES` explicitly when a host exposes several GPUs.

After an interruption, rerun the same command. Completed stages with matching
fingerprints are skipped; training resumes from `latest.json`. Incomplete
preparation/calibration/report directories are preserved with a `.partial-...`
suffix before retrying. Partial checkpoint directories are preserved with an
`.incomplete-...` suffix. If training was interrupted before its first checkpoint,
archive that incomplete training directory and rerun; no progress can be recovered.
Do not edit code/config mid-run. The runner rejects source/config drift on resume.

Monitor:

```bash
cat runs/clinc-qwen3-v1/status.json
tail -f runs/clinc-qwen3-v1/train-lora.log
nvidia-smi
```

Final models and reports are in `outputs/clinc-qwen3-v1/`; resumable checkpoints
and provenance are in `runs/clinc-qwen3-v1/`. Copy both off the instance before
stopping/deleting it. Keep prepared data and the original source bundle to resume
elsewhere. Filesystem persistence does not provide off-instance backup.
The pipeline records completion but leaves instance lifecycle decisions to you;
**billing continues after training finishes**.

## Brev access

Brev is installed locally but was logged out when this setup was prepared.
Authenticate with `brev login`, then inspect existing resources before creating:

```bash
brev ls
brev search --min-vram 24 --sort price
```

Select exactly one instance type and review its current hourly and storage cost.
After explicit provisioning approval, creation can use:

```bash
brev create amarken-train --type SELECTED_TYPE --min-disk 100
```

No provisioning command has been executed. Once the instance is available:

```bash
brev copy artifacts/cloud/amarken-source.tar.gz amarken-train:~/amarken-source.tar.gz
brev shell amarken-train
# On the remote host, select its actual persistent volume, extract the archive,
# and follow the setup/run steps above. /workspace is an example, not a Brev guarantee.
```

Do not copy local credential directories. Copy final artifacts back using
`brev copy` with the actual remote output path. Stopping or deleting an instance
requires explicit confirmation; this repository performs neither automatically.

For Tasksource, use a separate job name, set its dataset ID, and provide a reviewed
source allowlist via `review_file`. Put the review file on the remote host explicitly;
the default source bundle does not include arbitrary extra JSON files. Review the
unseen-source split separately; the default CLINC cloud job only trains intent choice.


## Experiment tracking

The `wandb` section of `cloud/clinc.json` logs offline by default. To use the live
W&B dashboard, run `.venv/bin/wandb login` on the host (or inject `WANDB_API_KEY`
through the provider's secret environment) and set `wandb.mode` to `online` before
the first job run. Optionally set `wandb.entity` to your workspace. Do not put an
API key into this config or the source bundle. Both training experiments log to
the configured project, grouped by job name. Each training directory records its
W&B run ID, so online resume uses the same dashboard run.

`wandb.log_checkpoints` is false by default; enabling it uploads weights-only
artifacts, excluding optimizer state, data rows, source snapshots and data manifests.
Offline logs are under each training directory's `wandb/` folder and persist on
the training volume. Sync those explicit directories later with `wandb sync`.
Read the root README for local tracking flags and offline resume semantics.

## Off-instance checkpoint recovery

Use a private S3-compatible bucket (AWS S3, Cloudflare R2, MinIO, etc.) independent
of the GPU instance. In `cloud/clinc.json`, set:

```json
"checkpoint_backup": {
  "uri": "s3://YOUR_BUCKET/amarken/clinc-qwen3-v1",
  "endpoint_url": null
}
```

The runner appends `/head` and `/lora` to keep the experiments separate. Use a
unique prefix for every independent run and exactly one writer per prefix. Leave
`uri` null to disable uploads. Set `endpoint_url` for non-AWS services, for example
`https://ACCOUNT_ID.r2.cloudflarestorage.com`; it must not include credentials.
Inject `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, optionally `AWS_SESSION_TOKEN`,
and `AWS_DEFAULT_REGION` through the cloud provider's secret environment, or use
an instance identity. Do not put keys in job JSON, Git, the source bundle or CLI
arguments. The SDK also supports standard AWS profiles and endpoint environment
configuration. Create/select the bucket separately; the job never creates one.

### Cloudflare R2

For R2, use the bucket's S3 API endpoint from the Cloudflare dashboard:

```json
"checkpoint_backup": {
  "uri": "s3://YOUR_R2_BUCKET/amarken/clinc-qwen3-v1",
  "endpoint_url": "https://YOUR_ACCOUNT_ID.r2.cloudflarestorage.com"
}
```

Keep the `s3://` URI scheme: the client uses R2's S3-compatible API. Supply the
R2 Access Key ID and Secret Access Key as `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY` through your GPU provider's secret environment, and set
`AWS_DEFAULT_REGION=auto`. Use the exact endpoint shown for your bucket, including
any jurisdiction-specific hostname. Use these same settings when restoring on a
replacement instance. The endpoint is the S3 API endpoint, not a public bucket URL
or custom download domain.

See [Cloudflare's boto3 setup](https://developers.cloudflare.com/r2/examples/aws/boto3/).

By default, local checkpoints and remote backups happen every 10 optimizer steps.
Validation runs every 100 steps and at epoch/end of training, updating the best
checkpoint by validation loss. Before the first validation there is no `best.json`;
`latest.json` already protects resumable progress. Validation itself can be costly.
Both intervals are configurable independently.

Each committed snapshot contains the latest full training checkpoint and a
reference to the current best validated checkpoint. Full checkpoints include
adapter/head weights, optimizer/scheduler/RNG states, progress and history. A
recovery context also backs up the exact prepared split files, manifests, run
metadata, W&B run identity and explicitly selected source/lock files. This can
include private training examples if you use a private dataset; choose the bucket
and access controls accordingly. Credential directories and process environment
values are never collected by this backup implementation. Context is uploaded
once per process; checkpoint objects use content-addressed names. Unchanged best
archives are reused during the process.

The job uploads complete archives and a snapshot manifest before publishing the
remote `latest.json`/`best.json` pointers. Restore verifies SHA-256 before installing
files. Interrupted uploads leave prior complete snapshots available. There is no
remote deletion or retention pruning. Protect referenced objects from expiration;
provider bucket versioning and abandoned-multipart cleanup can be configured
separately. The SDK retries failed requests up to five attempts, then stops
training with the local checkpoint intact instead of silently continuing without
backups. Check `RUN_DIR/backup/status.json` for the last upload status.

On a fresh instance with the project environment and bucket access:

```bash
uv run amarken restore \
  --backup-uri s3://YOUR_BUCKET/amarken/clinc-qwen3-v1/lora \
  --which latest \
  --output runs/restored-lora \
  --prepared-output data/restored-clinc
```

Add `--backup-endpoint-url URL` for a custom S3 endpoint. Both output directories
must be new. Use `--which best` to recover the best validated state instead of the
latest progress. Restoring latest also restores its referenced best checkpoint.
The source/lock snapshot is under `runs/restored-lora/source-package/`; use the same
source and locked environment when resuming. Download the pinned public backbone
again if it is not cached. W&B offline log sessions are not backed up; use online
tracking or separately sync them if you need off-instance dashboard history.

For the default cloud LoRA profile, resume with:

```bash
AMARKEN_CHECKPOINT=$(uv run python -c 'import json; print(json.load(open("runs/restored-lora/latest.json"))["checkpoint"])')
uv run amarken train \
  --prepared data/restored-clinc --output runs/restored-lora \
  --mode lora --epochs 3 --batch-size 16 --candidate-chunk-size 32 \
  --checkpoint-steps 10 --validation-steps 100 \
  --resume "runs/restored-lora/$AMARKEN_CHECKPOINT" \
  --backup-uri s3://YOUR_BUCKET/amarken/clinc-qwen3-v1/lora \
  --wandb-mode offline --wandb-project amarken
```

Match the original training and W&B project/entity settings if you changed them.
Do not resume against the same prefix while the old instance is still writing.
This restores the training stage; the cloud runner's orchestration status is not
restored automatically. Run calibration/evaluation commands after training, or
configure a new orchestration job explicitly. No periodic backup can recover the
steps since its last successful upload; reduce `checkpoint_steps` if that window
is too large. Real bucket authentication/uploads must be checked against your
chosen service before relying on the job.
