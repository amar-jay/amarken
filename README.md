# Amarken

A small typed decision model inspired by Jev, with candidate scoring, supervised
LoRA training, local data preparation, temperature calibration and evaluation.
Full benchmark quality must be established by completed training and held-out reports.

## Backbone

The default is **Qwen/Qwen3-0.6B**, with **Qwen/Qwen2.5-0.5B** as an explicit
fallback, in the order defined by `src/amarken/constants.py`. A load failure is
reported rather than silently changing the model. Both checkpoints are Apache-2.0.
Qwen3-0.6B is post-trained (thinking/non-thinking); Qwen2.5-0.5B is a base checkpoint.

| Model | Pinned revision |
| --- | --- |
| Qwen/Qwen3-0.6B | `c1899de289a04d12100db370d81485cdf75e47ca` |
| Qwen/Qwen2.5-0.5B | `060db6499f32faf8b98477b0a26969ef7d8b9987` |

Model and tokenizer load from the same revision with remote code disabled.
`AutoModel` loads transformer layers without the language-model output head.
The backbone returns `[batch, hidden_size]` features from the last non-padding
token, retaining causal attention and supporting gradients. All parameters remain
trainable; loading defaults to evaluation mode, CPU, and float32. Call `.train()`
when training. `BaseModel` returns features; `DecisionModel` supplies decision logits.

Plain text is tokenized without a chat template or generation. Versioned templates
encode JSON-escaped state, question and one candidate independently. Padding
is masked, generation caching is disabled, and inputs over the configured 2048-token
default raise an error instead of being truncated. This default is a resource cap,
not a claim about the checkpoints' maximum context lengths.

## Run

With Python 3.12 or 3.13 and the project's dependencies installed:

```bash
uv sync --locked
HF_HOME="$PWD/.cache/huggingface" uv run amarken
# Select the fallback deliberately:
HF_HOME="$PWD/.cache/huggingface" uv run amarken --fallback
# Reuse downloaded files without network access:
HF_HOME="$PWD/.cache/huggingface" uv run amarken --local-files-only
```

The first run downloads model weights. `--device cuda --dtype bfloat16` is
available for compatible hardware; the defaults also work without a GPU.
The smoke command prints the model ID, revision, feature shape, dtype, device,
and fails on non-finite features.

```python
from amarken.base import BaseConfig, BaseModel

model = BaseModel.load(BaseConfig())
features = model(
    **model.tokenize(["State: payment failed. Question: Is action needed?"])
)
```

Sources: [Qwen3 card](https://huggingface.co/Qwen/Qwen3-0.6B),
[Qwen2.5 card](https://huggingface.co/Qwen/Qwen2.5-0.5B),
[Transformers model loading](https://huggingface.co/docs/transformers/main/en/models).

## Data

Dataset IDs, revisions, subset names, question text, licenses and decision kinds
live in `constants.py`. `DataConfig` and `load_data` live in `data.py`.
Downloads occur only when the iterator is consumed; importing does not fetch data.

```python
from amarken.constants import TASKSOURCE
from amarken.data import DataConfig, load_data

baseline = load_data()  # CLINC150 OOS+ (plus), including the oos class
mixture = load_data(DataConfig(dataset_id=TASKSOURCE))
example = next(baseline)
validation = load_data(split="validation")
calibration = load_data(split="calibration")
test = load_data(split="test")
```

Every row uses `decision-v1`: `state`, `question`, `kind`, `options`, `target`,
`id`, `group_id`, `source`, `split`, `dataset_id`, `revision`, `license`, and
`license_use`. Targets are distributions aligned with options; `noul` has empty
options and a single probability of true. Scores retain their ordered options.
Malformed supervision raises an error. Training consumes validated local prepared files.

CLINC uses all 151 labels (150 intents plus `oos`) and preserves its official
training and test partitions. The chosen mirror's `plus` variant has 250 OOS
training examples, rather than the original full variant's 100. It is CC BY 3.0.
Tasksource caps do not apply to CLINC; audited overlaps and duplicates are excluded during preparation.

Tasksource uses the pinned `default` text mixture. Training uses seeded buffered
shuffle, up to 100,000 examples, up to 2,000 per source and approximately one third
per decision kind as upper bounds. This can produce fewer than 100,000 rows;
these are caps, not guarantees of balance or representative random sampling.
Only rows marked `commercial` are included by default. This source annotation is
a filter, not independent license verification; inspect original terms before use.
`commercial_only=False` includes other terms explicitly.

Both datasets retain the upstream test split. Upstream validation is divided
approximately 50/50 into validation and calibration using a seeded hash of group
identity. CLINC groups equal normalized text; Tasksource uses its supplied group
IDs. Training subsets never resplit individual variants. Upstream train/test
leakage, cross-source duplicates and held-out-task evaluation still require an
explicit audit before training or claims of generalization. Persist selected IDs
and `DataConfig` with each experiment; do not tune against the test set.

The loader first downloads pinned Parquet files into the local Hub cache, then
iterates local rows. It does not rely on remote range reads. Low-level `load_data`
supports buffered sampling; the preparation command instead sorts by seeded ID
hash before applying caps and writes reproducible local files.

## Decision sampler

`Sampler` in `sampler.py` turns decision-head logits into typed output; it does
not sample training data or generate text. `SamplerConfig` defaults to argmax,
temperature 1 and seed 42. Supported mode strings live in `constants.py`.

```python
from amarken.sampler import Sampler, SamplerConfig

sampler = Sampler()
choice = sampler.sample([2.0, -1.0], kind="choice", options=["route", "review"])
probability = sampler.sample([1.5], kind="noul")  # one true-vs-false logit
score = sampler.sample(
    [-1.0, 2.0, 0.0],
    kind="score",
    options=["low", "medium", "high"],
    score_values=[1, 2, 3],
)
random_sampler = Sampler(SamplerConfig(mode="sample", seed=42))
```

Choice and score return the chosen label, index, full probability distribution,
and `selected_probability`. Score additionally returns its selected numeric anchor
and expected score. Supply strictly increasing numeric anchors explicitly; the
sampler does not infer numbers from tier names. Argmax ties use the first supplied
option. Sampling uses a private seeded RNG and advances its state on each
choice/score call. Noul always returns `probability_true`, without sampling a
boolean. Calls validate finite logits, label uniqueness and scale alignment.

Temperature rescales logits before normalization; it is not automatically fitted
calibration. Returned probabilities are not verified correctness estimates.
Escalation thresholds belong to application policy. The current `BaseModel`
returns hidden features; `DecisionModel` adds the scalar candidate head that supplies
these logits. This sampler is our baseline implementation, not Jev's proprietary
parallel sampler.


## End-to-end pipeline

All model and dataset cache files default to `.cache/huggingface`. The following
commands create new output directories and refuse to overwrite existing artifacts.
Use the same prepared dataset for training and calibration. Preparation checks
normalized duplicate states and source groups across splits, excludes lower-priority
overlaps, records invalid/oversized rows, and saves selected IDs and file checksums.
Near-duplicate semantics and unknown source aliases still need review.

```bash
uv sync --locked
uv run amarken prepare --output data/clinc-v1
# Independent frozen-backbone baseline and LoRA experiment:
uv run amarken train --prepared data/clinc-v1 --output runs/clinc-head --mode head
uv run amarken train --prepared data/clinc-v1 --output runs/clinc-lora --mode lora
```

Full training requires CUDA. For a bounded GPU check use `--smoke --max-steps 1
--batch-size 1`; for a CPU-only check use `--cpu-smoke --max-steps 1`. Smoke runs
limit training/validation examples and must not be used as benchmark results.
The default candidate chunk size is one to accommodate the observed 4 GB GPU.
The effective question batch size is 16; all candidates are scored, including the
151 CLINC labels. Runtime therefore grows with candidate count.

Training defaults: LoRA rank 16, alpha 32, dropout 0.05 on attention q/k/v/o
projections; AdamW with adapter LR 2e-4, head LR 1e-3, weight decay 0.01,
5% warmup, gradient clipping at 1, seed 42. CLINC defaults to three epochs;
Tasksource to one. BF16 is used when supported, otherwise FP32. Parameters outside
the adapters and head are frozen. The two-pass gradient algorithm replays RNG
states so dropout masks match between scoring and recomputation. No candidate
negative subsampling changes the categorical objective.

Each run writes `run.json`, `history.json`, `data-manifest.json`, immutable
`step-NNNNNNN` directories and `best.json` / `latest.json` checkpoint pointers.
Only validation loss selects the best checkpoint. Resume with the original config:

```bash
uv run amarken train --prepared data/clinc-v1 --output runs/clinc-lora \
  --resume runs/clinc-lora/step-0000100
```

Substitute the checkpoint recorded in `best.json` below:

```bash
uv run amarken calibrate --model runs/clinc-lora/step-0001000 \
  --prepared data/clinc-v1 --output outputs/clinc-calibrated --device cuda --dtype bfloat16
uv run amarken evaluate --model outputs/clinc-calibrated --prepared data/clinc-v1 \
  --output outputs/clinc-test --device cuda --dtype bfloat16 --robustness
uv run amarken predict --model outputs/clinc-calibrated --input request.json
```

Calibration fits one bounded positive temperature per kind on calibration data
only. Kinds with fewer than 20 examples or failed fitting retain temperature 1
with `not_fitted` status. A calibrated artifact is a new copy, leaving the original
checkpoint intact. Sampling temperature is an additional explicit multiplier.
Predicted probabilities do not establish safety or production readiness.

Evaluation writes predictions, JSON metrics and Matplotlib reliability plots.
It includes uniform and training-prior baselines, per-kind/source metrics,
accuracy, macro-F1, NLL, Brier score, 10-bin ECE, Wilson accuracy intervals,
score tier error, binary/OOS metrics and coverage/error curves. Accuracy/F1 use
argmax reference labels; soft-target calibration uses target probability mass.
Latency includes warmed end-to-end and model-only p50/p95, first-forward time,
throughput, candidate counts, token lengths and peak GPU memory. Artifact loading
is excluded from forward timing. Use `--limit N` only for explicitly partial checks.
Use `--challenge reviewed.jsonl` for labeled ambiguity/injected-state cases in the
same row schema with an optional `challenge_type`; no resilience claim is inferred
without such a reviewed set. `--robustness` checks candidate permutations.

Request format:

```json
{
  "state": "The customer reports being charged twice.",
  "questions": [
    {"id": "route", "kind": "choice", "question": "Which team should handle this?", "options": ["billing", "technical"]},
    {"id": "urgent", "kind": "noul", "question": "Does the message describe an urgent outage?"},
    {"id": "severity", "kind": "score", "question": "Rate the severity: low, medium, high.", "options": ["low", "medium", "high"], "score_values": [1, 2, 3]}
  ]
}
```

Tasksource preparation requires an explicit license-reviewed source list:

```json
{
  "reviewed_sources": ["exact-source-name-from-dataset"],
  "source_families": {"exact-source-name-from-dataset": "canonical-family-name"}
}
```

```bash
uv run amarken prepare --dataset tasksource/tasksource-jev-typed-decisions \
  --review source-review.json --output data/tasksource-v1
uv run amarken train --prepared data/tasksource-v1 --output runs/tasksource-lora
```

The source names above are placeholders, not approvals. Review original terms and
aliases before filling them in. CLINC-derived sources are excluded; 10% of source
families are deterministically reserved for `unseen_test` using their upstream test
rows. Training and tuning do not see those families. Evaluate this split separately
with `--split unseen_test`; an empty split is reported rather than scored.
Checkpoint packages contain adapter/head weights, tokenizer, schema/template IDs,
base model revision, data/run manifests, calibration and integrity checksums.
The pinned backbone is fetched from cache/Hub when loading an adapter artifact.
Do not distribute `training.pt` unless optimizer/RNG resume state is needed.


## Validation status in this workspace

The pinned Qwen3 checkpoint downloads and runs on the local RTX 3050 (4 GB).
`data/clinc-v1` contains 15,245 training, 1,542 validation, 1,558 calibration and
5,500 test rows after overlap/duplicate exclusions. Prepared files and model
weights remain local and ignored by Git.

Offline tiny-Qwen checks verified exact two-pass gradients with dropout, overlap
exclusion, head-only and LoRA updates, checkpoint reload, resumed-run parity,
calibration and report/plot generation. Real Qwen3/CLINC GPU smoke runs also
completed training, calibration and evaluation. Canonical internal candidate
batching fixed a measured BF16 option-permutation drift; the repeated real check
reported zero logit difference when options were reversed.

Smoke artifacts are under `runs/qwen3-gpu-smoke-chunk8`,
`outputs/qwen3-calibrated-smoke` and `outputs/qwen3-smoke-canonical-evaluation`.
They are implementation checks, **not a trained release or representative accuracy
results**. Choice-only CLINC does not train or calibrate the score/noul tasks.

Eight-candidate chunks completed the observed training question in about 7.6
seconds, versus about 27 seconds with one-candidate chunks. A simple extrapolation
puts three full CLINC epochs near four days before baseline runs and evaluation;
actual runtime depends on question lengths, thermal behavior and batch settings.
Full-duration training and a reviewed Tasksource source mixture remain outstanding.
No multi-day training job has been left running. Use `--candidate-chunk-size 8`
for the measured short-input configuration, or the safer default of one for long
inputs. No production-readiness claim is made from smoke results.


## Cloud GPU setup

See [cloud/README.md](cloud/README.md) for the portable GPU bootstrap, resumable
full training job, persistent storage, source bundle, and Brev launch workflow.
Preview the job with `python3 cloud/run.py --dry-run`; build the upload archive
with `python3 cloud/bundle.py`. No paid instance is provisioned automatically.


## Weights & Biases

W&B tracking is optional and provider-independent. Constants live in
`constants.py`; `WandbConfig` and `WandbTracker` live in `tracking.py`.
Plain training defaults to disabled tracking. To see live charts:

```bash
uv run wandb login
uv run amarken train --prepared data/clinc-v1 --output runs/clinc-wandb \
  --candidate-chunk-size 8 --checkpoint-steps 10 \
  --wandb-mode online --wandb-project amarken --wandb-name clinc-qwen3
```

Use `--wandb-entity TEAM_OR_USER` when selecting a workspace explicitly. Supply
credentials via the W&B login prompt or `WANDB_API_KEY` in the process environment;
never place keys in repository files, job JSON or command arguments. The run URL
and ID are recorded in the run directory's `wandb-run.json`.

Metrics include training loss, gradient norm, both learning rates, step duration,
question throughput, peak allocated CUDA memory and remaining-training-time
estimates. Validation loss and argmax accuracy are recorded at epoch/end-of-run
validation; smoke accuracy is not a representative benchmark. W&B also collects
system utilization metrics. Configuration contains model/dataset revisions,
hyperparameters, counts and fingerprints. Dataset examples, source snapshots,
Git state and console output are not uploaded by this integration.

Offline mode requires no login and stores logs under the run directory:

```bash
uv run amarken train --prepared data/clinc-v1 --output runs/clinc-offline \
  --candidate-chunk-size 8 --wandb-mode offline
# Later, after login, sync the specific directory printed by W&B:
uv run wandb sync runs/clinc-offline/wandb/offline-run-TIMESTAMP-RUN_ID
```

Online resumes reuse the ID recorded in `wandb-run.json`; project/entity must stay
consistent. Pass the same tracking options with `--resume`. Metrics use `train/step`
as a custom x-axis, so checkpoint rollbacks can produce repeated step values
rather than having replayed metrics silently discarded. Offline restarts create
separate local session directories; sync their explicit paths together with
`wandb sync --id RUN_ID PATH1 PATH2` to target the recorded run. Offline mode alone
does not provide a live dashboard. `--wandb-mode disabled` never initializes W&B.

Checkpoint upload is off by default. `--wandb-checkpoints` publishes weights-only
model artifacts with `latest` and `best` aliases: adapters/head, tokenizer, model
metadata, calibration and checksums. Optimizer state, run source snapshots and data
manifests remain local. Downloaded artifacts are compatible with `DecisionModel.load`
and still reference the pinned public backbone. Existing checkpoints from before
tracking was added can be resumed with tracking enabled.

The cloud job's `wandb` section defaults to offline logging. Set its mode to
`online` and optionally set its entity **before starting a new cloud job** to
stream metrics remotely; `head` and `lora` runs share a group and have distinct
names. Changing an existing cloud job config requires a new job name, as before.

## Bounded local learning check

`configs/local-learning.json` defines a one-epoch RTX 3050 experiment: 128 training
and 64 validation examples balanced across 16 intents (including OOS), retaining
all 151 candidates. It uses eight-candidate chunks, four questions per optimizer
step, checkpoints every four steps, and offline W&B. The test set is not sampled.

```bash
uv run python scripts/local_learning.py --config configs/local-learning.json
```

The script compares the same initial and trained model on the selected validation
examples, without temperature fitting. Progress is in
`outputs/local-learning-v1/status.json`; the final comparison is written to
`outputs/local-learning-v1/comparison.json`. Run/checkpoint state is under
`runs/local-learning-v1`. Use a new config name for a repeat experiment; existing
outputs are never overwritten. Validation selection is small and cannot establish
full-dataset performance. The exact executed experiment source is retained locally
alongside its protocol when this run was launched.

## Bucket checkpoint backups

Training supports `--backup-uri s3://BUCKET/unique-run-prefix` and optional
`--backup-endpoint-url URL` for S3-compatible storage. Every completed local
checkpoint is backed up, including optimizer/RNG state. Latest and best pointers
are published after complete uploads. `--validation-steps N` adds periodic best
checkpoint selection; checkpoint frequency is controlled by `--checkpoint-steps`.
Remote backup is disabled until a URI is supplied.

`amarken restore --backup-uri ... --which latest --output runs/restored
--prepared-output data/restored` recovers verified resumable state, its referenced
best checkpoint, exact prepared splits and a source snapshot. Use `--which best`
for the best validated state. See [cloud/README.md](cloud/README.md#off-instance-checkpoint-recovery)
for credentials, provider endpoints, upload failure behavior and resume commands.
