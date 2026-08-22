# Training preflight

No proxy or full training run should start until the shared correctness suite
passes for every candidate:

```bash
python -m src.training.correctness \
  --device cpu \
  --output experiments/correctness_gates.json
```

The suite uses identical tiny configs and a four-sequence synthetic corpus. It
checks substantial loss reduction, exact memorized continuation, causal future
invariance, left/all-padding safety, exact model checkpoint round-trip, and
bit-identical interrupted/resumed optimization with stochastic batch selection.
It also reports analytical model statistics and measured training throughput.

This is a blocking implementation gate, not a capability benchmark. Passing only
shows that a model can learn and that the surrounding experiment machinery is
coherent. Tokenizer/data/evaluation gates remain separate.

## Shared trainer

`Trainer` is the sole optimizer loop for Glimmer and Bit. Callers provide
`TokenizedExample` records with an explicit assistant-target mask, pack them, and
then pass the same dataset and trainer configuration to either registered model:

```python
from pathlib import Path
from src.models import create_model, load_config
from src.training import PackedSequenceDataset, TokenizedExample, Trainer, TrainerConfig

examples = [TokenizedExample((1, 20, 21, 2), (False, False, True, True))]
dataset = PackedSequenceDataset(examples, sequence_length=2048, eos_token_id=2, pad_token_id=3)
model = create_model(load_config("configs/model.json"))
trainer = Trainer(model, dataset, TrainerConfig(output_dir=Path("runs/proxy")), device="cuda")
trainer.train(1_000)
```

Packing masks padding, inserted EOS targets, every record's first token, every
block's first token, and all non-assistant tokens with label `-100`. Documents
may share a causal context block for efficiency, but no loss crosses a document
or block boundary. Validation should use separately constructed validation
blocks; the trainer never mixes dataset objects or invents a split.

The loop uses AdamW, token-weighted gradient accumulation, unified `torch.amp`,
global-norm clipping, and optional non-reentrant whole-model activation
checkpointing with RNG preservation. Glimmer separates decayed matrices from
norm/embedding parameters. Bit additionally isolates every `BitLinear` FP master
in a named group with independent learning-rate multiplier and decay settings.

Atomic checkpoints are written only after completed optimizer updates. They
contain model weights/config, optimizer groups and moments, AMP scaler, data
epoch/block cursor, dataset fingerprint, counters, and Python/CPU/CUDA RNG state.
Loading rejects model, optimizer-policy, or dataset mismatches. `metrics.jsonl`
records loss, per-group learning rates, token counts, timing, pre-clip gradient
health, and AMP scale. Bit also records trit sign/zero fractions, the distribution
of layer absmean scales, and ternary-master gradient finite/zero fractions. These
statistics diagnose quantizer collapse; they are not quality objectives.

## 10M proxy tournament

Run the first matched control experiment with:

```bash
python -m src.training.proxy_experiment \
  --config configs/proxy_10m.json \
  --report experiments/proxy_10m.json
```

The configuration fixes the 12k tokenizer, proxy-v1 train/validation hashes,
64-token context, BF16, AdamW hyperparameters, batch/accumulation schedule,
optimizer updates and initialization/shuffle seed. DT and Bit use ten 256-wide
layers; Glimmer uses nine because its attention gate adds a fifth projection.
This yields 9.57–9.64M parameters and keeps analytical FLOPs/token within 1.1%.
Bit's named ternary group receives the same LR and weight decay in this control.

The runner evaluates the same first 32 validation blocks before and after
training, writes per-step metrics and exact-resume checkpoints, and records all
configuration, tokenizer, data, checkpoint and software hashes. The default
4,096 consumed tokens per arm are enough to catch optimization failures, not to
rank language capability or justify architecture promotion.

## Three-seed scaling preflight

`configs/proxy_scaling.json` repeats the tokenizer, proxy data, 64-token context,
32-update/4,096-token schedule, AdamW settings and BF16 precision at approximately
25M and 60M parameters for seeds 2026–2028. Within each scale, parameter and
analytical FLOP spreads remain below 2%. Architecture order rotates by seed to
reduce systematic cold-start and thermal bias.

CUDA runs require host GPU access and deterministic cuBLAS workspace setup. The
runner sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing Torch, then enables
hard deterministic algorithms. Reproducibility is scoped to the recorded Torch,
CUDA, driver, device and platform; it is not promised across platforms/releases.

```bash
python -m src.training.scaling_experiment \
  --config configs/proxy_scaling.json \
  --report experiments/proxy_scaling.json
```

Every run retains per-step JSONL metrics. To control disk use, only seed 2026
retains model-only weights at each scale/architecture. These are evaluation
artifacts and cannot exactly resume AdamW because optimizer moments are omitted.
Exact trainer resume remains a blocking gate. No capability promotion may be
based on this short optimization preflight.
