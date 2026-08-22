![amarken](assets/banner.png)
# Amarken
Goal: best measured chat intelligence under `P <= 60M`; secondary objective: intelligence/artifact-bit and intelligence/runtime-byte. No architecture privileged. Unlimited sequential experiments; fixed tests decide.

## Current status (2026-08-22)

The repository now contains a deterministic EN/TR model-tournament harness, three
implemented decoder families (`dt`, `glimmer`, and native-ternary `bit`), shared
training/checkpointing, tokenizer runtime support, and deterministic evaluation.
The first matched 10M/context-512 tournament is complete: Glimmer achieved the
best mean validation loss, but every architecture remained at chance-level
capability on the small proxy benchmark. This is therefore **not** a released
assistant or a model-quality claim.

The selected production tokenizer is `tiktoken-style-apostrophe-bpe-12k`; the
legacy SentencePiece tokenizer remains an experimental control. The proxy-v2
learning curve is paused: although validation loss improved, the OPUS/Python
proxy corpus did not produce the intended concise EN/TR assistant behavior. The
current critical path is qualifying a local teacher and building an auditable,
filtered SFT corpus for concise English/Turkish QA, short reasoning, and tool
calling. Code generation is out of scope. See [REPORT.md](REPORT.md) for
results, provenance, and next gated actions.

## Development

Install the runtime dependencies plus the test dependencies into an activated
virtual environment, then run the repository's configured correctness suite:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

Pytest configuration lives in `pytest.toml`; discovery is deliberately limited
to `tests/test_*.py` so generated corpora and experiment artifacts are excluded.

## Local inference CLI/TUI

The same interface loads exact trainer checkpoints, standalone checkpoints, and
model-only tournament artifacts. CUDA and BF16 are selected automatically when
available; use `--device cpu --precision fp32` for a portable fallback. Pass the
tokenizer used to train the checkpoint; newer checkpoints reject a mismatched
tokenizer fingerprint.

```bash
# One-shot, script-friendly generation.
python -m src.inference.cli \
  --checkpoint MODEL.pt \
  --tokenizer artifacts/tokenizers/v2/tiktoken-style-apostrophe-bpe-12k.json \
  --prompt "Türkiye'nin başkenti" --max-new-tokens 32 --show-info --show-stats

# Interactive terminal session with retained plain-text turns.
python -m src.inference.cli \
  --checkpoint MODEL.pt \
  --tokenizer artifacts/tokenizers/v2/tiktoken-style-apostrophe-bpe-12k.json \
  --chat

# Pipe a prompt and keep stdout suitable for another program.
printf 'Türkiye’nin başkenti nedir?' | python -m src.inference.cli \
  --checkpoint MODEL.pt \
  --tokenizer artifacts/tokenizers/v2/tiktoken-style-apostrophe-bpe-12k.json
```

Interactive commands are `/help`, `/reset`, `/settings`, `/max-new N`,
`/temperature X`, `/top-k N|none`, `/seed N`, and `/quit`. Current retained
weights are experimental base models, not instruction-tuned assistants; `--chat`
only supplies consistent `User:`/`Assistant:` text and cannot create capabilities
the checkpoint has not learned.

## Tokenizer v2: selected artifact and controls

The completed v2 sweep trained compact tokenizers—including base,
apostrophe-aware, and Turkish-weighted tiktoken-regex byte-BPE candidates—on
fixed English, Turkish, and Python slices, then compared them with a
revision-pinned SmolLM2 49k control. It selected the apostrophe-aware 12k
artifact for the current EN/TR assistant objective:

```bash
python -m src.tokenization.v2_sweep --config configs/tokenizer_v2.json
```

Use `--evaluate-only` to regenerate metrics from existing artifacts without
retraining deterministic models. The report includes realized
training-token shares, EN/TR fertility, code token density, whitespace and byte
behavior, Turkish morphology fragmentation, exact round trips, indentation
overhead, vocabulary/embedding cost, artifact hashes, a provisional metric-only
choice, and candidates promoted to a downstream control-model probe. Tokenizer
metrics alone never authorize an architecture tournament.

Visualize deterministic random dataset samples with colored token boundaries:

```bash
python -m src.tokenization.visualize --samples 3 --language tr --legend

# Compare candidates on the exact same random documents.
python -m src.tokenization.visualize \
  --tokenizer tiktoken=artifacts/tokenizers/v2/tiktoken-style-tr-weighted-bpe-12k.json \
  --tokenizer sentencepiece=artifacts/tokenizers/v2/sp-bpe-12k.model \
  --language tr --samples 5
```

Whitespace is rendered visibly (`·`, `→`, `↵`), repeated `--tokenizer` arguments
share the same sampled documents, and `--no-color` emits bracketed boundaries for
logs or redirected output. Sampling uses a fixed-seed streaming reservoir, so it
does not load the 89MB JSONL dataset into memory.

## Invariants

- Decoder LM unless an alternative wins.
- Count embeddings, heads, shared parameters once; report total/active parameters, FLOPs/token, model bytes, KV bytes.
- Custom EN/TR tokenizer sweep: `8k, 12k, 16k`; byte fallback; tied input/output embeddings.
- Final selection uses hidden, contamination-checked EN/TR chat tests: instruction following, compositional reasoning, state tracking, grounded QA, tool routing, calibration, fluency; plus latency/RAM/energy.
- Never select on teacher/judge score alone. Deterministic tests > blinded humans > multiple independent judges.

## Architecture tournament

The completed first generation matched tokenizer, tokens, optimizer-compute,
data, context, and evaluation across DT, Glimmer, and two Bit scaling variants.
It established an optimization-loss result only; architecture promotion is
deferred until the replacement data program demonstrates capability. Current
implemented and planned candidates are:

- **Amarken-DT**: deep-thin RMSNorm/RoPE/SwiGLU/GQA Transformer; baseline near `18L × 512d × 1344ff × 8Q/2KV`, ~57M.
- **Amarken-Glimmer**: repeat `(Local-RoPE, Local-RoPE, Local-RoPE, Global-NoPE)`; local-window sweep `256/512/1024`; gated-GQA sweep `8:1/16:1`; per-head RMS QK-norm + learned query scale. Hypothesis: local layers learn order/composition cheaply; periodic position-free global layers carry long-range content without RoPE distance distortion.
- **Amarken-Bit**: natively ternary weights; FP activations/norms; compare with post-trained INT4 DT winner.

The completed 10M/context-512 sweep used three seeds and approximately one
million tokens per arm. Earlier 10M/25M/60M short runs are retained as optimizer
preflights, not model rankings. Future architecture sweeps resume only after a
DT control proves capability gains with the new data program.

Muse Glimmer provenance: released decoder = 52 layers of `(SWA-RoPE ×3, Full-NoPE ×1)`, 32Q/2KV gated GQA, per-head RMS QK-norm, extra query scaling. Transfer only these testable components. Optional DFlash improves decoding speed, not student intelligence, and counts against artifact parameters/bytes if tested. Release states Spark→Glimmer distillation but publishes no reproducible method; it supplies no Amarken distillation recipe. Sources: [model card](https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/main/README.md), [architecture](https://huggingface.co/blog/muse-glimmer#architecture).

## Distillation: executable definition

This section records the target data/training policy; bulk distillation has not
started. The implemented next gate is a frozen, resumable 200-case local-teacher
qualification suite for concise EN/TR QA, short reasoning, tool routing, and
post-tool answers. It deliberately excludes code-generation tasks:

```bash
# Rebuild the frozen suite if required.
python -m src.distillation.ollama_qualification build

# Qualify a local Ollama teacher (the default is qwen3.5:2b-q4_K_M).
python -m src.distillation.ollama_qualification run
```

Qualification is pending a GPU-capable Ollama runtime; do not generate a bulk
training corpus until its results and filter gates are reviewed.

### Teachers

- `T0`: strongest available general teacher.
- `T1`: open-weight 1B–3B bridge teacher.
- `S`: Amarken student.
- If vocabularies differ: sequence/on-policy distillation only.
- If vocabulary is shared: sequence + sparse-logit distillation.

### Dataset construction

For each task specification `x`:

1. Generate ground truth `g(x)` by program/database/source text where possible.
2. Sample `K` T0 answers under varied seeds/styles.
3. Verify mechanically; reject incorrect/unfaithful outputs.
4. Normalize to short student-capacity answers; retain answer, tool action, cited evidence, difficulty, provenance.
5. Deduplicate train/test semantically and lexically.
6. Train T1 on accepted traces; use T1 to generate simpler boundary examples.

Data units: `(prompt, evidence?, action?, answer, verifier, difficulty)`. No unverifiable chain-of-thought target; distill concise answer plus optional explicit intermediate state required by the task.

### Losses

Base causal loss, assistant tokens only:

`L_CE = -sum_t log p_S(y_t | x,y_<t)`

Shared-vocabulary sparse logits: store teacher top-`k` (`k=32/64`) logits and residual mass per token:

`L_KD = tau^2 * KL(q_T^tau || p_S^tau)`

Reverse-KL branch:

`L_RKL = tau^2 * KL(p_S^tau || q_T^tau)`

Representation branch, selected layers only:

`L_H = sum_l || normalize(P_l h_S^l) - normalize(h_T^m(l)) ||_2^2`

Total sweep:

`L = a*L_CE + b*L_KD/RKL + c*L_H + d*L_tool`

Tune `a,b,c,d,tau,k`; CE-only is mandatory baseline. Hidden-state loss survives only by ablation.

### On-policy distillation

Avoid pure teacher forcing/exposure mismatch:

1. Sample prefix/completion from current `S`.
2. Query T1/T0 on the same student prefix.
3. Obtain teacher correction, preference, or shared-vocab top-k distribution.
4. Train on states actually visited by `S`.
5. Increase teacher-target interpolation as S improves: `q_i=(1-r_i)p_S+r_i p_T`, `r_i: 0→1` (adaptive schedule chosen by validation).

### Curriculum

`language → atomic skills → composed skills → chat → tools/RAG → adversarial recovery`.

Maintain a difficulty ladder per skill. Sample near S's frontier (`success ≈ 30–70%`). Generate harder variants after mastery; decompose always-failed tasks. Replay earlier stages to prevent forgetting.

### Preference compression

Generate pairs from S, not only T. Deterministically filter where possible; otherwise blinded multi-judge/human ranking. Train DPO/ORPO branch against frozen SFT reference. Reject if capability/calibration regresses.

## Training

- Scratch pretraining; AdamW baseline; BF16/FP16; gradient checkpointing; packed sequences; deterministic resumable shards.
- Token sweep: `0.5B, 1B, 3B, 6B+`; stop only on held-out capability saturation, not epoch count.
- Mixture sampling driven by per-skill validation deficits, capped to prevent benchmark overfitting.
- Checkpoints retain config, tokenizer hash, data manifest/hash, code revision, RNG state and full metrics.
- Every claimed gain requires matched-compute ablation and confidence interval.

## Compression tournament

From best FP model:

- INT8 baseline.
- Weight-only INT4.
- GPTQ/AWQ-style post-training variants.
- Quantization-aware training.
- Mixed-bit allocation by per-layer sensitivity under a fixed byte budget.
- Native ternary Amarken-Bit trained from scratch.
- Optional structured pruning followed by recovery distillation.

Score deployed artifact, including weights/scales/tokenizer; separately report runtime/KV memory. Quantized model must rerun the entire hidden suite.

## System boundary

Weights learn language, routing, extraction, short composition and calibration. Calculator/search/retrieval/database own exact computation and mutable facts. Tool calls use a minimal grammar with constrained decoding. RAG provides few reranked evidence sentences, never document dumps.

## Selection

Maintain Pareto frontier over:

`Q = hidden capability`, `B = artifact bits`, `M = peak runtime bytes`, `F = FLOPs/token`, `L = latency`, `E = energy/token`.

Primary winner: maximum `Q` subject to `P<=60M` and deployment constraints. Efficiency winners: maximize `Q/B`, `Q/M`, `Q/F`. Keep distinct winners if objectives disagree; never collapse them into one opaque score.
