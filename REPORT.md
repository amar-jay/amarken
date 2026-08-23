# Amarken Project Report

Status date: 2026-08-22 (Europe/Istanbul)

This report records the project chronologically. Completed stages are backed by repository commits, configurations, and experiment artifacts. The original context-512 tournament has completed. A new apostrophe-BPE DT learning curve through 104,857,600 nominal tokens is active; only milestones with retained checkpoints and evaluation artifacts are reported as complete.

## Objective and experimental policy

Amarken aims to maximize measured English/Turkish chat intelligence under a hard limit of 60 million parameters. Secondary objectives include intelligence per artifact bit and intelligence per runtime byte. No architecture is privileged: matched data, tokenizer, token exposure, optimizer compute, context, seeds, and deterministic evaluations decide promotion.

The active architecture set is:

- DT: a full-precision deep-thin RMSNorm/RoPE/SwiGLU/GQA Transformer control.
- Glimmer: local RoPE attention interleaved with periodic position-free global attention, gated GQA, QK normalization, and learned query scaling.
- Bit: a native ternary-weight Transformer with floating-point master weights, activations, embeddings, and normalization.

## Timeline

### 2026-08-21 — Project initialization

The repository was initialized with the sub-60M objective, model-tournament policy, distillation policy, compression tournament, and Pareto selection criteria.

Commit: `17e0bba` (`init`)

### 2026-08-21 — Initial Glimmer implementation

A compact Glimmer decoder was implemented as the first experimental architecture.

Commit: `5ec4a33` (`mini glimmer model`)

### 2026-08-21 — Native ternary Bit implementation

A BitNet-inspired model was added. Its learned projections retain floating-point master weights during training but use absmean-scaled ternary effective weights through a straight-through estimator. Packed ternary accounting and export diagnostics were also introduced.

Commit: `dcdf84d` (`bit model - modelled after bitnet`)

### 2026-08-21 — Common model contract

DT/Glimmer/Bit-compatible infrastructure began with a common causal-LM contract covering:

- Identical `input_ids`, `attention_mask`, and `labels` behavior.
- A shared causal-LM output type.
- Parameter, FLOP, artifact, and KV-cache accounting.
- Configuration serialization and registry-based construction.
- Generation.
- Atomic model checkpoints and optimizer/RNG restoration.

Commit: `0bc5a64` (`common model`)

### 2026-08-21 — Blocking correctness gates

The first executable pre-training gates were implemented and run:

- Tiny-corpus overfitting and exact short-sequence memorization.
- Causal invariance under future-token mutation.
- Padding invariance and all-padding numerical safety.
- Checkpoint round-trip.
- Optimizer and RNG restoration.
- Exact interrupted-versus-uninterrupted resume.
- Short autoregressive generation.

Both active models passed the original suite. DT was later integrated into the same gates.

Artifacts: `experiments/correctness_gates.json`

Commit: `565d9a8` (`Implemented and ran the blocking correctness suite`)

### 2026-08-21 — EN/TR tokenizer tournament

Matched BPE tokenizers were trained on OPUS-100 English/Turkish data with byte fallback, fixed special tokens, deterministic ordering, and a single trainer thread.

The sweep measured:

- English and Turkish fertility.
- Byte-fallback rate.
- Unknown-token count.
- Exact round-trip correctness.
- Turkish morphology fragmentation.
- Unicode and whitespace stress behavior.
- Model and vocabulary artifact bytes.
- Corpus and tokenizer hashes.

All three candidates passed exact vocabulary-size, unknown-token, and round-trip gates. The 12k tokenizer was selected for subsequent matched experiments.

Artifacts: `artifacts/tokenizers/`, `experiments/tokenizer_sweep.json`

Commit: `73d6545` (`Implemented and completed the EN/TR tokenizer sweep`)

### 2026-08-22 — Tokenizer v2 visual audit and final selection

The original tokenizer selection was reopened after inference and
token-boundary visualization showed that aggregate fertility did not adequately
measure boundary quality. The v2 sweep compared byte-level BPE, tiktoken-style
regex BPE, a Turkish-apostrophe-aware tiktoken variant, a Turkish-weighted
tiktoken variant, byte-level BPE controls, and the
SmolLM2 tokenizer as an external reference. All compact finalists used a 12k
vocabulary except the explicit 16k byte-BPE capacity control.

The final tokenizer for the Amarken EN/TR/code use case is
`tiktoken-style-apostrophe-bpe-12k`. It provides the best overall balance:

- English fertility is effectively tied with byte-BPE 12k: 1.551 versus 1.544
  tokens per word.
- Turkish fertility is also effectively tied: 1.966 tokens per word for both.
- Code fertility is materially better: 2.895 versus 3.188 tokens per word,
  approximately 9.2% fewer tokens.
- Its pre-tokenization prevents cross-word pieces while retaining useful code
  units such as `_config`, `.get`, and `_time`.
- The Turkish apostrophe branch keeps productive proper-noun suffixes such as
  `'dan`, `'da`, and `'nin` coherent without treating ordinary agglutinative
  suffixation as apostrophe-dependent.
- It uses the compact 12k embedding vocabulary and passes unknown-token,
  byte-fallback, indentation, and exact round-trip gates.

The settled ranking is:

1. **Tiktoken apostrophe 12k — selected production tokenizer.** Best overall
   balance across English, Turkish, code, structural boundaries, and embedding
   cost.
2. **Tiktoken TR-weighted 12k — retained Turkish-first option.** It achieves
   the best Turkish fertility (1.895 tokens/word) and morphology result (3.60
   mean tokens/form), but gives back English efficiency and does not use the
   apostrophe-specific pre-tokenization branch.
3. **Byte-BPE 12k — retained structural control.** It has highly regular,
   conservative boundaries and zero cross-word pieces, but fragments code more
   heavily than both tiktoken variants.
The 16k byte-BPE candidate was not selected because its modest fertility gain
does not justify 4,000 additional embedding rows under the fixed model-parameter
budget. Visual inspection is treated here as a diagnostic backed by explicit
structural-boundary measurements, not as a substitute for downstream language-
model evaluation.

Artifacts: `artifacts/tokenizers/v2/`, `experiments/tokenizer_v2.json`

### 2026-08-21 — Proxy dataset v1

A deterministic, non-distilled English/Turkish/code proxy corpus was built with:

- NFKC normalization.
- Exact and SimHash near-duplicate removal.
- Deterministic sampling and group-based train/validation splitting.
- Document-level source, locator, language, domain, group, and SHA-256 provenance.
- Exact and token-window contamination checks.
- Quarantine of contaminated records.

Materialized results:

| Measure | Count |
|---|---:|
| Sampled documents | 200,808 |
| Deduplicated documents | 181,961 |
| Clean documents | 181,959 |
| Contaminated documents | 2 |
| Exact duplicates removed | 17,824 |
| Near duplicates removed | 1,023 |
| Training documents | 180,156 |
| Validation documents | 1,803 |

No distillation data was used.

Commit: `8881417` (`Built and materialized proxy-v1, without distillation`)

### 2026-08-22 — Shared trainer

A shared exact-resumable trainer was implemented with:

- Fixed-length packed sequences.
- Explicit assistant/source-token loss masks.
- AdamW and model-specific optimizer groups.
- BF16/FP16 autocast and FP16 gradient scaling.
- Token-weighted gradient accumulation.
- Gradient checkpointing and clipping.
- Deterministic epoch/block order.
- Atomic checkpoints containing model, optimizer, scaler, cursor, dataset fingerprint, and Python/CPU/CUDA RNG state.
- JSONL metrics.
- General gradient-health diagnostics.
- Bit trit sign/zero fractions, scale distribution, and ternary-master gradient health.

Commit: `cfbe6aa` (`Implemented the shared trainer for Glimmer and Bit`)

### 2026-08-22 — First matched approximately 10M experiment

A full-precision DT control was added and compared with approximately 10M-parameter Glimmer and Bit models. The first run held tokenizer, proxy data, context, seed, AdamW settings, precision, batch size, accumulation, update count, and validation procedure constant.

| Model | Parameters | FLOPs/token | Initial validation loss | Final validation loss | Train tok/s |
|---|---:|---:|---:|---:|---:|
| DT | 9,630,976 | 19.584M | 9.393 | 7.671 | 76.2 |
| Glimmer | 9,569,572 | 19.369M | 9.354 | 7.879 | 72.4 |
| Bit | 9,639,936 | 19.584M | 9.435 | 7.734 | 68.9 |

Each model consumed only 4,096 tokens over 32 optimizer updates. This established optimization health, not model quality. Bit remained numerically healthy with approximately 31% zero trits and fully finite ternary gradients.

Artifact: `experiments/proxy_10m.json`

Commit: `0aba26b` (`Completed the first matched ~10M proxy experiment`)

### 2026-08-22 — Deterministic bilingual capability evaluation

The first deterministic EN/TR capability and systems evaluation covered instruction following, compositional reasoning, retrieval, state tracking, tool syntax, choice NLL, calibration, latency, throughput, RSS, KV memory, and artifact bytes.

All models scored 7/30 (23.3%) after only 4,096 training tokens, consistent with four-way random chance (25%). The evaluation therefore correctly produced no capability winner.

| Model | Full validation loss | EN accuracy | TR accuracy | Overall | Choice NLL | ECE |
|---|---:|---:|---:|---:|---:|---:|
| DT | 6.852 | 20.0% | 26.7% | 23.3% | 1.530 | 0.168 |
| Glimmer | 7.001 | 20.0% | 26.7% | 23.3% | 1.525 | 0.226 |
| Bit | 6.882 | 26.7% | 20.0% | 23.3% | 1.416 | 0.114 |

Systems measurements showed DT as the fastest reference implementation, Glimmer with the smallest KV cache, and Bit with the smallest artifact. Bit's ordinary PyTorch fake-quantization path was correctly identified as unrepresentative of packed ternary inference.

Additional gates found zero 13-token overlaps across 180,156 training records, balanced answer positions, complete validation coverage, and exact reproduction of losses, predictions, probabilities, and calibration on an independent rerun.

Artifact: `experiments/capability_10m.json`

Commit: `83f582e` (`Implemented and ran the deterministic bilingual capability evaluation`)

### 2026-08-22 — Three-seed 25M/60M scaling preflight

The full parameter-scaling matrix ran on an RTX 3050 Laptop GPU using seeds 2026, 2027, and 2028. Every arm used the same 12k tokenizer, proxy stream, 64-token context, 32 optimizer updates, 4,096 consumed tokens, BF16 precision, matched AdamW settings, deterministic kernels, and rotated architecture order.

| Scale | Model | Parameters | Validation loss mean ± σ | Improvement | Throughput |
|---|---|---:|---:|---:|---:|
| 25M | DT | 24,426,624 | 7.2366 ± 0.0266 | 2.2407 | 1,156 tok/s |
| 25M | Glimmer | 24,680,904 | 7.5014 ± 0.0280 | 1.8513 | 957 tok/s |
| 25M | Bit | 24,444,928 | 7.3154 ± 0.0837 | 2.1634 | 716 tok/s |
| 60M | DT | 58,657,280 | 7.0984 ± 0.0408 | 2.3600 | 661 tok/s |
| 60M | Glimmer | 58,675,856 | 7.3152 ± 0.0190 | 2.0444 | 510 tok/s |
| 60M | Bit | 58,692,992 | 7.1154 ± 0.0364 | 2.3706 | 390 tok/s |

DT won all three 25M seeds and two of three 60M seeds. Bit narrowed its mean loss gap to DT from 0.079 at 25M to 0.017 at 60M. Peak CUDA allocation remained below approximately 1.14 GiB. A repeated 25M DT run reproduced its validation loss exactly.

These runs remained optimization preflights because each model saw only 4,096 tokens.

Artifact: `experiments/proxy_scaling.json`

Commit: `258ebdf` (`Completed the full scaling matrix on the RTX 3050 Laptop GPU`)

### 2026-08-22 — Context-512 learning-rate screen

Before longer training, 10M DT, Glimmer, tensor-scale Bit, and output-channel-scale Bit variants were screened independently at context 512. Each LR candidate consumed 65,536 tokens.

Selected learning rates:

| Variant | Selected LR | Screen validation loss |
|---|---:|---:|
| DT | 0.0010 | 5.1693 |
| Glimmer | 0.0015 | 4.9555 |
| Bit tensor-scale | 0.0020 | 5.2168 |
| Bit channel-scale | 0.0015 | 5.2659 |

The screen changed the earlier interpretation: with a suitable learning rate and longer context, Glimmer became the strongest early optimizer. It also established tensor- and output-channel-scale Bit variants as explicit tournament arms.

Artifact: `experiments/lr_screen_10m.json`

### 2026-08-22 — Context-512, one-million-token tournament (active)

The corrected tournament uses:

- Approximately 10M parameters per model.
- Context length 512.
- Seeds 2026, 2027, and 2028.
- Four 512-token microbatches per update.
- 512 optimizer updates.
- Exactly 1,048,576 consumed tokens per arm and seed.
- A larger non-repeating packed training pool.
- BF16, gradient checkpointing, cosine scheduling, warmup, deterministic CUDA execution, and rotated architecture order.
- DT, Glimmer, tensor-scale Bit, and output-channel-scale Bit arms.

The final report `experiments/proxy_tournament_10m_context512.json` had not yet been emitted when this timeline was written.

#### Provisional completed results

| Seed | Variant | Final validation loss | Approx. throughput | Status |
|---:|---|---:|---:|---|
| 2026 | DT | 3.5243 | 11.5k tok/s | Complete |
| 2026 | Glimmer | 3.0483 | 8.9k tok/s | Complete |
| 2026 | Bit tensor-scale | 3.6415 | 5.7k tok/s | Complete |
| 2026 | Bit channel-scale | 3.5992 | — | Complete |
| 2027 | Glimmer | 3.0670 | — | Complete |
| 2027 | Bit tensor-scale | 3.6261 | — | Complete |
| 2027 | Bit channel-scale | 3.6526 | — | Complete |
| 2027 | DT | — | — | Running at last update |

Provisional observations:

- Glimmer leads seed 2026 by 0.4760 nats over DT and is stable across its first two seeds.
- DT remains the faster full-precision control.
- Bit tensor-scale is stable across its first two seeds.
- Bit channel scaling reversed direction across seeds: it improved over tensor scaling by 0.0423 in seed 2026 but trailed by 0.0265 in seed 2027. No scale-granularity winner has therefore been established.
- Bit's slower reference training path is expected because floating-point master weights are fake-quantized during every forward. It is not a packed ternary inference benchmark.
- No arm encountered memory, determinism, or finite-gradient failures in the supplied updates.

These results must remain provisional until all 12 arms complete and the final hashed report is written.

### 2026-08-22 — Context-512 tournament completion and capability diagnosis

All 12 context-512 arms subsequently completed and the final report passed its
paired-seed and trajectory gates. Mean final validation loss across three seeds
was 3.0523 for Glimmer, 3.5187 for DT, 3.6465 for tensor-scale Bit, and 3.6631
for channel-scale Bit. Glimmer therefore retained a clear optimization-loss
advantage, while DT remained the throughput leader.

The retained-checkpoint capability evaluation did not convert this loss
ordering into meaningful task capability. DT scored 23.3% and Glimmer and both
Bit variants scored 26.7% on the 30-item four-choice benchmark. These values
remain consistent with chance and do not authorize architecture promotion.
The result motivated reopening the tokenizer/data boundary and establishing a
meaningful training-exposure curve before another architecture tournament.

Artifacts: `experiments/proxy_tournament_10m_context512.json`,
`experiments/capability_tournament_10m_context512.json`

### 2026-08-22 — Unified tokenizer API and clean proxy-v2 freeze

A common tokenizer contract was added for JSON artifacts. Training, learning-rate screening,
scaling, evaluation, and inference now use the same encode/decode, vocabulary,
special-token, artifact-byte, and fingerprint behavior. New checkpoints bind
the tokenizer fingerprint, and inference rejects a same-size but incorrect
tokenizer.

The repaired corpus was frozen as `proxy-v2-clean` using the historical sampling
seed so the tokenizer comparison would not introduce a new corpus-selection
confound. Its manifest records:

| Measure | Count |
|---|---:|
| Clean documents | 181,962 |
| Contaminated documents quarantined | 11 |
| Encoding-corrupted lines rejected | 125 |
| Encoding repairs accepted | 1,216 |
| Exact duplicates removed | 17,829 |
| Near duplicates removed | 1,025 |
| Training documents | 180,160 |
| Validation documents | 1,802 |

All output files, source inputs, cleaning policies, repair counts, contamination
references, and train/validation splits are hash-bound in
`data/processed/proxy-v2-clean/manifest.json`.

### 2026-08-22 — Downstream tokenizer probe

A matched 9,630,976-parameter DT probe compared the structurally qualified
tokenizer controls on clean proxy-v2 data. The
65,536-token CPU run was single-seed directional evidence, not a promotion-grade
experiment.

| Tokenizer | Validation loss/token | Estimated bits/source byte | Training source bytes covered |
|---|---:|---:|---:|
| Tiktoken apostrophe BPE 12k | 7.4061 | 2.7664 | 231,736 |
| Tiktoken TR-weighted BPE 12k | 7.4498 | 2.8150 | 229,123 |
| Byte-BPE 12k | 6.9448 | 2.7884 | 221,903 |

Loss per tokenizer token can be misleading because candidates cover different
amounts of source text at a matched budget. `tiktoken-style-apostrophe-bpe-12k` is deliberately
fixed for the EN/TR/code use case because its word, Turkish proper-noun suffix,
and code-boundary behavior matches the deployment objective.

Artifact: `experiments/tokenizer_probe_dt_65k_cpu.json`

### 2026-08-22 — Meaningful-scale evaluation suite

The original 30-item benchmark was retained as a regression suite. A larger
deterministic development benchmark was added with:

- 240 English/Turkish multiple-choice tasks, with 24 cases per language and
  category across instruction following, compositional reasoning, retrieval,
  state tracking, and tool syntax.
- Exact balance across the four authored answer positions.
- Four cyclic answer-order evaluations per task, producing 960 scored decisions
  and explicit content-invariance measurements.
- 120 greedy generative exact-match tasks covering EN/TR arithmetic, exact copy,
  and Python completion.
- Wilson 95% confidence intervals by language and category.
- 68 corpus-derived held-out continuation probes: 30 English, 30 Turkish, and
  all eight available validation-code documents.
- Zero detected 13-token overlaps between the authored task prompts and the
  clean-v2 training split.

This is a frozen development suite, not a secret final test. Final capability
claims still require a separately held suite whose concrete items and seed were
never exposed during model, data-mixture, or hyperparameter selection.

Artifacts: `benchmarks/meaningful_scale_v2.json`,
`src/evaluation/meaningful_scale.py`

### 2026-08-22 — Apostrophe-BPE meaningful-scale DT learning curve (stopped)

One continuous 9.63M-parameter DT control trajectory is running with the fixed
apostrophe-BPE tokenizer, clean proxy-v2 data, context 512, BF16, four
microbatches per update, cosine scheduling, and exact-resume milestone
checkpoints. The milestones are 1,048,576; 8,388,608; 33,554,432; and
104,857,600 nominal tokens. The fixed packed training pool contains 4,194,304
tokens, so the later milestones are explicitly multi-epoch exposure rather than
non-repeating data.

Completed milestone results:

| Milestone | Full validation loss | v1 accuracy | v2 MC accuracy | Generative exact match | Overall LM perplexity | EN PPL | TR PPL | Code PPL |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,048,576 | 5.7413 | 20.0% | 25.4% | 0.0% | 540.2 | 471.0 | 1,556.3 | 141.6 |
| 8,388,608 | 4.4921 | 23.3% | 24.2% | 0.0% | 181.0 | 213.3 | 422.0 | 43.3 |

The 1M-to-8M interval shows strong held-out language-model learning without
measurable capability emergence: both multiple-choice suites remain consistent
with chance and greedy exact match remains zero. This directly supports the need
for the 32M and 100M observations before selecting a meaningful architecture-
tournament exposure. Gradient values remain finite, although clipping has
occurred on most updates; this is retained as a post-trajectory ablation target
and will not be changed mid-protocol.

The trajectory later passed approximately 21.7M realized tokens but did not
reach the 32M checkpoint. It is no longer running. The user-facing generations,
zero exact-match score, and chance-level choice behavior were sufficient to
reject the current corpus as a route to the intended assistant, regardless of
whether continued repetition might further reduce validation loss. The 1M and
8M checkpoints remain useful controls; resuming toward 32M/100M is paused until
the training-data program changes.

Configuration: `configs/learning_curve_dt_apostrophe_100m.json`

Milestone artifacts:
`runs/dt-apostrophe-bpe-meaningful-scale-100m-v1/`

### 2026-08-22 — Data-program pivot and local teacher qualification

The product objective is now explicitly limited to concise general English and
Turkish question answering, short reasoning when necessary, and native tool
calling. Code generation is out of scope. The current proxy-v2 mixture—small,
fragmentary OPUS-derived text plus Python standard-library material—is therefore
not suitable as the primary capability corpus.

The next data program will use a qualified local Ollama teacher to produce
structured supervised conversations, not copy arbitrary free-running model
output. The proposed teacher is `qwen3.5:2b-q4_K_M`, run with thinking disabled,
a 2,048-token context, deterministic decoding for qualification, and a strict
concise bilingual system contract. Generated records must preserve messages,
tool schemas, tool calls, tool results, provenance, generator settings, and
automatic validation decisions. Training examples will be filtered for language,
answerability, concision, tool necessity, schema validity, and contamination.

A frozen 200-case teacher qualification suite now covers:

- 50 concise English general-QA cases;
- 50 concise Turkish general-QA cases;
- 40 exact short-reasoning cases, balanced across EN/TR;
- 40 tool-routing cases, including unnecessary-tool checks;
- 20 post-tool-result answer cases.

The runner is deterministic, resumable JSONL, and scores correctness, verbosity,
tool choice, argument accuracy, and unnecessary tool use. A five-case pilot
showed that a system-level concision contract is necessary: without it, correct
answers routinely expanded into unwanted explanations. The contracted pilot
produced concise correct answers.

The incomplete user-local Ollama installation was repaired with the official
bundle. Ollama 0.32.15 now detects the RTX 3050 as CUDA compute 8.6, selects its
CUDA 13 runner, and loads the 2B model in approximately 1.4 GiB VRAM at context
2,048. The frozen qualification then produced:

| Slice | Passed | Rate |
|---|---:|---:|
| Concise English QA | 49/50 | 98% |
| Concise Turkish QA | 34/50 | 68% |
| English short reasoning | 1/20 | 5% |
| Turkish short reasoning | 0/20 | 0% |
| Tool routing / unnecessary-tool checks | 38/40 | 95% |
| Post-tool-result answers | 20/20 | 100% |
| Overall | 142/200 | 71% |

The conclusion is not to bulk-generate unconstrained answers. The model is a
good candidate for concise English and tool-use surface realization, but it is
not a trustworthy sole source for Turkish facts or reasoning labels. A low-
thinking follow-up correctly computed a sample internally but repeatedly spent
the entire 512-token budget restating formatting checks and emitted no final
answer; it does not repair the production behavior. The data generator must
therefore be grounded: authored or retrieved answer keys for QA, deterministic
programs/calculators for verifiable reasoning, and executable schema validation
for tools. Qwen may verbalize those verified records, but it must not define
their truth.

Artifacts: `benchmarks/ollama_teacher_qualification_v1.json`,
`src/distillation/ollama_qualification.py`,
`experiments/ollama_teacher_qwen3_5_2b.jsonl`,
`experiments/ollama_teacher_qwen3_5_2b.summary.json`

### 2026-08-22 — Grounded assistant pilot v1

The first grounded, no-code SFT pilot was generated with Qwen restricted to
surface realization. Truth was supplied by 16 paired EN/TR authored answer
keys, a restricted deterministic arithmetic evaluator, and executable weather
and calculator tool fixtures. EN/TR variants of the same fact or scenario are
kept in the same split. Exact prompt contamination checks against both retained
student benchmarks run before generation, and a second audit replays every
calculation and tool call after generation.

Of 96 requested conversations, 95 passed all gates: 75 train and 20 validation.
The accepted set contains all 32 reasoning and all 32 tool conversations, 16/16
English QA, and 15/16 Turkish QA. The single exhausted item was rejected because
Qwen repeatedly corrupted the verified Turkish phrase for “blue whale.” The
post-generation audit passed 95/95 accepted records with zero exact prompt
overlaps against the retained benchmarks. The rejection is a desired result:
the verifier, not the teacher, controls admission.

Artifacts: `data/grounded/answer_keys_v1.json`,
`configs/grounded_pilot_v1.json`, `src/distillation/grounded_pilot.py`,
`data/processed/grounded-pilot-v1/manifest.json`

## Current position

The project has progressed from architecture prototypes to a deterministic, provenance-aware tournament harness. The major conclusions supported so far are:

1. `tiktoken-style-apostrophe-bpe-12k` is fixed as the production tokenizer for
   the EN/TR assistant objective.
2. Clean proxy-v2, tokenizer fingerprints, shared tokenizer loading, exact resume,
   and expanded deterministic evaluation operate end to end.
3. The completed generation-1 tournament established Glimmer as the strongest
   loss optimizer at 10M/context-512, but no architecture demonstrated capability
   above chance after approximately one million tokens.
4. The DT learning curve shows substantial held-out LM improvement through 8M
   tokens without capability emergence; the current corpus is rejected for the
   intended assistant and the 32M/100M continuation is paused.
5. A local-teacher SFT data program is now the critical path. Teacher capability
   and filtering quality must be measured before bulk generation.
6. DT remains the control and reference-speed model; architecture retesting is
   deferred until the new data produces capability at a meaningful exposure.
7. Final claims require both the frozen development suites and a separately held
   secret evaluation.

## Immediate next actions

1. Build a grounded pilot generator with no code tasks: verified EN/TR answer
   keys, deterministic reasoning generators/calculator results, and executable
   tool transcripts. Use Qwen only to phrase already-verified targets.
2. Add rejection and retry rules for Turkish language, concision, answer-key
   preservation, tool necessity, tool schema, and argument equivalence.
3. Generate a small auditable pilot dataset, deduplicate it, and manually review
   stratified samples before any bulk generation.
4. Train an apostrophe-BPE DT control on the pilot and require capability gains
   on frozen held-out tasks before scaling synthetic-data volume.
5. Only after that gate, expand the balanced EN/TR general-QA, short-reasoning,
   and tool-use corpus and re-establish the 1M/8M/32M learning curve.
