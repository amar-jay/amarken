# Amarken Project Report

Status date: 2026-08-22 (Europe/Istanbul)

This report records the project chronologically. Completed stages are backed by repository commits, configurations, and experiment artifacts. The context-512 tournament is still running; its results are explicitly marked provisional and reflect the latest operator updates available when this report was written.

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

Matched 8k, 12k, and 16k SentencePiece BPE tokenizers were trained on OPUS-100 English/Turkish data with byte fallback, fixed special tokens, identity normalization, deterministic ordering, and a single trainer thread.

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

The original SentencePiece-only selection was reopened after inference and
token-boundary visualization showed that aggregate fertility did not adequately
measure boundary quality. The v2 sweep compared byte-level BPE, tiktoken-style
regex BPE, a Turkish-apostrophe-aware tiktoken variant, a Turkish-weighted
tiktoken variant, corrected SentencePiece BPE/unigram candidates, and the
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
4. **SentencePiece unigram 12k — rejected.** Its favorable aggregate token
   count is partly obtained through structurally undesirable pieces spanning
   words, indentation, and lexemes. On a deterministic 250-document-per-domain
   audit, 12.263% of English tokens, 4.516% of Turkish tokens, and 3.983% of code
   tokens crossed internal word boundaries; 4.749% of code tokens combined
   indentation with a lexeme. The selected tiktoken and byte-BPE candidates had
   zero such cross-word pieces.

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

## Current position

The project has progressed from architecture prototypes to a deterministic, provenance-aware tournament harness. The major conclusions supported so far are:

1. `tiktoken-style-apostrophe-bpe-12k` is the selected production tokenizer;
   the Turkish-weighted tiktoken variant and byte-BPE 12k remain the Turkish-first
   and structural-control alternatives, respectively.
2. The tokenizer, data, model, training, resume, evaluation, and scaling paths operate end to end.
3. The 4k-token runs were useful health checks but could not support capability selection.
4. Learning-rate tuning and one-million-token exposure materially changed the apparent ordering: Glimmer is the provisional quality leader in the active 10M/context-512 tournament.
5. DT remains the essential full-precision control and reference-speed leader.
6. Bit remains the artifact-efficiency candidate; tensor versus channel scaling is unresolved.
7. Final promotion must wait for the complete three-seed report and deterministic capability/system evaluation of retained checkpoints.

## Immediate next actions

1. Allow every context-512 tournament arm to complete without changing the protocol.
2. Emit and hash `experiments/proxy_tournament_10m_context512.json`.
3. Verify exact token counts, validation coverage, finite gradients, LR selections, checkpoint hashes, and paired seed completeness.
4. Report paired per-seed differences: Glimmer−DT, Bit tensor−DT, Bit channel−DT, and Bit channel−Bit tensor.
5. Run the frozen deterministic capability and systems evaluator on retained checkpoints.
6. Promote distinct Pareto winners for quality, artifact size, KV memory, and runtime rather than collapsing all objectives into one score.
