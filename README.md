# Amarken

Goal: best measured chat intelligence under `P <= 60M`; secondary objective: intelligence/artifact-bit and intelligence/runtime-byte. No architecture privileged. Unlimited sequential experiments; fixed tests decide.

## Invariants

- Decoder LM unless an alternative wins.
- Count embeddings, heads, shared parameters once; report total/active parameters, FLOPs/token, model bytes, KV bytes.
- Custom EN/TR tokenizer sweep: `8k, 12k, 16k`; byte fallback; tied input/output embeddings.
- Final selection uses hidden, contamination-checked EN/TR chat tests: instruction following, compositional reasoning, state tracking, grounded QA, tool routing, calibration, fluency; plus latency/RAM/energy.
- Never select on teacher/judge score alone. Deterministic tests > blinded humans > multiple independent judges.

## Student tournament

Matched tokenizer, tokens, optimizer-compute, data, context and evaluation:

- **Amarken-DT**: deep-thin RMSNorm/RoPE/SwiGLU/GQA Transformer; baseline near `18L × 512d × 1344ff × 8Q/2KV`, ~57M.
- **Amarken-Share**: fewer unique blocks, recurrent/immediate block reuse; spend saved parameters on width/depth; report logical and unique layers.
- **Amarken-Recur**: recurrent-depth Transformer with step embeddings and adaptive/fixed iterations.
- **Amarken-SSM**: gated state-space/convolutional mixer; matched parameters and FLOPs.
- **Amarken-Hybrid**: sparse attention layers interleaved with recurrent/SSM layers.
- **Amarken-Glimmer**: repeat `(Local-RoPE, Local-RoPE, Local-RoPE, Global-NoPE)`; local-window sweep `256/512/1024`; gated-GQA sweep `8:1/16:1`; per-head RMS QK-norm + learned query scale. Hypothesis: local layers learn order/composition cheaply; periodic position-free global layers carry long-range content without RoPE distance distortion.
- **Amarken-MoE**: <=60M total sparse experts; matched active FLOPs; only survives if total-byte score wins.
- **Amarken-Bit**: natively ternary weights; FP activations/norms; compare with post-trained INT4 DT winner.

Run 10M/25M proxy sweeps; promote Pareto winners to 60M. Ablate depth, width, FFN ratio, Q:KV ratio, sharing pattern, context, tokenizer, positional scheme. Repeat finalists across >=3 seeds.

Muse Glimmer provenance: released decoder = 52 layers of `(SWA-RoPE ×3, Full-NoPE ×1)`, 32Q/2KV gated GQA, per-head RMS QK-norm, extra query scaling. Transfer only these testable components. Optional DFlash improves decoding speed, not student intelligence, and counts against artifact parameters/bytes if tested. Release states Spark→Glimmer distillation but publishes no reproducible method; it supplies no Amarken distillation recipe. Sources: [model card](https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/main/README.md), [architecture](https://huggingface.co/blog/muse-glimmer#architecture).

## Distillation: executable definition

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
