# Amarken-Bit research record

## Primary evidence

- [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)
  introduced BitLinear and native quantization-aware training from scratch.
- [The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764) defines b1.58
  absmean weight quantization: one matrix scale, round-and-clip to `{-1,0,+1}`;
  it uses per-token symmetric 8-bit activations and LLaMA-like RMSNorm, SwiGLU,
  RoPE, and bias-free projections.
- [BitNet b1.58 2B4T technical report](https://arxiv.org/abs/2504.12285)
  documents the scaled release: native-from-scratch ternary forward weights,
  8-bit per-token absmax activations, SubLN, ReLU2, RoPE, and no biases.
- [Released 2B configuration](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-bf16/blob/main/config.json)
  specifies 30x2560, 6912 intermediate, 20Q/5KV GQA, head dimension 128,
  `relu2`, theta 500k, epsilon 1e-5, and tied embeddings.
- [Microsoft BitNet repository](https://github.com/microsoft/BitNet) supplies
  packed ternary CPU/GPU inference. Packing/runtime gains require specialized
  kernels and must not be inferred from ordinary PyTorch fake quantization.
- [Transformers BitNet integration](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/bitnet.py)
  confirms online FP-master weight quantization with an identity STE, FP32
  absmean calculation, and practical four-trits-per-byte packing plus scale.

## Decisions transferred

- Every attention and FFN projection is a bias-free BitLinear.
- Each forward uses `gamma=mean(abs(W))`, trits `round(W/gamma)` clipped to
  `[-1,1]`, and effective weights `gamma*trits`.
- Optimization uses FP masters and an identity straight-through gradient. Thus
  training RAM is not 1.58-bit; deployment weights can be packed losslessly.
- Full causal RoPE attention, 4:1 GQA, ReLU2-gated FFN, attention SubLN before
  output projection, and FFN SubLN before down projection follow the release.
- Tied FP embeddings/head and FP RMSNorm parameters follow the released artifact
  boundary and the Amarken tokenizer invariant.

## Intentional Amarken differences

- Activations stay floating point because the tournament definition explicitly
  asks for ternary weights with FP activations/norms. Activation INT8/a4.8/v2
  belongs in a separate ablation; combining it here would confound attribution.
- Dimensions become 18x512 with 1472 intermediate, 8Q/2KV, and a 12k vocabulary.
  This gives 58,692,992 parameters, of which 52,494,336 (89.44%) are ternary
  projection parameters. It preserves the 60M count constraint rather than
  spending storage savings on an unfairly larger parameter count.
- Plain PyTorch materializes effective weights and repeated GQA K/V tensors. It
  validates training behavior but makes no inference speed or energy claim.

## Byte boundaries

For the default model, `artifact_report()` reports approximately:

- FP32 learned training masters only: 223.90 MiB (optimizer/gradients excluded);
- theoretical `log2(3)` projections plus BF16 FP tensors/scales: 21.74 MiB;
- practical 2-bit projections plus BF16 FP tensors/scales: 24.34 MiB.

Tokenizer files, container metadata/alignment, runtime workspaces, activations,
and KV cache are outside these weight-only figures and must be reported separately
in a deployment experiment.
