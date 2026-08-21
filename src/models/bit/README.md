# Amarken-Bit

Native ternary-weight decoder derived from BitNet b1.58. It is trained from
scratch with FP master weights and absmean ternarization on every forward pass;
this is QAT, not post-training quantization. All Transformer projection weights
execute as scaled `{-1, 0, +1}` values. Activations, RMSNorm parameters, tied
token embeddings, and the tied LM head remain floating point by experiment
definition.

Default architecture: 18 layers, 512 hidden, 1472 ReLU2-GLU intermediate,
8Q/2KV GQA, 64 head dimension, full causal RoPE attention, four RMS norms per
layer including attention/FFN SubLN, 12k tied vocabulary, no biases. It remains
below 60M parameters; `artifact_report()` separately reports FP training master
bytes, the information-theoretic `log2(3)` bound, and practical 2-bit packing.

```python
import torch
from src.models.bit import BitCausalLM

model = BitCausalLM()
tokens = torch.randint(0, model.config.vocab_size, (2, 128))
output = model(tokens, labels=tokens)
output.loss.backward()
print(model.artifact_report())
```

`export_ternary()` emits packed projection payloads in memory. It is not a
runtime: actual speed/energy benefits require BitNet-aware ternary kernels.

Research basis: BitNet b1.58 paper, BitNet b1.58 2B4T technical report,
Microsoft BitNet inference repository, and the released 2B model configuration.
The detailed evidence, transfers, deliberate deviations, and byte boundaries are
recorded in [RESEARCH.md](RESEARCH.md).
