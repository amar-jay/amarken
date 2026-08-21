# Amarken-Glimmer

A trainable, text-only, sub-60M decoder derived from the released Muse Glimmer
text architecture. This is not the 30B checkpoint and does not include its
vision tower or DFlash drafter.

Transferred components:

- hybrid attention, with full/NoPE layers counted backward from the final layer;
- sliding-window/RoPE attention in all other layers;
- gated grouped-query attention (8 query heads to 1 KV head by default);
- scaleless per-head Q/K RMS normalization and learnable per-query-head scaling;
- normalized embeddings, centered pre/post sublayer RMSNorm, SwiGLU;
- output scaling, tanh logit soft-capping, and tied token/output embeddings.

The default `12k vocab × 18 layers × 512 hidden × 1344 FFN` configuration is
about 58.6M parameters. It uses a 512-token local window; the project should
sweep 256/512/1024 and tokenizer sizes as specified in the root README.

```python
import torch
from src.models.glimmer import GlimmerCausalLM

model = GlimmerCausalLM()
tokens = torch.randint(0, model.config.vocab_size, (2, 256))
result = model(tokens, labels=tokens)
result.loss.backward()
```

Run the focused tests with `pytest -q tests/test_glimmer.py`.
