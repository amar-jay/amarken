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
