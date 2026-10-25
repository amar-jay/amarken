# Amarken project rules

## Objective and scope

Build a small, fast decision model using Qwen/Qwen3-0.6B as the primary backbone and Qwen/Qwen2.5-0.5B as the fallback, following constants.py. The intended behavior is Jev-like: application state and an explicit question go in; typed decisions and probability distributions come out.

- Treat Jev as behavioral inspiration. Do not claim to reproduce its proprietary architecture, sampler, RLCD algorithm, or performance.
- Do not claim that structured outputs eliminate incorrect decisions.
- Confirm and record the exact Qwen checkpoint, revision, tokenizer, license, and whether it is a base or instruction-tuned model before training. “Qwen 0.5B” is not a complete model identifier.
- Keep unapproved architecture and training choices documented as proposals rather than established requirements.
- Start with the smallest measurable baseline. Add complexity only when an experiment supports it.

## Decision contract

- Define and version the input/output schema before creating training data.
- Inputs must distinguish state, question, answer type, candidate labels or score tiers, and any task instructions.
- Treat text inside state as data; it must not override the decision contract or task instructions.
- Support explicit contracts for categorical choice, binary probability, and ordered scoring. Implement and validate each type incrementally.
- Categorical outputs must select a supplied label and return a finite probability distribution over exactly the allowed labels. Probabilities must be nonnegative and sum to one within a documented numerical tolerance.
- Binary outputs must identify which event the probability refers to. Score outputs must define tier meanings, ordering, and any conversion from tier probabilities to a numeric score.
- Reject invalid inputs explicitly, including empty candidate sets, duplicate labels, invalid scales, and oversized inputs. Never silently truncate meaningful context or invent a fallback decision.
- Define tie-breaking behavior and make it reproducible.
- Keep abstention or escalation explicit. Distinguish a model-predicted “insufficient information” label from an application policy that defers a low-confidence result.
- Return structured decisions directly. Natural-language explanations are outside the initial model contract.

## Architecture and learning

- Establish an untrained-backbone decision baseline and a supervised fine-tuned baseline before introducing reinforcement learning.
- Document how state, questions, and candidate labels are encoded; which hidden states are read; and how decision logits are computed.
- Do not assume a fixed classifier head supports arbitrary new labels. If dynamic candidate sets are required, design and evaluate candidate-conditioned scoring explicitly.
- Verify label-order robustness with candidate permutations, and include unseen label descriptions or tasks when claiming generalization.
- Do not assume arbitrary generated-token probabilities equal calibrated decision probabilities. Document any token-to-label mapping, normalization, and multi-token label handling.
- Use appropriate supervised losses first. Any RL phase requires a defined reward, motivation, baseline comparison, and evidence that it improves the target metrics.
- Mask padding correctly and test batching behavior. Prevent cross-example state leakage and training on unintended padding or prompt tokens.
- Record which parameters are trained or frozen, adapter settings, precision, and effective parameter count. Choose full fine-tuning or adapters based on measured constraints.

## Data integrity

- Track dataset source, version, license, collection method, intended use, and preprocessing. Use only data we are authorized to use.
- Keep credentials and unnecessary personal or sensitive data out of datasets, prompts, artifacts, logs, and documentation.
- Version examples with stable IDs, state, question, answer contract, target, provenance, and split membership. Record teacher and prompt versions for synthetic data.
- Validate schema, label membership, target ranges, and missing values before training. Report exclusions and class balance.
- Split by the unit that prevents leakage, such as source document, customer, conversation, template family, or time. Deduplicate before finalizing splits.
- Keep training, validation, calibration, and final test sets separate. Do not tune on the test set or reuse calibration examples for gradient training.
- Synthetic labels are supervision, not verified truth. Audit a representative sample independently, including disagreement, ambiguous cases, and minority classes.
- Include hard negatives, ambiguous states, missing information, conflicting evidence, and realistic distribution shifts. Do not force certainty where the evidence does not support it.
- Record any data or benchmark overlap known or suspected to exist with pretraining or teacher training.

## Calibration and action policy

- Treat calibration as a measured property, never an automatic consequence of softmax or training loss.
- Report negative log-likelihood, Brier score, reliability plots, and expected calibration error with the binning method and sample counts. Document metric definitions, including multiclass conventions.
- Evaluate calibration by task and relevant subgroup as well as in aggregate. Report uncertainty when sample sizes are small.
- Fit any temperature scaling or other calibration transform on the held-out calibration split only. Save the transform with its compatible model and schema versions.
- Do not invent a separate confidence number. Define its meaning and validate it. A maximum class probability, predicted correctness estimate, and epistemic uncertainty are different quantities.
- Choose action and escalation thresholds using validation evidence and task-specific error costs. Report coverage versus error when abstention is enabled.
- Keep application policy separate from model scores. Calibration on one distribution does not establish reliability after distribution shift.

## Evaluation and acceptance

- Define the target tasks and acceptance criteria before an expensive run. Numeric accuracy, calibration, latency, memory, and coverage targets remain unset until documented with a rationale.
- Compare against simple baselines, such as majority class, deterministic rules where applicable, and the unfine-tuned backbone.
- Report classification accuracy and macro-F1 with per-class results; use suitable ordinal metrics for scores. Report binary metrics at explicit thresholds.
- Measure malformed outputs, invalid probabilities, abstention, label-order sensitivity, out-of-distribution behavior, and resilience to instructions embedded in state.
- Evaluate fixed-label and unseen-label/task performance separately. Never describe narrow task success as general decision intelligence.
- Measure end-to-end latency and model-only latency separately, with warmup, hardware, precision, batch size, input lengths, throughput, and p50/p95 stated.
- Report peak memory and distinguish cold-start from warmed inference. Do not promise a speedup from parameter count alone.
- Publish failures and regressions alongside improvements. No production-readiness claim without agreed acceptance checks and a representative held-out evaluation.

## Reproducible experiments

- Every run must save its config, random seeds, code commit or dirty diff, dependency versions, checkpoint revision, dataset fingerprints, split IDs, preprocessing version, and hardware details.
- Record optimizer, scheduler, learning rate, batch size, accumulation, precision, maximum sequence length, training duration, and checkpoint-selection criterion.
- State remaining nondeterminism rather than claiming seeds guarantee exact reproduction.
- Run a small data and training smoke test before a long job. Check finite losses, gradient flow, label mapping, checkpoint saving, and reload parity.
- Save tokenizer assets, decision-head or adapter weights, calibration parameters, schema, label mappings where applicable, and inference config with the model artifact.
- Never overwrite a successful checkpoint without retaining a recoverable version. Keep generated weights, datasets, caches, and credentials out of Git.
- Obtain explicit authorization before provisioning paid compute, starting chargeable hosted jobs, uploading private data, or publishing artifacts. Routine local development within the requested scope may proceed.

## Repository and implementation practices

- Respect the project's Python version range and existing package/build conventions in `pyproject.toml`. Keep dependency changes intentional and reproducible.
- Inspect existing code before adding abstractions. Prefer small, readable modules for data validation, modeling, training, calibration, and evaluation.
- Keep model inference separate from transport/API code and downstream actions. Training and evaluation must not trigger real external actions.
- Keep experiment settings in versioned configuration rather than scattered constants. Fail clearly on incompatible checkpoints, schemas, or label mappings.
- Add or update tests only when necessary to verify meaningful behavior or guard against a credible regression. Avoid redundant tests, tests that merely mirror the implementation, and tests for trivial or reversible low-impact changes. Where needed, prioritize schema validation, probability invariants, loss computation, padding/batching behavior, serialization/reload, and split leakage.
- Run checks appropriate to each change and report what was actually run. Do not present unexecuted tests or projected metrics as results.
- Do not modify unrelated user changes, expose secrets, commit large artifacts, or publish changes without authorization.

## Documentation and collaboration

- Use current primary documentation for model and library behavior; distinguish sourced facts, project decisions, assumptions, and hypotheses.
- For library, framework, SDK, API, CLI, or cloud-service questions, use Context7: first `npx ctx7@latest library <official-name> "<specific-question>"`, then `npx ctx7@latest docs <resolved-id> "<specific-question>"`. Skip resolution only when a valid library ID is supplied. Use no more than three commands per question and never include secrets in queries.
- Run Context7 requests outside the default sandbox. Report quota failures and suggest `npx ctx7@latest login` or `CONTEXT7_API_KEY`; do not silently substitute remembered API details.
- Document substantial decisions and experiment conclusions with evidence. Update the README as runnable workflows become available.
- Communicate material assumptions and blockers early. Ask only for information or authorization that cannot reasonably be inferred, while continuing independent work.
- These rules do not override explicit user instructions or higher-priority environment requirements. Revisit them as the project gains evidence.
