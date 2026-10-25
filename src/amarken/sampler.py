"""Turn decision logits into typed outputs without generating text."""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import exp, fsum, isfinite
from random import Random

from amarken.constants import DECISION_KINDS, SAMPLER_MODES


@dataclass(frozen=True)
class SamplerConfig:
    mode: str = SAMPLER_MODES[0]
    temperature: float = 1.0
    seed: int = 42

    def __post_init__(self):
        if self.mode not in SAMPLER_MODES:
            raise ValueError(f"Supported sampler modes: {SAMPLER_MODES}")
        if not isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")


class Sampler:
    def __init__(self, config: SamplerConfig | None = None):
        self.config = config or SamplerConfig()
        self._random = Random(self.config.seed)

    def sample(
        self,
        logits: Sequence[float],
        *,
        kind: str,
        options: Sequence[str] = (),
        score_values: Sequence[float] | None = None,
    ) -> dict:
        """Decode one question. Noul expects one true-vs-false logit.

        Choice/score logits align with options. Score requires explicit strictly
        increasing numeric anchors; labels alone do not define a numeric scale.
        Ties select the first option in argmax mode. Probabilities are model
        probabilities, not a claim of calibrated correctness.
        """
        if kind not in DECISION_KINDS:
            raise ValueError(f"Supported decision kinds: {DECISION_KINDS}")
        if isinstance(options, str):
            raise TypeError("options must be a sequence of labels")
        options = list(options)
        values = [float(value) for value in logits]
        if not values or any(not isfinite(value) for value in values):
            raise ValueError("logits must be nonempty and finite")
        if kind != "score" and score_values is not None:
            raise ValueError("score_values applies only to score questions")
        if kind == "noul":
            if options or len(values) != 1:
                raise ValueError("noul requires one logit and no options")
            # A two-class softmax [false=0, true=logit] avoids sigmoid overflow.
            probabilities = self._probabilities([0.0, values[0]])
            # Return the event probability directly, without drawing a boolean.
            return {"kind": kind, "probability_true": probabilities[1]}
        if (
            len(options) < 2
            or len(options) != len(values)
            or any(
                not isinstance(option, str) or not option.strip() for option in options
            )
        ):
            raise ValueError("Provide at least two labels and one logit per label")
        if len(set(options)) != len(options):
            raise ValueError("Option labels must be unique")
        anchors = None
        if kind == "score":
            if score_values is None:
                raise ValueError(
                    "score requires numeric score_values aligned with options"
                )
            anchors = [float(value) for value in score_values]
            if (
                len(anchors) != len(options)
                or any(not isfinite(value) for value in anchors)
                or any(a >= b for a, b in pairwise(anchors))
            ):
                raise ValueError(
                    "Score anchors must be finite, strictly increasing and aligned with options"
                )
        probabilities = self._probabilities(values)
        if self.config.mode == "argmax":
            index = max(range(len(probabilities)), key=probabilities.__getitem__)
        else:
            index = self._random.choices(
                range(len(options)), weights=probabilities, k=1
            )[0]
        result = {
            "kind": kind,
            "choice": options[index],
            "index": index,
            "probabilities": dict(zip(options, probabilities)),
            "selected_probability": probabilities[index],
        }
        if anchors is not None:
            result.update(
                score=anchors[index],
                expected_score=fsum(
                    p * value for p, value in zip(probabilities, anchors)
                ),
            )
        return result

    def _probabilities(self, logits: list[float]) -> list[float]:
        # Subtract the maximum BEFORE temperature scaling, including tiny T.
        maximum = max(logits)
        weights = [exp((value - maximum) / self.config.temperature) for value in logits]
        total = fsum(weights)
        return [weight / total for weight in weights]
