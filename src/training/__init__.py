"""Training and preflight correctness infrastructure."""

from .correctness import GateResult, SuiteResult, run_correctness_suite

__all__ = ["GateResult", "SuiteResult", "run_correctness_suite"]
