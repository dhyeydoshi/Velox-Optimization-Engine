from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from code.code_optimizer_ai.core.validation_engine import ValidationResult
from code.code_optimizer_ai.evolutionary.constants import EvaluationGateResult


@dataclass(frozen=True)
class GateDecision:
    result: EvaluationGateResult
    reason: str


def evaluate_validation_gates(
    validation: ValidationResult,
    *,
    min_run_count: int,
    max_cv: float,
) -> GateDecision:
    if not validation.syntax_ok or validation.status == "failed_syntax":
        return GateDecision(EvaluationGateResult.FAILED_SYNTAX, validation.error or "syntax gate failed")

    if validation.unit_tests_passed is False or validation.status == "failed_tests":
        return GateDecision(EvaluationGateResult.FAILED_TESTS, validation.error or "unit tests failed")

    if validation.status == "failed_benchmark":
        return GateDecision(EvaluationGateResult.FAILED_BENCHMARK, validation.error or "benchmark failed")

    run_count = validation.run_count
    if run_count is not None and run_count < int(min_run_count):
        return GateDecision(
            EvaluationGateResult.FAILED_RUN_COUNT,
            f"run_count={run_count} is below required minimum {min_run_count}",
        )

    cv_value: Optional[float] = validation.cv
    if cv_value is not None and cv_value > float(max_cv):
        return GateDecision(
            EvaluationGateResult.FAILED_CV,
            f"cv={cv_value:.4f} exceeds threshold {max_cv:.4f}",
        )

    if validation.status != "passed":
        return GateDecision(
            EvaluationGateResult.FAILED_BENCHMARK,
            validation.error or f"unexpected validation status: {validation.status}",
        )

    return GateDecision(EvaluationGateResult.PASSED, "all gates passed")
