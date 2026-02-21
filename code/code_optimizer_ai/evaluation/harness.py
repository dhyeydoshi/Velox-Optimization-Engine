from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from code.code_optimizer_ai.config.evolutionary import BENCHMARK_RUN_COUNT_MIN, BENCHMARK_WARMUP_RUNS
from code.code_optimizer_ai.evaluation.gates import evaluate_validation_gates
from code.code_optimizer_ai.evaluation.sandbox_client import SandboxValidationClient
from code.code_optimizer_ai.evolutionary.constants import EvaluationGateResult
from code.code_optimizer_ai.evolutionary.population import CandidateRecord
from code.code_optimizer_ai.evolutionary.scoring import (
    compute_code_size_delta_norm,
    compute_composite_score,
    static_quality_score,
)


@dataclass
class EvaluationResultV2:
    candidate_id: str
    gate_result: EvaluationGateResult
    gate_reason: str
    syntax_valid: bool
    test_pass: Optional[bool]
    median_runtime_ms: Optional[float]
    mean_runtime_ms: Optional[float]
    runtime_std_ms: Optional[float]
    cv: Optional[float]
    run_count: Optional[int]
    warmup_runs_discarded: Optional[int]
    runtime_delta_pct: Optional[float]
    memory_delta_pct: Optional[float]
    code_size_delta_norm: float
    static_quality_score: float
    composite_score: Optional[float]
    error: Optional[str]


class EvaluationHarness:
    def __init__(self, sandbox_client: Optional[SandboxValidationClient] = None):
        self.sandbox_client = sandbox_client or SandboxValidationClient()

    async def evaluate_candidates(
        self,
        *,
        request_context: Any,
        candidates: List[CandidateRecord],
        file_path: str,
        run_unit_tests: bool,
        unit_test_command: Optional[str],
    ) -> List[EvaluationResultV2]:
        benchmark_runs = max(BENCHMARK_RUN_COUNT_MIN, int(getattr(request_context, "benchmark_runs", 1)))
        warmup_runs = max(BENCHMARK_WARMUP_RUNS, int(getattr(request_context, "warmup_runs", 0)))
        max_cv = float(getattr(request_context, "cv_gate_threshold", 0.15))

        results: List[EvaluationResultV2] = []
        for candidate in candidates:
            validation = await self.sandbox_client.validate_candidate(
                original_code=candidate.original_code,
                candidate_code=candidate.code_patch,
                file_path=file_path,
                run_unit_tests=run_unit_tests,
                unit_test_command=unit_test_command,
                benchmark_mode=True,
                benchmark_runs=benchmark_runs,
                warmup_runs=warmup_runs,
            )
            gate = evaluate_validation_gates(
                validation,
                min_run_count=benchmark_runs,
                max_cv=max_cv,
            )

            quality = static_quality_score(candidate.code_patch)
            size_delta_norm = compute_code_size_delta_norm(candidate.original_code, candidate.code_patch)
            score_breakdown = compute_composite_score(
                runtime_delta_pct=validation.runtime_delta_pct,
                memory_delta_pct=validation.memory_delta_pct,
                static_quality=quality,
                cv=validation.cv,
                code_size_delta_norm=size_delta_norm,
                runtime_weight=float(getattr(request_context, "runtime_weight", 0.5)),
                memory_weight=float(getattr(request_context, "memory_weight", 0.3)),
                quality_weight=float(getattr(request_context, "quality_weight", 0.2)),
                unit_tests_passed=validation.unit_tests_passed,
                synthetic_fallback=bool(getattr(request_context, "representative_input_warning", False)),
            )

            composite_score: Optional[float]
            if gate.result == EvaluationGateResult.PASSED or validation.unit_tests_passed is False:
                composite_score = score_breakdown.composite_score
            else:
                composite_score = None

            results.append(
                EvaluationResultV2(
                    candidate_id=candidate.candidate_id,
                    gate_result=gate.result,
                    gate_reason=gate.reason,
                    syntax_valid=validation.syntax_ok,
                    test_pass=validation.unit_tests_passed,
                    median_runtime_ms=validation.median_runtime_ms,
                    mean_runtime_ms=validation.mean_runtime_ms,
                    runtime_std_ms=validation.runtime_std_ms,
                    cv=validation.cv,
                    run_count=validation.run_count,
                    warmup_runs_discarded=validation.warmup_runs_discarded,
                    runtime_delta_pct=validation.runtime_delta_pct,
                    memory_delta_pct=validation.memory_delta_pct,
                    code_size_delta_norm=size_delta_norm,
                    static_quality_score=quality,
                    composite_score=composite_score,
                    error=validation.error,
                )
            )

        return results
