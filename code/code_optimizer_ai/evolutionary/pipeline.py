from __future__ import annotations

import time
import uuid
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

from code.code_optimizer_ai.config.evolutionary import (
    BENCHMARK_RUN_COUNT_MIN,
    BENCHMARK_WARMUP_RUNS,
    PHASE_A_MAX_SUGGESTIONS,
    PHASE_A_PROMPT_TEMPLATE_VERSION,
    PHASE_A_TRANSFORM_LIB_VERSION,
    REQUEST_TIME_BUDGET_MS,
    PhaseAWeights,
)
from code.code_optimizer_ai.core.llm_analyzer import CodeAnalyzerLLM
from code.code_optimizer_ai.evolutionary.constants import BottleneckStatus
from code.code_optimizer_ai.database.connection import db_manager
from code.code_optimizer_ai.evaluation.harness import EvaluationHarness, EvaluationResultV2
from code.code_optimizer_ai.evolutionary.population import CandidateRecord, PopulationGenerator
from code.code_optimizer_ai.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RequestContext:
    request_id: str
    created_at: datetime
    request_scope: str
    file_path: str
    representative_input: List[str]
    representative_input_warning: bool
    runtime_weight: float
    memory_weight: float
    quality_weight: float
    benchmark_runs: int
    warmup_runs: int
    max_suggestions: int
    request_time_budget_ms: int
    prompt_template_version: str
    transform_lib_version: str
    cv_gate_threshold: float = 0.15
    metadata: Dict[str, Any] = field(default_factory=dict)


class EvolutionaryPipeline:
    def __init__(
        self,
        *,
        analyzer: Optional[CodeAnalyzerLLM] = None,
        population_generator: Optional[PopulationGenerator] = None,
        evaluation_harness: Optional[EvaluationHarness] = None,
    ):
        self.analyzer = analyzer or CodeAnalyzerLLM()
        self.population_generator = population_generator or PopulationGenerator(self.analyzer)
        self.evaluation_harness = evaluation_harness or EvaluationHarness()
        self._requests_total = 0
        self._last_request_at: Optional[str] = None

    @staticmethod
    def _normalize_weights(objective_weights: Optional[Dict[str, float]]) -> tuple[float, float, float]:
        defaults = PhaseAWeights()
        weights = dict(objective_weights or {})

        runtime = float(weights.get("runtime", weights.get("runtime_weight", defaults.runtime_weight)))
        memory = float(weights.get("memory", weights.get("memory_weight", defaults.memory_weight)))
        quality = float(weights.get("quality", weights.get("quality_weight", defaults.quality_weight)))

        runtime = max(0.0, runtime)
        memory = max(0.0, memory)
        quality = max(0.0, quality)
        total = runtime + memory + quality
        if total <= 0:
            return defaults.runtime_weight, defaults.memory_weight, defaults.quality_weight
        return runtime / total, memory / total, quality / total

    @staticmethod
    def _clean_representative_input(representative_input: Optional[List[str]]) -> List[str]:
        cleaned = [str(item).strip() for item in (representative_input or []) if str(item).strip()]
        return cleaned

    async def _store_optimization_request(
        self,
        context: RequestContext,
        status: str,
        *,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not db_manager.connection_pool:
            return

        metadata = dict(context.metadata)
        metadata["request_scope"] = context.request_scope
        metadata["benchmark_runs"] = context.benchmark_runs
        metadata["warmup_runs"] = context.warmup_runs
        metadata["max_suggestions"] = context.max_suggestions
        if extra_metadata:
            metadata.update(extra_metadata)

        query = """
            INSERT INTO optimization_requests
            (id, request_scope, status, file_path, representative_input_warning, metadata, created_at, updated_at)
            VALUES ($1::uuid, $2, $3, $4, $5, $6::jsonb, NOW(), NOW())
            ON CONFLICT (id) DO UPDATE SET
                status = EXCLUDED.status,
                representative_input_warning = EXCLUDED.representative_input_warning,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
        """
        try:
            async with db_manager.connection_pool.acquire() as conn:
                await conn.execute(
                    query,
                    context.request_id,
                    context.request_scope,
                    status,
                    context.file_path,
                    context.representative_input_warning,
                    json.dumps(metadata),
                )
        except Exception as exc:
            logger.warning(
                "v2_request_tracking_failed",
                request_id=context.request_id,
                status=status,
                error=str(exc),
            )

    async def _store_candidate_evaluations(
        self,
        *,
        context: RequestContext,
        candidates: List[CandidateRecord],
        evaluations: List[EvaluationResultV2],
        latency_ms: int,
    ) -> None:
        if not db_manager.connection_pool:
            return
        if not evaluations:
            return

        candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
        rows: List[tuple[Any, ...]] = []
        for evaluation in evaluations:
            candidate = candidate_by_id.get(evaluation.candidate_id)
            if candidate is None:
                continue
            rows.append(
                (
                    str(uuid.uuid4()),
                    context.request_id,
                    evaluation.candidate_id,
                    candidate.generation_path.value,
                    candidate.family_tag.value,
                    evaluation.gate_result.value,
                    evaluation.gate_reason,
                    evaluation.syntax_valid,
                    evaluation.test_pass,
                    evaluation.run_count,
                    evaluation.warmup_runs_discarded,
                    evaluation.cv,
                    evaluation.runtime_delta_pct,
                    evaluation.memory_delta_pct,
                    evaluation.composite_score,
                    context.representative_input_warning,
                    context.representative_input_warning,
                    context.benchmark_runs,
                    latency_ms,
                )
            )

        if not rows:
            return

        query = """
            INSERT INTO phase_a_candidate_evaluations (
                id,
                request_id,
                candidate_id,
                generation_path,
                family_tag,
                gate_result,
                gate_reason,
                syntax_valid,
                test_pass,
                run_count,
                warmup_runs_discarded,
                cv,
                runtime_delta_pct,
                memory_delta_pct,
                composite_score,
                representative_input_warning,
                synthetic_fallback_penalty_applied,
                benchmark_runs_min,
                request_latency_ms,
                created_at
            )
            VALUES (
                $1::uuid,
                $2::uuid,
                $3::uuid,
                $4,
                $5,
                $6,
                $7,
                $8,
                $9,
                $10,
                $11,
                $12,
                $13,
                $14,
                $15,
                $16,
                $17,
                $18,
                $19,
                NOW()
            )
            ON CONFLICT (request_id, candidate_id) DO UPDATE SET
                gate_result = EXCLUDED.gate_result,
                gate_reason = EXCLUDED.gate_reason,
                syntax_valid = EXCLUDED.syntax_valid,
                test_pass = EXCLUDED.test_pass,
                run_count = EXCLUDED.run_count,
                warmup_runs_discarded = EXCLUDED.warmup_runs_discarded,
                cv = EXCLUDED.cv,
                runtime_delta_pct = EXCLUDED.runtime_delta_pct,
                memory_delta_pct = EXCLUDED.memory_delta_pct,
                composite_score = EXCLUDED.composite_score,
                representative_input_warning = EXCLUDED.representative_input_warning,
                synthetic_fallback_penalty_applied = EXCLUDED.synthetic_fallback_penalty_applied,
                benchmark_runs_min = EXCLUDED.benchmark_runs_min,
                request_latency_ms = EXCLUDED.request_latency_ms
        """
        try:
            async with db_manager.connection_pool.acquire() as conn:
                await conn.executemany(query, rows)
        except Exception as exc:
            logger.warning(
                "v2_candidate_eval_persist_failed",
                request_id=context.request_id,
                error=str(exc),
            )

    def build_request_context(
        self,
        *,
        request_scope: str,
        file_path: str,
        representative_input: Optional[List[str]],
        objective_weights: Optional[Dict[str, float]],
        max_suggestions: Optional[int],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RequestContext:
        runtime_weight, memory_weight, quality_weight = self._normalize_weights(objective_weights)
        cleaned_input = self._clean_representative_input(representative_input)
        return RequestContext(
            request_id=str(uuid.uuid4()),
            created_at=datetime.now(UTC),
            request_scope=request_scope,
            file_path=file_path,
            representative_input=cleaned_input,
            representative_input_warning=(len(cleaned_input) == 0),
            runtime_weight=runtime_weight,
            memory_weight=memory_weight,
            quality_weight=quality_weight,
            benchmark_runs=max(BENCHMARK_RUN_COUNT_MIN, int(BENCHMARK_RUN_COUNT_MIN)),
            warmup_runs=max(BENCHMARK_WARMUP_RUNS, int(BENCHMARK_WARMUP_RUNS)),
            max_suggestions=max(1, min(int(max_suggestions or PHASE_A_MAX_SUGGESTIONS), 3)),
            request_time_budget_ms=REQUEST_TIME_BUDGET_MS,
            prompt_template_version=PHASE_A_PROMPT_TEMPLATE_VERSION,
            transform_lib_version=PHASE_A_TRANSFORM_LIB_VERSION,
            metadata=metadata or {},
        )

    @staticmethod
    def _rank_candidates(
        candidates: List[CandidateRecord],
        evaluations: List[EvaluationResultV2],
        max_suggestions: int,
    ) -> List[Dict[str, Any]]:
        eval_by_id = {item.candidate_id: item for item in evaluations}
        ranked_payload: List[Dict[str, Any]] = []
        for candidate in candidates:
            evaluation = eval_by_id.get(candidate.candidate_id)
            if evaluation is None or evaluation.composite_score is None:
                continue
            ranked_payload.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "code_patch": candidate.code_patch,
                    "family_tag": candidate.family_tag.value,
                    "generation_path": candidate.generation_path.value,
                    "generation_produced": candidate.generation_number,
                    "composite_score": evaluation.composite_score,
                    "runtime_delta_pct": evaluation.runtime_delta_pct,
                    "memory_delta_pct": evaluation.memory_delta_pct,
                    "cv": evaluation.cv,
                    "test_pass": evaluation.test_pass,
                    "run_count": evaluation.run_count,
                    "warmup_runs_discarded": evaluation.warmup_runs_discarded,
                    "median_runtime_ms": evaluation.median_runtime_ms,
                    "mean_runtime_ms": evaluation.mean_runtime_ms,
                    "runtime_std_ms": evaluation.runtime_std_ms,
                }
            )
        ranked_payload.sort(key=lambda item: (item["composite_score"], item.get("runtime_delta_pct") or -999.0), reverse=True)
        return ranked_payload[:max(1, max_suggestions)]

    async def optimize_inline(
        self,
        *,
        code: str,
        file_path: str,
        representative_input: Optional[List[str]],
        objective_weights: Optional[Dict[str, float]],
        max_suggestions: Optional[int],
        run_unit_tests: bool,
        unit_test_command: Optional[str],
        analysis_static_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        context = self.build_request_context(
            request_scope="inline",
            file_path=file_path,
            representative_input=representative_input,
            objective_weights=objective_weights,
            max_suggestions=max_suggestions,
            metadata={"analysis_static_metrics": bool(analysis_static_metrics)},
        )
        await self._store_optimization_request(context, status="running")

        try:
            analysis = await self.analyzer.analyze_code(
                code,
                file_path=file_path,
                static_metrics=analysis_static_metrics or {},
                baseline_metrics=None,
            )

            candidates = await self.population_generator.generate_phase_a_population(
                request_context=context,
                original_code=code,
                analysis=analysis,
                max_candidates=context.max_suggestions + 1,
            )
            evaluations = await self.evaluation_harness.evaluate_candidates(
                request_context=context,
                candidates=candidates,
                file_path=file_path,
                run_unit_tests=run_unit_tests,
                unit_test_command=unit_test_command,
            )
            top_suggestions = self._rank_candidates(candidates, evaluations, context.max_suggestions)

            self._requests_total += 1
            self._last_request_at = datetime.now(UTC).isoformat()
            latency_ms = int((time.perf_counter() - started) * 1000)

            gate_counts: Dict[str, int] = {}
            measured_candidates = 0
            cv_pass_candidates = 0
            for evaluation in evaluations:
                gate_name = evaluation.gate_result.value
                gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
                if evaluation.run_count is not None and evaluation.run_count >= context.benchmark_runs and evaluation.cv is not None:
                    measured_candidates += 1
                    if evaluation.cv < context.cv_gate_threshold:
                        cv_pass_candidates += 1

            cv_pass_rate = (cv_pass_candidates / measured_candidates) if measured_candidates else None
            avg_top_runtime_delta = (
                sum((item.get("runtime_delta_pct") or 0.0) for item in top_suggestions) / len(top_suggestions)
                if top_suggestions
                else None
            )

            await self._store_candidate_evaluations(
                context=context,
                candidates=candidates,
                evaluations=evaluations,
                latency_ms=latency_ms,
            )
            payload = {
                "request_id": context.request_id,
                "phase": "A",
                "suggestions": top_suggestions,
                "bottleneck_status": BottleneckStatus.NO_CHANGE.value,
                "representative_input_warning": context.representative_input_warning,
                "synthetic_fallback_penalty_applied": context.representative_input_warning,
                "latency_ms": latency_ms,
                "candidate_count": len(candidates),
                "evaluated_count": len(evaluations),
            }
            await self._store_optimization_request(
                context,
                status="completed",
                extra_metadata={
                    "latency_ms": latency_ms,
                    "candidate_count": len(candidates),
                    "evaluated_count": len(evaluations),
                    "suggestions_count": len(top_suggestions),
                    "avg_top_runtime_delta_pct": avg_top_runtime_delta,
                    "cv_pass_rate": cv_pass_rate,
                    "gate_counts": gate_counts,
                },
            )
            return payload
        except Exception as exc:
            await self._store_optimization_request(
                context,
                status="failed",
                extra_metadata={"error": str(exc)},
            )
            raise

    def get_status(self) -> Dict[str, Any]:
        return {
            "phase": "A",
            "feature": "evolutionary_search",
            "requests_total": self._requests_total,
            "last_request_at": self._last_request_at,
            "benchmark_runs_min": BENCHMARK_RUN_COUNT_MIN,
            "warmup_runs": BENCHMARK_WARMUP_RUNS,
            "time_budget_ms": REQUEST_TIME_BUDGET_MS,
        }


evolutionary_pipeline = EvolutionaryPipeline()
