import asyncio
import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate

from code.code_optimizer_ai.config.settings import settings
from code.code_optimizer_ai.core.llm_gateway import LLMGateway
from code.code_optimizer_ai.evolutionary.constants import FamilyTag
from code.code_optimizer_ai.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CodeAnalysisResult:

    semantic_summary: str
    performance_bottlenecks: List[str]
    optimization_opportunities: List[str]
    complexity_score: float
    maintainability_score: float
    security_issues: List[str]
    best_practices_violations: List[str]
    confidence_score: float
    processing_time: float
    hotspot_embedding: Optional[List[float]] = None
    family_seed_tags: Optional[List[str]] = None


@dataclass
class OptimizationSuggestion:

    suggestion_id: str
    category: str
    priority: str
    description: str
    original_code: str
    optimized_code: str
    patch_diff: str = ""
    expected_improvement: str = ""
    expected_runtime_delta_pct: Optional[float] = None
    expected_memory_delta_pct: Optional[float] = None
    expected_weighted_score: Optional[float] = None
    implementation_effort: str = "medium"
    confidence: float = 0.5
    reasoning: str = ""
    validation_status: str = "not_validated"
    model_trace: Dict[str, Any] = field(default_factory=dict)


class CodeAnalyzerLLM:

    def __init__(self):
        self.gateway = LLMGateway()
        self.analysis_prompt = self._create_analysis_prompt()
        self.optimization_prompt = self._create_optimization_prompt()
        self.repository_optimization_prompt = self._create_repository_optimization_prompt()

    def _create_analysis_prompt(self) -> PromptTemplate:
        template = """
You are an expert software engineer focused on performance analysis. Respond with JSON only (no prose, no Markdown). If the input is not valid Python, return {{"error": {{"type": "invalid_input", "reason": "<short reason>"}}}}.

Context:
- file_path: {file_path}
- identifier: {identifier}
- static_metrics: {static_metrics}
- baseline_metrics: {baseline_metrics}
- code:
```python
{code}
```

Return exactly this JSON shape:
{{
  "semantic_summary": "<=80 words on purpose/behavior",
  "performance_bottlenecks": ["<identifier>: <issue> (<why>)", ...],
  "optimization_opportunities": ["<identifier>: <idea> (<why>)", ...],
  "complexity_score": 0.0,
  "maintainability_score": 0.0,
  "security_issues": ["<issue>", ...],
  "best_practices_violations": ["<issue>", ...],
  "confidence_score": 0.0,
  "reasoning": "<=120 words showing the key evidence used"
}}

Rules: keep lists to max 5 items; cite the target function/class in each list entry; prefer concrete observations (allocations, loops, I/O, globals). Do not invent metrics beyond provided context.
"""
        return PromptTemplate(
            template=template,
            input_variables=["code", "file_path", "identifier", "static_metrics", "baseline_metrics"],
        )

    def _create_optimization_prompt(self) -> PromptTemplate:
        template = """
You are an expert performance optimization engineer. Respond with JSON only (no prose, no Markdown). If the analysis context indicates an error, return {{"suggestions": [], "error": "<reason>"}}.

Inputs:
- analysis_context (JSON string): {analysis_context}
- code:
```python
{code}
```

Allowed categories (match RL action space): ["algorithm_change","data_structure_optimization","caching_strategy","parallelization","memory_optimization","io_optimization","database_optimization","network_optimization","security_optimization","code_quality_optimization","function_optimization","string_optimization","mathematical_optimization","loop_optimization","exception_handling_optimization","import_optimization","configuration_optimization","api_optimization","concurrency_optimization","resource_management_optimization","logging_optimization","testing_optimization","build_optimization","documentation_optimization","no_change"]
Allowed priorities: ["low","medium","high","critical"]

Return exactly this JSON shape:
{{
  "suggestions": [
    {{
      "suggestion_id": "opt_{identifier}_{{n}}",
      "category": "<one of allowed categories>",
      "priority": "low|medium|high|critical",
      "description": "<=50 words, concrete change",
      "optimized_code": "<either minimal diff block or full replacement, no commentary>",
      "expected_improvement": "<quantify speed/memory/I/O>",
      "expected_runtime_delta_pct": 0.0,
      "expected_memory_delta_pct": 0.0,
      "implementation_effort": "low|medium|high",
      "confidence": 0.0,
      "reasoning": "<=80 words, cite the exact bottleneck addressed>"
    }}
  ]
}}

Rules: only emit 1-3 high-signal suggestions; avoid adding dependencies; prefer backward-compatible changes; if any change is breaking, state it in description and set priority to critical; keep optimized_code minimal and self-contained.
        """
        return PromptTemplate(template=template, input_variables=["analysis_context", "code", "identifier"])

    def _create_repository_optimization_prompt(self) -> PromptTemplate:
        template = """
You are an expert Python performance engineer optimizing a connected set of files together. Respond with JSON only (no prose, no Markdown).

Inputs:
- batch_context (JSON string): {batch_context}
- max_suggestions_per_file: {max_suggestions_per_file}
- files (JSON string): {files_json}

Allowed categories (match RL action space): ["algorithm_change","data_structure_optimization","caching_strategy","parallelization","memory_optimization","io_optimization","database_optimization","network_optimization","security_optimization","code_quality_optimization","function_optimization","string_optimization","mathematical_optimization","loop_optimization","exception_handling_optimization","import_optimization","configuration_optimization","api_optimization","concurrency_optimization","resource_management_optimization","logging_optimization","testing_optimization","build_optimization","documentation_optimization","no_change"]
Allowed priorities: ["low","medium","high","critical"]

Return exactly this JSON shape:
{{
  "results": [
    {{
      "file_path": "<must match one input file_path>",
      "suggestions": [
        {{
          "suggestion_id": "batch_opt_<n>",
          "category": "<allowed category>",
          "priority": "low|medium|high|critical",
          "description": "<=60 words; include cross-file impact if any>",
          "optimized_code": "<full replacement for this file only, no markdown>",
          "expected_improvement": "<runtime/memory estimate>",
          "expected_runtime_delta_pct": 0.0,
          "expected_memory_delta_pct": 0.0,
          "implementation_effort": "low|medium|high",
          "confidence": 0.0,
          "reasoning": "<=90 words referencing concrete lines/flows>"
        }}
      ]
    }}
  ]
}}

Rules:
- Keep behavior unchanged unless explicitly noted.
- Ensure compatibility across files in this batch (imports/interfaces/call signatures).
- Emit 0 to max_suggestions_per_file suggestions per file.
- Prefer the smallest safe change set and avoid new dependencies.
"""
        return PromptTemplate(
            template=template,
            input_variables=["batch_context", "files_json", "max_suggestions_per_file"],
        )

    @staticmethod
    def _float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        try:
            start = response_text.find("{")
            if start == -1:
                return json.loads(response_text)

            # Walk through closing braces to find the first balanced JSON object.
            pos = start
            while True:
                end = response_text.find("}", pos)
                if end == -1:
                    break
                try:
                    return json.loads(response_text[start : end + 1])
                except json.JSONDecodeError:
                    pos = end + 1

            # Fallback: try the full text.
            return json.loads(response_text)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse JSON response", error=str(exc))
            return {}

    @staticmethod
    def _deterministic_hotspot_embedding(payload: str, dim: int = 768) -> List[float]:
        # Cheap deterministic encoder for seam compatibility in Phase A.
        values: List[float] = []
        counter = 0
        normalized_payload = payload or ""
        while len(values) < dim:
            digest = hashlib.sha256(f"{normalized_payload}:{counter}".encode("utf-8")).digest()
            for idx in range(0, len(digest), 2):
                pair = digest[idx : idx + 2]
                if len(pair) < 2:
                    continue
                unit = int.from_bytes(pair, byteorder="big", signed=False) / 65535.0
                values.append((unit * 2.0) - 1.0)
                if len(values) >= dim:
                    break
            counter += 1
        return values

    @staticmethod
    def _infer_family_seed_tags(
        performance_bottlenecks: List[str],
        optimization_opportunities: List[str],
        security_issues: List[str],
    ) -> List[str]:
        text = " ".join(
            list(performance_bottlenecks or [])
            + list(optimization_opportunities or [])
            + list(security_issues or [])
        ).lower()

        tag_rules = (
            (FamilyTag.DATA_STRUCTURE.value, ("dict", "set", "hash", "lookup", "list")),
            (FamilyTag.LOOP_RESTRUCTURE.value, ("loop", "nested", "iteration", "comprehension")),
            (FamilyTag.VECTORIZATION.value, ("vector", "numpy", "broadcast")),
            (FamilyTag.CACHING.value, ("cache", "memo", "recompute", "reuse")),
            (FamilyTag.PARALLELIZATION.value, ("parallel", "concurrent", "thread", "process", "async")),
            (FamilyTag.ALGORITHMIC.value, ("algorithm", "complexity", "o(", "search", "sort")),
            (FamilyTag.IO_BATCHING.value, ("i/o", "io", "database", "network", "batch", "query")),
            (FamilyTag.MEMORY_LAYOUT.value, ("memory", "allocation", "buffer", "object churn")),
        )

        matched: List[str] = []
        for tag, keywords in tag_rules:
            if any(keyword in text for keyword in keywords):
                matched.append(tag)
        return matched

    async def _invoke_json(self, prompt: str, *, trace_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        response = await self.gateway.ainvoke(
            [
                SystemMessage(content="You are an expert code analyst and optimizer."),
                HumanMessage(content=prompt),
            ],
            trace_id=trace_id,
        )
        data = self._parse_json_response(response.content)
        trace = {
            "provider": response.provider,
            "model": response.model,
            "latency_ms": round(response.latency_ms, 2),
            "attempts": [attempt.__dict__ for attempt in response.attempts],
            "fallback_count": max(0, len(response.attempts) - 1),
        }
        return data, trace

    async def analyze_code(
        self,
        code: str,
        file_path: str = "",
        identifier: str = "",
        static_metrics: Optional[Dict[str, Any]] = None,
        baseline_metrics: Optional[Dict[str, Any]] = None,
    ) -> CodeAnalysisResult:
        start_time = datetime.now()
        try:
            prompt = self.analysis_prompt.format(
                code=code,
                file_path=file_path,
                identifier=identifier,
                static_metrics=json.dumps(static_metrics or {}),
                baseline_metrics=json.dumps(baseline_metrics) if baseline_metrics is not None else "null",
            )
            analysis_data, trace = await self._invoke_json(
                prompt,
                trace_id=f"analysis:{file_path or 'inline'}:{int(start_time.timestamp())}",
            )
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(
                "analysis_completed",
                file_path=file_path,
                processing_time=round(processing_time, 3),
                model=trace.get("model"),
                provider=trace.get("provider"),
            )

            family_seed_tags = self._infer_family_seed_tags(
                analysis_data.get("performance_bottlenecks", []),
                analysis_data.get("optimization_opportunities", []),
                analysis_data.get("security_issues", []),
            )
            embedding_payload = json.dumps(
                {
                    "code": code,
                    "file_path": file_path,
                    "identifier": identifier,
                    "semantic_summary": analysis_data.get("semantic_summary", ""),
                    "performance_bottlenecks": analysis_data.get("performance_bottlenecks", []),
                    "optimization_opportunities": analysis_data.get("optimization_opportunities", []),
                    "security_issues": analysis_data.get("security_issues", []),
                },
                sort_keys=True,
            )

            return CodeAnalysisResult(
                semantic_summary=analysis_data.get("semantic_summary", ""),
                performance_bottlenecks=analysis_data.get("performance_bottlenecks", []),
                optimization_opportunities=analysis_data.get("optimization_opportunities", []),
                complexity_score=self._float(analysis_data.get("complexity_score"), 0.0),
                maintainability_score=self._float(analysis_data.get("maintainability_score"), 0.0),
                security_issues=analysis_data.get("security_issues", []),
                best_practices_violations=analysis_data.get("best_practices_violations", []),
                confidence_score=self._float(analysis_data.get("confidence_score"), 0.0),
                processing_time=processing_time,
                hotspot_embedding=self._deterministic_hotspot_embedding(embedding_payload),
                family_seed_tags=family_seed_tags or None,
            )
        except Exception as exc:
            logger.error("Code analysis failed", error=str(exc), file_path=file_path)
            return CodeAnalysisResult(
                semantic_summary="Analysis failed",
                performance_bottlenecks=[],
                optimization_opportunities=[],
                complexity_score=0.0,
                maintainability_score=0.0,
                security_issues=[],
                best_practices_violations=[],
                confidence_score=0.0,
                processing_time=(datetime.now() - start_time).total_seconds(),
                hotspot_embedding=None,
                family_seed_tags=None,
            )

    async def generate_optimizations(
        self,
        code: str,
        analysis: CodeAnalysisResult,
        identifier: str = "",
    ) -> List[OptimizationSuggestion]:
        analysis_context = json.dumps(
            {
                "semantic_summary": analysis.semantic_summary,
                "performance_bottlenecks": analysis.performance_bottlenecks,
                "optimization_opportunities": analysis.optimization_opportunities,
                "complexity_score": analysis.complexity_score,
                "maintainability_score": analysis.maintainability_score,
                "security_issues": analysis.security_issues,
                "best_practices_violations": analysis.best_practices_violations,
                "confidence_score": analysis.confidence_score,
                "objective": {
                    "runtime_weight": settings.OBJECTIVE_RUNTIME_WEIGHT,
                    "memory_weight": settings.OBJECTIVE_MEMORY_WEIGHT,
                },
            }
        )

        prompt = self.optimization_prompt.format(
            analysis_context=analysis_context,
            code=code,
            identifier=identifier,
        )

        try:
            optimization_data, trace = await self._invoke_json(
                prompt,
                trace_id=f"opt:{identifier or 'inline'}:{int(datetime.now().timestamp())}",
            )
            suggestions: List[OptimizationSuggestion] = []
            for idx, suggestion_data in enumerate(optimization_data.get("suggestions", [])):
                suggestions.append(
                    OptimizationSuggestion(
                        suggestion_id=suggestion_data.get("suggestion_id", f"suggestion_{idx}"),
                        category=suggestion_data.get("category", "code_quality_optimization"),
                        priority=suggestion_data.get("priority", "medium"),
                        description=suggestion_data.get("description", ""),
                        original_code=code,
                        optimized_code=suggestion_data.get("optimized_code", ""),
                        expected_improvement=suggestion_data.get("expected_improvement", ""),
                        expected_runtime_delta_pct=self._float(
                            suggestion_data.get("expected_runtime_delta_pct"), 0.0
                        ),
                        expected_memory_delta_pct=self._float(
                            suggestion_data.get("expected_memory_delta_pct"), 0.0
                        ),
                        implementation_effort=suggestion_data.get("implementation_effort", "medium"),
                        confidence=self._float(suggestion_data.get("confidence"), 0.5),
                        reasoning=suggestion_data.get("reasoning", ""),
                        model_trace=dict(trace),
                    )
                )
            return suggestions
        except Exception as exc:
            logger.error("Optimization generation failed", error=str(exc))
            return []

    async def generate_repository_optimizations(
        self,
        files: List[Dict[str, Any]],
        *,
        max_suggestions_per_file: int = 2,
        shared_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[OptimizationSuggestion]]:
        if not files:
            return {}

        normalized_inputs: List[Dict[str, Any]] = []
        code_by_path: Dict[str, str] = {}
        for entry in files:
            if not isinstance(entry, dict):
                continue
            file_path = str(entry.get("file_path") or "").strip()
            code = str(entry.get("code") or "")
            static_metrics = entry.get("static_metrics", {})
            if not file_path:
                continue
            if not isinstance(static_metrics, dict):
                static_metrics = {}
            normalized_inputs.append(
                {
                    "file_path": file_path,
                    "code": code,
                    "static_metrics": static_metrics,
                }
            )
            code_by_path[file_path] = code

        if not normalized_inputs:
            return {}

        limit = max(1, int(max_suggestions_per_file))
        prompt = self.repository_optimization_prompt.format(
            batch_context=json.dumps(shared_context or {}),
            files_json=json.dumps(normalized_inputs),
            max_suggestions_per_file=limit,
        )

        try:
            batch_data, trace = await self._invoke_json(
                prompt,
                trace_id=f"batch_opt:{int(datetime.now().timestamp())}",
            )
            raw_results = batch_data.get("results", batch_data.get("file_results", []))
            if not isinstance(raw_results, list):
                return {}

            suggestions_by_path: Dict[str, List[OptimizationSuggestion]] = {}
            for file_result in raw_results:
                if not isinstance(file_result, dict):
                    continue
                file_path = str(file_result.get("file_path") or "").strip()
                if file_path not in code_by_path:
                    continue

                raw_suggestions = file_result.get("suggestions", [])
                if not isinstance(raw_suggestions, list):
                    raw_suggestions = []

                mapped: List[OptimizationSuggestion] = []
                for idx, suggestion_data in enumerate(raw_suggestions[:limit]):
                    if not isinstance(suggestion_data, dict):
                        continue
                    mapped.append(
                        OptimizationSuggestion(
                            suggestion_id=suggestion_data.get("suggestion_id", f"batch_suggestion_{idx}"),
                            category=suggestion_data.get("category", "code_quality_optimization"),
                            priority=suggestion_data.get("priority", "medium"),
                            description=suggestion_data.get("description", ""),
                            original_code=code_by_path[file_path],
                            optimized_code=suggestion_data.get("optimized_code", ""),
                            expected_improvement=suggestion_data.get("expected_improvement", ""),
                            expected_runtime_delta_pct=self._float(
                                suggestion_data.get("expected_runtime_delta_pct"), 0.0
                            ),
                            expected_memory_delta_pct=self._float(
                                suggestion_data.get("expected_memory_delta_pct"), 0.0
                            ),
                            implementation_effort=suggestion_data.get("implementation_effort", "medium"),
                            confidence=self._float(suggestion_data.get("confidence"), 0.5),
                            reasoning=suggestion_data.get("reasoning", ""),
                            model_trace={**trace, "batch_mode": True},
                        )
                    )

                suggestions_by_path[file_path] = mapped

            return suggestions_by_path
        except Exception as exc:
            logger.error("Repository optimization generation failed", error=str(exc))
            return {}

    async def batch_analyze(self, code_files: List[Tuple[str, str, str]]) -> List[CodeAnalysisResult]:
        tasks = [self.analyze_code(code, file_path, identifier) for code, file_path, identifier in code_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        analysis_results: List[CodeAnalysisResult] = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Batch analysis failed", file=code_files[idx][1], error=str(result))
            else:
                analysis_results.append(result)
        return analysis_results


code_analyzer = CodeAnalyzerLLM()
