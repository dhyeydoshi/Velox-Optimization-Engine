"""
RL-Based Code Optimizer
Combines LLM analysis with DQN strategy selection and safe suggestion validation.
"""
from __future__ import annotations

import asyncio
import difflib
import threading
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil

from code.code_optimizer_ai.config.settings import settings
from code.code_optimizer_ai.core.llm_analyzer import (
    CodeAnalysisResult,
    CodeAnalyzerLLM,
    OptimizationSuggestion,
)
from code.code_optimizer_ai.core.performance_profiler import PerformanceMetrics, performance_profiler
from code.code_optimizer_ai.core.validation_engine import ValidationResult, validation_engine
from code.code_optimizer_ai.ml.policy_manager import PolicyFeedback, PolicyManager
from code.code_optimizer_ai.ml.rl_agent import get_rl_trainer
from code.code_optimizer_ai.ml.training_semantics import (
    PRODUCTION_ACTION_TYPES,
    build_optimizer_state_vector,
    normalize_objective_weights,
    weighted_score,
)
from code.code_optimizer_ai.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizationDecision:
    """Decision made by RL agent for code optimization."""

    action_type: str
    confidence: float
    expected_runtime_delta_pct: float
    expected_memory_delta_pct: float
    reasoning: str
    parameters: Dict[str, Any]
    priority: str


@dataclass
class OptimizationContext:
    """Context for optimization decision making."""

    code_analysis: CodeAnalysisResult
    performance_baseline: Optional[PerformanceMetrics]
    recent_optimizations: List[str]
    system_load: float
    business_requirements: Dict[str, Any]
    objective_weights: Dict[str, float]


class RLCodeOptimizer:
    """RL-powered code optimization engine."""

    ACTION_TYPES: List[str] = list(PRODUCTION_ACTION_TYPES)

    def __init__(self):
        self.code_analyzer = CodeAnalyzerLLM()
        self.policy_manager = PolicyManager(get_rl_trainer().agent)
        self.validation_engine = validation_engine
        self.optimization_history: deque[Dict[str, Any]] = deque(maxlen=1000)

        # High-signal templates; non-covered actions fall back to LLM generation.
        self.optimization_templates = {
            "algorithm_change": {
                "description": "Replace inefficient algorithmic patterns",
                "patterns": [("nested loops", "hash lookup")],
            },
            "data_structure_optimization": {
                "description": "Switch to lookup-optimized structures",
                "patterns": [("list membership", "set membership")],
            },
            "caching_strategy": {
                "description": "Memoize expensive pure functions",
                "patterns": [("recursive", "lru_cache")],
            },
            "memory_optimization": {
                "description": "Reduce allocations and peak memory",
                "patterns": [("list comprehensions", "generator expressions")],
            },
            "string_optimization": {
                "description": "Avoid repeated string concatenation",
                "patterns": [("+= in loops", "join")],
            },
            "no_change": {
                "description": "Code appears sufficiently optimized for current objective",
                "patterns": [],
            },
        }

    @staticmethod
    def _normalize_weights(objective_weights: Optional[Dict[str, float]]) -> Dict[str, float]:
        return normalize_objective_weights(
            objective_weights,
            default_runtime_weight=settings.OBJECTIVE_RUNTIME_WEIGHT,
            default_memory_weight=settings.OBJECTIVE_MEMORY_WEIGHT,
        )

    @staticmethod
    def _weighted_score(runtime_delta_pct: float, memory_delta_pct: float, weights: Dict[str, float]) -> float:
        return weighted_score(runtime_delta_pct, memory_delta_pct, weights)

    async def optimize_code(
        self,
        code: str,
        file_path: str = "",
        context: Optional[OptimizationContext] = None,
        objective_weights: Optional[Dict[str, float]] = None,
        max_suggestions: Optional[int] = None,
        run_validation: bool = True,
        unit_test_command: Optional[str] = None,
        analysis_static_metrics: Optional[Dict[str, Any]] = None,
        analysis_baseline_metrics: Optional[Dict[str, Any]] = None,
    ) -> List[OptimizationSuggestion]:
        """Optimize code using LLM analysis + RL strategy selection."""

        started_at = datetime.now()
        objective = self._normalize_weights(objective_weights)
        max_count = max(1, max_suggestions or settings.DEFAULT_MAX_SUGGESTIONS)
        unit_cmd = unit_test_command or settings.UNIT_TEST_COMMAND

        analysis_result = await self.code_analyzer.analyze_code(
            code,
            file_path,
            static_metrics=analysis_static_metrics,
            baseline_metrics=analysis_baseline_metrics,
        )
        if not context:
            context = await self._create_optimization_context(analysis_result, objective)

        state_vector = self._analysis_to_state_vector(analysis_result, context)
        decision = await self._get_rl_optimization_decision(analysis_result, context, state_vector)

        if decision.action_type == "no_change":
            self._store_optimization_history(file_path, analysis_result, decision, [])
            return []

        suggestions = await self._generate_optimization_suggestions(code, analysis_result, decision)
        enriched = await self._enrich_suggestions(
            original_code=code,
            file_path=file_path,
            suggestions=suggestions,
            decision=decision,
            objective=objective,
            run_validation=run_validation,
            unit_test_command=unit_cmd,
        )

        ranked = await self._validate_and_rank_suggestions(enriched, max_count)
        self._store_optimization_history(file_path, analysis_result, decision, ranked)

        reward = ranked[0].expected_weighted_score if ranked else 0.0
        self._record_shadow_feedback(state_vector, decision.action_type, reward)

        elapsed = (datetime.now() - started_at).total_seconds()
        logger.info(
            "optimization_completed",
            file_path=file_path,
            suggestions=len(ranked),
            action_type=decision.action_type,
            latency_seconds=round(elapsed, 3),
        )
        return ranked

    async def _create_optimization_context(
        self,
        analysis: CodeAnalysisResult,
        objective_weights: Dict[str, float],
    ) -> OptimizationContext:
        performance_summary = performance_profiler.get_performance_summary()
        baseline_metrics = None

        if "error" not in performance_summary:
            context_metrics = performance_summary.get("summary_by_context", {}).get("system_monitoring", {})
            if context_metrics:
                baseline_metrics = PerformanceMetrics(
                    execution_time=context_metrics.get("avg_execution_time", 0),
                    cpu_usage=context_metrics.get("avg_cpu_usage", 0),
                    memory_usage=context_metrics.get("avg_memory_usage", 0),
                    peak_memory=context_metrics.get("max_memory_usage", 0),
                    function_calls=0,
                    unique_functions=0,
                    cpu_time=0,
                    wall_time=0,
                    timestamp=datetime.now(),
                    context="baseline",
                )

        recent = [item.get("decision", {}).get("action_type", "unknown") for item in list(self.optimization_history)[-10:]]

        business_requirements = {
            "performance_priority": "high",
            "development_speed": "medium",
            "maintainability_weight": 0.2,
            "performance_weight": 0.8,
            "implementation_effort_weight": 0.2,
        }

        system_load = await asyncio.to_thread(psutil.cpu_percent, 0.1)

        return OptimizationContext(
            code_analysis=analysis,
            performance_baseline=baseline_metrics,
            recent_optimizations=recent,
            system_load=system_load,
            business_requirements=business_requirements,
            objective_weights=objective_weights,
        )

    async def _get_rl_optimization_decision(
        self,
        analysis: CodeAnalysisResult,
        context: OptimizationContext,
        state_vector: np.ndarray,
    ) -> OptimizationDecision:
        recommendation = self.policy_manager.recommend_action(state_vector, self.ACTION_TYPES)

        action_type = recommendation.get("recommended_action", "code_quality_optimization")
        if action_type not in self.ACTION_TYPES:
            action_type = "code_quality_optimization"

        confidence = float(recommendation.get("confidence", 0.5))
        runtime_delta, memory_delta = self._estimate_expected_improvements(action_type, analysis)
        reasoning = self._generate_optimization_reasoning(action_type, analysis, context)
        parameters = self._get_optimization_parameters(action_type, analysis, context)

        weighted = self._weighted_score(runtime_delta, memory_delta, context.objective_weights)
        priority = self._determine_priority(confidence, weighted, context)

        return OptimizationDecision(
            action_type=action_type,
            confidence=confidence,
            expected_runtime_delta_pct=runtime_delta,
            expected_memory_delta_pct=memory_delta,
            reasoning=reasoning,
            parameters=parameters,
            priority=priority,
        )

    def _analysis_to_state_vector(self, analysis: CodeAnalysisResult, context: OptimizationContext) -> np.ndarray:
        return build_optimizer_state_vector(
            complexity_score=analysis.complexity_score,
            maintainability_score=analysis.maintainability_score,
            confidence_score=analysis.confidence_score,
            bottlenecks_count=len(analysis.performance_bottlenecks),
            opportunities_count=len(analysis.optimization_opportunities),
            security_issues_count=len(analysis.security_issues),
            best_practices_violations_count=len(analysis.best_practices_violations),
            recent_optimizations=context.recent_optimizations,
            system_load_pct=context.system_load,
            action_types=self.ACTION_TYPES,
        )

    def _estimate_expected_improvements(self, action_type: str, analysis: CodeAnalysisResult) -> Tuple[float, float]:
        base = {
            "algorithm_change": (30.0, 10.0),
            "data_structure_optimization": (20.0, 20.0),
            "caching_strategy": (25.0, -5.0),
            "parallelization": (35.0, -10.0),
            "memory_optimization": (8.0, 30.0),
            "io_optimization": (20.0, 5.0),
            "loop_optimization": (18.0, 10.0),
            "string_optimization": (12.0, 12.0),
            "no_change": (0.0, 0.0),
        }
        runtime, memory = base.get(action_type, (10.0, 5.0))

        if "nested" in " ".join(analysis.performance_bottlenecks).lower() and action_type == "algorithm_change":
            runtime += 10.0
        if action_type == "memory_optimization" and analysis.complexity_score > 0.7:
            memory += 5.0
        return runtime, memory

    def _generate_optimization_reasoning(
        self,
        action_type: str,
        analysis: CodeAnalysisResult,
        context: OptimizationContext,
    ) -> str:
        pieces = [f"Selected action: {action_type}"]
        if analysis.performance_bottlenecks:
            pieces.append("Bottlenecks: " + ", ".join(analysis.performance_bottlenecks[:3]))
        if analysis.optimization_opportunities:
            pieces.append("Opportunities: " + ", ".join(analysis.optimization_opportunities[:3]))
        pieces.append(
            f"Objective runtime={context.objective_weights['runtime']:.2f}, memory={context.objective_weights['memory']:.2f}"
        )
        return " | ".join(pieces)

    def _get_optimization_parameters(
        self,
        action_type: str,
        analysis: CodeAnalysisResult,
        context: OptimizationContext,
    ) -> Dict[str, Any]:
        return {
            "intensity": 0.8 if analysis.complexity_score > 0.7 else 0.5,
            "risk_level": 0.3,
            "system_load": context.system_load,
            "objective": context.objective_weights,
            "action_type": action_type,
        }

    def _determine_priority(
        self,
        confidence: float,
        weighted_score: float,
        context: OptimizationContext,
    ) -> str:
        score = (confidence * 0.4) + (max(weighted_score, 0) * 0.6)
        if context.system_load > 80:
            score *= 1.1

        if score >= 0.7:
            return "high"
        if score >= 0.4:
            return "medium"
        return "low"

    async def _generate_optimization_suggestions(
        self,
        code: str,
        analysis: CodeAnalysisResult,
        decision: OptimizationDecision,
    ) -> List[OptimizationSuggestion]:
        action_type = decision.action_type
        if action_type == "no_change":
            return []

        heuristic_code = self._apply_heuristic_transform(code, action_type)
        if heuristic_code and heuristic_code != code:
            return [
                OptimizationSuggestion(
                    suggestion_id=f"{action_type}_{int(datetime.now().timestamp())}",
                    category=action_type,
                    priority=decision.priority,
                    description=f"Heuristic optimization for {action_type}",
                    original_code=code,
                    optimized_code=heuristic_code,
                    expected_improvement=(
                        f"runtime +{decision.expected_runtime_delta_pct:.1f}%, "
                        f"memory +{decision.expected_memory_delta_pct:.1f}%"
                    ),
                    expected_runtime_delta_pct=decision.expected_runtime_delta_pct,
                    expected_memory_delta_pct=decision.expected_memory_delta_pct,
                    implementation_effort=self._assess_implementation_effort(action_type),
                    confidence=decision.confidence,
                    reasoning=decision.reasoning,
                )
            ]

        llm_suggestions = await self.code_analyzer.generate_optimizations(code, analysis, action_type)
        for suggestion in llm_suggestions:
            if suggestion.category not in self.ACTION_TYPES:
                suggestion.category = action_type
            if suggestion.priority not in {"low", "medium", "high", "critical"}:
                suggestion.priority = decision.priority
            if suggestion.expected_runtime_delta_pct is None:
                suggestion.expected_runtime_delta_pct = decision.expected_runtime_delta_pct
            if suggestion.expected_memory_delta_pct is None:
                suggestion.expected_memory_delta_pct = decision.expected_memory_delta_pct
        return llm_suggestions

    def _apply_heuristic_transform(self, code: str, action_type: str) -> str:
        if action_type == "data_structure_optimization":
            return self._optimize_list_membership(code)
        if action_type == "string_optimization":
            return self._optimize_string_concat(code)
        if action_type == "caching_strategy":
            return self._optimize_recursive_cache(code)
        return code

    def _optimize_list_membership(self, code: str) -> str:
        """Replace list-literal assignments with sets when safe for membership tests.

        Only transforms single-line ``NAME = [literal, ...]`` where:
        - the name appears later in an ``in NAME`` membership test,
        - elements are simple literals (str/int/float/bool/None),
        - there are no duplicates (set would silently change semantics).
        """
        if " in " not in code:
            return code

        import ast as _ast

        try:
            tree = _ast.parse(code)
        except SyntaxError:
            return code

        # Collect names used in ``x in NAME`` comparisons.
        membership_targets: set[str] = set()
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Compare):
                for op, comparator in zip(node.ops, node.comparators):
                    if isinstance(op, _ast.In) and isinstance(comparator, _ast.Name):
                        membership_targets.add(comparator.id)

        if not membership_targets:
            return code

        lines = code.split("\n")
        changed = False

        for node in _ast.walk(tree):
            if not isinstance(node, _ast.Assign):
                continue
            if len(node.targets) != 1 or not isinstance(node.targets[0], _ast.Name):
                continue
            name = node.targets[0].id
            if name not in membership_targets:
                continue
            value = node.value
            if not isinstance(value, _ast.List) or not value.elts:
                continue
            # Only allow simple hashable constants.
            if not all(isinstance(elt, _ast.Constant) for elt in value.elts):
                continue
            raw_values = [elt.value for elt in value.elts]  # type: ignore[union-attr]
            if len(raw_values) != len(set(raw_values)):
                continue  # duplicates -- set would change semantics

            # Safe to convert: replace ``[`` with ``{`` and ``]`` with ``}``
            start_line = node.lineno - 1  # AST lines are 1-based
            end_line = getattr(node, "end_lineno", node.lineno) - 1

            if start_line == end_line:
                # Single-line list
                line = lines[start_line]
                first_bracket = line.find("[")
                last_bracket = line.rfind("]")
                if first_bracket == -1 or last_bracket == -1 or last_bracket <= first_bracket:
                    continue
                lines[start_line] = (
                    line[:first_bracket] + "{"
                    + line[first_bracket + 1:last_bracket]
                    + "}" + line[last_bracket + 1:]
                )
            else:
                # Multi-line list: replace [ on first line, ] on last line
                first_line = lines[start_line]
                last_line = lines[end_line]
                fb = first_line.find("[")
                lb = last_line.rfind("]")
                if fb == -1 or lb == -1:
                    continue
                lines[start_line] = first_line[:fb] + "{" + first_line[fb + 1:]
                lines[end_line] = last_line[:lb] + "}" + last_line[lb + 1:]
            changed = True

        return "\n".join(lines) if changed else code

    def _optimize_string_concat(self, code: str) -> str:
        """Replace string += accumulation inside for-loops with list + join.

        Detects the pattern::

            name = ""
            for <target> in <iter>:
                name += <expr>

        and rewrites to list accumulation + ``"".join(...)``.
        Only transforms when the loop body is a single ``+=`` augmented assign.
        """
        if "+=" not in code:
            return code

        import ast as _ast

        try:
            tree = _ast.parse(code)
        except SyntaxError:
            return code

        # Map line-number → AST node for top-level statements.
        body = tree.body
        if not body:
            return code

        lines = code.split("\n")
        # Collect (init_line_idx, loop_node, target_name, loop_start, loop_end) candidates.
        rewrites: List[tuple] = []  # (init_line_0based, loop_node, target_var)

        for i, node in enumerate(body):
            # Look for: name = "" (or name = '') immediately before a For loop.
            if not isinstance(node, _ast.Assign):
                continue
            if len(node.targets) != 1 or not isinstance(node.targets[0], _ast.Name):
                continue
            if not isinstance(node.value, _ast.Constant) or node.value.value != "":
                continue
            var_name = node.targets[0].id

            # Next statement must be a For loop.
            if i + 1 >= len(body):
                continue
            loop_node = body[i + 1]
            if not isinstance(loop_node, _ast.For):
                continue
            # Loop body must be exactly one AugAssign: var_name += <expr>
            if len(loop_node.body) != 1:
                continue
            aug = loop_node.body[0]
            if not isinstance(aug, _ast.AugAssign):
                continue
            if not isinstance(aug.op, _ast.Add):
                continue
            if not isinstance(aug.target, _ast.Name) or aug.target.id != var_name:
                continue

            rewrites.append((node.lineno - 1, loop_node, var_name))

        if not rewrites:
            return code

        # Apply rewrites bottom-up to keep line indices stable.
        for init_line_0, loop_node, var_name in reversed(rewrites):
            loop_start = loop_node.lineno - 1  # 0-based
            loop_end = loop_node.end_lineno  # 1-based exclusive
            if loop_end is None:
                continue

            # Extract indentation from the for-line.
            for_line = lines[loop_start]
            indent = for_line[: len(for_line) - len(for_line.lstrip())]
            body_indent = indent + "    "

            # Build the += expression source (the RHS of the AugAssign).
            aug_node = loop_node.body[0]
            rhs_source = _ast.get_source_segment(code, aug_node.value)
            if not rhs_source:
                continue

            # Reconstruct the loop target and iter as source.
            loop_target_src = _ast.get_source_segment(code, loop_node.target)
            loop_iter_src = _ast.get_source_segment(code, loop_node.iter)
            if not loop_target_src or not loop_iter_src:
                continue

            parts_var = f"_{var_name}_parts"
            new_lines = [
                f"{indent}{parts_var} = []",
                f"{indent}for {loop_target_src} in {loop_iter_src}:",
                f"{body_indent}{parts_var}.append({rhs_source})",
                f"{indent}{var_name} = \"\".join({parts_var})",
            ]

            # Replace init line + loop lines.
            lines[init_line_0:loop_end] = new_lines

        return "\n".join(lines)

    def _optimize_recursive_cache(self, code: str) -> str:
        """Add ``@lru_cache`` only to genuinely recursive, cache-safe functions.

        Safety checks:
        1. Function must call itself (recursive).
        2. No mutable default parameter values (list/dict/set literals).
        3. No ``global`` or ``nonlocal`` statements in the body.
        4. ``@lru_cache`` not already present.
        """
        if "def " not in code or "return" not in code:
            return code
        if "@lru_cache" in code:
            return code

        import ast as _ast

        try:
            tree = _ast.parse(code)
        except SyntaxError:
            return code

        _MUTABLE_TYPES = (_ast.List, _ast.Dict, _ast.Set)

        def _is_recursive(func_node: _ast.FunctionDef) -> bool:
            """Return True if *func_node* contains a call to itself."""
            for child in _ast.walk(func_node):
                if isinstance(child, _ast.Call):
                    callee = child.func
                    if isinstance(callee, _ast.Name) and callee.id == func_node.name:
                        return True
            return False

        def _has_mutable_defaults(func_node: _ast.FunctionDef) -> bool:
            for default in func_node.args.defaults + func_node.args.kw_defaults:
                if default is not None and isinstance(default, _MUTABLE_TYPES):
                    return True
            return False

        def _uses_global_nonlocal(func_node: _ast.FunctionDef) -> bool:
            for child in _ast.walk(func_node):
                if isinstance(child, (_ast.Global, _ast.Nonlocal)):
                    return True
            return False

        # Find the first eligible recursive function.
        target_func: _ast.FunctionDef | None = None
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.FunctionDef):
                continue
            if not _is_recursive(node):
                continue
            if _has_mutable_defaults(node):
                continue
            if _uses_global_nonlocal(node):
                continue
            target_func = node
            break

        if target_func is None:
            return code

        # Collect ALL eligible recursive functions.
        eligible: list[_ast.FunctionDef] = []
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.FunctionDef):
                continue
            if not _is_recursive(node):
                continue
            if _has_mutable_defaults(node):
                continue
            if _uses_global_nonlocal(node):
                continue
            eligible.append(node)

        if not eligible:
            return code

        lines = code.split("\n")

        # Process from bottom to top so inserted lines don't shift later indices.
        for func_node in sorted(eligible, key=lambda n: n.lineno, reverse=True):
            func_line_idx = func_node.lineno - 1
            func_indent = lines[func_line_idx][: len(lines[func_line_idx]) - len(lines[func_line_idx].lstrip())]
            lines.insert(func_line_idx, f"{func_indent}@lru_cache(maxsize=128)")

        # Add import if not already present.
        if "from functools import lru_cache" not in code and "import functools" not in code:
            lines.insert(0, "from functools import lru_cache")

        return "\n".join(lines)

    @staticmethod
    def _build_patch_diff(original_code: str, optimized_code: str, file_path: str) -> str:
        from_name = f"a/{file_path or 'inline_code.py'}"
        to_name = f"b/{file_path or 'inline_code.py'}"
        diff = difflib.unified_diff(
            original_code.splitlines(),
            optimized_code.splitlines(),
            fromfile=from_name,
            tofile=to_name,
            lineterm="",
        )
        return "\n".join(diff)

    async def _enrich_suggestions(
        self,
        *,
        original_code: str,
        file_path: str,
        suggestions: List[OptimizationSuggestion],
        decision: OptimizationDecision,
        objective: Dict[str, float],
        run_validation: bool,
        unit_test_command: str,
    ) -> List[OptimizationSuggestion]:
        enriched: List[OptimizationSuggestion] = []

        for suggestion in suggestions:
            if not suggestion.optimized_code or suggestion.optimized_code == original_code:
                continue

            suggestion.patch_diff = self._build_patch_diff(original_code, suggestion.optimized_code, file_path)
            validation: Optional[ValidationResult] = None
            if run_validation:
                import asyncio as _aio
                validation = await _aio.to_thread(
                    self.validation_engine.validate_candidate,
                    original_code=original_code,
                    candidate_code=suggestion.optimized_code,
                    file_path=file_path,
                    run_unit_tests=True,
                    unit_test_command=unit_test_command,
                )
                suggestion.validation_status = validation.status
            else:
                suggestion.validation_status = "not_validated"

            runtime_delta = suggestion.expected_runtime_delta_pct or decision.expected_runtime_delta_pct
            memory_delta = suggestion.expected_memory_delta_pct or decision.expected_memory_delta_pct

            if validation and validation.status == "passed":
                runtime_delta = validation.runtime_delta_pct or runtime_delta
                memory_delta = validation.memory_delta_pct or memory_delta

            suggestion.expected_runtime_delta_pct = runtime_delta
            suggestion.expected_memory_delta_pct = memory_delta
            suggestion.expected_weighted_score = self._weighted_score(runtime_delta, memory_delta, objective)

            trace = dict(suggestion.model_trace or {})
            trace["decision"] = asdict(decision)
            trace["objective_weights"] = objective
            if validation:
                trace["validation"] = asdict(validation)
            suggestion.model_trace = trace
            enriched.append(suggestion)

        return enriched

    async def _validate_and_rank_suggestions(
        self,
        suggestions: List[OptimizationSuggestion],
        max_count: int,
    ) -> List[OptimizationSuggestion]:
        if not suggestions:
            return []

        def score(item: OptimizationSuggestion) -> Tuple[int, float, float]:
            status_score = 1 if item.validation_status == "passed" else 0
            weighted = item.expected_weighted_score if item.expected_weighted_score is not None else -1.0
            return (status_score, weighted, item.confidence)

        ranked = [s for s in suggestions if s.confidence >= 0.3]
        ranked.sort(key=score, reverse=True)
        return ranked[:max_count]

    def _assess_implementation_effort(self, action_type: str) -> str:
        effort_mapping = {
            "algorithm_change": "high",
            "data_structure_optimization": "medium",
            "caching_strategy": "medium",
            "parallelization": "high",
            "memory_optimization": "low",
            "io_optimization": "medium",
            "string_optimization": "low",
            "loop_optimization": "low",
            "no_change": "low",
        }
        return effort_mapping.get(action_type, "medium")

    def _record_shadow_feedback(self, state: np.ndarray, action_type: str, reward: Optional[float]):
        action_idx = self.ACTION_TYPES.index(action_type) if action_type in self.ACTION_TYPES else 0
        value = float(reward) if reward is not None else 0.0
        feedback = PolicyFeedback(
            state=state,
            action_idx=action_idx,
            reward=value,
            next_state=np.zeros_like(state),   # terminal -- no successor
            done=True,
        )
        self.policy_manager.record_shadow_feedback(feedback)
        self.policy_manager.train_shadow(max_updates=5)

    def _store_optimization_history(
        self,
        file_path: str,
        analysis: CodeAnalysisResult,
        decision: OptimizationDecision,
        suggestions: List[OptimizationSuggestion],
    ):
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "file_path": file_path,
            "analysis": asdict(analysis),
            "decision": asdict(decision),
            "suggestions_count": len(suggestions),
            "suggestion_ids": [s.suggestion_id for s in suggestions],
        }
        self.optimization_history.append(history_entry)


_rl_optimizer_instance: Optional["RLCodeOptimizer"] = None
_rl_optimizer_lock = threading.Lock()


def get_rl_optimizer() -> "RLCodeOptimizer":
    """Return the module-level RLCodeOptimizer singleton (lazy, thread-safe)."""
    global _rl_optimizer_instance
    with _rl_optimizer_lock:
        if _rl_optimizer_instance is None:
            _rl_optimizer_instance = RLCodeOptimizer()
    return _rl_optimizer_instance
