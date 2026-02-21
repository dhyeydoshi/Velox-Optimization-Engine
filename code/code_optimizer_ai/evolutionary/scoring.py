from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Optional

from code.code_optimizer_ai.config.evolutionary import (
    CV_PENALTY_COEFFICIENT,
    SIZE_PENALTY_COEFFICIENT,
    SYNTHETIC_FALLBACK_SCORE_PENALTY,
)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def normalize_delta_pct(delta_pct: Optional[float]) -> float:
    if delta_pct is None:
        return 0.0
    return _clamp(float(delta_pct) / 100.0, low=-1.0, high=1.0)


def compute_code_size_delta_norm(original_code: str, candidate_code: str) -> float:
    baseline = max(1, len(original_code or ""))
    candidate = len(candidate_code or "")
    return _clamp((candidate - baseline) / baseline, low=-1.0, high=1.0)


def _cyclomatic_complexity(code: str) -> float:
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return 100.0

    complexity = 1
    for node in ast.walk(tree):
        if isinstance(
            node,
            (
                ast.If,
                ast.For,
                ast.While,
                ast.AsyncFor,
                ast.ExceptHandler,
                ast.And,
                ast.Or,
                ast.comprehension,
            ),
        ):
            complexity += 1
    return float(complexity)


def _comment_density(code: str) -> float:
    lines = [line for line in (code or "").splitlines() if line.strip()]
    if not lines:
        return 0.0
    comment_lines = sum(1 for line in lines if line.lstrip().startswith("#"))
    return comment_lines / len(lines)


def _readability_score(code: str) -> float:
    lines = [line for line in (code or "").splitlines() if line.strip()]
    if not lines:
        return 0.0

    avg_line_len = sum(len(line.rstrip()) for line in lines) / len(lines)
    avg_indent = sum(len(line) - len(line.lstrip(" ")) for line in lines) / len(lines)

    line_length_component = _clamp(1.0 - max(0.0, (avg_line_len - 88.0) / 88.0))
    nesting_component = _clamp(1.0 - max(0.0, (avg_indent / 4.0 - 3.0) / 6.0))
    return _clamp((line_length_component * 0.65) + (nesting_component * 0.35))


def static_quality_score(code: str) -> float:
    """
    Deterministic Phase A static quality score in [0, 1].

    Components and fixed weights:
    - readability (0.45)
    - cyclomatic complexity quality (0.40)
    - comment density (0.15)
    """
    readability = _readability_score(code)
    cyclomatic = _cyclomatic_complexity(code)
    cyclomatic_quality = _clamp(1.0 - min(cyclomatic / 25.0, 1.0))
    comment_quality = _clamp(_comment_density(code) / 0.20)
    return _clamp((readability * 0.45) + (cyclomatic_quality * 0.40) + (comment_quality * 0.15))


@dataclass(frozen=True)
class ScoreBreakdown:
    runtime_delta_norm: float
    memory_delta_norm: float
    static_quality_score: float
    cv: float
    code_size_delta_norm: float
    composite_score: float


def compute_composite_score(
    *,
    runtime_delta_pct: Optional[float],
    memory_delta_pct: Optional[float],
    static_quality: float,
    cv: Optional[float],
    code_size_delta_norm: float,
    runtime_weight: float,
    memory_weight: float,
    quality_weight: float,
    unit_tests_passed: Optional[bool],
    synthetic_fallback: bool = False,
) -> ScoreBreakdown:
    """
    PA-03 composite score formula:
      score = (runtime_weight * runtime_delta_norm)
            + (memory_weight * memory_delta_norm)
            + (quality_weight * static_quality_score)
            - (CV_PENALTY_COEFFICIENT * cv)
            - (SIZE_PENALTY_COEFFICIENT * code_size_delta_norm)
    """
    if unit_tests_passed is False:
        return ScoreBreakdown(
            runtime_delta_norm=normalize_delta_pct(runtime_delta_pct),
            memory_delta_norm=normalize_delta_pct(memory_delta_pct),
            static_quality_score=_clamp(static_quality),
            cv=float(cv or 0.0),
            code_size_delta_norm=_clamp(code_size_delta_norm, low=-1.0, high=1.0),
            composite_score=-1.0,
        )

    runtime_delta_norm = normalize_delta_pct(runtime_delta_pct)
    memory_delta_norm = normalize_delta_pct(memory_delta_pct)
    quality_score = _clamp(static_quality)
    cv_value = _clamp(float(cv or 0.0), low=0.0, high=5.0)
    size_norm = _clamp(code_size_delta_norm, low=-1.0, high=1.0)

    score = (
        (float(runtime_weight) * runtime_delta_norm)
        + (float(memory_weight) * memory_delta_norm)
        + (float(quality_weight) * quality_score)
        - (CV_PENALTY_COEFFICIENT * cv_value)
        - (SIZE_PENALTY_COEFFICIENT * size_norm)
    )
    if synthetic_fallback:
        score -= SYNTHETIC_FALLBACK_SCORE_PENALTY

    return ScoreBreakdown(
        runtime_delta_norm=runtime_delta_norm,
        memory_delta_norm=memory_delta_norm,
        static_quality_score=quality_score,
        cv=cv_value,
        code_size_delta_norm=size_norm,
        composite_score=float(score),
    )
