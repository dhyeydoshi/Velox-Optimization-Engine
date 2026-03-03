from __future__ import annotations

from typing import List, Tuple

from code.code_optimizer_ai.config.evolutionary import (
    LLM_CALL_OVERHEAD_MS,
    PLATEAU_ABSOLUTE_FLOOR,
    PLATEAU_RATIO_THRESHOLD,
    REFINEMENT_BUDGET_SAFETY_MARGIN,
)
from code.code_optimizer_ai.evolutionary.constants import StoppingReason
from code.code_optimizer_ai.evolutionary.tier_config import TierConfig


def _relative_ratio(delta: float, prior_deltas: List[float]) -> float:
    positive_priors = [value for value in prior_deltas if value > 0]
    if not positive_priors:
        return 0.0
    baseline = max(positive_priors)
    if baseline <= 0:
        return 0.0
    return delta / baseline


def should_stop_generation(
    generation_scores: List[float],
    elapsed_ms: float,
    tier_config: TierConfig,
    avg_candidate_eval_ms: float,
    offspring_count: int,
) -> Tuple[bool, str]:
    generation_count = len(generation_scores or [])
    if generation_count >= int(tier_config.max_generations):
        return True, StoppingReason.MAX_GENERATIONS.value

    remaining_ms = float(tier_config.time_budget_ms) - float(elapsed_ms)
    projected_cost = (
        float(LLM_CALL_OVERHEAD_MS) + max(0, int(offspring_count)) * max(0.0, float(avg_candidate_eval_ms))
    ) * float(REFINEMENT_BUDGET_SAFETY_MARGIN)
    if remaining_ms < projected_cost:
        return True, StoppingReason.TIME_BUDGET.value

    # Need at least 3 generation scores to evaluate two consecutive deltas.
    if generation_count < 3:
        return False, ""

    deltas = [generation_scores[idx] - generation_scores[idx - 1] for idx in range(1, generation_count)]
    last_delta = deltas[-1]
    prev_delta = deltas[-2]

    if last_delta < float(PLATEAU_ABSOLUTE_FLOOR) and prev_delta < float(PLATEAU_ABSOLUTE_FLOOR):
        return True, StoppingReason.PLATEAU.value

    ratio_last = _relative_ratio(last_delta, deltas[:-1])
    ratio_prev = _relative_ratio(prev_delta, deltas[:-2]) if len(deltas) > 2 else 0.0
    if ratio_last < float(PLATEAU_RATIO_THRESHOLD) and ratio_prev < float(PLATEAU_RATIO_THRESHOLD):
        return True, StoppingReason.PLATEAU.value

    return False, ""
