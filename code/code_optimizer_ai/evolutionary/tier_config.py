from __future__ import annotations

from dataclasses import dataclass

from code.code_optimizer_ai.config.evolutionary import (
    OFFSPRING_PER_GENERATION,
    PARENTS_PER_GENERATION,
    TIER_DEEP_MAX_GENERATIONS,
    TIER_DEEP_POP_MAX,
    TIER_DEEP_POP_MIN,
    TIER_DEEP_TIME_BUDGET_MS,
    TIER_EXTENDED_MAX_GENERATIONS,
    TIER_EXTENDED_POP_MAX,
    TIER_EXTENDED_POP_MIN,
    TIER_EXTENDED_TIME_BUDGET_MS,
    TIER_STANDARD_MAX_GENERATIONS,
    TIER_STANDARD_POP_MAX,
    TIER_STANDARD_POP_MIN,
    TIER_STANDARD_TIME_BUDGET_MS,
)
from code.code_optimizer_ai.evolutionary.constants import OptimizationTier


@dataclass(frozen=True)
class TierConfig:
    time_budget_ms: int
    population_size_min: int
    population_size_max: int
    max_generations: int
    offspring_per_generation: int
    parents_per_generation: int
    enable_text_feedback: bool
    enable_diversity_enforcement: bool
    enable_novelty_weighting: bool
    enable_prompt_template_selection: bool


TIER_CONFIGS = {
    OptimizationTier.STANDARD.value: TierConfig(
        time_budget_ms=TIER_STANDARD_TIME_BUDGET_MS,
        population_size_min=TIER_STANDARD_POP_MIN,
        population_size_max=TIER_STANDARD_POP_MAX,
        max_generations=TIER_STANDARD_MAX_GENERATIONS,
        offspring_per_generation=2,
        parents_per_generation=2,
        enable_text_feedback=False,
        enable_diversity_enforcement=False,
        enable_novelty_weighting=False,
        enable_prompt_template_selection=False,
    ),
    OptimizationTier.EXTENDED.value: TierConfig(
        time_budget_ms=TIER_EXTENDED_TIME_BUDGET_MS,
        population_size_min=TIER_EXTENDED_POP_MIN,
        population_size_max=TIER_EXTENDED_POP_MAX,
        max_generations=TIER_EXTENDED_MAX_GENERATIONS,
        offspring_per_generation=OFFSPRING_PER_GENERATION,
        parents_per_generation=PARENTS_PER_GENERATION,
        enable_text_feedback=True,
        enable_diversity_enforcement=False,
        enable_novelty_weighting=False,
        enable_prompt_template_selection=False,
    ),
    OptimizationTier.DEEP.value: TierConfig(
        time_budget_ms=TIER_DEEP_TIME_BUDGET_MS,
        population_size_min=TIER_DEEP_POP_MIN,
        population_size_max=TIER_DEEP_POP_MAX,
        max_generations=TIER_DEEP_MAX_GENERATIONS,
        offspring_per_generation=6,
        parents_per_generation=4,
        enable_text_feedback=True,
        enable_diversity_enforcement=True,
        enable_novelty_weighting=True,
        enable_prompt_template_selection=True,
    ),
}


def validate_tier(tier_name: str) -> TierConfig:
    normalized = (tier_name or "").strip().lower()
    config = TIER_CONFIGS.get(normalized)
    if config is None:
        valid = ", ".join(sorted(TIER_CONFIGS.keys()))
        raise ValueError(f"Invalid optimization_tier '{tier_name}'. Valid values: {valid}")
    return config
