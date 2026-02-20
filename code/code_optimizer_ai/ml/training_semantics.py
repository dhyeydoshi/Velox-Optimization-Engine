"""
Shared production semantics for RL optimizer state vectors and reward.

This module is the canonical source for:
- action catalog used by policy/action indices
- runtime+memory weighted score
- production 27-dim state vector used by RLCodeOptimizer
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np

PRODUCTION_ACTION_TYPES: List[str] = [
    "algorithm_change",
    "data_structure_optimization",
    "caching_strategy",
    "parallelization",
    "memory_optimization",
    "io_optimization",
    "database_optimization",
    "network_optimization",
    "security_optimization",
    "code_quality_optimization",
    "function_optimization",
    "string_optimization",
    "mathematical_optimization",
    "loop_optimization",
    "exception_handling_optimization",
    "import_optimization",
    "configuration_optimization",
    "api_optimization",
    "concurrency_optimization",
    "resource_management_optimization",
    "logging_optimization",
    "testing_optimization",
    "build_optimization",
    "documentation_optimization",
    "no_change",
]

def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))

def normalize_objective_weights(
    objective_weights: Optional[Dict[str, float]],
    *,
    default_runtime_weight: float = 0.5,
    default_memory_weight: float = 0.5,
) -> Dict[str, float]:
    runtime = default_runtime_weight
    memory = default_memory_weight

    if objective_weights:
        runtime = float(objective_weights.get("runtime", runtime))
        memory = float(objective_weights.get("memory", memory))

    total = runtime + memory
    if total <= 0:
        return {"runtime": 0.5, "memory": 0.5}
    return {"runtime": runtime / total, "memory": memory / total}

def weighted_score(runtime_delta_pct: float, memory_delta_pct: float, weights: Dict[str, float]) -> float:
    runtime_score = float(runtime_delta_pct) / 100.0
    memory_score = float(memory_delta_pct) / 100.0
    return runtime_score * float(weights["runtime"]) + memory_score * float(weights["memory"])

def build_optimizer_state_vector(
    *,
    complexity_score: float,
    maintainability_score: float,
    confidence_score: float,
    bottlenecks_count: int,
    opportunities_count: int,
    security_issues_count: int,
    best_practices_violations_count: int,
    recent_optimizations: Iterable[str],
    system_load_pct: float,
    action_types: Optional[List[str]] = None,
) -> np.ndarray:
    """Build the production 27-dim state vector used by RLCodeOptimizer.

    Layout (0-indexed):
      [0]  complexity_score              – overall code complexity (0-1)
      [1]  bottlenecks_fraction          – bottleneck count / 10 (coarse)
      [2]  maintainability               – maintainability index (0-1)
      [3]  inverse_confidence            – 1 - confidence (uncertainty signal)
      [4]  opportunities_fraction        – opportunity count / 5 (coarse)
      [5]  security_issues_fraction      – security issues / 5
      [6-15] recent_action_encoding      – one-hot for last 10 actions
      [16] opportunity_to_issue_ratio    – opportunities / max(bottlenecks+opps, 1)
      [17] issue_severity_ratio          – (security+violations) / max(total_issues, 1)
      [18] code_health                   – maintainability * confidence
      [19] best_practices_fraction       – violations / 10
      [20] improvement_headroom          – complexity * (1 - maintainability)
      [21] recent_success_rate           – fraction of recent actions != no_change/unknown
      [22] system_load                   – system load pct / 100
      [23] confidence                    – raw confidence score
      [24] recent_depth                  – len(recent) / 10
      [25] action_diversity              – unique recent action types / max(len, 1)
      [26] optimization_pressure         – complexity * bottleneck_frac * (1-confidence)
    """
    active_actions = action_types or PRODUCTION_ACTION_TYPES
    recent_list = list(recent_optimizations)
    state = np.zeros(27, dtype=np.float32)

    maintainability = float(maintainability_score)
    if maintainability > 1.0:
        maintainability = maintainability / 100.0

    complexity = _clamp(complexity_score)
    confidence = _clamp(confidence_score)
    maint = _clamp(maintainability)
    bottleneck_frac = _clamp(float(bottlenecks_count) / 10.0)
    opp_frac = _clamp(float(opportunities_count) / 5.0)
    sec_frac = _clamp(float(security_issues_count) / 5.0)
    violations = best_practices_violations_count
    total_issues = bottlenecks_count + opportunities_count + security_issues_count + violations

    # --- primary features (0-5) ---
    state[0] = complexity
    state[1] = bottleneck_frac
    state[2] = maint
    state[3] = 1.0 - confidence
    state[4] = opp_frac
    state[5] = sec_frac

    # --- recent action encoding (6-15) ---
    # Encode the normalised action index so the DQN can distinguish which
    # action was applied (previous encoding was always 1.0 for any valid action).
    num_actions = len(active_actions)
    action_to_idx = {name: i for i, name in enumerate(active_actions)}
    for idx, action_name in enumerate(recent_list[-10:]):
        ai = action_to_idx.get(action_name)
        state[6 + idx] = (ai + 1) / num_actions if ai is not None else 0.0

    # --- derived features (16-26) ---
    state[16] = _clamp(float(opportunities_count) / max(bottlenecks_count + opportunities_count, 1))
    state[17] = _clamp(float(security_issues_count + violations) / max(total_issues, 1))
    state[18] = _clamp(maint * confidence)
    state[19] = _clamp(float(violations) / 10.0)
    state[20] = _clamp(complexity * (1.0 - maint))
    productive = sum(1 for a in recent_list if a not in ("no_change", "unknown"))
    state[21] = _clamp(float(productive) / max(len(recent_list), 1))
    state[22] = _clamp(float(system_load_pct) / 100.0)
    state[23] = confidence
    state[24] = _clamp(float(len(recent_list)) / 10.0)
    unique_actions = len(set(recent_list)) if recent_list else 0
    state[25] = _clamp(float(unique_actions) / max(len(recent_list), 1))
    state[26] = _clamp(complexity * bottleneck_frac * (1.0 - confidence))
    return state
