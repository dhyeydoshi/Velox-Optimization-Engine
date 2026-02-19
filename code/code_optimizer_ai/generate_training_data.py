from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Ensure repository root is on path for package imports.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from code.code_optimizer_ai.config.settings import settings
from code.code_optimizer_ai.ml.training_semantics import (
    PRODUCTION_ACTION_TYPES,
    build_optimizer_state_vector,
    normalize_objective_weights,
    weighted_score,
)
from code.code_optimizer_ai.utils.logger import get_logger

logger = get_logger(__name__)

ACTION_INDEX = {name: idx for idx, name in enumerate(PRODUCTION_ACTION_TYPES)}
ACTION_ALIASES = {
    "algorithm_optimization": "algorithm_change",
    "algorithm": "algorithm_change",
    "vectorization": "loop_optimization",
    "list_comprehension": "loop_optimization",
    "add_lru_cache": "caching_strategy",
    "cache": "caching_strategy",
    "data_structure": "data_structure_optimization",
}


@dataclass
class TransitionRecord:
    state: List[float]
    action_idx: int
    reward: float
    next_state: List[float]
    done: bool


@dataclass
class BuildStats:
    input_files: int = 0
    records_read: int = 0
    transitions_written: int = 0
    records_skipped: int = 0
    synthetic_records: int = 0


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    else:
        path = path.resolve()
    return path


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def _to_normalized_complexity(raw: Any) -> float:
    value = float(raw if raw is not None else 0.5)
    if value > 1.0:
        value = value / 100.0
    return _clamp(value)


def _to_int_count(raw: Any, default: int = 0) -> int:
    if raw is None:
        return default
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return default


def _to_float(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _extract_nested(record: Dict[str, Any], *keys: str) -> Any:
    current: Any = record
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def _pct_delta(before: Any, after: Any) -> Optional[float]:
    before_val = _to_float(before)
    after_val = _to_float(after)
    if before_val is None or after_val is None:
        return None
    if before_val <= 0:
        return None
    return ((before_val - after_val) / before_val) * 100.0


def normalize_action_type(raw_action: str) -> str:
    key = (raw_action or "").strip().lower()
    if not key:
        return "no_change"
    normalized = ACTION_ALIASES.get(key, key)
    if normalized not in ACTION_INDEX:
        return "no_change"
    return normalized


def _estimate_signal_counts(context_features: Dict[str, Any], complexity: float) -> Tuple[int, int, int, int]:
    algorithm_text = str(context_features.get("algorithm_type", "")).lower()
    bottlenecks = _to_int_count(context_features.get("bottlenecks_count"), int(round(complexity * 6)))
    opportunities = _to_int_count(context_features.get("opportunities_count"), int(round(complexity * 5)))
    security_issues = _to_int_count(context_features.get("security_issues_count"), 0)
    violations = _to_int_count(
        context_features.get("best_practices_violations_count"),
        int(round(complexity * 3)),
    )

    if "recursive" in algorithm_text or "nested" in algorithm_text:
        bottlenecks = max(bottlenecks, 2)
        opportunities = max(opportunities, 2)
    return bottlenecks, opportunities, security_issues, violations


def _reward_from_record(record: Dict[str, Any], objective_weights: Dict[str, float], reward_scale: float) -> float:
    runtime_delta = _to_float(record.get("runtime_delta_pct"))
    memory_delta = _to_float(record.get("memory_delta_pct"))

    if runtime_delta is None:
        runtime_delta = _to_float(_extract_nested(record, "outcome", "runtime_delta_pct"))
    if memory_delta is None:
        memory_delta = _to_float(_extract_nested(record, "outcome", "memory_delta_pct"))

    if runtime_delta is None:
        runtime_delta = _pct_delta(
            _extract_nested(record, "outcome", "execution_time_before_ms"),
            _extract_nested(record, "outcome", "execution_time_after_ms"),
        )
    if memory_delta is None:
        memory_delta = _pct_delta(
            _extract_nested(record, "outcome", "memory_before_mb"),
            _extract_nested(record, "outcome", "memory_after_mb"),
        )

    if runtime_delta is not None or memory_delta is not None:
        return float(
            weighted_score(
                float(runtime_delta if runtime_delta is not None else 0.0),
                float(memory_delta if memory_delta is not None else 0.0),
                objective_weights,
            )
        )

    improvement_factor = record.get("improvement_factor")
    if improvement_factor is None:
        improvement_factor = _extract_nested(record, "outcome", "improvement_factor")
    if improvement_factor is not None:
        value = float(improvement_factor)
        if abs(value) <= 5:
            return float(_clamp(value, -1.0, 1.0))

    raw_reward = record.get("reward")
    if raw_reward is None:
        raw_reward = _extract_nested(record, "outcome", "reward")
    raw_reward = _to_float(raw_reward) or 0.0
    if reward_scale <= 0:
        reward_scale = 10000.0
    return float(np.tanh(float(raw_reward) / reward_scale))


def _next_state_counts_from_reward(
    reward: float,
    *,
    complexity: float,
    maintainability: float,
    bottlenecks: int,
    opportunities: int,
    security_issues: int,
    violations: int,
) -> Tuple[float, float, int, int, int, int]:
    delta = max(-0.25, min(0.25, reward))
    improved = max(0.0, delta)
    regressed = max(0.0, -delta)

    next_complexity = _clamp(complexity - (improved * 0.35) + (regressed * 0.15))
    next_maintainability = _clamp(maintainability + (improved * 0.40) - (regressed * 0.20))

    next_bottlenecks = max(0, int(round(bottlenecks * (1.0 - (improved * 0.5) + (regressed * 0.2)))))
    next_opportunities = max(0, int(round(opportunities * (1.0 - (improved * 0.4) + (regressed * 0.1)))))
    next_security_issues = max(0, int(round(security_issues * (1.0 - (improved * 0.3) + (regressed * 0.3)))))
    next_violations = max(0, int(round(violations * (1.0 - (improved * 0.35) + (regressed * 0.25)))))

    return (
        next_complexity,
        next_maintainability,
        next_bottlenecks,
        next_opportunities,
        next_security_issues,
        next_violations,
    )


def legacy_record_to_transition(
    record: Dict[str, Any],
    *,
    objective_weights: Dict[str, float],
    reward_scale: float,
    recent_optimizations: List[str],
    default_system_load_pct: float,
) -> TransitionRecord:
    context_features = record.get("context_features") or {}

    complexity = _to_normalized_complexity(
        context_features.get("complexity_score", record.get("complexity_score", 0.5))
    )
    maintainability = context_features.get("maintainability_score")
    if maintainability is None:
        maintainability = max(0.0, 1.0 - (complexity * 0.6))
    maintainability = _to_normalized_complexity(maintainability)

    confidence = (
        _extract_nested(record, "action_taken", "parameters", "confidence")
        or context_features.get("confidence_score")
        or _extract_nested(record, "metadata", "confidence_score")
        or 0.7
    )
    confidence = _clamp(float(confidence))

    system_load_pct = float(context_features.get("system_load_pct", default_system_load_pct))
    system_load_pct = _clamp(system_load_pct, 0.0, 100.0)

    bottlenecks, opportunities, security_issues, violations = _estimate_signal_counts(
        context_features,
        complexity,
    )

    raw_action = (
        (record.get("action_taken") or {}).get("type")
        or record.get("optimization_type")
        or "no_change"
    )
    action_name = normalize_action_type(str(raw_action))
    action_idx = ACTION_INDEX[action_name]

    reward = _reward_from_record(record, objective_weights, reward_scale)
    done_raw = record.get("done")
    if done_raw is None:
        done_raw = record.get("terminal")
    done = bool(done_raw) if done_raw is not None else True

    state = build_optimizer_state_vector(
        complexity_score=complexity,
        maintainability_score=maintainability,
        confidence_score=confidence,
        bottlenecks_count=bottlenecks,
        opportunities_count=opportunities,
        security_issues_count=security_issues,
        best_practices_violations_count=violations,
        recent_optimizations=recent_optimizations,
        system_load_pct=system_load_pct,
        action_types=PRODUCTION_ACTION_TYPES,
    )

    (
        next_complexity,
        next_maintainability,
        next_bottlenecks,
        next_opportunities,
        next_security_issues,
        next_violations,
    ) = _next_state_counts_from_reward(
        reward,
        complexity=complexity,
        maintainability=maintainability,
        bottlenecks=bottlenecks,
        opportunities=opportunities,
        security_issues=security_issues,
        violations=violations,
    )

    next_history = [*recent_optimizations, action_name][-10:]
    next_state = build_optimizer_state_vector(
        complexity_score=next_complexity,
        maintainability_score=next_maintainability,
        confidence_score=confidence,
        bottlenecks_count=next_bottlenecks,
        opportunities_count=next_opportunities,
        security_issues_count=next_security_issues,
        best_practices_violations_count=next_violations,
        recent_optimizations=next_history,
        system_load_pct=system_load_pct,
        action_types=PRODUCTION_ACTION_TYPES,
    )

    return TransitionRecord(
        state=state.tolist(),
        action_idx=action_idx,
        reward=float(_clamp(reward, -1.0, 1.0)),
        next_state=next_state.tolist(),
        done=done,
    )


def _iter_legacy_records(input_dir: Path) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    patterns = ("episode_*.json", "training_episode_*.json")
    files: List[Path] = []
    for pattern in patterns:
        files.extend(sorted(input_dir.glob(pattern)))

    for file_path in files:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            yield file_path, payload
        elif isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    yield file_path, item


def _synthetic_legacy_records(count: int, seed: int) -> List[Dict[str, Any]]:
    random.seed(seed)

    # Each template: (algorithm_type, action_type, complexity_range,
    #                  runtime_delta_range, memory_delta_range, neg_weight)
    # neg_weight controls how often a negative-reward variant is generated
    # (0.0 = always positive, 1.0 = always negative)
    _TEMPLATES = [
        # --- algorithm & data structure ---
        ("recursive_fibonacci", "caching_strategy",      (0.50, 0.95), (15.0, 40.0),  (-8.0, 10.0),  0.15),
        ("nested_loops",        "algorithm_change",       (0.55, 0.95), (20.0, 50.0),  (5.0, 25.0),   0.10),
        ("membership_checks",   "data_structure_optimization", (0.35, 0.80), (10.0, 35.0), (5.0, 25.0), 0.15),
        ("sorting",             "algorithm_change",       (0.40, 0.85), (15.0, 45.0),  (0.0, 15.0),   0.10),
        ("matrix_multiply",     "mathematical_optimization", (0.50, 0.90), (10.0, 30.0), (-5.0, 10.0), 0.20),

        # --- caching & memoization ---
        ("repeated_computation","caching_strategy",       (0.40, 0.85), (20.0, 50.0),  (-15.0, 5.0),  0.15),

        # --- parallelization & concurrency ---
        ("cpu_bound_batch",     "parallelization",        (0.55, 0.95), (25.0, 55.0),  (-15.0, 0.0),  0.25),
        ("async_io_serial",     "concurrency_optimization",(0.40, 0.80),(15.0, 40.0),  (-5.0, 5.0),   0.20),

        # --- memory ---
        ("large_list_build",    "memory_optimization",    (0.45, 0.90), (5.0, 15.0),   (15.0, 40.0),  0.15),
        ("object_allocation",   "memory_optimization",    (0.35, 0.75), (2.0, 10.0),   (10.0, 35.0),  0.15),

        # --- I/O ---
        ("unbuffered_file_read","io_optimization",        (0.40, 0.85), (15.0, 40.0),  (0.0, 10.0),   0.15),
        ("sync_http_calls",     "io_optimization",        (0.50, 0.90), (20.0, 50.0),  (-5.0, 5.0),   0.20),

        # --- database ---
        ("n_plus_one_query",    "database_optimization",  (0.50, 0.90), (25.0, 55.0),  (0.0, 10.0),   0.15),
        ("unindexed_lookup",    "database_optimization",  (0.45, 0.85), (15.0, 40.0),  (0.0, 5.0),    0.15),

        # --- network ---
        ("chatty_api_calls",    "network_optimization",   (0.40, 0.80), (15.0, 35.0),  (-5.0, 5.0),   0.20),

        # --- security ---
        ("sql_injection",       "security_optimization",  (0.30, 0.70), (0.0, 2.0),    (0.0, 2.0),    0.10),
        ("hardcoded_secret",    "security_optimization",  (0.25, 0.60), (0.0, 1.0),    (0.0, 1.0),    0.10),

        # --- code quality ---
        ("god_class",           "code_quality_optimization",(0.55, 0.90),(0.0, 5.0),   (0.0, 5.0),    0.20),
        ("deep_nesting",        "code_quality_optimization",(0.45, 0.85),(2.0, 10.0),  (0.0, 5.0),    0.20),

        # --- function optimization ---
        ("long_function",       "function_optimization",  (0.45, 0.85), (2.0, 8.0),    (0.0, 5.0),    0.20),

        # --- string ---
        ("string_concat_loop",  "string_optimization",    (0.35, 0.75), (10.0, 30.0),  (5.0, 20.0),   0.10),

        # --- loop ---
        ("list_processing",     "loop_optimization",      (0.40, 0.80), (10.0, 30.0),  (5.0, 15.0),   0.15),

        # --- exception handling ---
        ("broad_except",        "exception_handling_optimization", (0.30, 0.65), (0.0, 3.0), (0.0, 2.0), 0.15),

        # --- import ---
        ("star_imports",        "import_optimization",    (0.20, 0.55), (1.0, 5.0),    (0.0, 3.0),    0.15),

        # --- configuration ---
        ("hardcoded_config",    "configuration_optimization",(0.25, 0.60),(0.0, 2.0),  (0.0, 2.0),    0.15),

        # --- api ---
        ("unversioned_endpoint","api_optimization",       (0.35, 0.70), (0.0, 5.0),    (0.0, 3.0),    0.20),

        # --- resource management ---
        ("leaked_file_handle",  "resource_management_optimization",(0.35, 0.75),(0.0, 5.0),(5.0, 15.0),0.15),

        # --- logging ---
        ("excessive_logging",   "logging_optimization",   (0.25, 0.60), (3.0, 12.0),   (2.0, 8.0),    0.15),

        # --- testing ---
        ("no_unit_tests",       "testing_optimization",   (0.30, 0.70), (0.0, 2.0),    (0.0, 2.0),    0.20),

        # --- build ---
        ("slow_build_pipeline", "build_optimization",     (0.35, 0.70), (5.0, 20.0),   (0.0, 5.0),    0.20),

        # --- documentation ---
        ("undocumented_api",    "documentation_optimization",(0.20, 0.55),(0.0, 1.0),  (0.0, 1.0),    0.20),

        # --- no change (already optimal) ---
        ("well_optimized_code", "no_change",              (0.10, 0.35), (0.0, 1.0),    (0.0, 1.0),    0.80),
    ]

    rows: List[Dict[str, Any]] = []
    for _ in range(count):
        (
            algorithm_type, action_type,
            (cplx_lo, cplx_hi),
            (rt_lo, rt_hi),
            (mem_lo, mem_hi),
            neg_weight,
        ) = random.choice(_TEMPLATES)

        complexity = random.uniform(cplx_lo, cplx_hi)
        system_load = random.uniform(15.0, 85.0)
        confidence = random.uniform(0.50, 0.95)

        # Decide if this is a negative example (wrong/unhelpful action)
        is_negative = random.random() < neg_weight
        if is_negative:
            runtime_delta = random.uniform(-15.0, 2.0)
            memory_delta = random.uniform(-10.0, 2.0)
        else:
            runtime_delta = random.uniform(rt_lo, rt_hi)
            memory_delta = random.uniform(mem_lo, mem_hi)

        # Derive additional context features for richer state vectors
        bottlenecks = max(0, int(round(complexity * random.uniform(3.0, 8.0))))
        opportunities = max(0, int(round(complexity * random.uniform(2.0, 7.0))))
        security_issues = random.randint(0, 3) if "security" in action_type else 0
        violations = max(0, int(round(complexity * random.uniform(1.0, 5.0))))

        rows.append(
            {
                "context_features": {
                    "complexity_score": complexity,
                    "algorithm_type": algorithm_type,
                    "system_load_pct": system_load,
                    "maintainability_score": max(0.0, 1.0 - complexity * 0.6 + random.uniform(-0.1, 0.1)),
                    "confidence_score": confidence,
                    "bottlenecks_count": bottlenecks,
                    "opportunities_count": opportunities,
                    "security_issues_count": security_issues,
                    "best_practices_violations_count": violations,
                },
                "action_taken": {
                    "type": action_type,
                    "parameters": {"confidence": confidence},
                },
                "runtime_delta_pct": runtime_delta,
                "memory_delta_pct": memory_delta,
                "success": (runtime_delta + memory_delta) > 0,
            }
        )
    return rows


def generate_transitions_from_legacy(
    *,
    input_dir: Path,
    output_jsonl: Path,
    objective_weights: Dict[str, float],
    reward_scale: float = 10000.0,
    default_system_load_pct: float = 35.0,
    strict: bool = False,
    synthetic_samples: int = 0,
    seed: int = 42,
) -> BuildStats:
    stats = BuildStats()
    transitions: List[TransitionRecord] = []
    recent_history: List[str] = []

    patterns = ("episode_*.json", "training_episode_*.json")
    input_files: List[Path] = []
    for pattern in patterns:
        input_files.extend(sorted(input_dir.glob(pattern)))
    stats.input_files = len(input_files)

    for file_path, payload in _iter_legacy_records(input_dir):
        stats.records_read += 1
        try:
            transition = legacy_record_to_transition(
                payload,
                objective_weights=objective_weights,
                reward_scale=reward_scale,
                recent_optimizations=recent_history,
                default_system_load_pct=default_system_load_pct,
            )
            transitions.append(transition)
            action_name = PRODUCTION_ACTION_TYPES[transition.action_idx]
            recent_history = [*recent_history, action_name][-10:]
        except Exception as exc:
            stats.records_skipped += 1
            if strict:
                raise
            logger.warning(
                "Skipping legacy training record",
                file=str(file_path),
                error=str(exc),
            )

    if synthetic_samples > 0:
        for synthetic_payload in _synthetic_legacy_records(synthetic_samples, seed):
            transition = legacy_record_to_transition(
                synthetic_payload,
                objective_weights=objective_weights,
                reward_scale=reward_scale,
                recent_optimizations=recent_history,
                default_system_load_pct=default_system_load_pct,
            )
            transitions.append(transition)
            action_name = PRODUCTION_ACTION_TYPES[transition.action_idx]
            recent_history = [*recent_history, action_name][-10:]
            stats.synthetic_records += 1

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for transition in transitions:
            payload = {
                "state": transition.state,
                "action_idx": transition.action_idx,
                "reward": transition.reward,
                "next_state": transition.next_state,
                "done": transition.done,
            }
            handle.write(json.dumps(payload))
            handle.write("\n")

    stats.transitions_written = len(transitions)
    return stats


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate production-aligned transition JSONL for DQN pretraining."
    )
    parser.add_argument(
        "--input-dir",
        default=str(Path(settings.TRAINING_DATA_PATH)),
        help="Directory containing legacy episode JSON files.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=str(Path(settings.TRAINING_DATA_PATH) / "pretrain_transitions.jsonl"),
        help="Output transition JSONL path.",
    )
    parser.add_argument("--runtime-weight", type=float, default=settings.OBJECTIVE_RUNTIME_WEIGHT)
    parser.add_argument("--memory-weight", type=float, default=settings.OBJECTIVE_MEMORY_WEIGHT)
    parser.add_argument("--reward-scale", type=float, default=10000.0)
    parser.add_argument("--default-system-load-pct", type=float, default=35.0)
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=0,
        help="Add synthetic legacy records when real records are sparse.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict", action="store_true")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    input_dir = _resolve(args.input_dir)
    output_jsonl = _resolve(args.output_jsonl)
    input_dir.mkdir(parents=True, exist_ok=True)

    objective_weights = normalize_objective_weights(
        {"runtime": args.runtime_weight, "memory": args.memory_weight},
        default_runtime_weight=settings.OBJECTIVE_RUNTIME_WEIGHT,
        default_memory_weight=settings.OBJECTIVE_MEMORY_WEIGHT,
    )

    stats = generate_transitions_from_legacy(
        input_dir=input_dir,
        output_jsonl=output_jsonl,
        objective_weights=objective_weights,
        reward_scale=max(1.0, float(args.reward_scale)),
        default_system_load_pct=float(args.default_system_load_pct),
        strict=args.strict,
        synthetic_samples=max(0, int(args.synthetic_samples)),
        seed=int(args.seed),
    )

    if stats.transitions_written == 0:
        raise RuntimeError("No transitions generated. Check input files or add --synthetic-samples.")

    print("Generated transition dataset")
    print(f"  input_dir: {input_dir}")
    print(f"  output_jsonl: {output_jsonl}")
    print(f"  input_files: {stats.input_files}")
    print(f"  records_read: {stats.records_read}")
    print(f"  records_skipped: {stats.records_skipped}")
    print(f"  synthetic_records: {stats.synthetic_records}")
    print(f"  transitions_written: {stats.transitions_written}")


if __name__ == "__main__":
    main()
