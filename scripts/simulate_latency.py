from __future__ import annotations

import argparse
import statistics
import time

from code.code_optimizer_ai.core.validation_engine import ValidationEngine


SAMPLE_ORIGINAL = """
def slow_sum(items):
    total = 0
    for item in items:
        total += item
    return total
"""

SAMPLE_CANDIDATE = """
def slow_sum(items):
    return sum(items)
"""


def run_simulation(iterations: int, benchmark_runs: int, warmup_runs: int) -> dict:
    engine = ValidationEngine()
    latencies_ms = []
    cvs = []

    for _ in range(iterations):
        started = time.perf_counter()
        result = engine.validate_candidate(
            original_code=SAMPLE_ORIGINAL,
            candidate_code=SAMPLE_CANDIDATE,
            file_path="",
            run_unit_tests=False,
            unit_test_command=None,
            benchmark_mode=True,
            benchmark_runs=benchmark_runs,
            warmup_runs=warmup_runs,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latencies_ms.append(elapsed_ms)
        if result.cv is not None:
            cvs.append(result.cv)

    return {
        "iterations": iterations,
        "benchmark_runs": benchmark_runs,
        "warmup_runs": warmup_runs,
        "latency": {
            "mean_ms": round(statistics.fmean(latencies_ms), 3),
            "median_ms": round(statistics.median(latencies_ms), 3),
            "p95_ms": round(sorted(latencies_ms)[max(0, int(len(latencies_ms) * 0.95) - 1)], 3),
        },
        "cv": {
            "mean": round(statistics.fmean(cvs), 5) if cvs else None,
            "median": round(statistics.median(cvs), 5) if cvs else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase A latency simulation helper")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--benchmark-runs", type=int, default=12)
    parser.add_argument("--warmup-runs", type=int, default=2)
    args = parser.parse_args()

    report = run_simulation(
        iterations=max(1, args.iterations),
        benchmark_runs=max(1, args.benchmark_runs),
        warmup_runs=max(0, args.warmup_runs),
    )

    print(report)


if __name__ == "__main__":
    main()
