from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

import asyncpg

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.code_optimizer_ai.config.settings import settings


@dataclass
class PhaseAGateInputs:
    date_from: Optional[str]
    date_to: Optional[str]
    min_requests: int
    cv_threshold: float
    cv_pass_target: float
    latency_sla_ms: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Phase A gate evidence from Postgres")
    parser.add_argument("--date-from", default=None, help="Inclusive ISO8601 start timestamp")
    parser.add_argument("--date-to", default=None, help="Exclusive ISO8601 end timestamp")
    parser.add_argument("--min-requests", type=int, default=100)
    parser.add_argument("--cv-threshold", type=float, default=0.15)
    parser.add_argument("--cv-pass-target", type=float, default=0.90)
    parser.add_argument("--latency-sla-ms", type=int, default=4500)
    parser.add_argument(
        "--output-json",
        default=str(REPO_ROOT / "data" / "phase_a_gate_report.json"),
        help="Output JSON report path",
    )
    parser.add_argument(
        "--output-md",
        default=str(REPO_ROOT / "data" / "phase_a_gate_report.md"),
        help="Output Markdown report path",
    )
    return parser.parse_args()


async def _fetch_completed_requests(
    conn: asyncpg.Connection,
    *,
    date_from: Optional[str],
    date_to: Optional[str],
) -> List[Dict[str, Any]]:
    query = """
        SELECT
            id::text AS request_id,
            created_at,
            metadata
        FROM optimization_requests
        WHERE status = 'completed'
          AND ($1::timestamptz IS NULL OR created_at >= $1::timestamptz)
          AND ($2::timestamptz IS NULL OR created_at < $2::timestamptz)
        ORDER BY created_at ASC
    """
    rows = await conn.fetch(query, date_from, date_to)
    return [dict(row) for row in rows]


async def _fetch_candidate_gate_stats(
    conn: asyncpg.Connection,
    request_ids: List[str],
    cv_threshold: float,
) -> Dict[str, Any]:
    if not request_ids:
        return {
            "evaluated_candidates": 0,
            "measured_candidates": 0,
            "cv_pass_candidates": 0,
            "cv_pass_rate": None,
        }
    request_uuids = [UUID(value) for value in request_ids]
    query = """
        SELECT
            COUNT(*) AS evaluated_candidates,
            COUNT(*) FILTER (WHERE run_count >= 12 AND cv IS NOT NULL) AS measured_candidates,
            COUNT(*) FILTER (WHERE run_count >= 12 AND cv IS NOT NULL AND cv < $2) AS cv_pass_candidates
        FROM phase_a_candidate_evaluations
        WHERE request_id = ANY($1::uuid[])
    """
    row = await conn.fetchrow(query, request_uuids, float(cv_threshold))
    evaluated = int(row["evaluated_candidates"] or 0)
    measured = int(row["measured_candidates"] or 0)
    passed = int(row["cv_pass_candidates"] or 0)
    rate = (passed / measured) if measured else None
    return {
        "evaluated_candidates": evaluated,
        "measured_candidates": measured,
        "cv_pass_candidates": passed,
        "cv_pass_rate": rate,
    }


async def _fetch_top_runtime_delta(
    conn: asyncpg.Connection,
    request_ids: List[str],
) -> Dict[str, Any]:
    if not request_ids:
        return {"requests_with_top_candidate": 0, "avg_runtime_delta_pct": None}
    request_uuids = [UUID(value) for value in request_ids]
    query = """
        WITH ranked AS (
            SELECT
                request_id,
                runtime_delta_pct,
                ROW_NUMBER() OVER (
                    PARTITION BY request_id
                    ORDER BY composite_score DESC NULLS LAST, runtime_delta_pct DESC NULLS LAST
                ) AS rn
            FROM phase_a_candidate_evaluations
            WHERE request_id = ANY($1::uuid[])
              AND gate_result = 'PASSED'
              AND run_count >= 12
              AND composite_score IS NOT NULL
        )
        SELECT
            COUNT(*) AS requests_with_top_candidate,
            AVG(runtime_delta_pct) AS avg_runtime_delta_pct
        FROM ranked
        WHERE rn = 1
    """
    row = await conn.fetchrow(query, request_uuids)
    return {
        "requests_with_top_candidate": int(row["requests_with_top_candidate"] or 0),
        "avg_runtime_delta_pct": (
            float(row["avg_runtime_delta_pct"])
            if row["avg_runtime_delta_pct"] is not None
            else None
        ),
    }


def _p95(values: List[int]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(len(ordered) * 0.95) - 1))
    return float(ordered[idx])


def _safe_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _build_markdown(report: Dict[str, Any]) -> str:
    gates = report["gates"]
    return "\n".join(
        [
            "# Phase A Gate Evidence Report",
            "",
            f"- Generated at: `{report['generated_at_utc']}`",
            f"- Date window: `{report['window']['date_from']}` -> `{report['window']['date_to']}`",
            f"- Completed requests: `{report['counts']['completed_requests']}`",
            f"- Evaluated candidates: `{report['counts']['evaluated_candidates']}`",
            f"- Measured candidates (run_count>=12): `{report['counts']['measured_candidates']}`",
            "",
            "## Gate Results",
            f"1. Positive avg improvement (>0): `{gates['positive_avg_improvement']['passed']}` "
            f"(value={gates['positive_avg_improvement']['value']})",
            f"2. CV pass rate >= target: `{gates['cv_reliability']['passed']}` "
            f"(value={gates['cv_reliability']['value']}, target={gates['cv_reliability']['target']})",
            f"3. Latency p95 <= SLA: `{gates['latency_sla']['passed']}` "
            f"(value={gates['latency_sla']['value']}, target={gates['latency_sla']['target']})",
            "",
            f"## Overall: `{report['overall_phase_a_gate_passed']}`",
            "",
        ]
    )


async def _run_report(inputs: PhaseAGateInputs, output_json: Path, output_md: Path) -> int:
    conn = await asyncpg.connect(settings.DATABASE_URL)
    try:
        completed = await _fetch_completed_requests(
            conn,
            date_from=inputs.date_from,
            date_to=inputs.date_to,
        )
        request_ids = [row["request_id"] for row in completed]
        candidate_stats = await _fetch_candidate_gate_stats(conn, request_ids, inputs.cv_threshold)
        top_runtime_stats = await _fetch_top_runtime_delta(conn, request_ids)
    except asyncpg.UndefinedTableError:
        print(
            json.dumps(
                {
                    "error": "Phase A evidence tables are missing.",
                    "hint": "Run: python -m code.code_optimizer_ai.database.migrate --migrate-v2",
                }
            )
        )
        return 2
    finally:
        await conn.close()

    latency_values: List[int] = []
    for row in completed:
        metadata = row.get("metadata") or {}
        latency_raw = metadata.get("latency_ms")
        if latency_raw is None:
            continue
        try:
            latency_values.append(int(latency_raw))
        except (TypeError, ValueError):
            continue

    completed_requests = len(completed)
    p95_latency = _p95(latency_values)
    avg_runtime_delta = _safe_float(top_runtime_stats["avg_runtime_delta_pct"])
    cv_pass_rate = _safe_float(candidate_stats["cv_pass_rate"])

    gate_positive_avg = (avg_runtime_delta is not None) and (avg_runtime_delta > 0.0)
    gate_cv = (cv_pass_rate is not None) and (cv_pass_rate >= inputs.cv_pass_target)
    gate_latency = (p95_latency is not None) and (p95_latency <= float(inputs.latency_sla_ms))
    gate_sample = completed_requests >= inputs.min_requests

    overall_pass = gate_sample and gate_positive_avg and gate_cv and gate_latency

    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "window": {"date_from": inputs.date_from, "date_to": inputs.date_to},
        "counts": {
            "completed_requests": completed_requests,
            "requests_with_top_candidate": top_runtime_stats["requests_with_top_candidate"],
            "evaluated_candidates": candidate_stats["evaluated_candidates"],
            "measured_candidates": candidate_stats["measured_candidates"],
            "cv_pass_candidates": candidate_stats["cv_pass_candidates"],
            "latency_samples": len(latency_values),
        },
        "metrics": {
            "avg_top_runtime_delta_pct": avg_runtime_delta,
            "cv_pass_rate": cv_pass_rate,
            "latency_p95_ms": p95_latency,
            "latency_mean_ms": _safe_float(statistics.fmean(latency_values)) if latency_values else None,
            "latency_median_ms": _safe_float(statistics.median(latency_values)) if latency_values else None,
        },
        "gates": {
            "minimum_sample_size": {
                "passed": gate_sample,
                "value": completed_requests,
                "target": inputs.min_requests,
            },
            "positive_avg_improvement": {
                "passed": gate_positive_avg,
                "value": avg_runtime_delta,
                "target": "> 0.0",
            },
            "cv_reliability": {
                "passed": gate_cv,
                "value": cv_pass_rate,
                "target": inputs.cv_pass_target,
            },
            "latency_sla": {
                "passed": gate_latency,
                "value": p95_latency,
                "target": inputs.latency_sla_ms,
            },
        },
        "overall_phase_a_gate_passed": overall_pass,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output_md.write_text(_build_markdown(report), encoding="utf-8")

    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md), "overall_pass": overall_pass}))
    return 0


def main() -> int:
    args = _parse_args()
    inputs = PhaseAGateInputs(
        date_from=args.date_from,
        date_to=args.date_to,
        min_requests=max(1, int(args.min_requests)),
        cv_threshold=float(args.cv_threshold),
        cv_pass_target=float(args.cv_pass_target),
        latency_sla_ms=max(1, int(args.latency_sla_ms)),
    )
    return asyncio.run(
        _run_report(
            inputs,
            output_json=Path(args.output_json),
            output_md=Path(args.output_md),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
