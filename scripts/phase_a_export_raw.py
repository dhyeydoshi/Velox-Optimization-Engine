from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

import asyncpg

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.code_optimizer_ai.config.settings import settings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export raw Phase A evidence from Postgres")
    parser.add_argument("--date-from", default=None, help="Inclusive ISO8601 start timestamp")
    parser.add_argument("--date-to", default=None, help="Exclusive ISO8601 end timestamp")
    parser.add_argument("--status", default="completed", help="Request status filter (default: completed)")
    parser.add_argument("--limit", type=int, default=2000, help="Max requests to export")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "data" / "phase_a_evidence_raw.jsonl"),
        help="Output JSONL path",
    )
    return parser.parse_args()


async def _fetch_requests(
    conn: asyncpg.Connection,
    *,
    status: str,
    date_from: Optional[str],
    date_to: Optional[str],
    limit: int,
) -> List[Dict[str, Any]]:
    query = """
        SELECT
            id::text AS request_id,
            request_scope,
            status,
            file_path,
            representative_input_warning,
            metadata,
            created_at,
            updated_at
        FROM optimization_requests
        WHERE ($1::text IS NULL OR status = $1::text)
          AND ($2::timestamptz IS NULL OR created_at >= $2::timestamptz)
          AND ($3::timestamptz IS NULL OR created_at < $3::timestamptz)
        ORDER BY created_at DESC
        LIMIT $4
    """
    rows = await conn.fetch(query, status, date_from, date_to, max(1, limit))
    return [dict(row) for row in rows]


async def _fetch_candidates(conn: asyncpg.Connection, request_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    if not request_ids:
        return {}
    request_uuids = [UUID(value) for value in request_ids]
    query = """
        SELECT
            request_id::text AS request_id,
            candidate_id::text AS candidate_id,
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
        FROM phase_a_candidate_evaluations
        WHERE request_id = ANY($1::uuid[])
        ORDER BY created_at ASC
    """
    rows = await conn.fetch(query, request_uuids)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        payload = dict(row)
        grouped.setdefault(payload["request_id"], []).append(payload)
    return grouped


async def _run_export(args: argparse.Namespace) -> int:
    conn = await asyncpg.connect(settings.DATABASE_URL)
    try:
        requests = await _fetch_requests(
            conn,
            status=args.status,
            date_from=args.date_from,
            date_to=args.date_to,
            limit=args.limit,
        )
        request_ids = [item["request_id"] for item in requests]
        candidates_by_request = await _fetch_candidates(conn, request_ids)
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

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for request_row in requests:
            request_id = request_row["request_id"]
            payload = {
                "request": request_row,
                "candidates": candidates_by_request.get(request_id, []),
            }
            handle.write(json.dumps(payload, default=str) + "\n")

    print(
        json.dumps(
            {
                "exported_requests": len(requests),
                "output": str(output_path),
                "date_from": args.date_from,
                "date_to": args.date_to,
                "status": args.status,
            }
        )
    )
    return 0


def main() -> int:
    args = _parse_args()
    return asyncio.run(_run_export(args))


if __name__ == "__main__":
    raise SystemExit(main())
