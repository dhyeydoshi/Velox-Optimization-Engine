from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import List, Optional

from code.code_optimizer_ai.config.evolutionary import QD_IMPROVEMENT_THRESHOLD, QD_REGRESSION_THRESHOLD
from code.code_optimizer_ai.database.connection import db_manager
from code.code_optimizer_ai.evolutionary.constants import FamilyTag
from code.code_optimizer_ai.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ArchiveEntry:
    id: str
    family_tag: FamilyTag
    performance_tier: str
    best_candidate_id: Optional[str]
    best_request_id: Optional[str]
    composite_score: float
    measured_runtime_delta: float
    code_pattern_hash: Optional[str]
    transform_summary: Optional[str]


class QDArchive:
    @staticmethod
    def _performance_tier(runtime_delta_pct: float) -> str:
        if runtime_delta_pct > QD_REGRESSION_THRESHOLD:
            return "REGRESSION"
        if runtime_delta_pct < QD_IMPROVEMENT_THRESHOLD:
            return "IMPROVEMENT"
        return "NEUTRAL"

    async def get_entries(self, family_tags: Optional[List[FamilyTag]] = None) -> List[ArchiveEntry]:
        if not db_manager.connection_pool:
            return []

        tags = [tag.value for tag in (family_tags or [])]
        query = """
            SELECT
                id::text AS id,
                family_tag,
                performance_tier,
                best_candidate_id::text AS best_candidate_id,
                best_request_id::text AS best_request_id,
                composite_score,
                measured_runtime_delta,
                code_pattern_hash,
                transform_summary
            FROM qd_archive
            WHERE ($1::text[] IS NULL OR family_tag = ANY($1::text[]))
            ORDER BY family_tag, performance_tier
        """
        try:
            async with db_manager.connection_pool.acquire() as conn:
                rows = await conn.fetch(query, tags if tags else None)
        except Exception as exc:
            logger.warning("qd_archive_get_failed", error=str(exc))
            return []

        entries: List[ArchiveEntry] = []
        for row in rows:
            try:
                entries.append(
                    ArchiveEntry(
                        id=row["id"],
                        family_tag=FamilyTag(row["family_tag"]),
                        performance_tier=row["performance_tier"],
                        best_candidate_id=row["best_candidate_id"],
                        best_request_id=row["best_request_id"],
                        composite_score=float(row["composite_score"] or 0.0),
                        measured_runtime_delta=float(row["measured_runtime_delta"] or 0.0),
                        code_pattern_hash=row["code_pattern_hash"],
                        transform_summary=row["transform_summary"],
                    )
                )
            except Exception:
                continue
        return entries

    async def update_cell(
        self,
        *,
        family_tag: FamilyTag,
        best_candidate_id: Optional[str],
        best_request_id: Optional[str],
        composite_score: float,
        measured_runtime_delta: float,
        code_pattern_hash: Optional[str],
        transform_summary: Optional[str],
    ) -> bool:
        if not db_manager.connection_pool:
            return False

        performance_tier = self._performance_tier(float(measured_runtime_delta))
        upsert_sql = """
            INSERT INTO qd_archive (
                id,
                family_tag,
                performance_tier,
                best_candidate_id,
                best_request_id,
                composite_score,
                measured_runtime_delta,
                code_pattern_hash,
                transform_summary,
                updated_at
            )
            VALUES (
                $1::uuid,
                $2,
                $3,
                $4::uuid,
                $5::uuid,
                $6,
                $7,
                $8,
                $9,
                NOW()
            )
            ON CONFLICT (family_tag, performance_tier) DO UPDATE SET
                best_candidate_id = EXCLUDED.best_candidate_id,
                best_request_id = EXCLUDED.best_request_id,
                composite_score = EXCLUDED.composite_score,
                measured_runtime_delta = EXCLUDED.measured_runtime_delta,
                code_pattern_hash = EXCLUDED.code_pattern_hash,
                transform_summary = EXCLUDED.transform_summary,
                updated_at = NOW()
        """

        try:
            async with db_manager.connection_pool.acquire() as conn:
                await conn.execute(
                    upsert_sql,
                    str(uuid.uuid4()),
                    family_tag.value,
                    performance_tier,
                    best_candidate_id,
                    best_request_id,
                    float(composite_score),
                    float(measured_runtime_delta),
                    code_pattern_hash,
                    transform_summary,
                )
            return True
        except Exception as exc:
            logger.warning("qd_archive_update_failed", error=str(exc))
            return False


qd_archive = QDArchive()
