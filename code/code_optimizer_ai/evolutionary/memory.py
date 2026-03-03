from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from typing import List, Optional, Sequence

from code.code_optimizer_ai.config.evolutionary import (
    BENCHMARK_RUN_COUNT_MIN,
    RAG_MIN_COMPOSITE_SCORE,
    RAG_RETRIEVAL_LIMIT,
    RAG_SIMILARITY_THRESHOLD,
    RECENCY_WEIGHT_MULTIPLIER,
    RECENCY_WEIGHT_WINDOW_DAYS,
)
from code.code_optimizer_ai.database.connection import db_manager
from code.code_optimizer_ai.evolutionary.constants import FamilyTag
from code.code_optimizer_ai.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TransitionRecord:
    id: str
    family_tag: FamilyTag
    transform_summary: str
    code_diff: str
    measured_runtime_delta: float
    composite_score: float
    similarity: float


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in values) + "]"


class TransitionMemory:
    async def query_similar(
        self,
        embedding: Sequence[float],
        family_tags: Optional[List[FamilyTag]] = None,
        min_score: float = RAG_SIMILARITY_THRESHOLD,
        limit: int = RAG_RETRIEVAL_LIMIT,
    ) -> List[TransitionRecord]:
        if not db_manager.connection_pool or not embedding:
            return []

        tags = [tag.value for tag in (family_tags or [])]
        query = """
            SELECT
                id::text AS id,
                family_tag,
                transform_summary,
                code_diff,
                measured_runtime_delta,
                composite_score,
                (1 - (hotspot_embedding <=> $1::vector)) AS similarity,
                (1 - (hotspot_embedding <=> $1::vector))
                    * CASE
                        WHEN created_at >= (NOW() - make_interval(days => $4::int))
                            THEN $5::float
                        ELSE 1.0
                      END AS weighted_similarity
            FROM transition_memory
            WHERE (1 - (hotspot_embedding <=> $1::vector)) >= $2::float
              AND ($3::text[] IS NULL OR family_tag = ANY($3::text[]))
            ORDER BY weighted_similarity DESC, composite_score DESC, created_at DESC
            LIMIT $6::int
        """

        try:
            async with db_manager.connection_pool.acquire() as conn:
                rows = await conn.fetch(
                    query,
                    _vector_literal(embedding),
                    float(min_score),
                    tags if tags else None,
                    int(RECENCY_WEIGHT_WINDOW_DAYS),
                    float(RECENCY_WEIGHT_MULTIPLIER),
                    max(1, int(limit)),
                )
        except Exception as exc:
            logger.warning("transition_memory_query_failed", error=str(exc))
            return []

        records: List[TransitionRecord] = []
        for row in rows:
            try:
                records.append(
                    TransitionRecord(
                        id=row["id"],
                        family_tag=FamilyTag(row["family_tag"]),
                        transform_summary=row["transform_summary"] or "",
                        code_diff=row["code_diff"] or "",
                        measured_runtime_delta=float(row["measured_runtime_delta"] or 0.0),
                        composite_score=float(row["composite_score"] or 0.0),
                        similarity=float(row["similarity"] or 0.0),
                    )
                )
            except Exception:
                continue
        return records

    async def store_transition(
        self,
        *,
        request_id: str,
        embedding: Sequence[float],
        family_tag: FamilyTag,
        original_code: str,
        transform_summary: str,
        code_diff: str,
        measured_runtime_delta: float,
        measured_memory_delta: Optional[float],
        composite_score: float,
        cv: float,
        run_count: int,
        test_pass: bool,
        model_id: Optional[str],
        prompt_template_version: str,
        cv_gate_threshold: float = 0.15,
    ) -> bool:
        if not db_manager.connection_pool:
            return False
        if not embedding:
            return False
        if composite_score <= RAG_MIN_COMPOSITE_SCORE:
            logger.debug("transition_rejected_low_score", score=composite_score)
            return False
        if not test_pass:
            logger.debug("transition_rejected_test_failure")
            return False
        if cv > cv_gate_threshold:
            logger.debug("transition_rejected_high_cv", cv=cv)
            return False
        if run_count < BENCHMARK_RUN_COUNT_MIN:
            logger.debug("transition_rejected_low_run_count", run_count=run_count)
            return False

        insert_sql = """
            INSERT INTO transition_memory (
                id,
                request_id,
                hotspot_embedding,
                family_tag,
                original_code_hash,
                transform_summary,
                code_diff,
                measured_runtime_delta,
                measured_memory_delta,
                composite_score,
                cv,
                run_count,
                model_id,
                prompt_template_version
            )
            VALUES (
                $1::uuid,
                $2::uuid,
                $3::vector,
                $4,
                $5,
                $6,
                $7,
                $8,
                $9,
                $10,
                $11,
                $12,
                $13,
                $14
            )
        """

        code_hash = hashlib.sha256((original_code or "").encode("utf-8")).hexdigest()
        try:
            async with db_manager.connection_pool.acquire() as conn:
                await conn.execute(
                    insert_sql,
                    str(uuid.uuid4()),
                    request_id,
                    _vector_literal(embedding),
                    family_tag.value,
                    code_hash,
                    transform_summary or "",
                    code_diff or "",
                    float(measured_runtime_delta),
                    None if measured_memory_delta is None else float(measured_memory_delta),
                    float(composite_score),
                    float(cv),
                    int(run_count),
                    model_id,
                    prompt_template_version,
                )
            return True
        except Exception as exc:
            logger.warning("transition_memory_store_failed", error=str(exc))
            return False


transition_memory = TransitionMemory()
