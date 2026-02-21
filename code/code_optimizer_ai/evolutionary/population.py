from __future__ import annotations

import asyncio
import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from code.code_optimizer_ai.core.llm_analyzer import CodeAnalysisResult, CodeAnalyzerLLM
from code.code_optimizer_ai.evolutionary.constants import FamilyTag, GenerationPath


_CATEGORY_TO_FAMILY: Dict[str, FamilyTag] = {
    "algorithm_change": FamilyTag.ALGORITHMIC,
    "data_structure_optimization": FamilyTag.DATA_STRUCTURE,
    "caching_strategy": FamilyTag.CACHING,
    "parallelization": FamilyTag.PARALLELIZATION,
    "memory_optimization": FamilyTag.MEMORY_LAYOUT,
    "loop_optimization": FamilyTag.LOOP_RESTRUCTURE,
    "io_optimization": FamilyTag.IO_BATCHING,
    "database_optimization": FamilyTag.IO_BATCHING,
    "network_optimization": FamilyTag.IO_BATCHING,
}


@dataclass
class CandidateRecord:
    candidate_id: str
    request_id: str
    generation_path: GenerationPath
    family_tag: FamilyTag
    generation_number: int
    parent_id: Optional[str]
    model_id: Optional[str]
    prompt_template_version: str
    transform_lib_version: str
    original_code: str
    code_patch: str
    created_at: str
    provenance: Dict[str, Any] = field(default_factory=dict)


class PopulationGenerator:
    def __init__(self, analyzer: Optional[CodeAnalyzerLLM] = None):
        self.analyzer = analyzer or CodeAnalyzerLLM()

    @staticmethod
    def _default_family_tag(analysis: CodeAnalysisResult) -> FamilyTag:
        if analysis.family_seed_tags:
            first = analysis.family_seed_tags[0]
            for tag in FamilyTag:
                if tag.value == first:
                    return tag
        return FamilyTag.ALGORITHMIC

    def _category_to_family_tag(self, category: Optional[str], analysis: CodeAnalysisResult) -> FamilyTag:
        if not category:
            return self._default_family_tag(analysis)
        mapped = _CATEGORY_TO_FAMILY.get(category.strip().lower())
        if mapped:
            return mapped
        return self._default_family_tag(analysis)

    @staticmethod
    def _build_candidate(
        *,
        request_id: str,
        generation_path: GenerationPath,
        family_tag: FamilyTag,
        prompt_template_version: str,
        transform_lib_version: str,
        original_code: str,
        code_patch: str,
        model_id: Optional[str],
        parent_id: Optional[str] = None,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> CandidateRecord:
        return CandidateRecord(
            candidate_id=str(uuid.uuid4()),
            request_id=request_id,
            generation_path=generation_path,
            family_tag=family_tag,
            generation_number=0,
            parent_id=parent_id,
            model_id=model_id,
            prompt_template_version=prompt_template_version,
            transform_lib_version=transform_lib_version,
            original_code=original_code,
            code_patch=code_patch,
            created_at=datetime.utcnow().isoformat(),
            provenance=provenance or {},
        )

    @staticmethod
    def _ast_heuristic_transform(code: str) -> str:
        # Safe deterministic transform in Phase A: strip trailing whitespace and
        # enforce a single trailing newline for stable benchmark behavior.
        stripped = "\n".join(line.rstrip() for line in (code or "").splitlines()).strip("\n")
        if not stripped:
            return code
        return stripped + "\n"

    async def _generate_ast_candidate(
        self,
        *,
        request_context: Any,
        original_code: str,
        analysis: CodeAnalysisResult,
    ) -> CandidateRecord:
        code_patch = self._ast_heuristic_transform(original_code)
        return self._build_candidate(
            request_id=request_context.request_id,
            generation_path=GenerationPath.AST_HEURISTIC,
            family_tag=self._default_family_tag(analysis),
            prompt_template_version=request_context.prompt_template_version,
            transform_lib_version=request_context.transform_lib_version,
            original_code=original_code,
            code_patch=code_patch,
            model_id=None,
            provenance={"strategy": "ast_heuristic"},
        )

    async def _generate_temperature_candidates(
        self,
        *,
        request_context: Any,
        original_code: str,
        analysis: CodeAnalysisResult,
        max_candidates: int,
    ) -> List[CandidateRecord]:
        suggestions = await self.analyzer.generate_optimizations(
            original_code,
            analysis,
            identifier="v2_temperature",
        )
        records: List[CandidateRecord] = []
        for suggestion in suggestions[: max(1, max_candidates)]:
            patch = suggestion.optimized_code or original_code
            trace = suggestion.model_trace or {}
            records.append(
                self._build_candidate(
                    request_id=request_context.request_id,
                    generation_path=GenerationPath.LLM_TEMPERATURE,
                    family_tag=self._category_to_family_tag(suggestion.category, analysis),
                    prompt_template_version=request_context.prompt_template_version,
                    transform_lib_version=request_context.transform_lib_version,
                    original_code=original_code,
                    code_patch=patch,
                    model_id=trace.get("model"),
                    provenance={
                        "strategy": "llm_temperature",
                        "provider": trace.get("provider"),
                        "category": suggestion.category,
                    },
                )
            )
        return records

    async def _generate_critique_refine_candidate(
        self,
        *,
        request_context: Any,
        base_candidate: CandidateRecord,
        analysis: CodeAnalysisResult,
    ) -> Optional[CandidateRecord]:
        suggestions = await self.analyzer.generate_optimizations(
            base_candidate.code_patch,
            analysis,
            identifier="v2_critique_refine",
        )
        if not suggestions:
            return None
        suggestion = suggestions[0]
        patch = suggestion.optimized_code or base_candidate.code_patch
        trace = suggestion.model_trace or {}
        return self._build_candidate(
            request_id=request_context.request_id,
            generation_path=GenerationPath.LLM_CRITIQUE_REFINE,
            family_tag=self._category_to_family_tag(suggestion.category, analysis),
            prompt_template_version=request_context.prompt_template_version,
            transform_lib_version=request_context.transform_lib_version,
            original_code=base_candidate.code_patch,
            code_patch=patch,
            model_id=trace.get("model"),
            parent_id=base_candidate.candidate_id,
            provenance={
                "strategy": "llm_critique_refine",
                "provider": trace.get("provider"),
                "category": suggestion.category,
            },
        )

    @staticmethod
    def _dedupe_candidates(candidates: List[CandidateRecord]) -> List[CandidateRecord]:
        seen_hashes: set[str] = set()
        unique: List[CandidateRecord] = []
        for candidate in candidates:
            digest = hashlib.sha256((candidate.code_patch or "").encode("utf-8")).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            unique.append(candidate)
        return unique

    async def generate_phase_a_population(
        self,
        *,
        request_context: Any,
        original_code: str,
        analysis: CodeAnalysisResult,
        max_candidates: int,
    ) -> List[CandidateRecord]:
        ast_task = asyncio.create_task(
            self._generate_ast_candidate(
                request_context=request_context,
                original_code=original_code,
                analysis=analysis,
            )
        )
        temp_task = asyncio.create_task(
            self._generate_temperature_candidates(
                request_context=request_context,
                original_code=original_code,
                analysis=analysis,
                max_candidates=max(1, max_candidates - 1),
            )
        )

        ast_candidate, temperature_candidates = await asyncio.gather(ast_task, temp_task)
        candidates: List[CandidateRecord] = [ast_candidate, *temperature_candidates]

        # Critique-refine starts only after at least one temperature candidate exists.
        if temperature_candidates:
            critique_candidate = await self._generate_critique_refine_candidate(
                request_context=request_context,
                base_candidate=temperature_candidates[0],
                analysis=analysis,
            )
            if critique_candidate is not None:
                candidates.append(critique_candidate)

        deduped = self._dedupe_candidates(candidates)
        if not deduped:
            deduped = [ast_candidate]
        return deduped[: max(1, max_candidates)]
