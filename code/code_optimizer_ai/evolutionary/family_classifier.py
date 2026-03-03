from __future__ import annotations

from typing import Dict, Iterable, List

from code.code_optimizer_ai.config.evolutionary import FAMILY_CLASSIFICATION_AMBIGUITY_THRESHOLD
from code.code_optimizer_ai.evolutionary.constants import FamilyTag, GenerationPath


_KEYWORDS: Dict[FamilyTag, Iterable[str]] = {
    FamilyTag.DATA_STRUCTURE: ("dict", "set", "hash", "map", "lookup"),
    FamilyTag.LOOP_RESTRUCTURE: ("for ", "while ", "comprehension", "enumerate", "zip("),
    FamilyTag.VECTORIZATION: ("numpy", "np.", "vector", "broadcast"),
    FamilyTag.CACHING: ("cache", "memo", "lru_cache"),
    FamilyTag.PARALLELIZATION: ("thread", "process", "async", "await", "concurrent"),
    FamilyTag.ALGORITHMIC: ("algorithm", "sort", "search", "complexity", "binary"),
    FamilyTag.IO_BATCHING: ("batch", "query", "io", "read(", "write(", "fetch"),
    FamilyTag.MEMORY_LAYOUT: ("memory", "buffer", "allocation", "__slots__", "generator"),
}


def _seed_fallback(analysis_family_seeds: List[str]) -> FamilyTag:
    for raw in analysis_family_seeds or []:
        try:
            return FamilyTag(raw)
        except ValueError:
            continue
    return FamilyTag.ALGORITHMIC


def classify_family_tag(
    code_diff: str,
    generation_path: GenerationPath,
    analysis_family_seeds: List[str],
) -> FamilyTag:
    text = (code_diff or "").lower()
    if not text:
        return _seed_fallback(analysis_family_seeds)

    scores: Dict[FamilyTag, int] = {tag: 0 for tag in FamilyTag}
    for tag, keywords in _KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                scores[tag] += 1

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_tag, best_score = ranked[0]
    if best_score <= 0:
        return _seed_fallback(analysis_family_seeds)

    second_score = ranked[1][1] if len(ranked) > 1 else 0
    if best_score > 0:
        margin = (best_score - second_score) / float(best_score)
        if margin < FAMILY_CLASSIFICATION_AMBIGUITY_THRESHOLD:
            return _seed_fallback(analysis_family_seeds)

    # A deterministic path prior helps with sparse keyword matches.
    if generation_path == GenerationPath.AST_HEURISTIC and best_score == 1:
        return _seed_fallback(analysis_family_seeds)

    return best_tag
