from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


_START = "<<<<<<< SEARCH"
_SEP = "======="
_END = ">>>>>>> REPLACE"


@dataclass(frozen=True)
class SearchReplacePatch:
    search: str
    replace: str


def _normalize_newlines(value: str) -> str:
    return (value or "").replace("\r\n", "\n").replace("\r", "\n")


def parse_search_replace_blocks(text: str) -> List[SearchReplacePatch]:
    """Parse SEARCH/REPLACE blocks; return empty list on malformed input."""
    normalized = _normalize_newlines(text)
    if _START not in normalized:
        return []

    patches: List[SearchReplacePatch] = []
    cursor = 0
    while True:
        start_idx = normalized.find(_START, cursor)
        if start_idx == -1:
            break

        line_end = normalized.find("\n", start_idx)
        if line_end == -1:
            return []
        search_start = line_end + 1

        sep_marker = "\n" + _SEP + "\n"
        sep_idx = normalized.find(sep_marker, search_start)
        if sep_idx == -1:
            return []

        end_marker = "\n" + _END
        end_idx = normalized.find(end_marker, sep_idx + len(sep_marker))
        if end_idx == -1:
            return []
        end_payload_start = end_idx + len(end_marker)
        end_line_end = normalized.find("\n", end_payload_start)
        if end_line_end == -1:
            end_line_end = len(normalized)
        if normalized[end_payload_start:end_line_end].strip():
            return []

        search_text = normalized[search_start:sep_idx]
        replace_text = normalized[sep_idx + len(sep_marker):end_idx]
        if not search_text or replace_text == "":
            return []

        patches.append(SearchReplacePatch(search=search_text, replace=replace_text))
        cursor = end_line_end

    return patches


def apply_diff_patches(original: str, patches: List[SearchReplacePatch]) -> Optional[str]:
    """Apply SEARCH/REPLACE patches in order. Returns None if any patch fails."""
    updated = _normalize_newlines(original)
    for patch in patches:
        if patch.search not in updated:
            return None
        updated = updated.replace(patch.search, patch.replace, 1)
    return updated
