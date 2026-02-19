from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from code.code_optimizer_ai.config.settings import settings


def parse_csv(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


@lru_cache
def allowed_code_roots(fallback: Path | None = None) -> List[Path]:
    roots: List[Path] = []
    for token in parse_csv(settings.ALLOWED_CODE_ROOTS):
        candidate = Path(token)
        if not candidate.is_absolute():
            candidate = (fallback or Path.cwd()) / candidate
        roots.append(candidate.resolve())
    if not roots:
        roots.append((fallback or Path.cwd()).resolve())
    return roots
