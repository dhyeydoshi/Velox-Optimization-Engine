from __future__ import annotations

import asyncio
from typing import Optional

from code.code_optimizer_ai.core.validation_engine import ValidationEngine, ValidationResult, validation_engine


class SandboxValidationClient:
    def __init__(self, engine: Optional[ValidationEngine] = None):
        self.engine = engine or validation_engine

    async def validate_candidate(
        self,
        *,
        original_code: str,
        candidate_code: str,
        file_path: str,
        run_unit_tests: bool,
        unit_test_command: Optional[str],
        benchmark_mode: bool,
        benchmark_runs: int,
        warmup_runs: int,
    ) -> ValidationResult:
        return await asyncio.to_thread(
            self.engine.validate_candidate,
            original_code=original_code,
            candidate_code=candidate_code,
            file_path=file_path,
            run_unit_tests=run_unit_tests,
            unit_test_command=unit_test_command,
            benchmark_mode=benchmark_mode,
            benchmark_runs=benchmark_runs,
            warmup_runs=warmup_runs,
        )
