from __future__ import annotations

import ast
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
import textwrap
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from code.code_optimizer_ai.config.settings import settings
from code.code_optimizer_ai.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkMetrics:
    runtime_ms: float
    peak_kb: float


@dataclass
class ValidationResult:
    status: str
    syntax_ok: bool
    unit_tests_passed: Optional[bool]
    baseline_runtime_ms: Optional[float]
    candidate_runtime_ms: Optional[float]
    baseline_peak_kb: Optional[float]
    candidate_peak_kb: Optional[float]
    runtime_delta_pct: Optional[float]
    memory_delta_pct: Optional[float]
    error: Optional[str] = None


class ValidationEngine:

    SANDBOX_COPY_PATHS: Sequence[str] = (
        "code",
        "pytest.ini",
        "pyproject.toml",
        "setup.cfg",
        "tox.ini",
        "requirements.txt",
    )
    SANDBOX_IGNORED_PARTS = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".idea",
        ".vscode",
        ".tmp_validation",
        "data",
        "models",
    }
    SANDBOX_IGNORED_PREFIXES = ("pytest-cache-files-",)

    def __init__(self, project_root: Optional[Path] = None, sandbox_root: Optional[Path] = None):
        self.timeout_seconds = max(10, settings.VALIDATION_TIMEOUT_SECONDS)
        self.project_root = (project_root or Path.cwd()).resolve()
        default_sandbox = self.project_root / ".tmp_validation"
        self.sandbox_root = (sandbox_root or default_sandbox).resolve()
        self.sandbox_root.mkdir(parents=True, exist_ok=True)

    # Builtins denied to user code during micro-benchmarking.
    # Blocks import hijacking, file access, and nested code execution.
    _DENIED_BUILTINS = frozenset({
        "__import__", "open", "exec", "eval", "compile",
        "breakpoint", "exit", "quit", "input", "help",
    })

    def _microbench_script(self, code: str) -> str:
        return textwrap.dedent(
            f"""
            import ast
            import builtins
            import json
            import time
            import tracemalloc

            # Restrict builtins available to user code: deny imports,
            # file access, and nested exec/eval to limit attack surface.
            _DENIED = frozenset({{
                "__import__", "open", "exec", "eval", "compile",
                "breakpoint", "exit", "quit", "input", "help",
            }})
            _safe_builtins = {{k: v for k, v in builtins.__dict__.items() if k not in _DENIED}}

            code = {code!r}
            code_obj = compile(ast.parse(code, filename="<candidate>"), "<candidate>", "exec")

            # Try execution first; fall back to compile-only measurement.
            try:
                exec(code_obj, {{"__builtins__": _safe_builtins}})
                can_exec = True
            except Exception:
                can_exec = False

            tracemalloc.start()
            start = time.perf_counter()
            if can_exec:
                for _ in range(3):
                    exec(code_obj, {{"__builtins__": _safe_builtins}})
            else:
                for _ in range(3):
                    compile(ast.parse(code, filename="<candidate>"), "<candidate>", "exec")
            end = time.perf_counter()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            print(json.dumps({{"runtime_ms": (end-start) * 1000.0, "peak_kb": peak / 1024.0}}))
            """
        ).strip()

    def _run_microbench(self, code: str) -> BenchmarkMetrics:
        proc = subprocess.run(
            [sys.executable, "-I", "-S", "-c", self._microbench_script(code)],
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            env=self._subprocess_env(),
        )
        if proc.returncode != 0:
            stderr = proc.stderr.strip() or "Unknown benchmark error"
            raise RuntimeError(stderr)
        payload = json.loads(proc.stdout.strip() or "{}")
        return BenchmarkMetrics(
            runtime_ms=float(payload.get("runtime_ms", 0.0)),
            peak_kb=float(payload.get("peak_kb", 0.0)),
        )

    @staticmethod
    def _subprocess_env() -> dict:
        env = {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"}
        for key in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP"):
            if key in os.environ:
                env[key] = os.environ[key]
        return env

    def _safe_relative_file(self, file_path: str) -> Optional[Path]:
        if not file_path:
            return None
        path = Path(file_path)
        if not path.is_absolute():
            path = self.project_root / path
        if not path.exists():
            return None
        try:
            return path.resolve().relative_to(self.project_root)
        except ValueError:
            return None

    def _is_ignored(self, relative_path: Path) -> bool:
        names = tuple(part.lower() for part in relative_path.parts)
        if any(part in self.SANDBOX_IGNORED_PARTS for part in names):
            return True
        name = relative_path.name.lower()
        return any(name.startswith(prefix) for prefix in self.SANDBOX_IGNORED_PREFIXES)

    def _copy_file(self, source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with source.open("rb") as src_handle, destination.open("wb") as dst_handle:
                shutil.copyfileobj(src_handle, dst_handle)
        except OSError as exc:
            logger.warning(
                "validation_sandbox_copy_file_skipped",
                source=str(source),
                destination=str(destination),
                error=str(exc),
            )

    def _copy_directory(self, source: Path, destination: Path) -> None:
        for root, dirs, files in os.walk(source):
            root_path = Path(root)
            rel_root = root_path.relative_to(source)
            dirs[:] = [directory for directory in dirs if not self._is_ignored(rel_root / directory)]

            for file_name in files:
                rel_file = rel_root / file_name
                if self._is_ignored(rel_file):
                    continue
                self._copy_file(root_path / file_name, destination / rel_file)

    def _copy_path(self, relative_path: Path, sandbox_dir: Path) -> None:
        if self._is_ignored(relative_path):
            return
        source = self.project_root / relative_path
        destination = sandbox_dir / relative_path
        if not source.exists():
            return
        if source.is_dir():
            self._copy_directory(source, destination)
            return
        self._copy_file(source, destination)

    def _parse_unit_test_command(self, unit_test_command: str) -> list[str]:
        command = (unit_test_command or "").strip()
        if not command:
            return [sys.executable, "-m", "pytest", "code/code_optimizer_ai/tests", "-q"]

        try:
            args = shlex.split(command, posix=False)
        except ValueError as exc:
            raise ValueError(f"Invalid unit test command: {exc}") from exc

        if not args:
            return [sys.executable, "-m", "pytest", "code/code_optimizer_ai/tests", "-q"]

        head = args[0].strip("'\"").lower()
        tail: list[str]
        if head == "pytest":
            tail = args[1:]
            return [sys.executable, "-m", "pytest", *tail]
        if head in {"python", "python3", "py"} or Path(head).name.lower().startswith("python"):
            if len(args) < 3 or args[1] != "-m" or args[2] != "pytest":
                raise ValueError("Only 'python -m pytest ...' commands are allowed")
            tail = args[3:]
            return [sys.executable, "-m", "pytest", *tail]
        raise ValueError("Only python/pytest unit test commands are allowed")

    def _create_sandbox(self, relative_file: Path) -> Path:
        sandbox_dir: Optional[Path] = None
        for _ in range(5):
            candidate = self.sandbox_root / f"validation_{uuid.uuid4().hex}"
            try:
                candidate.mkdir(parents=True, exist_ok=False)
                sandbox_dir = candidate
                break
            except FileExistsError:
                continue

        if sandbox_dir is None:
            raise RuntimeError("Failed to create validation sandbox directory")

        for path_str in self.SANDBOX_COPY_PATHS:
            self._copy_path(Path(path_str), sandbox_dir)

        top_level = relative_file.parts[0]
        self._copy_path(Path(top_level), sandbox_dir)
        return sandbox_dir

    @staticmethod
    def _sandbox_file_path(sandbox_dir: Path, relative_file: Path) -> Path:
        target = (sandbox_dir / relative_file).resolve()
        try:
            target.relative_to(sandbox_dir.resolve())
        except ValueError as exc:
            raise ValueError("Candidate file path escapes validation sandbox") from exc
        return target

    def _run_unit_tests_on_candidate(
        self,
        *,
        file_path: str,
        candidate_code: str,
        unit_test_command: str,
    ) -> bool:
        relative_file = self._safe_relative_file(file_path)
        if relative_file is None:
            raise ValueError(f"File path is not inside project root or not found: {file_path}")

        command_args = self._parse_unit_test_command(unit_test_command)
        sandbox_dir = self._create_sandbox(relative_file)

        try:
            sandbox_file = self._sandbox_file_path(sandbox_dir, relative_file)
            sandbox_file.parent.mkdir(parents=True, exist_ok=True)
            sandbox_file.write_text(candidate_code, encoding="utf-8")

            proc = subprocess.run(
                command_args,
                cwd=sandbox_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                env=self._subprocess_env(),
            )
            if proc.returncode != 0:
                logger.warning(
                    "validation_unit_tests_failed",
                    file_path=file_path,
                    sandbox_dir=str(sandbox_dir),
                    command=" ".join(command_args),
                    stderr_tail=(proc.stderr or "")[-500:],
                )
                return False
            return True
        finally:
            self._cleanup_sandbox(sandbox_dir)

    def _cleanup_sandbox(self, sandbox_dir: Path) -> None:
        def _onerror(func, path, _exc_info):
            try:
                os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
                func(path)
            except OSError as exc:
                logger.warning(
                    "validation_sandbox_cleanup_item_failed",
                    path=str(path),
                    error=str(exc),
                )

        for attempt in range(3):
            try:
                shutil.rmtree(sandbox_dir, onerror=_onerror)
            except OSError as exc:
                logger.warning(
                    "validation_sandbox_cleanup_retry",
                    sandbox_dir=str(sandbox_dir),
                    attempt=attempt + 1,
                    error=str(exc),
                )
            if not sandbox_dir.exists():
                return
            time.sleep(0.05 * (attempt + 1))

        if sandbox_dir.exists():
            logger.warning(
                "validation_sandbox_cleanup_incomplete",
                sandbox_dir=str(sandbox_dir),
            )

    def validate_candidate(
        self,
        *,
        original_code: str,
        candidate_code: str,
        file_path: str = "",
        run_unit_tests: bool = True,
        unit_test_command: Optional[str] = None,
    ) -> ValidationResult:
        # Syntax gate
        try:
            ast.parse(candidate_code)
        except SyntaxError as exc:
            return ValidationResult(
                status="failed_syntax",
                syntax_ok=False,
                unit_tests_passed=None,
                baseline_runtime_ms=None,
                candidate_runtime_ms=None,
                baseline_peak_kb=None,
                candidate_peak_kb=None,
                runtime_delta_pct=None,
                memory_delta_pct=None,
                error=str(exc),
            )

        # Unit tests gate
        unit_status: Optional[bool] = None
        try:
            if run_unit_tests and file_path and unit_test_command:
                unit_status = self._run_unit_tests_on_candidate(
                    file_path=file_path,
                    candidate_code=candidate_code,
                    unit_test_command=unit_test_command,
                )
                if not unit_status:
                    return ValidationResult(
                        status="failed_tests",
                        syntax_ok=True,
                        unit_tests_passed=False,
                        baseline_runtime_ms=None,
                        candidate_runtime_ms=None,
                        baseline_peak_kb=None,
                        candidate_peak_kb=None,
                        runtime_delta_pct=None,
                        memory_delta_pct=None,
                        error="Unit tests failed for candidate patch",
                    )
            elif run_unit_tests:
                unit_status = None
        except subprocess.TimeoutExpired:
            return ValidationResult(
                status="failed_tests",
                syntax_ok=True,
                unit_tests_passed=False,
                baseline_runtime_ms=None,
                candidate_runtime_ms=None,
                baseline_peak_kb=None,
                candidate_peak_kb=None,
                runtime_delta_pct=None,
                memory_delta_pct=None,
                error="Unit test execution timed out",
            )
        except Exception as exc:
            return ValidationResult(
                status="failed_tests",
                syntax_ok=True,
                unit_tests_passed=False,
                baseline_runtime_ms=None,
                candidate_runtime_ms=None,
                baseline_peak_kb=None,
                candidate_peak_kb=None,
                runtime_delta_pct=None,
                memory_delta_pct=None,
                error=str(exc),
            )

        # Micro-benchmark gate
        try:
            baseline = self._run_microbench(original_code)
            candidate = self._run_microbench(candidate_code)
        except subprocess.TimeoutExpired:
            return ValidationResult(
                status="failed_benchmark",
                syntax_ok=True,
                unit_tests_passed=unit_status,
                baseline_runtime_ms=None,
                candidate_runtime_ms=None,
                baseline_peak_kb=None,
                candidate_peak_kb=None,
                runtime_delta_pct=None,
                memory_delta_pct=None,
                error="Benchmark execution timed out",
            )
        except Exception as exc:
            return ValidationResult(
                status="failed_benchmark",
                syntax_ok=True,
                unit_tests_passed=unit_status,
                baseline_runtime_ms=None,
                candidate_runtime_ms=None,
                baseline_peak_kb=None,
                candidate_peak_kb=None,
                runtime_delta_pct=None,
                memory_delta_pct=None,
                error=str(exc),
            )

        runtime_delta = 0.0
        if baseline.runtime_ms > 0:
            runtime_delta = ((baseline.runtime_ms - candidate.runtime_ms) / baseline.runtime_ms) * 100.0

        memory_delta = 0.0
        if baseline.peak_kb > 0:
            memory_delta = ((baseline.peak_kb - candidate.peak_kb) / baseline.peak_kb) * 100.0

        return ValidationResult(
            status="passed",
            syntax_ok=True,
            unit_tests_passed=unit_status,
            baseline_runtime_ms=baseline.runtime_ms,
            candidate_runtime_ms=candidate.runtime_ms,
            baseline_peak_kb=baseline.peak_kb,
            candidate_peak_kb=candidate.peak_kb,
            runtime_delta_pct=runtime_delta,
            memory_delta_pct=memory_delta,
        )


validation_engine = ValidationEngine()
