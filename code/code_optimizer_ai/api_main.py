import sys
import hmac
import os
import re
import shutil
import subprocess
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from urllib.parse import urlparse

# Ensure imports work by adding project root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import asyncio
import uvicorn

from code.code_optimizer_ai.agents.pipeline_orchestrator import (
    PipelineOrchestrator,
    TaskPriority,
    pipeline_orchestrator,
)
from code.code_optimizer_ai.core.code_scanner import code_scanner
from code.code_optimizer_ai.core.llm_analyzer import code_analyzer
from code.code_optimizer_ai.core.performance_profiler import performance_profiler
from code.code_optimizer_ai.database.connection import cache_manager, db_manager
from code.code_optimizer_ai.database.migrate import create_database_schema, seed_sample_data
from code.code_optimizer_ai.ml.rl_optimizer import get_rl_optimizer
from code.code_optimizer_ai.ml.training_semantics import PRODUCTION_ACTION_TYPES
from code.code_optimizer_ai.ml.training_runner import (
    TrainingConfig,
    TrainingJobStatus,
    create_job_id,
    create_job_status,
    load_job_status,
    list_jobs,
    run_training_job,
)
from code.code_optimizer_ai.config.settings import settings
from code.code_optimizer_ai.utils.logger import get_logger
from code.code_optimizer_ai.utils.paths import parse_csv, allowed_code_roots

logger = get_logger(__name__)
ERROR_GENERIC = "Internal server error"
_RATE_LIMIT_WINDOWS: Dict[str, deque] = defaultdict(deque)
_GITHUB_ALLOWED_HOSTS = {"github.com", "www.github.com"}
_GIT_REF_PATTERN = re.compile(r"^[A-Za-z0-9._/-]{1,128}$")


def _allowed_code_roots() -> List[Path]:
    return allowed_code_roots(REPO_ROOT)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _normalize_user_path(
    raw_path: str,
    *,
    must_exist: bool = True,
    expect_file: bool = False,
    expect_directory: bool = False,
) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate

    # Reject symbolic links before resolving (defense-in-depth against traversal)
    if candidate.is_symlink():
        raise HTTPException(status_code=400, detail="Symbolic links are not allowed")

    candidate = candidate.resolve()

    if not any(_is_within(candidate, root) for root in _allowed_code_roots()):
        raise HTTPException(status_code=400, detail="Path is outside allowed code roots")

    if must_exist and not candidate.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {raw_path}")

    if expect_file and must_exist and not candidate.is_file():
        raise HTTPException(status_code=400, detail="Expected a file path")
    if expect_directory and must_exist and not candidate.is_dir():
        raise HTTPException(status_code=400, detail="Expected a directory path")

    return candidate


def _normalize_github_repo_url(repo_url: str) -> str:
    url = (repo_url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="Repository URL is required")

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or host not in _GITHUB_ALLOWED_HOSTS:
        raise HTTPException(
            status_code=400,
            detail="Only HTTPS GitHub repository URLs are supported",
        )
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise HTTPException(status_code=400, detail="Repository URL format is invalid")
    if parsed.port is not None:
        raise HTTPException(status_code=400, detail="Custom ports are not supported")

    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) != 2:
        raise HTTPException(
            status_code=400,
            detail="Repository URL must be in the format https://github.com/<owner>/<repo>",
        )

    owner, repo = parts
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", owner):
        raise HTTPException(status_code=400, detail="Repository owner is invalid")
    if repo.endswith(".git"):
        repo = repo[:-4]
    if not repo or not re.fullmatch(r"[A-Za-z0-9_.-]+", repo):
        raise HTTPException(status_code=400, detail="Repository name is invalid")

    return f"https://github.com/{owner}/{repo}.git"


def _normalize_git_ref(git_ref: Optional[str]) -> Optional[str]:
    if git_ref is None:
        return None
    value = git_ref.strip()
    if not value:
        return None
    if value.startswith("-") or ".." in value or value.endswith("/"):
        raise HTTPException(status_code=400, detail="Invalid git ref")
    if not _GIT_REF_PATTERN.fullmatch(value):
        raise HTTPException(status_code=400, detail="Invalid git ref")
    return value


def _repo_clone_root() -> Path:
    candidate = Path(settings.GITHUB_CLONE_ROOT)
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if not any(_is_within(candidate, root) for root in _allowed_code_roots()):
        raise HTTPException(
            status_code=400,
            detail="Configured GITHUB_CLONE_ROOT is outside allowed code roots",
        )

    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def _clone_github_repository(repo_url: str, git_ref: Optional[str]) -> Path:
    normalized_url = _normalize_github_repo_url(repo_url)
    normalized_ref = _normalize_git_ref(git_ref)

    clone_dir = _repo_clone_root() / f"repo_{uuid.uuid4().hex}"
    timeout_seconds = max(15, int(settings.GITHUB_CLONE_TIMEOUT_SECONDS))
    command = ["git", "clone", "--depth", "1", "--single-branch"]
    if normalized_ref:
        command.extend(["--branch", normalized_ref])
    command.extend([normalized_url, str(clone_dir)])

    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
            check=False,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail="git executable is not available") from exc
    except subprocess.TimeoutExpired as exc:
        _cleanup_clone_directory(clone_dir)
        raise HTTPException(status_code=504, detail="Repository clone timed out") from exc

    if result.returncode != 0:
        _cleanup_clone_directory(clone_dir)
        # Sanitize stderr: strip potential auth tokens, credentials, or internal
        # paths before including in the client-facing response.
        raw_stderr = (result.stderr or "").strip()
        # Remove lines that look like credential or token leaks.
        sanitized_lines = [
            line for line in raw_stderr.splitlines()
            if not any(kw in line.lower() for kw in (
                "password", "token", "authorization", "credential",
                "secret", "bearer", "x-oauth",
            ))
        ]
        stderr_tail = "\n".join(sanitized_lines)[-200:]
        detail = "Failed to clone repository"
        if stderr_tail:
            detail = f"{detail}: {stderr_tail}"
        raise HTTPException(status_code=400, detail=detail)

    return clone_dir


def _cleanup_clone_directory(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


async def _read_upload_limited(upload: UploadFile, max_bytes: int) -> bytes:
    chunks: List[bytes] = []
    total_size = 0
    while True:
        chunk = await upload.read(64 * 1024)
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded file exceeds max size of {max_bytes} bytes",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def require_api_auth(
    request: Request,
    x_api_token: Optional[str] = Header(default=None, alias="X-API-Token")
) -> None:
    expected = settings.API_AUTH_TOKEN
    if not expected:
        return
    
    if not hmac.compare_digest(x_api_token or "", expected):
        # Audit log for security monitoring
        client_ip = request.client.host if request.client else "unknown"
        endpoint = f"{request.method} {request.url.path}"
        failure_reason = "missing_token" if not x_api_token else "invalid_token"
        
        logger.warning(
            "Authentication failed",
            extra={
                "event": "auth_failure",
                "client_ip": client_ip,
                "endpoint": endpoint,
                "reason": failure_reason,
                "timestamp": datetime.now().isoformat()
            }
        )
        raise HTTPException(status_code=401, detail="Unauthorized")



@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME}...")

    try:
        if not settings.TESTING and settings.SECRET_KEY == "your-secret-key-change-in-production":
            raise RuntimeError("Insecure default SECRET_KEY is not allowed")
        
        # Configurable auth token requirement
        if not settings.TESTING:
            if settings.REQUIRE_AUTH_TOKEN and not settings.API_AUTH_TOKEN:
                raise RuntimeError("API_AUTH_TOKEN must be set when REQUIRE_AUTH_TOKEN=True")
            elif not settings.API_AUTH_TOKEN:
                logger.warning(
                    "Security Warning: Running without API_AUTH_TOKEN - API endpoints are unprotected. "
                    "Set REQUIRE_AUTH_TOKEN=True to enforce authentication."
                )

        cors_origins = parse_csv(settings.CORS_ALLOWED_ORIGINS)
        if "*" in cors_origins and not settings.TESTING:
            raise RuntimeError("Wildcard CORS origin is not allowed")

        if settings.TESTING:
            logger.info("Running in TESTING mode - skipping external service initialization")
            yield
            return

        # Initialize core services
        await db_manager.initialize()
        await cache_manager.initialize()
        await create_database_schema()

        if settings.DEBUG:
            await seed_sample_data()

        # Start the optimization pipeline
        await pipeline_orchestrator.start_pipeline()
        logger.info("API ready on port 8000")

        yield

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    finally:
        logger.info("Shutting down...")
        await pipeline_orchestrator.stop_pipeline()
        await db_manager.close()
        await cache_manager.close()


# FastAPI application
docs_enabled = settings.DEBUG or settings.ENABLE_API_DOCS
app = FastAPI(
    title=f"{settings.APP_NAME} API",
    description="Intelligent code optimization using LLM, RL, and multi-agent pipeline",
    version=settings.APP_VERSION,
    docs_url="/docs" if docs_enabled else None,
    redoc_url="/redoc" if docs_enabled else None,
    openapi_url="/openapi.json" if docs_enabled else None,
    lifespan=lifespan,
)

# Add CORS middleware
cors_origins = parse_csv(settings.CORS_ALLOWED_ORIGINS) or ["http://localhost", "http://127.0.0.1"]
allow_any_origin = "*" in cors_origins
allow_credentials = settings.CORS_ALLOW_CREDENTIALS and not allow_any_origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Token"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if not settings.RATE_LIMIT_ENABLED or settings.TESTING:
        return await call_next(request)

    client_host = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = _RATE_LIMIT_WINDOWS[client_host]
    cutoff = now - 60.0

    while window and window[0] < cutoff:
        window.popleft()

    # Evict stale clients every 1000 requests to prevent unbounded growth
    if len(_RATE_LIMIT_WINDOWS) > 1000:
        stale_keys = [k for k, v in _RATE_LIMIT_WINDOWS.items() if not v or v[-1] < cutoff]
        for k in stale_keys:
            del _RATE_LIMIT_WINDOWS[k]

    limit = max(1, settings.RATE_LIMIT_REQUESTS_PER_MINUTE)
    if len(window) >= limit:
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

    window.append(now)
    response = await call_next(request)
    return response


# Request/Response Models
class OptimizationRequest(BaseModel):
    code: Optional[str] = None
    file_path: Optional[str] = None
    priority: str = "medium"
    context: Optional[Dict[str, Any]] = None
    objective_weights: Optional[Dict[str, float]] = None
    max_suggestions: int = Field(default=settings.DEFAULT_MAX_SUGGESTIONS, ge=1, le=10)
    run_validation: bool = False
    unit_test_command: Optional[str] = None


class OptimizationResponse(BaseModel):
    task_id: str
    status: str
    suggestions: List[Dict[str, Any]] = Field(default_factory=list)
    analysis: Optional[Dict[str, Any]] = None
    processing_time: float
    timestamp: str


# Helper functions
def parse_priority(priority_value: str) -> TaskPriority:
    try:
        return TaskPriority(priority_value.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid priority '{priority_value}'. Use: low, medium, high, critical"
        )


def _serialize_suggestions(suggestions: List[Any]) -> List[Dict[str, Any]]:
    suggestion_list: List[Dict[str, Any]] = []
    allowed_keys = {
        "suggestion_id",
        "category",
        "priority",
        "description",
        "original_code",
        "optimized_code",
        "patch_diff",
        "expected_improvement",
        "expected_runtime_delta_pct",
        "expected_memory_delta_pct",
        "expected_weighted_score",
        "implementation_effort",
        "confidence",
        "reasoning",
        "validation_status",
        "model_trace",
    }

    def _jsonable(value: Any) -> bool:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return True
        if isinstance(value, list):
            return all(_jsonable(v) for v in value)
        if isinstance(value, dict):
            return all(isinstance(k, str) and _jsonable(v) for k, v in value.items())
        return False

    for suggestion in suggestions or []:
        if isinstance(suggestion, dict):
            suggestion_list.append(suggestion)
            continue
        try:
            suggestion_list.append(asdict(suggestion))
            continue
        except Exception:
            pass

        mapped: Dict[str, Any] = {}
        for key in allowed_keys:
            if hasattr(suggestion, key):
                value = getattr(suggestion, key)
                if _jsonable(value):
                    mapped[key] = value
        if mapped:
            suggestion_list.append(mapped)
        else:
            suggestion_list.append({"value": str(suggestion)})

    return suggestion_list


def build_response(task_id: str, suggestions: List, started_at: datetime,
                   status: str = "completed") -> OptimizationResponse:
    return OptimizationResponse(
        task_id=task_id,
        status=status,
        suggestions=_serialize_suggestions(suggestions),
        processing_time=(datetime.now() - started_at).total_seconds(),
        timestamp=started_at.isoformat()
    )


def get_pipeline() -> PipelineOrchestrator:
    return pipeline_orchestrator


async def _scan_directory_summary(directory_path: str, recursive: bool, include_analysis: bool):
    scan_result = await code_scanner.scan_directory(directory_path, recursive)
    statistics = code_scanner.get_scan_statistics(scan_result)
    complexity_hotspots = statistics.get("high_complexity_file_paths", [])
    optimization_candidates = statistics.get("performance_candidate_files", [])

    if include_analysis and scan_result.code_files:
        sample_files = scan_result.code_files[:3]
        code_snippets = await code_scanner.get_code_snippets(sample_files, 500)
        if code_snippets:
            await code_analyzer.batch_analyze(code_snippets)

    return scan_result, statistics, complexity_hotspots, optimization_candidates


def _select_repository_files(scan_result, statistics: Dict[str, Any], max_files: int):
    selected_paths: set[str] = set()
    selected_files = []
    code_files = list(scan_result.code_files or [])
    by_path = {entry.file_path: entry for entry in code_files}

    for file_path in statistics.get("performance_candidate_files", []):
        code_file = by_path.get(file_path)
        if not code_file or code_file.file_path in selected_paths:
            continue
        selected_files.append(code_file)
        selected_paths.add(code_file.file_path)
        if len(selected_files) >= max_files:
            return selected_files

    ranked = sorted(
        code_files,
        key=lambda file: (
            float(file.complexity_metrics.get("cyclomatic_complexity", 0.0)),
            len(file.functions),
            len(file.classes),
        ),
        reverse=True,
    )
    for code_file in ranked:
        if code_file.file_path in selected_paths:
            continue
        selected_files.append(code_file)
        selected_paths.add(code_file.file_path)
        if len(selected_files) >= max_files:
            break

    return selected_files


def _relative_repo_path(file_path: str, repo_root: Path) -> str:
    try:
        relative = Path(file_path).resolve().relative_to(repo_root.resolve())
        return relative.as_posix()
    except Exception:
        return Path(file_path).name


def _module_name_from_relative_path(relative_path: str) -> str:
    relative = Path(relative_path)
    if relative.suffix.lower() != ".py":
        return relative.as_posix().replace("/", ".")

    parts = list(relative.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_import_to_module(import_name: str, known_modules: set[str]) -> Optional[str]:
    pieces = [piece for piece in import_name.split(".") if piece]
    for idx in range(len(pieces), 0, -1):
        candidate = ".".join(pieces[:idx])
        if candidate in known_modules:
            return candidate
    return None


def _build_repository_file_context(scan_result, repo_root: Path) -> Dict[str, Dict[str, Any]]:
    code_files = list(scan_result.code_files or [])
    if not code_files:
        return {}

    module_to_file: Dict[str, str] = {}
    module_to_relpath: Dict[str, str] = {}
    module_imports: Dict[str, List[str]] = {}
    file_to_module: Dict[str, str] = {}

    for code_file in code_files:
        file_path = str(getattr(code_file, "file_path", "") or "")
        if not file_path:
            continue
        rel_path = _relative_repo_path(file_path, repo_root)
        module_name = _module_name_from_relative_path(rel_path)
        if not module_name:
            continue

        raw_imports = getattr(code_file, "imports", None)
        if isinstance(raw_imports, str):
            import_entries = [raw_imports]
        elif isinstance(raw_imports, (list, tuple, set)):
            import_entries = list(raw_imports)
        else:
            import_entries = []

        imports = [
            entry.strip()
            for entry in import_entries
            if isinstance(entry, str) and entry.strip()
        ]

        module_to_file[module_name] = file_path
        module_to_relpath[module_name] = rel_path
        module_imports[module_name] = imports
        file_to_module[file_path] = module_name

    known_modules = set(module_to_file.keys())
    outgoing: Dict[str, set[str]] = {module: set() for module in known_modules}
    incoming: Dict[str, set[str]] = {module: set() for module in known_modules}

    for module_name, imports in module_imports.items():
        for import_name in imports[:100]:
            target = _resolve_import_to_module(import_name, known_modules)
            if target is None or target == module_name:
                continue
            outgoing[module_name].add(target)
            incoming[target].add(module_name)

    total_files = len(code_files)
    file_context: Dict[str, Dict[str, Any]] = {}
    for code_file in code_files:
        file_path = str(getattr(code_file, "file_path", "") or "")
        module_name = file_to_module.get(file_path)
        if not file_path or not module_name:
            continue

        related_modules = sorted(outgoing[module_name] | incoming[module_name])
        related_files = [module_to_relpath[module] for module in related_modules if module in module_to_relpath]

        file_context[file_path] = {
            "project_total_files": total_files,
            "relative_path": _relative_repo_path(file_path, repo_root),
            "module_name": module_name,
            "imports": sorted(outgoing[module_name])[:20],
            "imported_by": sorted(incoming[module_name])[:20],
            "related_python_files": related_files[:15],
        }

    return file_context


def _truncate_code_excerpt(content: Any, max_chars: int) -> str:
    if not isinstance(content, str) or max_chars <= 0:
        return ""
    cleaned = content.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    marker = "\n# ... truncated ..."
    cutoff = max(0, max_chars - len(marker))
    return f"{cleaned[:cutoff].rstrip()}{marker}"


def _build_cross_file_context(
    file_path: str,
    project_context: Dict[str, Dict[str, Any]],
    code_file_map: Dict[str, Any],
    *,
    max_related_files: int,
    snippet_chars: int,
) -> Dict[str, Any]:
    context = project_context.get(file_path, {})
    if not isinstance(context, dict):
        context = {}

    imports = [
        value for value in context.get("imports", [])
        if isinstance(value, str) and value
    ]
    imported_by = [
        value for value in context.get("imported_by", [])
        if isinstance(value, str) and value
    ]

    module_to_path: Dict[str, str] = {}
    for candidate_path, candidate_context in project_context.items():
        if not isinstance(candidate_context, dict):
            continue
        module_name = candidate_context.get("module_name")
        if isinstance(module_name, str) and module_name:
            module_to_path[module_name] = candidate_path

    related_files: List[Dict[str, Any]] = []
    for module_name in sorted(set(imports + imported_by)):
        related_path = module_to_path.get(module_name)
        if not related_path:
            continue
        related_entry = project_context.get(related_path, {})
        relationship = "bidirectional"
        in_imports = module_name in imports
        in_imported_by = module_name in imported_by
        if in_imports and not in_imported_by:
            relationship = "imports"
        elif in_imported_by and not in_imports:
            relationship = "imported_by"

        related_code_file = code_file_map.get(related_path)
        related_content = getattr(related_code_file, "content", "") if related_code_file else ""
        related_files.append(
            {
                "module": module_name,
                "path": related_entry.get("relative_path", Path(related_path).name),
                "relationship": relationship,
                "code_excerpt": _truncate_code_excerpt(related_content, snippet_chars),
            }
        )
        if len(related_files) >= max_related_files:
            break

    return {
        "enabled": True,
        "current_file": context.get("relative_path", Path(file_path).name),
        "current_module": context.get("module_name", ""),
        "related_files": related_files,
    }


def _build_target_dependency_graph(
    target_paths: List[str],
    project_context: Dict[str, Dict[str, Any]],
) -> Dict[str, set[str]]:
    target_set = set(target_paths)
    graph: Dict[str, set[str]] = {path: set() for path in target_paths}
    module_to_path: Dict[str, str] = {}

    for candidate_path, candidate_context in project_context.items():
        if not isinstance(candidate_context, dict):
            continue
        module_name = candidate_context.get("module_name")
        if isinstance(module_name, str) and module_name:
            module_to_path[module_name] = candidate_path

    for path in target_paths:
        context = project_context.get(path, {})
        if not isinstance(context, dict):
            continue
        dependencies: List[str] = []
        for key in ("imports", "imported_by"):
            values = context.get(key, [])
            if isinstance(values, list):
                dependencies.extend(value for value in values if isinstance(value, str))
        for module_name in dependencies:
            related_path = module_to_path.get(module_name)
            if related_path and related_path in target_set and related_path != path:
                graph[path].add(related_path)
                graph[related_path].add(path)

    return graph


def _build_coordination_batches(
    target_paths: List[str],
    project_context: Dict[str, Dict[str, Any]],
    *,
    max_group_size: int,
) -> List[List[str]]:
    if not target_paths:
        return []
    if max_group_size < 2:
        return [[path] for path in target_paths]

    graph = _build_target_dependency_graph(target_paths, project_context)
    visited: set[str] = set()
    components: List[List[str]] = []

    for start in target_paths:
        if start in visited:
            continue
        queue: deque[str] = deque([start])
        component: List[str] = []
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            for neighbor in sorted(graph.get(current, set())):
                if neighbor not in visited:
                    queue.append(neighbor)
        components.append(component)

    batches: List[List[str]] = []
    for component in components:
        if len(component) <= max_group_size:
            batches.append(component)
            continue
        for index in range(0, len(component), max_group_size):
            batches.append(component[index : index + max_group_size])
    return batches


class ScanRequest(BaseModel):
    directory_path: str
    recursive: bool = True
    include_analysis: bool = False


class ScanResponse(BaseModel):
    scan_id: str
    status: str
    files_scanned: int
    total_size_mb: float
    scan_duration: float
    statistics: Dict[str, Any]
    complexity_hotspots: List[str]
    optimization_candidates: List[str]
    timestamp: str


class RepositoryScanRequest(BaseModel):
    repo_url: str
    ref: Optional[str] = None
    recursive: bool = True
    include_analysis: bool = False


class RepositoryScanResponse(ScanResponse):
    repository_url: str
    ref: Optional[str] = None


class RepositoryOptimizationRequest(BaseModel):
    repo_url: str
    ref: Optional[str] = None
    max_files: int = Field(default=5, ge=1, le=50)
    max_suggestions_per_file: int = Field(default=settings.DEFAULT_MAX_SUGGESTIONS, ge=1, le=10)
    objective_weights: Optional[Dict[str, float]] = None
    run_validation: bool = False
    unit_test_command: Optional[str] = None
    enable_cross_file_context: bool = False
    cross_file_context_max_files: int = Field(default=3, ge=1, le=10)
    cross_file_context_snippet_chars: int = Field(default=1000, ge=200, le=5000)
    enable_batch_coordination: bool = False
    batch_coordination_max_group_size: int = Field(default=3, ge=2, le=10)
    batch_coordination_max_code_chars_per_file: int = Field(default=8000, ge=1000, le=30000)


class RepositoryOptimizationFileResult(BaseModel):
    file_path: str
    suggestions: List[Dict[str, Any]] = Field(default_factory=list)


class RepositoryOptimizationResponse(BaseModel):
    repository_url: str
    ref: Optional[str] = None
    status: str
    files_scanned: int
    files_analyzed: int
    files_with_suggestions: int
    results: List[RepositoryOptimizationFileResult] = Field(default_factory=list)
    processing_time: float
    timestamp: str


class PipelineStatusResponse(BaseModel):
    pipeline_status: str
    agents: Dict[str, Dict[str, Any]]
    active_tasks: int
    configuration: Dict[str, Any]
    timestamp: str


class TrainingRequest(BaseModel):
    synthetic_samples: int = Field(default=10_000, ge=0, le=100_000)
    epochs: int = Field(default=40, ge=1, le=200)
    steps_per_epoch: int = Field(default=1000, ge=100, le=10_000)
    batch_size: int = Field(default=64, ge=16, le=512)
    learning_rate: float = Field(default=1e-3, gt=0)
    holdout_fraction: float = Field(default=0.20, ge=0.05, le=0.50)
    seed: int = Field(default=42)
    require_td_improvement: bool = False
    background: bool = Field(default=True, description="Run training asynchronously")


class TrainingResponse(BaseModel):
    training_id: str
    status: str
    episodes_completed: int = 0
    final_reward: float = 0.0
    training_stats: Dict[str, Any] = Field(default_factory=dict)
    model_path: Optional[str] = None
    timestamp: str


@app.get("/")
async def root():
    endpoints = [
        "/health",
        "/optimize",
        "/optimize/repository",
        "/scan",
        "/scan/repository",
        "/pipeline/status",
        "/train",
    ]
    if docs_enabled:
        endpoints = ["/docs", *endpoints]
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "endpoints": endpoints,
    }


@app.get("/health")
async def health_check():
    try:
        pipeline_status = await pipeline_orchestrator.get_pipeline_status()

        pipeline_ok = pipeline_status["pipeline_status"] == "running"
        db_ok = bool(db_manager.connection_pool)
        cache_ok = bool(cache_manager.redis_client)
        overall = "healthy" if (pipeline_ok and db_ok and cache_ok) else "degraded"

        return {
            "status": overall,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.get("/health/details")
async def health_check_details(_: None = Depends(require_api_auth)):
    pipeline_status = await pipeline_orchestrator.get_pipeline_status()
    components = {
        "pipeline": "ok" if pipeline_status["pipeline_status"] == "running" else "degraded",
        "database": "ok" if db_manager.connection_pool else "starting",
        "cache": "ok" if cache_manager.redis_client else "starting",
    }
    return {
        "status": "healthy" if all(v == "ok" for v in components.values()) else "degraded",
        "components": components,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/database/status")
async def database_status(_: None = Depends(require_api_auth)):

    try:
        await db_manager.initialize()
        await cache_manager.initialize()

        async with db_manager.connection_pool.acquire() as conn:
            record_count = await conn.fetchval("SELECT COUNT(*) FROM optimization_records")
            episode_count = await conn.fetchval("SELECT COUNT(*) FROM training_episodes")
            pattern_count = await conn.fetchval("SELECT COUNT(*) FROM knowledge_patterns")

        redis_info = await cache_manager.redis_client.info()

        return {
            "postgresql": {
                "status": "connected",
                "optimization_records": record_count,
                "training_episodes": episode_count,
                "knowledge_patterns": pattern_count,
            },
            "redis": {
                "status": "connected",
                "connected_clients": redis_info.get("connected_clients", 0),
                "used_memory": redis_info.get("used_memory_human", "N/A"),
                "total_commands_processed": redis_info.get("total_commands_processed", 0),
            },
        }
    except Exception as exc:
        logger.error("Database status check failed", error=str(exc))
        return {
            "postgresql": {"status": "error", "error": "Database status unavailable"},
            "redis": {"status": "error", "error": "Cache status unavailable"},
        }


@app.post("/database/migrate")
async def run_migration(_: None = Depends(require_api_auth)):

    try:
        await create_database_schema()
        return {"status": "success", "message": "Database migration completed"}
    except Exception as exc:
        logger.error("Migration failed", error=str(exc))
        return {"status": "error", "message": "Migration failed"}


@app.post("/database/seed")
async def seed_database(_: None = Depends(require_api_auth)):

    if not settings.DEBUG:
        return {"status": "error", "message": "Seeding only available in DEBUG mode"}

    try:
        await seed_sample_data()
        return {"status": "success", "message": "Sample data seeded"}
    except Exception as exc:
        logger.error("Seeding failed", error=str(exc))
        return {"status": "error", "message": "Seeding failed"}


@app.get("/demo/database")
async def database_demo(_: None = Depends(require_api_auth)):

    try:
        from code.code_optimizer_ai.database.demo import demo_database_operations

        await demo_database_operations()
        return {"status": "success", "message": "Database demo completed"}
    except Exception as exc:
        logger.error("Database demo failed", error=str(exc))
        return {"status": "error", "message": "Database demo failed"}


@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_code(
    request: OptimizationRequest,
    pipeline: PipelineOrchestrator = Depends(get_pipeline),
    _: None = Depends(require_api_auth),
):

    start_time = datetime.now()

    try:
        if not request.code and not request.file_path:
            raise HTTPException(
                status_code=400, detail="Either 'code' or 'file_path' must be provided"
            )

        if request.code and len(request.code) > settings.MAX_INLINE_CODE_CHARS:
            raise HTTPException(
                status_code=413,
                detail=f"Code payload exceeds max size of {settings.MAX_INLINE_CODE_CHARS} characters",
            )

        if request.unit_test_command and not settings.ENABLE_API_UNIT_TEST_COMMAND_OVERRIDE:
            raise HTTPException(
                status_code=400,
                detail="Overriding unit_test_command via API is disabled",
            )

        priority = parse_priority(request.priority)

        if request.code:
            suggestions = await get_rl_optimizer().optimize_code(
                request.code,
                "inline_code",
                objective_weights=request.objective_weights,
                max_suggestions=request.max_suggestions,
                run_validation=request.run_validation,
                unit_test_command=(
                    request.unit_test_command
                    if settings.ENABLE_API_UNIT_TEST_COMMAND_OVERRIDE
                    else None
                ),
            )
            return build_response(
                task_id=f"inline_{int(start_time.timestamp())}",
                suggestions=suggestions,
                started_at=start_time,
                status="completed"
            )

        normalized_file = _normalize_user_path(request.file_path or "", expect_file=True)
        if normalized_file.suffix.lower() != ".py":
            raise HTTPException(status_code=400, detail="Only Python files (.py) are supported")

        task_id = await pipeline.submit_optimization_request(
            str(normalized_file), priority
        )
        return build_response(task_id, [], start_time, status="pending")

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Optimization failed", error=str(exc))
        raise HTTPException(status_code=500, detail=ERROR_GENERIC) from exc


@app.post("/optimize/file", response_model=OptimizationResponse)
async def optimize_file(
    file: UploadFile = File(...),
    priority: str = "medium",
    max_suggestions: int = settings.DEFAULT_MAX_SUGGESTIONS,
    run_validation: bool = False,
    unit_test_command: Optional[str] = None,
    _: None = Depends(require_api_auth),
):

    start_time = datetime.now()

    try:
        if not file.filename or not file.filename.lower().endswith(".py"):
            raise HTTPException(
                status_code=400, detail="Only Python files (.py) are supported"
            )

        if unit_test_command and not settings.ENABLE_API_UNIT_TEST_COMMAND_OVERRIDE:
            raise HTTPException(
                status_code=400,
                detail="Overriding unit_test_command via API is disabled",
            )

        if run_validation:
            raise HTTPException(
                status_code=400,
                detail="run_validation is not supported for uploaded files",
            )

        parse_priority(priority)  # validate even though optimization is immediate

        content_bytes = await _read_upload_limited(file, settings.MAX_UPLOAD_SIZE_BYTES)
        try:
            content = content_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=400, detail="File must be UTF-8 encoded"
            ) from exc

        if len(content) > settings.MAX_INLINE_CODE_CHARS:
            raise HTTPException(
                status_code=413,
                detail=f"Code payload exceeds max size of {settings.MAX_INLINE_CODE_CHARS} characters",
            )

        safe_filename = Path(file.filename).name
        suggestions = await get_rl_optimizer().optimize_code(
            content,
            safe_filename,
            max_suggestions=max_suggestions,
            run_validation=False,
            unit_test_command=(
                unit_test_command if settings.ENABLE_API_UNIT_TEST_COMMAND_OVERRIDE else None
            ),
        )
        return build_response(
            task_id=f"file_{int(start_time.timestamp())}",
            suggestions=suggestions,
            started_at=start_time,
            status="completed"
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("File optimization failed", error=str(exc))
        raise HTTPException(status_code=500, detail=ERROR_GENERIC) from exc


@app.post("/scan", response_model=ScanResponse)
async def scan_code(request: ScanRequest, _: None = Depends(require_api_auth)):
    
    start_time = datetime.now()
    
    try:
        target_dir = _normalize_user_path(request.directory_path, expect_directory=True)
        scan_result, statistics, complexity_hotspots, optimization_candidates = await _scan_directory_summary(
            str(target_dir),
            request.recursive,
            request.include_analysis,
        )
        
        response = ScanResponse(
            scan_id=f"scan_{int(start_time.timestamp())}",
            status="completed",
            files_scanned=scan_result.scanned_files,
            total_size_mb=scan_result.total_size_bytes / (1024 * 1024),
            scan_duration=scan_result.scan_duration,
            statistics=statistics,
            complexity_hotspots=complexity_hotspots,
            optimization_candidates=optimization_candidates,
            timestamp=start_time.isoformat()
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Code scanning failed", error=str(e))
        raise HTTPException(status_code=500, detail=ERROR_GENERIC)


@app.post("/scan/repository", response_model=RepositoryScanResponse)
async def scan_repository(
    request: RepositoryScanRequest,
    _: None = Depends(require_api_auth),
):
    start_time = datetime.now()
    clone_dir: Optional[Path] = None

    try:
        clone_dir = await asyncio.to_thread(
            _clone_github_repository,
            request.repo_url,
            request.ref,
        )
        scan_result, statistics, complexity_hotspots, optimization_candidates = await _scan_directory_summary(
            str(clone_dir),
            request.recursive,
            request.include_analysis,
        )

        return RepositoryScanResponse(
            scan_id=f"repo_scan_{int(start_time.timestamp())}",
            repository_url=request.repo_url,
            ref=request.ref,
            status="completed",
            files_scanned=scan_result.scanned_files,
            total_size_mb=scan_result.total_size_bytes / (1024 * 1024),
            scan_duration=scan_result.scan_duration,
            statistics=statistics,
            complexity_hotspots=complexity_hotspots,
            optimization_candidates=optimization_candidates,
            timestamp=start_time.isoformat(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Repository scanning failed", error=str(exc))
        raise HTTPException(status_code=500, detail=ERROR_GENERIC) from exc
    finally:
        if clone_dir is not None:
            await asyncio.to_thread(_cleanup_clone_directory, clone_dir)


@app.post("/optimize/repository", response_model=RepositoryOptimizationResponse)
async def optimize_repository(
    request: RepositoryOptimizationRequest,
    _: None = Depends(require_api_auth),
):
    start_time = datetime.now()
    clone_dir: Optional[Path] = None

    try:
        if request.unit_test_command and not settings.ENABLE_API_UNIT_TEST_COMMAND_OVERRIDE:
            raise HTTPException(
                status_code=400,
                detail="Overriding unit_test_command via API is disabled",
            )
        if request.run_validation:
            raise HTTPException(
                status_code=400,
                detail="run_validation is not supported for repository optimization",
            )

        clone_dir = await asyncio.to_thread(
            _clone_github_repository,
            request.repo_url,
            request.ref,
        )
        scan_result, statistics, _, _ = await _scan_directory_summary(
            str(clone_dir),
            True,
            False,
        )
        target_files = _select_repository_files(scan_result, statistics, request.max_files)
        project_context = _build_repository_file_context(scan_result, clone_dir)
        code_file_map = {
            str(getattr(code_file, "file_path", "") or ""): code_file
            for code_file in (scan_result.code_files or [])
        }

        optimizer = get_rl_optimizer()
        file_results: List[RepositoryOptimizationFileResult] = []
        files_with_suggestions = 0
        unit_cmd = (
            request.unit_test_command
            if settings.ENABLE_API_UNIT_TEST_COMMAND_OVERRIDE
            else None
        )
        target_file_map = {
            str(getattr(code_file, "file_path", "") or ""): code_file
            for code_file in target_files
        }
        static_metrics_by_path: Dict[str, Dict[str, Any]] = {}
        for code_file in target_files:
            static_metrics = dict(project_context.get(
                code_file.file_path,
                {"project_total_files": scan_result.scanned_files},
            ))
            if request.enable_cross_file_context:
                static_metrics["cross_file_context"] = _build_cross_file_context(
                    code_file.file_path,
                    project_context,
                    code_file_map,
                    max_related_files=request.cross_file_context_max_files,
                    snippet_chars=request.cross_file_context_snippet_chars,
                )
            static_metrics_by_path[code_file.file_path] = static_metrics

        coordinated_suggestions_by_path: Dict[str, List[Any]] = {}
        if request.enable_batch_coordination:
            target_paths = [code_file.file_path for code_file in target_files]
            batches = _build_coordination_batches(
                target_paths,
                project_context,
                max_group_size=request.batch_coordination_max_group_size,
            )
            for batch_paths in batches:
                if len(batch_paths) < 2:
                    continue

                batch_payload: List[Dict[str, Any]] = []
                for batch_path in batch_paths:
                    candidate = target_file_map.get(batch_path)
                    if not candidate:
                        continue
                    batch_payload.append(
                        {
                            "file_path": batch_path,
                            "code": _truncate_code_excerpt(
                                getattr(candidate, "content", ""),
                                request.batch_coordination_max_code_chars_per_file,
                            ),
                            "static_metrics": static_metrics_by_path.get(batch_path, {}),
                        }
                    )

                if len(batch_payload) < 2:
                    continue

                batch_context = {
                    "repo_url": request.repo_url,
                    "ref": request.ref,
                    "objective_weights": request.objective_weights or {},
                    "batch_file_count": len(batch_payload),
                }
                batch_suggestions = await code_analyzer.generate_repository_optimizations(
                    batch_payload,
                    max_suggestions_per_file=request.max_suggestions_per_file,
                    shared_context=batch_context,
                )
                if not isinstance(batch_suggestions, dict):
                    continue

                for batch_path, suggestions in batch_suggestions.items():
                    if batch_path not in target_file_map or not isinstance(suggestions, list):
                        continue
                    coordinated_suggestions_by_path[batch_path] = suggestions[: request.max_suggestions_per_file]

        for code_file in target_files:
            suggestions = coordinated_suggestions_by_path.get(code_file.file_path, [])
            if not suggestions:
                suggestions = await optimizer.optimize_code(
                    code_file.content,
                    code_file.file_path,
                    objective_weights=request.objective_weights,
                    max_suggestions=request.max_suggestions_per_file,
                    run_validation=False,
                    unit_test_command=unit_cmd,
                    analysis_static_metrics=static_metrics_by_path.get(
                        code_file.file_path,
                        {"project_total_files": scan_result.scanned_files},
                    ),
                )
            serialized = _serialize_suggestions(suggestions)
            if serialized:
                files_with_suggestions += 1
            file_results.append(
                RepositoryOptimizationFileResult(
                    file_path=code_file.file_path,
                    suggestions=serialized,
                )
            )

        return RepositoryOptimizationResponse(
            repository_url=request.repo_url,
            ref=request.ref,
            status="completed",
            files_scanned=scan_result.scanned_files,
            files_analyzed=len(target_files),
            files_with_suggestions=files_with_suggestions,
            results=file_results,
            processing_time=(datetime.now() - start_time).total_seconds(),
            timestamp=start_time.isoformat(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Repository optimization failed", error=str(exc))
        raise HTTPException(status_code=500, detail=ERROR_GENERIC) from exc
    finally:
        if clone_dir is not None:
            await asyncio.to_thread(_cleanup_clone_directory, clone_dir)


@app.get("/pipeline/status", response_model=PipelineStatusResponse)
async def get_pipeline_status(
    pipeline: PipelineOrchestrator = Depends(get_pipeline),
    _: None = Depends(require_api_auth),
):
    
    try:
        status = await pipeline.get_pipeline_status()
        return PipelineStatusResponse(**status)
        
    except Exception as e:
        logger.error("Failed to get pipeline status", error=str(e))
        raise HTTPException(status_code=500, detail=ERROR_GENERIC)


@app.post("/train", response_model=TrainingResponse)
async def train_rl_model(request: TrainingRequest, _: None = Depends(require_api_auth)):

    start_time = datetime.now()
    training_id = create_job_id()

    try:
        config = TrainingConfig(
            synthetic_samples=request.synthetic_samples,
            seed=request.seed,
            holdout_fraction=request.holdout_fraction,
            epochs=request.epochs,
            steps_per_epoch=request.steps_per_epoch,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            shuffle=True,
            require_td_error_improvement=request.require_td_improvement,
        )

        if request.background:
            # Persist "initiated" record so GET /train/{id} works immediately.
            create_job_status(training_id)

            # Run in a thread so the API stays responsive
            future = asyncio.get_running_loop().run_in_executor(
                None, run_training_job, config, training_id
            )
            future.add_done_callback(
                lambda f: logger.error(
                    "background_training_failed",
                    training_id=training_id,
                    error=str(f.exception()),
                )
                if f.exception()
                else None
            )
            return TrainingResponse(
                training_id=training_id,
                status="initiated",
                training_stats={"mode": "offline_pretrain", "background": True},
                timestamp=start_time.isoformat(),
            )
        else:
            # Synchronous (blocking) -- useful for small jobs or testing
            result = await asyncio.to_thread(run_training_job, config, training_id)
            return TrainingResponse(
                training_id=result.job_id,
                status=result.status,
                training_stats={
                    "updates": result.updates,
                    "final_epoch_loss": result.final_epoch_loss,
                    "holdout_mean_td_error": result.holdout_mean_td_error,
                    "gate_passed": result.gate_passed,
                    "gate_reason": result.gate_reason,
                },
                model_path=result.model_path,
                timestamp=start_time.isoformat(),
            )

    except Exception as e:
        logger.error("Training request failed", error=str(e))
        raise HTTPException(status_code=500, detail=ERROR_GENERIC)


@app.get("/train/{job_id}")
async def get_training_status(job_id: str, _: None = Depends(require_api_auth)):
    status = load_job_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Training job '{job_id}' not found")
    from dataclasses import asdict
    return asdict(status)


@app.get("/train")
async def list_training_jobs(
    limit: int = Query(default=20, ge=1, le=100),
    _: None = Depends(require_api_auth),
):
    return {"jobs": list_jobs(limit=limit)}


@app.get("/metrics")
async def get_performance_metrics(
    pipeline: PipelineOrchestrator = Depends(get_pipeline),
    _: None = Depends(require_api_auth),
):
    
    try:
        # Get performance summary
        performance_summary = performance_profiler.get_performance_summary()
        
        # Get pipeline metrics
        pipeline_status = await pipeline.get_pipeline_status()
        
        # Combine metrics
        combined_metrics = {
            "timestamp": datetime.now().isoformat(),
            "performance_profiling": performance_summary,
            "pipeline_status": pipeline_status,
            "system_info": {
                "version": settings.APP_VERSION,
                "uptime": "running",  # Would calculate actual uptime
                "active_components": [
                    "code_scanner",
                    "llm_analyzer", 
                    "performance_profiler",
                    "rl_optimizer",
                    "pipeline_orchestrator"
                ]
            }
        }
        
        return combined_metrics
        
    except Exception as e:
        logger.error("Failed to get metrics", error=str(e))
        raise HTTPException(status_code=500, detail=ERROR_GENERIC)


@app.post("/monitor/directory")
async def add_monitored_directory(
    directory_path: str = Query(...),
    pipeline: PipelineOrchestrator = Depends(get_pipeline),
    _: None = Depends(require_api_auth),
):
    
    try:
        normalized_directory = _normalize_user_path(directory_path, expect_directory=True)
        pipeline.add_watched_directory(str(normalized_directory))
        
        return {
            "status": "success",
            "message": f"Directory {normalized_directory} added to monitoring",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to add monitored directory", error=str(e))
        raise HTTPException(status_code=500, detail=ERROR_GENERIC)


@app.get("/suggestions/categories")
async def get_optimization_categories():

    _CATEGORY_META: Dict[str, Dict[str, Any]] = {
        "algorithm_change": {
            "description": "Replace inefficient algorithms with optimized alternatives",
            "examples": ["O(n²) → O(n log n)", "nested loops → hash lookup", "linear search → binary search"],
            "typical_improvement": "20-50%",
            "implementation_effort": "high",
        },
        "data_structure_optimization": {
            "description": "Optimize data structure usage for better performance",
            "examples": ["list → set for membership", "list → dict for lookups", "nested lists → tuples"],
            "typical_improvement": "15-40%",
            "implementation_effort": "medium",
        },
        "caching_strategy": {
            "description": "Implement caching to reduce computation overhead",
            "examples": ["memoize pure functions", "cache database queries", "cache computed values"],
            "typical_improvement": "30-70%",
            "implementation_effort": "medium",
        },
        "parallelization": {
            "description": "Add parallel processing for CPU-intensive operations",
            "examples": ["threading for I/O", "multiprocessing for CPU", "async/await patterns"],
            "typical_improvement": "40-80%",
            "implementation_effort": "high",
        },
        "memory_optimization": {
            "description": "Optimize memory usage and garbage collection",
            "examples": ["generators vs lists", "slots on classes", "context managers for resources"],
            "typical_improvement": "10-30%",
            "implementation_effort": "low",
        },
        "io_optimization": {
            "description": "Optimize input/output operations for better performance",
            "examples": ["async I/O", "buffered operations", "batch file operations"],
            "typical_improvement": "25-60%",
            "implementation_effort": "medium",
        },
        "database_optimization": {
            "description": "Improve database queries and access patterns",
            "examples": ["add indexes", "batch inserts", "connection pooling"],
            "typical_improvement": "20-70%",
            "implementation_effort": "medium",
        },
        "network_optimization": {
            "description": "Reduce network latency and bandwidth usage",
            "examples": ["connection reuse", "payload compression", "request batching"],
            "typical_improvement": "15-50%",
            "implementation_effort": "medium",
        },
        "security_optimization": {
            "description": "Harden code against security vulnerabilities",
            "examples": ["input sanitization", "parameterized queries", "secure defaults"],
            "typical_improvement": "N/A (security)",
            "implementation_effort": "medium",
        },
        "code_quality_optimization": {
            "description": "Improve code readability, structure, and correctness",
            "examples": ["extract functions", "simplify conditionals", "remove dead code"],
            "typical_improvement": "5-15%",
            "implementation_effort": "low",
        },
        "function_optimization": {
            "description": "Optimize individual function performance and structure",
            "examples": ["reduce call overhead", "inline hot paths", "simplify signatures"],
            "typical_improvement": "10-25%",
            "implementation_effort": "low",
        },
        "string_optimization": {
            "description": "Optimize string operations and memory patterns",
            "examples": ["+= in loops → join()", "f-strings over format()", "pre-compile regex"],
            "typical_improvement": "10-30%",
            "implementation_effort": "low",
        },
        "mathematical_optimization": {
            "description": "Optimize mathematical computations",
            "examples": ["vectorize with numpy", "avoid redundant calculations", "use math identities"],
            "typical_improvement": "15-40%",
            "implementation_effort": "medium",
        },
        "loop_optimization": {
            "description": "Optimize loop structures and iteration patterns",
            "examples": ["loop unrolling", "early termination", "list comprehensions"],
            "typical_improvement": "10-30%",
            "implementation_effort": "low",
        },
        "exception_handling_optimization": {
            "description": "Improve exception handling for correctness and performance",
            "examples": ["narrow except clauses", "avoid exceptions for flow control", "LBYL vs EAFP"],
            "typical_improvement": "5-15%",
            "implementation_effort": "low",
        },
        "import_optimization": {
            "description": "Optimize import structure and lazy loading",
            "examples": ["lazy imports", "remove unused imports", "defer heavy modules"],
            "typical_improvement": "5-20% (startup)",
            "implementation_effort": "low",
        },
        "configuration_optimization": {
            "description": "Optimize runtime configuration and settings management",
            "examples": ["environment-based config", "cached settings", "validation at startup"],
            "typical_improvement": "5-10%",
            "implementation_effort": "low",
        },
        "api_optimization": {
            "description": "Optimize API design and request handling",
            "examples": ["pagination", "response caching", "field selection"],
            "typical_improvement": "20-50%",
            "implementation_effort": "medium",
        },
        "concurrency_optimization": {
            "description": "Improve concurrent execution and synchronization",
            "examples": ["reduce lock contention", "asyncio.gather", "thread pool sizing"],
            "typical_improvement": "20-60%",
            "implementation_effort": "high",
        },
        "resource_management_optimization": {
            "description": "Optimize resource lifecycle and cleanup",
            "examples": ["context managers", "connection pooling", "file handle limits"],
            "typical_improvement": "10-25%",
            "implementation_effort": "medium",
        },
        "logging_optimization": {
            "description": "Optimize logging for production performance",
            "examples": ["structured logging", "lazy formatting", "log level guards"],
            "typical_improvement": "5-15%",
            "implementation_effort": "low",
        },
        "testing_optimization": {
            "description": "Improve test performance and coverage efficiency",
            "examples": ["parallel test execution", "fixture reuse", "test deduplication"],
            "typical_improvement": "20-50% (CI time)",
            "implementation_effort": "medium",
        },
        "build_optimization": {
            "description": "Optimize build and deployment pipelines",
            "examples": ["incremental builds", "dependency caching", "multi-stage Docker"],
            "typical_improvement": "30-60% (build time)",
            "implementation_effort": "medium",
        },
        "documentation_optimization": {
            "description": "Improve code documentation and API docs",
            "examples": ["add docstrings", "type annotations", "usage examples"],
            "typical_improvement": "N/A (maintainability)",
            "implementation_effort": "low",
        },
        "no_change": {
            "description": "Code is already well-optimized for the given objectives",
            "examples": [],
            "typical_improvement": "0%",
            "implementation_effort": "none",
        },
    }

    categories: Dict[str, Dict[str, Any]] = {}
    for action_type in PRODUCTION_ACTION_TYPES:
        meta = _CATEGORY_META.get(action_type)
        if meta:
            categories[action_type] = meta
        else:
            # Fallback for any action type without hand-written metadata.
            categories[action_type] = {
                "description": action_type.replace("_", " ").title(),
                "examples": [],
                "typical_improvement": "varies",
                "implementation_effort": "medium",
            }

    return {
        "categories": categories,
        "total": len(categories),
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "code.code_optimizer_ai.api_main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
