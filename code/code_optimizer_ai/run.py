import asyncio
import os
import socket
import sys
import time
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import uvicorn


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]  # workspace root (two levels above code/code_optimizer_ai/)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def log_info(msg: str) -> None:
    print(f"[INFO] {msg}")


def log_warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def str_to_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def parse_host_port(url: str) -> Optional[Tuple[str, int]]:
    parsed = urlparse(url)
    if parsed.hostname and parsed.port:
        return parsed.hostname, parsed.port
    return None


def wait_for_service(name: str, host: str, port: int, timeout: int = 30) -> bool:
    log_info(f"Waiting for {name} at {host}:{port} (timeout {timeout}s)")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                log_info(f"{name} is ready")
                return True
        except OSError:
            time.sleep(1)
    log_warn(f"Timeout waiting for {name} at {host}:{port}")
    return False


async def run_migrations_if_needed(debug: bool) -> None:
    if debug:
        return
    try:
        from code.code_optimizer_ai.database.migrate import create_database_schema

        log_info("Running database migrations...")
        await create_database_schema()
    except Exception as exc:  # pragma: no cover - defensive logging
        log_warn(f"Migration failed, continuing anyway... ({exc})")


def ensure_directories() -> None:
    targets = [Path("/app/data"), Path("/app/models"), Path("/app/logs")]
    for target in targets:
        try:
            target.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            fallback = SCRIPT_DIR / target.name
            fallback.mkdir(parents=True, exist_ok=True)
            log_warn(f"Cannot create {target}, using {fallback} instead")


def main() -> None:
    debug = str_to_bool(os.getenv("DEBUG"), default=False)
    log_level = os.getenv("LOG_LEVEL", "info").lower()

    ensure_directories()

    # Wait for dependencies in production
    if not debug:
        db_url = os.getenv("DATABASE_URL")
        if db_url:
            parsed = parse_host_port(db_url)
            if parsed:
                wait_for_service("database", parsed[0], parsed[1], timeout=60)
            else:
                log_warn("DATABASE_URL found but host/port could not be parsed; skipping wait")

        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            parsed = parse_host_port(redis_url)
            if parsed:
                wait_for_service("redis", parsed[0], parsed[1], timeout=30)
            else:
                log_warn("REDIS_URL found but host/port could not be parsed; skipping wait")

    # Run migrations (production only)
    asyncio.run(run_migrations_if_needed(debug))

    log_info("Starting Velox Optimization Engine...")

    if debug:
        uvicorn.run(
            "code.code_optimizer_ai.api_main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level=log_level,
        )
    else:
        forwarded_allow_ips = os.getenv("FORWARDED_ALLOW_IPS", "127.0.0.1")
        if not forwarded_allow_ips.strip():
            forwarded_allow_ips = "127.0.0.1"
        if forwarded_allow_ips.strip() == "*":
            log_warn("FORWARDED_ALLOW_IPS='*' is insecure; forcing '127.0.0.1'")
            forwarded_allow_ips = "127.0.0.1"
        uvicorn.run(
            "code.code_optimizer_ai.api_main:app",
            host="0.0.0.0",
            port=8000,
            workers=4,
            log_level=log_level,
            access_log=True,
            proxy_headers=True,
            forwarded_allow_ips=forwarded_allow_ips,
        )


if __name__ == "__main__":
    main()
