import argparse
import asyncio
import hashlib
import sys
from pathlib import Path

# Ensure repository root is on path for package imports
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.code_optimizer_ai.database.connection import DatabaseManager, OptimizationRecord
from code.code_optimizer_ai.utils.logger import get_logger

logger = get_logger(__name__)
MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
ROLLBACK_DIR = MIGRATIONS_DIR / "rollback"
PGVECTOR_MIGRATION = "add_pgvector_extension.sql"
MIGRATION_TRACKING_TABLE = "schema_migrations"


def _migration_checksum(sql: str) -> str:
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()


def _ordered_migrations(paths: list[Path]) -> list[Path]:
    # Ensure pgvector extension runs before any vector-typed table migration.
    return sorted(
        paths,
        key=lambda path: (0 if path.name == PGVECTOR_MIGRATION else 1, path.name.lower()),
    )


def _read_migration_sql(migration_path: Path) -> str:
    return migration_path.read_text(encoding="utf-8").strip()


async def _ensure_tracking_table(conn) -> None:
    await conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {MIGRATION_TRACKING_TABLE} (
            version VARCHAR(255) PRIMARY KEY,
            checksum VARCHAR(64) NOT NULL,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )


async def _load_applied_checksums(conn) -> dict[str, str]:
    rows = await conn.fetch(
        f"""
        SELECT version, checksum
        FROM {MIGRATION_TRACKING_TABLE}
        """
    )
    return {str(row["version"]): str(row["checksum"]) for row in rows}


def _resolve_migration_paths(migration_name: str | None = None) -> list[Path]:
    migration_paths = _ordered_migrations(list(MIGRATIONS_DIR.glob("*.sql")))
    if migration_name:
        migration_paths = [path for path in migration_paths if path.name == migration_name]
        if not migration_paths:
            raise FileNotFoundError(f"Migration file not found: {migration_name}")
    return migration_paths


async def _apply_migrations_on_connection(conn, migration_paths: list[Path]) -> tuple[int, int]:
    await _ensure_tracking_table(conn)
    applied_checksums = await _load_applied_checksums(conn)

    applied_count = 0
    skipped_count = 0
    for migration_path in migration_paths:
        sql = _read_migration_sql(migration_path)
        if not sql:
            logger.warning("Skipping empty migration file", migration=migration_path.name)
            skipped_count += 1
            continue

        version = migration_path.name
        checksum = _migration_checksum(sql)
        existing_checksum = applied_checksums.get(version)
        if existing_checksum is not None:
            if existing_checksum != checksum:
                raise RuntimeError(
                    "Migration checksum mismatch detected. "
                    f"version={version} expected={existing_checksum} actual={checksum}. "
                    "Refuse to continue to prevent schema drift."
                )
            logger.info("Skipping already applied migration", migration=version)
            skipped_count += 1
            continue

        async with conn.transaction():
            await conn.execute(sql)
            await conn.execute(
                f"""
                INSERT INTO {MIGRATION_TRACKING_TABLE} (version, checksum, applied_at)
                VALUES ($1, $2, NOW())
                """,
                version,
                checksum,
            )

        applied_checksums[version] = checksum
        applied_count += 1
        logger.info("Applied additive migration", migration=version)

    return applied_count, skipped_count


async def _migration_status_on_connection(conn, migration_paths: list[Path]) -> list[dict[str, str]]:
    await _ensure_tracking_table(conn)
    applied_checksums = await _load_applied_checksums(conn)

    statuses: list[dict[str, str]] = []
    for migration_path in migration_paths:
        version = migration_path.name
        sql = _read_migration_sql(migration_path)
        checksum = _migration_checksum(sql) if sql else ""
        existing_checksum = applied_checksums.get(version)

        if not sql:
            status = "empty"
        elif existing_checksum is None:
            status = "pending"
        elif existing_checksum == checksum:
            status = "applied"
        else:
            status = "checksum_mismatch"

        rollback_exists = (ROLLBACK_DIR / version).exists()
        statuses.append(
            {
                "version": version,
                "status": status,
                "rollback": "present" if rollback_exists else "missing",
            }
        )

    known_versions = {item["version"] for item in statuses}
    for applied_version in sorted(set(applied_checksums.keys()) - known_versions):
        statuses.append(
            {
                "version": applied_version,
                "status": "applied_missing_file",
                "rollback": "unknown",
            }
        )

    return statuses


async def apply_additive_migrations(migration_name: str | None = None):
    """Apply additive SQL migrations from database/migrations/ in lexical order."""
    db_manager = DatabaseManager()

    try:
        await db_manager.initialize()
        migration_paths = _resolve_migration_paths(migration_name)
        if not migration_paths:
            logger.info("No additive migration files found")
            return

        async with db_manager.connection_pool.acquire() as conn:
            applied_count, skipped_count = await _apply_migrations_on_connection(conn, migration_paths)
            logger.info(
                "Additive migration run complete",
                applied=applied_count,
                skipped=skipped_count,
                total=len(migration_paths),
            )
    except Exception as exc:
        logger.error("Failed to apply additive migrations", error=str(exc))
        raise
    finally:
        await db_manager.close()


async def create_database_schema(include_additive_migrations: bool = True):
    db_manager = DatabaseManager()

    try:
        await db_manager.initialize()
        logger.info("Database schema created successfully")

        if include_additive_migrations:
            migration_paths = _resolve_migration_paths()
            if migration_paths:
                async with db_manager.connection_pool.acquire() as conn:
                    applied_count, skipped_count = await _apply_migrations_on_connection(conn, migration_paths)
                    logger.info(
                        "Schema additive migration run complete",
                        applied=applied_count,
                        skipped=skipped_count,
                        total=len(migration_paths),
                    )

        async with db_manager.connection_pool.acquire() as conn:
            tables = await conn.fetch(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_type = 'BASE TABLE'
                ORDER BY table_name
                """
            )
            logger.info("Current tables", count=len(tables))
            for table in tables:
                logger.info("table", name=table["table_name"])
    except Exception as exc:
        logger.error("Failed to create database schema", error=str(exc))
        raise
    finally:
        await db_manager.close()


async def migration_status(migration_name: str | None = None) -> list[dict[str, str]]:
    db_manager = DatabaseManager()
    try:
        await db_manager.initialize()
        migration_paths = _resolve_migration_paths(migration_name)
        async with db_manager.connection_pool.acquire() as conn:
            statuses = await _migration_status_on_connection(conn, migration_paths)

        for item in statuses:
            logger.info(
                "migration_status",
                version=item["version"],
                status=item["status"],
                rollback=item["rollback"],
            )
        return statuses
    except Exception as exc:
        logger.error("Failed to read migration status", error=str(exc))
        raise
    finally:
        await db_manager.close()


async def verify_migrations(migration_name: str | None = None) -> bool:
    statuses = await migration_status(migration_name)
    bad_statuses = {"checksum_mismatch", "empty", "applied_missing_file"}
    failures = [item for item in statuses if item["status"] in bad_statuses]
    if failures:
        logger.error("Migration verification failed", failures=failures)
        return False
    logger.info("Migration verification passed")
    return True


async def rollback_additive_migrations(steps: int = 1) -> None:
    db_manager = DatabaseManager()
    try:
        await db_manager.initialize()
        steps = max(1, int(steps))
        async with db_manager.connection_pool.acquire() as conn:
            await _ensure_tracking_table(conn)
            rows = await conn.fetch(
                f"""
                SELECT version
                FROM {MIGRATION_TRACKING_TABLE}
                ORDER BY applied_at DESC, version DESC
                LIMIT $1
                """,
                steps,
            )
            versions = [str(row["version"]) for row in rows]
            if not versions:
                logger.info("No applied additive migrations to roll back")
                return

            for version in versions:
                rollback_path = ROLLBACK_DIR / version
                if not rollback_path.exists():
                    raise FileNotFoundError(
                        f"Rollback file not found for migration '{version}': {rollback_path}"
                    )
                rollback_sql = _read_migration_sql(rollback_path)
                if not rollback_sql:
                    raise RuntimeError(f"Rollback file is empty for migration '{version}'")

                async with conn.transaction():
                    await conn.execute(rollback_sql)
                    await conn.execute(
                        f"DELETE FROM {MIGRATION_TRACKING_TABLE} WHERE version = $1",
                        version,
                    )
                logger.info("Rolled back additive migration", migration=version)
    except Exception as exc:
        logger.error("Failed to roll back additive migrations", error=str(exc))
        raise
    finally:
        await db_manager.close()


async def check_database_connection() -> bool:
    db_manager = DatabaseManager()
    try:
        await db_manager.initialize()
        await db_manager.close()
        logger.info("Database connection test passed")
        return True
    except Exception as exc:
        logger.error("Database connection test failed", error=str(exc))
        return False


async def check_phase_a_schema() -> bool:
    db_manager = DatabaseManager()
    try:
        await db_manager.initialize()
        required = {"optimization_requests", "phase_a_candidate_evaluations"}
        async with db_manager.connection_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_type = 'BASE TABLE'
                  AND table_name = ANY($1::text[])
                """,
                list(required),
            )
        existing = {row["table_name"] for row in rows}
        missing = sorted(required - existing)
        if missing:
            logger.error("Phase A schema check failed", missing_tables=missing)
            return False
        logger.info("Phase A schema check passed", tables=sorted(existing))
        return True
    except Exception as exc:
        logger.error("Phase A schema check failed", error=str(exc))
        return False
    finally:
        await db_manager.close()


async def check_raeo_schema() -> bool:
    db_manager = DatabaseManager()
    try:
        await db_manager.initialize()
        required = {"transition_memory", "qd_archive", "prompt_template_registry"}
        async with db_manager.connection_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_type = 'BASE TABLE'
                  AND table_name = ANY($1::text[])
                """,
                list(required),
            )
        existing = {row["table_name"] for row in rows}
        missing = sorted(required - existing)
        if missing:
            logger.error("RAEO schema check failed", missing_tables=missing)
            return False
        logger.info("RAEO schema check passed", tables=sorted(existing))
        return True
    except Exception as exc:
        logger.error("RAEO schema check failed", error=str(exc))
        return False
    finally:
        await db_manager.close()


async def check_pgvector_extension() -> bool:
    db_manager = DatabaseManager()
    try:
        await db_manager.initialize()
        async with db_manager.connection_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT EXISTS(
                    SELECT 1
                    FROM pg_extension
                    WHERE extname = 'vector'
                ) AS has_vector
                """
            )
        has_vector = bool(row["has_vector"])
        if not has_vector:
            logger.error("pgvector extension check failed", extension="vector")
            return False
        logger.info("pgvector extension check passed", extension="vector")
        return True
    except Exception as exc:
        logger.error("pgvector extension check failed", error=str(exc))
        return False
    finally:
        await db_manager.close()


async def seed_sample_data():
    db_manager = DatabaseManager()

    try:
        await db_manager.initialize()

        sample_records = [
            OptimizationRecord(
                code_hash="abc123def456",
                file_path="/test/slow_sort.py",
                complexity_before=75.0,
                complexity_after=25.0,
                execution_time_before=5.2,
                execution_time_after=0.8,
                memory_usage_before=128.5,
                memory_usage_after=45.2,
                optimization_type="algorithm_change",
                success=True,
                improvement_factor=6.5,
            ),
            OptimizationRecord(
                code_hash="def456ghi789",
                file_path="/test/fibonacci.py",
                complexity_before=95.0,
                complexity_after=15.0,
                execution_time_before=30.5,
                execution_time_after=0.000003,
                memory_usage_before=256.0,
                memory_usage_after=8.7,
                optimization_type="dynamic_programming",
                success=True,
                improvement_factor=10166666.67,
            ),
            OptimizationRecord(
                code_hash="ghi789jkl012",
                file_path="/test/duplicate_detection.py",
                complexity_before=85.0,
                complexity_after=20.0,
                execution_time_before=12.3,
                execution_time_after=0.1,
                memory_usage_before=145.2,
                memory_usage_after=32.1,
                optimization_type="data_structure",
                success=True,
                improvement_factor=123.0,
            ),
        ]

        for record in sample_records:
            record_id = await db_manager.store_optimization_record(record)
            logger.info("Seeded record", record_id=record_id, file_path=record.file_path)

        logger.info("Sample data seeded successfully")
    except Exception as exc:
        logger.error("Failed to seed sample data", error=str(exc))
        raise
    finally:
        await db_manager.close()


def main():
    parser = argparse.ArgumentParser(description="Database migration script")
    parser.add_argument("--check", action="store_true", help="Check database connection")
    parser.add_argument("--migrate", action="store_true", help="Create database schema")
    parser.add_argument("--migrate-v2", action="store_true", help="Apply additive V2 SQL migrations (legacy alias for --up)")
    parser.add_argument("--up", action="store_true", help="Apply additive SQL migrations")
    parser.add_argument("--down", action="store_true", help="Roll back latest additive SQL migration(s)")
    parser.add_argument("--steps", type=int, default=1, help="Number of additive migrations to roll back with --down")
    parser.add_argument("--status", action="store_true", help="Show additive migration status")
    parser.add_argument("--verify", action="store_true", help="Verify additive migrations (checksums and tracking integrity)")
    parser.add_argument("--migration-name", type=str, default=None, help="Apply one additive migration file")
    parser.add_argument("--check-phase-a", action="store_true", help="Verify Phase A evidence schema tables")
    parser.add_argument("--check-raeo", action="store_true", help="Verify RAEO schema tables")
    parser.add_argument("--check-pgvector", action="store_true", help="Verify pgvector extension is installed")
    parser.add_argument("--seed", action="store_true", help="Seed sample data")
    parser.add_argument("--all", action="store_true", help="Run all operations")

    args = parser.parse_args()

    if not any(
        [
            args.check,
            args.migrate,
            args.migrate_v2,
            args.up,
            args.down,
            args.status,
            args.verify,
            args.check_phase_a,
            args.check_raeo,
            args.check_pgvector,
            args.seed,
            args.all,
        ]
    ):
        parser.print_help()
        return

    async def run_operations():
        if args.check or args.all:
            await check_database_connection()

        if args.migrate or args.all:
            await create_database_schema(include_additive_migrations=True)

        if args.migrate_v2 or args.up:
            await apply_additive_migrations(args.migration_name)

        if args.down:
            await rollback_additive_migrations(args.steps)

        if args.status:
            await migration_status(args.migration_name)

        if args.verify:
            ok = await verify_migrations(args.migration_name)
            if not ok:
                raise RuntimeError("Migration verification failed")

        if args.check_phase_a:
            ok = await check_phase_a_schema()
            if not ok:
                raise RuntimeError("Phase A schema check failed")

        if args.check_raeo:
            ok = await check_raeo_schema()
            if not ok:
                raise RuntimeError("RAEO schema check failed")

        if args.check_pgvector:
            ok = await check_pgvector_extension()
            if not ok:
                raise RuntimeError("pgvector extension check failed")

        if args.seed or args.all:
            await seed_sample_data()

    try:
        asyncio.run(run_operations())
        logger.info("Database migration completed successfully")
    except KeyboardInterrupt:
        logger.info("Migration interrupted by user")
    except Exception as exc:
        logger.error("Migration failed", error=str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
