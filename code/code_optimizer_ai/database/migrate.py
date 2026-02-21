import argparse
import asyncio
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


async def apply_additive_migrations(migration_name: str | None = None):
    """Apply additive SQL migrations from database/migrations/ in lexical order."""
    db_manager = DatabaseManager()

    try:
        await db_manager.initialize()
        migration_paths = sorted(MIGRATIONS_DIR.glob("*.sql"))
        if migration_name:
            migration_paths = [path for path in migration_paths if path.name == migration_name]
            if not migration_paths:
                raise FileNotFoundError(f"Migration file not found: {migration_name}")

        if not migration_paths:
            logger.info("No additive migration files found")
            return

        async with db_manager.connection_pool.acquire() as conn:
            for migration_path in migration_paths:
                sql = migration_path.read_text(encoding="utf-8").strip()
                if not sql:
                    continue
                await conn.execute(sql)
                logger.info("Applied additive migration", migration=migration_path.name)
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
            migration_paths = sorted(MIGRATIONS_DIR.glob("*.sql"))
            if migration_paths:
                async with db_manager.connection_pool.acquire() as conn:
                    for migration_path in migration_paths:
                        sql = migration_path.read_text(encoding="utf-8").strip()
                        if not sql:
                            continue
                        await conn.execute(sql)
                        logger.info("Applied additive migration", migration=migration_path.name)

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
    parser.add_argument("--migrate-v2", action="store_true", help="Apply additive V2 SQL migrations")
    parser.add_argument("--migration-name", type=str, default=None, help="Apply one additive migration file")
    parser.add_argument("--check-phase-a", action="store_true", help="Verify Phase A evidence schema tables")
    parser.add_argument("--seed", action="store_true", help="Seed sample data")
    parser.add_argument("--all", action="store_true", help="Run all operations")

    args = parser.parse_args()

    if not any([args.check, args.migrate, args.migrate_v2, args.check_phase_a, args.seed, args.all]):
        parser.print_help()
        return

    async def run_operations():
        if args.check or args.all:
            await check_database_connection()

        if args.migrate or args.all:
            await create_database_schema(include_additive_migrations=True)

        if args.migrate_v2:
            await apply_additive_migrations(args.migration_name)

        if args.check_phase_a:
            ok = await check_phase_a_schema()
            if not ok:
                raise RuntimeError("Phase A schema check failed")

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
