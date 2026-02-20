import asyncio
import sys
import os
from pathlib import Path

# Ensure repository root is on path for package imports
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.code_optimizer_ai.database.connection import DatabaseManager, OptimizationRecord
from code.code_optimizer_ai.utils.logger import get_logger

logger = get_logger(__name__)

async def create_database_schema():
    db_manager = DatabaseManager()
    
    try:
        await db_manager.initialize()
        logger.info("Database schema created successfully!")
        
        # Display table information
        async with db_manager.connection_pool.acquire() as conn:
            # Get list of tables
            tables = await conn.fetch("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """)
            
            logger.info("Created tables:")
            for table in tables:
                logger.info(f"  • {table['table_name']}")
            
            # Get table counts
            try:
                count_result = await conn.fetchval(
                    "SELECT COUNT(*) FROM optimization_records"
                )
                logger.info(f"Optimization records table created with {count_result} records")
            except Exception:
                logger.info("Optimization records table created (empty)")
                
    except Exception as e:
        logger.error(f"Failed to create database schema: {e}")
        raise
    finally:
        await db_manager.close()

async def check_database_connection():
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize()
        await db_manager.close()
        logger.info("Database connection test passed")
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False

async def seed_sample_data():
    from code.code_optimizer_ai.database.connection import OptimizationRecord
    
    db_manager = DatabaseManager()
    
    try:
        await db_manager.initialize()
        
        # Sample optimization records
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
                improvement_factor=6.5
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
                improvement_factor=10166666.67
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
                improvement_factor=123.0
            )
        ]
        
        for record in sample_records:
            record_id = await db_manager.store_optimization_record(record)
            logger.info(f"Seeded record: {record_id} - {record.file_path}")
        
        logger.info("Sample data seeded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to seed sample data: {e}")
        raise
    finally:
        await db_manager.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Database migration script")
    parser.add_argument("--check", action="store_true", help="Check database connection")
    parser.add_argument("--migrate", action="store_true", help="Create database schema")
    parser.add_argument("--seed", action="store_true", help="Seed sample data")
    parser.add_argument("--all", action="store_true", help="Run all operations")
    
    args = parser.parse_args()
    
    if not any([args.check, args.migrate, args.seed, args.all]):
        parser.print_help()
        return
    
    async def run_operations():
        if args.check or args.all:
            await check_database_connection()
        
        if args.migrate or args.all:
            await create_database_schema()
            
        if args.seed or args.all:
            await seed_sample_data()
    
    # Run operations
    try:
        asyncio.run(run_operations())
        logger.info("Database migration completed successfully!")
    except KeyboardInterrupt:
        logger.info("Migration interrupted by user")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
