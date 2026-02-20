import asyncio
import json
from datetime import datetime
from typing import Dict, Any

import sys
from pathlib import Path

# Ensure repository root is on path for package imports
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.code_optimizer_ai.database.connection import (
    DatabaseManager,
    CacheManager,
    OptimizationRecord,
    TrainingEpisode,
    KnowledgePattern,
    store_optimization_experience,
    retrieve_optimization_knowledge,
    generate_code_hash
)
from code.code_optimizer_ai.utils.logger import get_logger

logger = get_logger(__name__)

async def demo_database_operations():
    print(" Database Integration Demo")
    db_manager = DatabaseManager()
    cache_manager = CacheManager()

    try:
        print(" Initializing Database Connections...")
        await db_manager.initialize()
        await cache_manager.initialize()
        print(" Database connections established")
        print()

        # 1. Store optimization experience
        print(" Storing Optimization Experience:")

        sample_code = """
def fibonacci_inefficient(n):
    if n <= 1:
        return n
    return fibonacci_inefficient(n-1) + fibonacci_inefficient(n-2)
"""

        optimization_result = {
            "complexity_before": 95.0,
            "complexity_after": 15.0,
            "time_before": 30.5,
            "time_after": 0.000003,
            "memory_before": 256.0,
            "memory_after": 8.7,
            "optimization_type": "dynamic_programming",
            "success": True,
            "improvement_factor": 10166666.67,
            "optimized_code": "@lru_cache(maxsize=None)\ndef fibonacci_efficient(n): return n if n <= 1 else fibonacci_efficient(n-1) + fibonacci_efficient(n-2)"
        }

        record_id = await store_optimization_experience(
            sample_code,
            "/demo/fibonacci.py",
            optimization_result
        )
        print(f" - Stored optimization record: {record_id}")
        print(f" - Code hash: {generate_code_hash(sample_code)[:8]}...")
        print(f" - Improvement: {optimization_result['improvement_factor']:,.0f}x faster")
        print()

        # 2. Retrieve optimization knowledge
        print(" Retrieving Optimization Knowledge:")

        code_context = "def fibonacci_inefficient(n):"
        knowledge = await retrieve_optimization_knowledge(code_context)

        if knowledge:
            print(f" - Found knowledge for: {code_context}")
            print(f" - Source: {knowledge.get('source', 'unknown')}")
            print(f" - Improvement factor: {knowledge.get('improvement_factor', 0):,.0f}x")
            print(f" - Optimization type: {knowledge.get('optimization_type', 'unknown')}")
        else:
            print(f" - No knowledge found for: {code_context}")
        print()

        # 3. Cache demonstration
        print(" Redis Cache Demonstration:")

        cache_data = {
            "pattern_type": "fibonacci_optimization",
            "success_rate": 0.95,
            "avg_improvement": 10000000.0,
            "last_used": datetime.now().isoformat()
        }

        cache_key = f"demo_pattern_{generate_code_hash('fibonacci_demo')}"
        await cache_manager.cache_optimization_pattern(cache_key, cache_data, ttl=300)

        cached_result = await cache_manager.get_cached_pattern(cache_key)
        if cached_result:
            print(f" - Cache key: {cache_key[:16]}...")
            print(f" - Pattern type: {cached_result['pattern_type']}")
            print(f" - Success rate: {cached_result['success_rate']:.2f}")
            print(" - Cache hit!")
        else:
            print(" - Cache miss")
        print()

        # 4. Training episode storage
        print(" RL Training Episode Storage:")

        episode = TrainingEpisode(
            episode_data={
                "state": {"complexity": 75.0, "performance": 0.3},
                "action": {"type": "algorithm_change", "target": "fibonacci"},
                "reward": 0.92,
                "success": True
            },
            reward=0.92,
            success=True,
            episode_length=10
        )

        episode_id = await db_manager.store_training_episode(episode)
        print(f" - Stored training episode: {episode_id}")
        print(f" - Episode reward: {episode.reward}")
        print(f" - Episode length: {episode.episode_length}")
        print()

        # 5. Knowledge pattern storage
        print(" Knowledge Pattern Storage:")

        pattern = KnowledgePattern(
            pattern_signature="fibonacci_recursive_to_memoized",
            code_context="recursive fibonacci function",
            optimization_strategy="Add LRU cache with maxsize=None",
            success_rate=0.95,
            average_improvement=10000000.0,
            usage_count=5
        )

        await db_manager.store_knowledge_pattern(pattern)
        print(f" - Stored pattern: {pattern.pattern_signature}")
        print(f" - Success rate: {pattern.success_rate:.2f}")
        print(f" - Average improvement: {pattern.average_improvement:,.0f}x")
        print()

        # 6. Performance metrics caching
        print(" Performance Metrics Caching:")

        code_hash = generate_code_hash(sample_code)
        metrics = {
            "execution_time": 0.000003,
            "memory_usage": 8.7,
            "complexity_score": 15.0,
            "success": True
        }

        await cache_manager.cache_performance_metrics(code_hash, metrics, ttl=1800)
        cached_metrics = await cache_manager.get_cached_metrics(code_hash)

        if cached_metrics:
            print(f" - Cached metrics for: {code_hash[:8]}...")
            print(f" - Execution time: {cached_metrics['execution_time']:.6f}s")
            print(f" - Memory usage: {cached_metrics['memory_usage']:.1f}MB")
            print(" - Metrics cached successfully")
        print()

        print(" RL Experience Buffer Demo:")

        experience = {
            "state": {"code_complexity": 85.0},
            "action": {"type": "algorithm_replacement"},
            "reward": 0.88,
            "success": True,
            "timestamp": datetime.now().isoformat()
        }

        await cache_manager.store_experience_buffer(experience)

        batch = await cache_manager.get_experience_batch(batch_size=5)
        print(" - Stored experience in buffer")
        print(f" - Retrieved batch of {len(batch)} experiences")
        print(f" - Latest reward: {batch[0]['reward'] if batch else 'N/A'}")
        print()

        # 8. Pattern usage statistics
        print(" Pattern Usage Statistics:")

        test_pattern = "fibonacci_optimization"
        await cache_manager.increment_pattern_usage(test_pattern)
        await cache_manager.increment_pattern_usage(test_pattern)

        usage_count = await cache_manager.get_pattern_usage_stats(test_pattern)
        print(f" - Pattern: {test_pattern}")
        print(f" - Usage count: {usage_count}")
        print()

        print(" Database Integration Demo Completed!")
        print(" All operations executed successfully")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f" Demo failed: {e}")
        raise
    finally:
        # Cleanup
        await db_manager.close()
        await cache_manager.close()

async def demo_database_migration():
    print("\n Database Migration Demo")

    try:
        # Import the migration function
        from code.code_optimizer_ai.database.migrate import (
            create_database_schema,
            check_database_connection,
        )
        connection_ok = await check_database_connection()
        if not connection_ok:
            print("Cannot proceed with migration - database connection failed")
            return
        await create_database_schema()
        print(" Database schema created successfully")

    except Exception as e:
        logger.error(f"Migration demo failed: {e}")
        print(f" Migration demo failed: {e}")

async def show_storage_architecture():
    print("\n Complete Storage Architecture")

    architecture = {
        "PostgreSQL (Primary Storage)": [
            "optimization_records table",
            "training_episodes table",
            "knowledge_patterns table",
            "performance_metrics table"
        ],
        "Redis (Real-time Cache)": [
            "Optimization patterns cache",
            "RL experience replay buffer",
            "Performance metrics cache",
            "Pattern usage statistics"
        ],
        "File System (Model Storage)": [
            "RL model weights (.pth)",
            "Training checkpoints",
            "Model version history"
        ]
    }

    for layer, components in architecture.items():
        print(f"\n {layer}:")
        for component in components:
            print(f"   - {component}")

    print("\n Data Flow:")
    print(" 1. Code Analysis -> PostgreSQL + Redis")
    print(" 2. Performance Measurement -> Redis ")
    print(" 3. RL Training -> Redis + File System")
    print(" 4. Pattern Learning -> PostgreSQL")
    print(" 5. Knowledge Retrieval -> Redis Cache")

async def main():
    try:
        # Run all demonstrations
        await demo_database_operations()
        await demo_database_migration()
        await show_storage_architecture()

        print("\n All Database Integration Demonstrations Complete!")
        print(" The Code Optimizer AI now has full Redis + PostgreSQL integration")

    except KeyboardInterrupt:
        print("\n Demo interrupted by user")
    except Exception as e:
        print(f"\n Demo failed: {e}")
        logger.error(f"Demo error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
