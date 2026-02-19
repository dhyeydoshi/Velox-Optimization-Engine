import asyncio
import asyncpg
import redis.asyncio as redis
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import hashlib

from code.code_optimizer_ai.config.settings import settings
from code.code_optimizer_ai.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizationRecord:
    id: Optional[int] = None
    code_hash: str = ""
    file_path: str = ""
    complexity_before: float = 0.0
    complexity_after: float = 0.0
    execution_time_before: float = 0.0
    execution_time_after: float = 0.0
    memory_usage_before: float = 0.0
    memory_usage_after: float = 0.0
    optimization_type: str = ""
    success: bool = False
    improvement_factor: float = 0.0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class TrainingEpisode:
    id: Optional[int] = None
    episode_data: Dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0
    success: bool = False
    episode_length: int = 0
    created_at: Optional[datetime] = None


@dataclass
class KnowledgePattern:
    id: Optional[int] = None
    pattern_signature: str = ""
    code_context: str = ""
    optimization_strategy: str = ""
    success_rate: float = 0.0
    average_improvement: float = 0.0
    usage_count: int = 0
    last_used: Optional[datetime] = None
    created_at: Optional[datetime] = None


class DatabaseManager:

    def __init__(self):
        self.connection_pool = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    def _require_pool(self):
        if not self._initialized or self.connection_pool is None:
            raise RuntimeError(
                "DatabaseManager not initialized. Call await initialize() first."
            )
    
    async def initialize(self):
        async with self._init_lock:
            if self._initialized:
                return
            
            try:
                self.connection_pool = await asyncpg.create_pool(
                    settings.DATABASE_URL,
                    min_size=5,
                    max_size=20,
                    command_timeout=60,
                    server_settings={'jit': 'off'}
                )
                await self.create_tables()
                self._initialized = True
                logger.info("Database initialized")
            except Exception as e:
                logger.error(f"Database init failed: {e}")
                raise
    
    async def create_tables(self):
        async with self.connection_pool.acquire() as conn:
            # Optimization records table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS optimization_records (
                    id SERIAL PRIMARY KEY,
                    code_hash VARCHAR(64) NOT NULL,
                    file_path TEXT NOT NULL,
                    complexity_before DOUBLE PRECISION NOT NULL,
                    complexity_after DOUBLE PRECISION,
                    execution_time_before DOUBLE PRECISION NOT NULL,
                    execution_time_after DOUBLE PRECISION,
                    memory_usage_before DOUBLE PRECISION NOT NULL,
                    memory_usage_after DOUBLE PRECISION,
                    optimization_type VARCHAR(100) NOT NULL,
                    success BOOLEAN NOT NULL,
                    improvement_factor DOUBLE PRECISION NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Training episodes table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS training_episodes (
                    id SERIAL PRIMARY KEY,
                    episode_data JSONB NOT NULL,
                    reward DOUBLE PRECISION NOT NULL,
                    success BOOLEAN NOT NULL,
                    episode_length INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Knowledge patterns table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_patterns (
                    id SERIAL PRIMARY KEY,
                    pattern_signature VARCHAR(255) NOT NULL UNIQUE,
                    code_context TEXT NOT NULL,
                    optimization_strategy TEXT NOT NULL,
                    success_rate DOUBLE PRECISION DEFAULT 0.0,
                    average_improvement DOUBLE PRECISION DEFAULT 0.0,
                    usage_count INTEGER DEFAULT 0,
                    last_used TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Performance metrics table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id SERIAL PRIMARY KEY,
                    metric_name VARCHAR(100) NOT NULL,
                    metric_value DOUBLE PRECISION NOT NULL,
                    code_hash VARCHAR(64),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better performance
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_optimization_records_hash ON optimization_records(code_hash)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_optimization_records_success ON optimization_records(success)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_training_episodes_reward ON training_episodes(reward)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_patterns_signature ON knowledge_patterns(pattern_signature)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_performance_metrics_hash ON performance_metrics(code_hash)")
            
            logger.info("Database tables created successfully")
    
    async def store_optimization_record(self, record: OptimizationRecord) -> int:
        self._require_pool()
        async with self.connection_pool.acquire() as conn:
            query = """
                INSERT INTO optimization_records 
                (code_hash, file_path, complexity_before, complexity_after, 
                 execution_time_before, execution_time_after, memory_usage_before,
                 memory_usage_after, optimization_type, success, improvement_factor)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                RETURNING id
            """
            
            record_id = await conn.fetchval(
                query,
                record.code_hash,
                record.file_path,
                record.complexity_before,
                record.complexity_after,
                record.execution_time_before,
                record.execution_time_after,
                record.memory_usage_before,
                record.memory_usage_after,
                record.optimization_type,
                record.success,
                record.improvement_factor
            )
            
            logger.debug(f"Stored optimization record: {record_id}")
            return record_id
    
    async def get_similar_optimizations(self, code_hash: str, limit: int = 10) -> List[Dict]:
        self._require_pool()
        async with self.connection_pool.acquire() as conn:
            query = """
                SELECT * FROM optimization_records
                WHERE LEFT(code_hash, 8) = $1
                AND success = true
                ORDER BY improvement_factor DESC, created_at DESC
                LIMIT $2
            """
            
            rows = await conn.fetch(query, code_hash[:8], limit)
            
            return [dict(row) for row in rows]
    
    async def store_training_episode(self, episode: TrainingEpisode) -> int:
        self._require_pool()
        async with self.connection_pool.acquire() as conn:
            query = """
                INSERT INTO training_episodes
                (episode_data, reward, success, episode_length)
                VALUES ($1, $2, $3, $4)
                RETURNING id
            """
            
            episode_id = await conn.fetchval(
                query,
                json.dumps(episode.episode_data),
                episode.reward,
                episode.success,
                episode.episode_length
            )
            
            logger.debug(f"Stored training episode: {episode_id}")
            return episode_id
    
    async def get_knowledge_patterns(self, code_context: str) -> List[Dict]:
        """Get applicable knowledge patterns"""
        self._require_pool()
        async with self.connection_pool.acquire() as conn:
            query = """
                SELECT * FROM knowledge_patterns
                WHERE code_context ILIKE $1
                ORDER BY success_rate DESC, usage_count DESC
                LIMIT 10
            """
            
            rows = await conn.fetch(query, f"%{code_context}%")
            return [dict(row) for row in rows]
    
    async def store_knowledge_pattern(self, pattern: KnowledgePattern):
        self._require_pool()
        async with self.connection_pool.acquire() as conn:
            query = """
                INSERT INTO knowledge_patterns
                (pattern_signature, code_context, optimization_strategy, success_rate, 
                 average_improvement, usage_count, last_used)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (pattern_signature)
                DO UPDATE SET
                    success_rate = (knowledge_patterns.success_rate * knowledge_patterns.usage_count + $4) / (knowledge_patterns.usage_count + 1),
                    average_improvement = (knowledge_patterns.average_improvement * knowledge_patterns.usage_count + $5) / (knowledge_patterns.usage_count + 1),
                    usage_count = knowledge_patterns.usage_count + 1,
                    last_used = $7
            """
            
            await conn.execute(
                query,
                pattern.pattern_signature,
                pattern.code_context,
                pattern.optimization_strategy,
                pattern.success_rate,
                pattern.average_improvement,
                pattern.usage_count,
                datetime.now()
            )
    
    async def close(self):
        if self.connection_pool:
            await self.connection_pool.close()
            self._initialized = False


class CacheManager:
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self.key_prefix = settings.REDIS_KEY_PREFIX.strip().strip(":")

    def _key(self, suffix: str) -> str:
        if not self.key_prefix:
            return suffix
        return f"{self.key_prefix}:{suffix}"
    
    async def initialize(self):
        async with self._init_lock:
            if self._initialized:
                return
            
            try:
                self.redis_client = redis.from_url(
                    settings.REDIS_URL,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True
                )
                
                # Test connection
                await self.redis_client.ping()
                self._initialized = True
                logger.info("Redis connection initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Redis: {e}")
                raise
    
    async def cache_optimization_pattern(self, code_signature: str, pattern_data: Dict, ttl: int = 3600):
        key = self._key(f"pattern:{hashlib.sha256(code_signature.encode()).hexdigest()}")
        await self.redis_client.setex(key, ttl, json.dumps(pattern_data))
        logger.debug(f"Cached pattern for {code_signature}")
    
    async def get_cached_pattern(self, code_signature: str) -> Optional[Dict]:
        key = self._key(f"pattern:{hashlib.sha256(code_signature.encode()).hexdigest()}")
        cached_data = await self.redis_client.get(key)
        
        if cached_data:
            return json.loads(cached_data)
        return None
    
    async def store_experience_buffer(self, experience: Dict):
        buffer_key = self._key("rl_experience_buffer")
        await self.redis_client.lpush(buffer_key, json.dumps(experience))
        
        # Keep buffer size manageable
        await self.redis_client.ltrim(buffer_key, 0, 9999)  # Keep last 10k experiences
    
    async def get_experience_batch(self, batch_size: int = 32) -> List[Dict]:
        buffer_key = self._key("rl_experience_buffer")
        experiences = await self.redis_client.lrange(buffer_key, 0, batch_size - 1)
        return [json.loads(exp) for exp in experiences]
    
    async def cache_performance_metrics(self, code_hash: str, metrics: Dict, ttl: int = 1800):
        key = self._key(f"metrics:{code_hash}")
        await self.redis_client.hset(key, mapping=metrics)
        await self.redis_client.expire(key, ttl)
    
    async def get_cached_metrics(self, code_hash: str) -> Optional[Dict]:
        key = self._key(f"metrics:{code_hash}")
        metrics = await self.redis_client.hgetall(key)
        
        if metrics:
            # Convert string values to appropriate types
            converted: Dict[str, Any] = {}
            for k, v in metrics.items():
                try:
                    converted[k] = float(v)
                except (TypeError, ValueError):
                    converted[k] = v
            return converted
        return None
    
    async def increment_pattern_usage(self, pattern_signature: str):
        key = self._key(f"pattern_usage:{pattern_signature}")
        await self.redis_client.incr(key)
        await self.redis_client.expire(key, 86400)  # 24 hours
    
    async def get_pattern_usage_stats(self, pattern_signature: str) -> int:
        key = self._key(f"pattern_usage:{pattern_signature}")
        count = await self.redis_client.get(key)
        return int(count) if count else 0
    
    async def close(self):
        if self.redis_client:
            await self.redis_client.aclose()
            self.redis_client = None
            self._initialized = False


# Global instances
db_manager = DatabaseManager()
cache_manager = CacheManager()


@asynccontextmanager
async def get_db_connection():
    if not db_manager._initialized:
        await db_manager.initialize()
    try:
        yield db_manager
    except Exception as e:
        logger.error(f"Database operation failed: {e}")
        raise


@asynccontextmanager
async def get_redis_connection():
    if not cache_manager._initialized:
        await cache_manager.initialize()
    try:
        yield cache_manager
    except Exception as e:
        logger.error(f"Redis operation failed: {e}")
        raise


# Utility functions
def generate_code_hash(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


async def store_optimization_experience(code: str, file_path: str, optimization_result: Dict) -> int:
    code_hash = generate_code_hash(code)
    
    # Store in database
    record = OptimizationRecord(
        code_hash=code_hash,
        file_path=file_path,
        complexity_before=optimization_result.get("complexity_before", 0.0),
        complexity_after=optimization_result.get("complexity_after", 0.0),
        execution_time_before=optimization_result.get("time_before", 0.0),
        execution_time_after=optimization_result.get("time_after", 0.0),
        memory_usage_before=optimization_result.get("memory_before", 0.0),
        memory_usage_after=optimization_result.get("memory_after", 0.0),
        optimization_type=optimization_result.get("optimization_type", ""),
        success=optimization_result.get("success", False),
        improvement_factor=optimization_result.get("improvement_factor", 0.0)
    )
    
    record_id = await db_manager.store_optimization_record(record)
    
    # Cache optimization pattern
    pattern_data = {
        "optimization_code": optimization_result.get("optimized_code", ""),
        "improvement_factor": optimization_result.get("improvement_factor", 0.0),
        "optimization_type": optimization_result.get("optimization_type", ""),
        "record_id": record_id
    }
    
    await cache_manager.cache_optimization_pattern(code_hash, pattern_data)
    
    return record_id


async def retrieve_optimization_knowledge(code_context: str) -> Optional[Dict]:
    code_hash = generate_code_hash(code_context)
    
    # Try cache first
    cached_pattern = await cache_manager.get_cached_pattern(code_hash)
    if cached_pattern:
        logger.debug(f"Found cached pattern for {code_context}")
        return cached_pattern
    
    # Query database for similar patterns
    similar_optimizations = await db_manager.get_similar_optimizations(code_hash)
    if similar_optimizations:
        best_optimization = similar_optimizations[0]
        
        # Cache the found pattern
        pattern_data = {
            "optimization_code": best_optimization.get("optimization_code", ""),
            "improvement_factor": best_optimization.get("improvement_factor", 0.0),
            "optimization_type": best_optimization.get("optimization_type", ""),
            "source": "database",
            "confidence": min(best_optimization.get("improvement_factor", 0.0) / 10.0, 1.0)
        }
        
        await cache_manager.cache_optimization_pattern(code_hash, pattern_data)
        return pattern_data
    
    return None
