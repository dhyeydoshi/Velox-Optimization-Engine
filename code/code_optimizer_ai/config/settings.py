from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import logging
logger = logging.getLogger(__name__)

# Resolve .env relative to the package root (code/code_optimizer_ai/)
_PACKAGE_DIR = Path(__file__).resolve().parent.parent
_ENV_FILE = _PACKAGE_DIR / ".env"


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )    
    # Application
    APP_NAME: str = Field(validation_alias="APP_NAME")
    APP_VERSION: str = Field(validation_alias="APP_VERSION")
    DEBUG: bool = Field(default=False)
    TESTING: Optional[bool] = Field(default=False)
    API_AUTH_TOKEN: Optional[str] = Field(default=None)
    REQUIRE_AUTH_TOKEN: bool = Field(default=False, validation_alias="REQUIRE_AUTH_TOKEN")
    ENABLE_API_DOCS: bool = Field(default=False, validation_alias="ENABLE_API_DOCS")
    FEATURE_EVOLUTIONARY_SEARCH: bool = Field(
        default=False,
        validation_alias="FEATURE_EVOLUTIONARY_SEARCH",
    )
    
    # Database
    DATABASE_URL: str = Field(
        default="postgresql://user:password@localhost:5432/code_optimizer",
        validation_alias="DATABASE_URL"
    )
    REDIS_URL: str = Field(
        default="redis://localhost:6379/11",
        validation_alias="REDIS_URL"
    )
    REDIS_KEY_PREFIX: str = Field(
        default="code_optimizer_ai",
        validation_alias="REDIS_KEY_PREFIX",
    )
    
    
    # LLM Configuration
    OPENAI_API_KEY: Optional[str] = Field(default=None, validation_alias="OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    GOOGLE_API_KEY: Optional[str] = Field(default=None, validation_alias="GOOGLE_API_KEY")
    OPENROUTER_API_KEY: Optional[str] = Field(default=None, validation_alias="OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL: str = Field(
        default="https://openrouter.ai/api/v1",
        validation_alias="OPENROUTER_BASE_URL",
    )
    OPENROUTER_PRIMARY_MODEL: str = Field(
        default="openai/gpt-4o-mini",
        validation_alias="OPENROUTER_PRIMARY_MODEL",
    )
    OPENROUTER_SECONDARY_MODEL: str = Field(
        default="anthropic/claude-3.5-sonnet",
        validation_alias="OPENROUTER_SECONDARY_MODEL",
    )
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL")
    OLLAMA_MODEL: str = Field(default="llama3", validation_alias="OLLAMA_MODEL")
    DEFAULT_LLM_PROVIDER: str = Field(default="openai", validation_alias="DEFAULT_LLM_PROVIDER")
    LLM_TIMEOUT_SECONDS: int = Field(default=45, validation_alias="LLM_TIMEOUT_SECONDS")
    
    # Security
    SECRET_KEY: str = Field(
        default="",
        validation_alias="SECRET_KEY"
    )
    CORS_ALLOWED_ORIGINS: str = Field(
        default="http://localhost,http://127.0.0.1",
        validation_alias="CORS_ALLOWED_ORIGINS",
    )
    CORS_ALLOW_CREDENTIALS: bool = Field(default=False, validation_alias="CORS_ALLOW_CREDENTIALS")
    FORWARDED_ALLOW_IPS: str = Field(default="127.0.0.1", validation_alias="FORWARDED_ALLOW_IPS")
    ALLOWED_CODE_ROOTS: str = Field(default=".", validation_alias="ALLOWED_CODE_ROOTS")
    MAX_INLINE_CODE_CHARS: int = Field(default=200_000, validation_alias="MAX_INLINE_CODE_CHARS")
    MAX_UPLOAD_SIZE_BYTES: int = Field(default=1_000_000, validation_alias="MAX_UPLOAD_SIZE_BYTES")
    ENABLE_API_UNIT_TEST_COMMAND_OVERRIDE: bool = Field(
        default=False,
        validation_alias="ENABLE_API_UNIT_TEST_COMMAND_OVERRIDE",
    )
    RATE_LIMIT_ENABLED: bool = Field(default=True, validation_alias="RATE_LIMIT_ENABLED")
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(
        default=120,
        validation_alias="RATE_LIMIT_REQUESTS_PER_MINUTE",
    )
    
    # Performance Monitoring
    METRICS_RETENTION_DAYS: int = Field(default=90, validation_alias="METRICS_RETENTION_DAYS")
    BASELINE_COLLECTION_INTERVAL: int = Field(
        default=300, validation_alias="BASELINE_COLLECTION_INTERVAL"
    )
    
    # Optimization Settings
    MAX_OPTIMIZATION_ATTEMPTS: int = Field(default=3, validation_alias="MAX_OPTIMIZATION_ATTEMPTS")
    MIN_PERFORMANCE_IMPROVEMENT: float = Field(
        default=0.05, validation_alias="MIN_PERFORMANCE_IMPROVEMENT"
    )
    ANALYSIS_TIMEOUT_SECONDS: int = Field(
        default=30, validation_alias="ANALYSIS_TIMEOUT_SECONDS"
    )
    MAX_SCAN_FILES: int = Field(default=2_000, validation_alias="MAX_SCAN_FILES")
    MAX_SCAN_TOTAL_BYTES: int = Field(
        default=200 * 1024 * 1024,
        validation_alias="MAX_SCAN_TOTAL_BYTES",
    )
    VALIDATION_TIMEOUT_SECONDS: int = Field(
        default=120, validation_alias="VALIDATION_TIMEOUT_SECONDS"
    )
    UNIT_TEST_COMMAND: str = Field(
        default="python -m pytest code/code_optimizer_ai/tests -q",
        validation_alias="UNIT_TEST_COMMAND",
    )
    DEFAULT_MAX_SUGGESTIONS: int = Field(default=3, validation_alias="DEFAULT_MAX_SUGGESTIONS")
    OBJECTIVE_RUNTIME_WEIGHT: float = Field(
        default=0.5, validation_alias="OBJECTIVE_RUNTIME_WEIGHT"
    )
    OBJECTIVE_MEMORY_WEIGHT: float = Field(
        default=0.5, validation_alias="OBJECTIVE_MEMORY_WEIGHT"
    )
    GITHUB_CLONE_ROOT: str = Field(
        default=".tmp_remote_repos",
        validation_alias="GITHUB_CLONE_ROOT",
    )
    GITHUB_CLONE_TIMEOUT_SECONDS: int = Field(
        default=120,
        validation_alias="GITHUB_CLONE_TIMEOUT_SECONDS",
    )
    
    # RL Training
    RL_MODEL_PATH: str = Field(
        default="./models/rl_policy",
        validation_alias="RL_MODEL_PATH"
    )
    TRAINING_DATA_PATH: str = Field(
        default="./data/training",
        validation_alias="TRAINING_DATA_PATH"
    )
    
    # AWS Configuration
    AWS_REGION: str = Field(default="us-east-1", validation_alias="AWS_REGION")
    AWS_S3_BUCKET: Optional[str] = Field(default=None, validation_alias="AWS_S3_BUCKET")
    
    # Agent Configuration
    MONITORING_INTERVAL: int = Field(default=60, validation_alias="MONITORING_INTERVAL")
    ANALYSIS_BATCH_SIZE: int = Field(default=10, validation_alias="ANALYSIS_BATCH_SIZE")
    MAX_CONCURRENT_ANALYSIS: int = Field(default=4, validation_alias="MAX_CONCURRENT_ANALYSIS")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    ENABLE_SENTRY: bool = Field(default=False, validation_alias="ENABLE_SENTRY")
    SENTRY_DSN: Optional[str] = Field(default=None, validation_alias="SENTRY_DSN")


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()

    return settings



# Global settings instance
settings = get_settings()
