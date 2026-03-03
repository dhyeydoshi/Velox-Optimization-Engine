from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from code.code_optimizer_ai.config.evolutionary import (
    BANDIT_MIN_EXPLORATION_RATE,
    BANDIT_MIN_OBSERVATIONS,
    BANDIT_PRIOR_MEAN,
    BANDIT_PRIOR_VARIANCE,
)
from code.code_optimizer_ai.config.settings import settings
from code.code_optimizer_ai.database.connection import cache_manager
from code.code_optimizer_ai.evolutionary.constants import FamilyTag
from code.code_optimizer_ai.utils.logger import get_logger

logger = get_logger(__name__)

LUA_ATOMIC_UPDATE = """
local key = KEYS[1]
local reward = tonumber(ARGV[1])
local n = redis.call('HINCRBY', key, 'n', 1)
local new_sum = redis.call('HINCRBYFLOAT', key, 'sum', reward)
local new_sum_sq = redis.call('HINCRBYFLOAT', key, 'sum_sq', reward * reward)
return {n, new_sum, new_sum_sq}
"""


@dataclass(frozen=True)
class BanditStats:
    n: int
    mean: float
    variance: float


class GaussianThompsonBanditSelector:
    def __init__(self, model_ids: Optional[List[str]] = None):
        defaults = [
            settings.OPENROUTER_PRIMARY_MODEL,
            settings.OPENROUTER_SECONDARY_MODEL,
            settings.OLLAMA_MODEL,
        ]
        self.model_ids = [item for item in (model_ids or defaults) if item]

    @staticmethod
    def _stats_key(family_tag: FamilyTag, model_id: str) -> str:
        base = f"bandit:{family_tag.value}:{model_id}"
        return cache_manager._key(base)  # Reuse existing prefixing scheme.

    async def _load_stats(self, family_tag: FamilyTag, model_id: str) -> BanditStats:
        client = cache_manager.redis_client
        if client is None:
            return BanditStats(n=0, mean=BANDIT_PRIOR_MEAN, variance=BANDIT_PRIOR_VARIANCE)

        key = self._stats_key(family_tag, model_id)
        values = await client.hmget(key, "n", "sum", "sum_sq")
        n = int(values[0] or 0)
        if n <= 0:
            return BanditStats(n=0, mean=BANDIT_PRIOR_MEAN, variance=BANDIT_PRIOR_VARIANCE)

        sum_value = float(values[1] or 0.0)
        sum_sq_value = float(values[2] or 0.0)
        mean = sum_value / float(n)
        variance = max(1e-9, (sum_sq_value / float(n)) - (mean * mean))
        return BanditStats(n=n, mean=mean, variance=variance)

    async def select_model(self, family_tag: FamilyTag) -> str:
        if not self.model_ids:
            raise ValueError("No model IDs configured for bandit selection")

        # Safety exploration floor.
        if random.random() < BANDIT_MIN_EXPLORATION_RATE:
            return random.choice(self.model_ids)

        sampled: Dict[str, float] = {}
        try:
            for model_id in self.model_ids:
                stats = await self._load_stats(family_tag, model_id)
                if stats.n < BANDIT_MIN_OBSERVATIONS:
                    mean = BANDIT_PRIOR_MEAN
                    variance = BANDIT_PRIOR_VARIANCE
                    effective_n = max(1, stats.n)
                else:
                    mean = stats.mean
                    variance = stats.variance
                    effective_n = max(1, stats.n)

                posterior_std = math.sqrt(max(1e-9, variance / float(effective_n)))
                sampled[model_id] = random.gauss(mean, posterior_std)
        except Exception as exc:
            logger.warning("bandit_select_failed", error=str(exc))
            return self.model_ids[0]

        return max(sampled, key=sampled.get)

    async def update(self, *, model_id: str, family_tag: FamilyTag, composite_score: float) -> None:
        client = cache_manager.redis_client
        if client is None:
            return
        if not model_id:
            return

        reward = max(float(composite_score), 0.0)
        key = self._stats_key(family_tag, model_id)
        try:
            await client.eval(LUA_ATOMIC_UPDATE, 1, key, str(reward))
            await client.expire(key, 60 * 60 * 24 * 30)
        except Exception as exc:
            logger.warning("bandit_update_failed", error=str(exc), key=key)


bandit_selector = GaussianThompsonBanditSelector()
