from __future__ import annotations

import asyncio
import time
from typing import Sequence

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from code.code_optimizer_ai.config.settings import settings
from code.code_optimizer_ai.core.llm_gateway import GatewayAttempt, GatewayResponse, LLMGateway
from code.code_optimizer_ai.utils.logger import get_logger

logger = get_logger(__name__)


class ModelRouter:
    def __init__(self, gateway: LLMGateway | None = None):
        self.gateway = gateway or LLMGateway()
        self.timeout_seconds = max(5, int(settings.LLM_TIMEOUT_SECONDS))

    async def ainvoke(
        self,
        messages: Sequence[BaseMessage],
        *,
        model_id: str,
        trace_id: str = "",
    ) -> GatewayResponse:
        if not model_id:
            return await self.gateway.ainvoke(messages, trace_id=trace_id)
        if not settings.OPENROUTER_API_KEY:
            return await self.gateway.ainvoke(messages, trace_id=trace_id)

        started = time.perf_counter()
        try:
            client = ChatOpenAI(
                api_key=settings.OPENROUTER_API_KEY,
                base_url=settings.OPENROUTER_BASE_URL,
                model=model_id,
                temperature=0.1,
                timeout=self.timeout_seconds,
                max_tokens=4000,
            )
            response = await asyncio.wait_for(client.ainvoke(list(messages)), timeout=self.timeout_seconds)
            content = str(getattr(response, "content", "")).strip()
            if not content:
                raise ValueError("Empty response content")
            latency_ms = (time.perf_counter() - started) * 1000
            return GatewayResponse(
                content=content,
                provider="openrouter_direct",
                model=model_id,
                latency_ms=latency_ms,
                attempts=[
                    GatewayAttempt(
                        provider="openrouter_direct",
                        model=model_id,
                        success=True,
                        latency_ms=latency_ms,
                    )
                ],
            )
        except Exception as exc:
            logger.warning("model_router_direct_failed", trace_id=trace_id, model=model_id, error=str(exc))
            return await self.gateway.ainvoke(messages, trace_id=trace_id)
