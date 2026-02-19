from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

try:
    from langchain_ollama import ChatOllama
    HAS_OLLAMA = True
except Exception:
    try:
        from langchain_community.chat_models import ChatOllama
        HAS_OLLAMA = True
    except Exception:  # pragma: no cover - optional dependency
        ChatOllama = None  # type: ignore
        HAS_OLLAMA = False

from code.code_optimizer_ai.config.settings import settings
from code.code_optimizer_ai.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GatewayAttempt:
    provider: str
    model: str
    success: bool
    latency_ms: float
    error: Optional[str] = None


@dataclass
class GatewayResponse:
    content: str
    provider: str
    model: str
    latency_ms: float
    attempts: List[GatewayAttempt]


class LLMGateway:

    def __init__(self):
        self.timeout_seconds = max(5, settings.LLM_TIMEOUT_SECONDS)
        self._openrouter_clients: Dict[str, ChatOpenAI] = {}
        self._ollama_client: Optional[ChatOllama] = None

    def _openrouter_client(self, model: str) -> ChatOpenAI:
        client = self._openrouter_clients.get(model)
        if client is not None:
            return client

        if not settings.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is required for OpenRouter routing")

        client = ChatOpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
            model=model,
            temperature=0.1,
            timeout=self.timeout_seconds,
            max_tokens=4000,
        )
        self._openrouter_clients[model] = client
        return client

    def _ollama(self) -> ChatOllama:
        if not HAS_OLLAMA:
            raise RuntimeError(
                "langchain_community is required for Ollama fallback but is not installed"
            )
        if self._ollama_client is None:
            self._ollama_client = ChatOllama(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0.1,
            )
        return self._ollama_client

    def _quality_order(self) -> List[Tuple[str, str]]:
        return [
            ("openrouter", settings.OPENROUTER_PRIMARY_MODEL),
            ("openrouter", settings.OPENROUTER_SECONDARY_MODEL),
            ("ollama", settings.OLLAMA_MODEL),
        ]

    async def ainvoke(
        self,
        messages: Sequence[BaseMessage],
        *,
        trace_id: str = "",
    ) -> GatewayResponse:
        attempts: List[GatewayAttempt] = []
        last_error: Optional[str] = None

        for provider, model in self._quality_order():
            start = time.perf_counter()
            try:
                if provider == "openrouter":
                    response = await asyncio.wait_for(
                        self._openrouter_client(model).ainvoke(list(messages)),
                        timeout=self.timeout_seconds,
                    )
                else:
                    response = await asyncio.wait_for(
                        self._ollama().ainvoke(list(messages)),
                        timeout=self.timeout_seconds,
                    )

                content = str(getattr(response, "content", "")).strip()
                elapsed_ms = (time.perf_counter() - start) * 1000

                if not content:
                    raise ValueError("Empty response content")

                attempts.append(
                    GatewayAttempt(
                        provider=provider,
                        model=model,
                        success=True,
                        latency_ms=elapsed_ms,
                    )
                )

                logger.info(
                    "llm_gateway_success",
                    trace_id=trace_id,
                    provider=provider,
                    model=model,
                    latency_ms=round(elapsed_ms, 2),
                    fallback_count=max(0, len(attempts) - 1),
                )
                return GatewayResponse(
                    content=content,
                    provider=provider,
                    model=model,
                    latency_ms=elapsed_ms,
                    attempts=attempts,
                )
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000
                last_error = str(exc)
                attempts.append(
                    GatewayAttempt(
                        provider=provider,
                        model=model,
                        success=False,
                        latency_ms=elapsed_ms,
                        error=last_error,
                    )
                )
                logger.warning(
                    "llm_gateway_attempt_failed",
                    trace_id=trace_id,
                    provider=provider,
                    model=model,
                    latency_ms=round(elapsed_ms, 2),
                    error=last_error,
                )

        raise RuntimeError(f"All model routes failed. Last error: {last_error}")
