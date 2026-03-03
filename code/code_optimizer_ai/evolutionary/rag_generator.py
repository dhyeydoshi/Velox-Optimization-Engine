from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from code.code_optimizer_ai.evolutionary.diff_parser import apply_diff_patches, parse_search_replace_blocks
from code.code_optimizer_ai.evolutionary.memory import TransitionRecord
from code.code_optimizer_ai.evolutionary.model_router import ModelRouter
from code.code_optimizer_ai.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RagGenerationResult:
    optimized_code: str
    code_diff: str
    model_id: Optional[str]
    used_patches: bool


class RAGGuidedGenerator:
    def __init__(self, model_router: Optional[ModelRouter] = None):
        self.model_router = model_router or ModelRouter()

    @staticmethod
    def _few_shot_block(transitions: List[TransitionRecord]) -> str:
        if not transitions:
            return "No prior transitions available."
        blocks: List[str] = []
        for idx, item in enumerate(transitions, start=1):
            blocks.append(
                "\n".join(
                    [
                        f"Example {idx}:",
                        f"Family: {item.family_tag.value}",
                        f"Summary: {item.transform_summary}",
                        f"Runtime Delta: {item.measured_runtime_delta:.4f}",
                        "Patch:",
                        item.code_diff,
                    ]
                )
            )
        return "\n\n".join(blocks)

    async def generate(
        self,
        *,
        original_code: str,
        analysis: Any,
        retrieved_transitions: List[TransitionRecord],
        model_id: Optional[str],
        request_context: Any,
    ) -> RagGenerationResult:
        prompt = (
            "Return only SEARCH/REPLACE blocks. No prose.\n"
            "Optimize for runtime and memory while preserving behavior.\n\n"
            f"Analysis Summary:\n{getattr(analysis, 'semantic_summary', '')}\n\n"
            f"Representative Input: {getattr(request_context, 'representative_input', [])}\n\n"
            "Relevant Prior Transitions:\n"
            f"{self._few_shot_block(retrieved_transitions)}\n\n"
            "Original Code:\n"
            "```python\n"
            f"{original_code}\n"
            "```"
        )

        response = await self.model_router.ainvoke(
            [
                SystemMessage(content="You are an expert Python optimization assistant."),
                HumanMessage(content=prompt),
            ],
            model_id=model_id or "",
            trace_id=f"raeo_rag:{getattr(request_context, 'request_id', '')}",
        )
        content = (response.content or "").strip()
        patches = parse_search_replace_blocks(content)
        if patches:
            updated = apply_diff_patches(original_code, patches)
            if updated is not None:
                return RagGenerationResult(
                    optimized_code=updated,
                    code_diff=content,
                    model_id=response.model,
                    used_patches=True,
                )

        # Whole-file fallback for malformed patch output.
        return RagGenerationResult(
            optimized_code=content or original_code,
            code_diff="",
            model_id=response.model,
            used_patches=False,
        )


rag_guided_generator = RAGGuidedGenerator()
