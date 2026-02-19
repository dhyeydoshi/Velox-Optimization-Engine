from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from code.code_optimizer_ai.ml.rl_agent import DQNAgent
from code.code_optimizer_ai.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PolicyFeedback:
    state: np.ndarray
    action_idx: int
    reward: float
    next_state: np.ndarray
    done: bool


class PolicyManager:
    """Owns active serving policy and shadow fine-tuning policy."""

    def __init__(self, active_agent: DQNAgent):
        self.active_agent = active_agent
        self.shadow_agent = DQNAgent(
            state_dim=active_agent.state_dim,
            action_dim=active_agent.action_dim,
        )
        self.shadow_agent.q_network.load_state_dict(active_agent.q_network.state_dict())
        self.shadow_agent.target_network.load_state_dict(active_agent.target_network.state_dict())
        self.shadow_buffer: List[PolicyFeedback] = []

    def recommend_action(self, state: np.ndarray, action_names: List[str]) -> Dict[str, Any]:
        """Always serve from active model."""
        return self.active_agent.get_action_recommendation(state, action_names)

    def record_shadow_feedback(self, feedback: PolicyFeedback):
        """Collect online feedback for shadow-only updates."""
        self.shadow_buffer.append(feedback)
        if len(self.shadow_buffer) > 5000:
            self.shadow_buffer = self.shadow_buffer[-5000:]

    def train_shadow(self, max_updates: int = 20) -> int:
        """Update shadow model using collected feedback only."""
        updates = 0
        if not self.shadow_buffer:
            return updates

        for feedback in self.shadow_buffer[-max_updates:]:
            self.shadow_agent.store_experience(
                feedback.state,
                feedback.action_idx,
                feedback.reward,
                feedback.next_state,
                feedback.done,
            )
            loss = self.shadow_agent.train()
            if loss is not None:
                updates += 1
        return updates

    def evaluate_shadow_promotion(
        self,
        *,
        active_score: float,
        shadow_score: float,
        active_invalid_rate: float,
        shadow_invalid_rate: float,
    ) -> bool:
        """Promotion gate: +3% score and <= +1% invalid-rate regression."""
        score_ok = shadow_score >= active_score * 1.03
        invalid_ok = shadow_invalid_rate <= active_invalid_rate + 0.01
        return score_ok and invalid_ok

    def promote_shadow(self):
        """Promote shadow to active after passing gates."""
        self.active_agent.q_network.load_state_dict(self.shadow_agent.q_network.state_dict())
        self.active_agent.target_network.load_state_dict(self.shadow_agent.target_network.state_dict())
        logger.info("shadow_policy_promoted")

