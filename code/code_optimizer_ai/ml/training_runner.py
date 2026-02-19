"""
Offline DQN training runner with evaluation gate and artifact management.

This is the PRIMARY training path. It orchestrates:
  1. Transition dataset generation (generate_training_data)
  2. Train/holdout split
  3. Offline DQN pretraining
  4. Holdout evaluation + comparison against previous checkpoint
  5. Conditional model publish or rollback

The online RL environment training (RLTrainer.train) is EXPERIMENTAL and
should not be used in production without manual review.
"""
from __future__ import annotations

import json
import math
import random
import secrets
import shutil
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from code.code_optimizer_ai.config.settings import settings
from code.code_optimizer_ai.ml.rl_agent import DQNAgent, QNetwork, ReplayBuffer
from code.code_optimizer_ai.ml.training_semantics import PRODUCTION_ACTION_TYPES
from code.code_optimizer_ai.utils.logger import get_logger

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
ACTION_INDEX = {name: idx for idx, name in enumerate(PRODUCTION_ACTION_TYPES)}

# Lock serialising the publish-to-canonical step across concurrent jobs.
_publish_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrainingJobStatus:
    job_id: str
    status: str = "initiated"  # initiated | generating | training | evaluating | publishing | completed | failed | rejected
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None

    # Dataset stats
    total_transitions: int = 0
    train_transitions: int = 0
    holdout_transitions: int = 0
    synthetic_samples: int = 0

    # Training stats
    epochs: int = 0
    steps_per_epoch: int = 0
    updates: int = 0
    final_epoch_loss: Optional[float] = None

    # Evaluation gate
    holdout_mean_td_error: Optional[float] = None
    holdout_mean_reward: Optional[float] = None
    prev_holdout_mean_td_error: Optional[float] = None
    gate_passed: Optional[bool] = None
    gate_reason: Optional[str] = None

    # Artifacts
    model_path: Optional[str] = None
    prev_model_path: Optional[str] = None
    dataset_path: Optional[str] = None
    holdout_path: Optional[str] = None


@dataclass
class TrainingConfig:
    """Parameters for a training run."""
    synthetic_samples: int = 10_000
    seed: int = 42
    holdout_fraction: float = 0.20
    epochs: int = 40
    steps_per_epoch: int = 1000
    batch_size: int = 64
    learning_rate: float = 1e-3
    gamma: float = 0.99
    hidden_dim: int = 256
    replay_buffer_size: int = 200_000
    reward_clip: Optional[float] = None
    shuffle: bool = True
    # Evaluation gate thresholds
    max_td_error_regression_pct: float = 10.0  # New model's holdout TD error must not be >10% worse
    require_td_error_improvement: bool = False  # If True, new model must strictly improve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p


def _jobs_dir() -> Path:
    d = _resolve(settings.TRAINING_DATA_PATH) / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_status(status: TrainingJobStatus) -> Path:
    path = _jobs_dir() / f"{status.job_id}.json"
    path.write_text(json.dumps(asdict(status), indent=2), encoding="utf-8")
    return path


def load_job_status(job_id: str) -> Optional[TrainingJobStatus]:
    path = _jobs_dir() / f"{job_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return TrainingJobStatus(**data)


def create_job_id() -> str:
    """Generate a collision-safe job ID (timestamp + random hex)."""
    return f"job_{int(time.time())}_{secrets.token_hex(4)}"


def create_job_status(job_id: str) -> TrainingJobStatus:
    """Create and persist an 'initiated' job record. Call before scheduling background work."""
    status = TrainingJobStatus(job_id=job_id, started_at=datetime.now().isoformat())
    _save_status(status)
    return status


def list_jobs(limit: int = 20) -> List[Dict[str, Any]]:
    jobs_path = _jobs_dir()
    files = sorted(jobs_path.glob("job_*.json"), reverse=True)[:limit]
    results = []
    for f in files:
        try:
            results.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    return results


# ---------------------------------------------------------------------------
# Dataset split
# ---------------------------------------------------------------------------

def _split_jsonl(
    source: Path,
    train_out: Path,
    holdout_out: Path,
    holdout_fraction: float,
    seed: int,
) -> Tuple[int, int]:
    """Split a JSONL file into train and holdout sets."""
    lines = source.read_text(encoding="utf-8").strip().splitlines()
    rng = random.Random(seed)
    rng.shuffle(lines)

    holdout_count = max(1, int(len(lines) * holdout_fraction))
    holdout_lines = lines[:holdout_count]
    train_lines = lines[holdout_count:]

    train_out.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    holdout_out.write_text("\n".join(holdout_lines) + "\n", encoding="utf-8")
    return len(train_lines), len(holdout_lines)


# ---------------------------------------------------------------------------
# Holdout evaluation
# ---------------------------------------------------------------------------

def _evaluate_on_holdout(
    agent: DQNAgent,
    holdout_path: Path,
    gamma: float,
) -> Dict[str, float]:
    """Compute mean TD error and mean reward on holdout transitions."""
    td_errors: List[float] = []
    rewards: List[float] = []

    with holdout_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            state = np.array(record["state"], dtype=np.float32)
            action_idx = int(record["action_idx"])
            reward = float(record["reward"])
            next_state = np.array(record["next_state"], dtype=np.float32)
            done = bool(record["done"])

            state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(agent.device)

            with torch.no_grad():
                q_current = agent.q_network(state_t)[0, action_idx].item()
                q_next_max = agent.target_network(next_state_t).max(1)[0].item()
                target = reward + (gamma * q_next_max * (0.0 if done else 1.0))
                td_error = abs(q_current - target)

            td_errors.append(td_error)
            rewards.append(reward)

    return {
        "mean_td_error": float(np.mean(td_errors)) if td_errors else float("inf"),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "count": len(td_errors),
    }


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def run_training_job(config: TrainingConfig, job_id: Optional[str] = None) -> TrainingJobStatus:
    """
    Execute the full offline training pipeline synchronously.

    Steps:
        1. Generate transitions (with synthetic augmentation)
        2. Split into train / holdout
        3. Pretrain DQN on train set
        4. Evaluate on holdout; compare with previous checkpoint
        5. Publish or reject the new model
    """
    if job_id is None:
        job_id = f"job_{int(time.time())}_{secrets.token_hex(4)}"

    # Re-use pre-persisted record if the API already created one, else create fresh.
    existing = load_job_status(job_id)
    if existing is not None:
        status = existing
    else:
        status = TrainingJobStatus(job_id=job_id, started_at=datetime.now().isoformat())
    _save_status(status)

    training_data_dir = _resolve(settings.TRAINING_DATA_PATH)
    model_dir = _resolve(settings.RL_MODEL_PATH)
    training_data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    canonical_model = model_dir / "dqn_model.pth"
    new_model = model_dir / f"dqn_model_{job_id}.pth"
    backup_model = model_dir / "dqn_model_prev.pth"

    dataset_path = training_data_dir / f"dataset_{job_id}.jsonl"
    train_path = training_data_dir / f"train_{job_id}.jsonl"
    holdout_path = training_data_dir / f"holdout_{job_id}.jsonl"

    try:
        # ------------------------------------------------------------------
        # Step 1: Generate dataset
        # ------------------------------------------------------------------
        status.status = "generating"
        status.synthetic_samples = config.synthetic_samples
        _save_status(status)

        from code.code_optimizer_ai.generate_training_data import (
            generate_transitions_from_legacy,
        )
        from code.code_optimizer_ai.ml.training_semantics import normalize_objective_weights

        objective_weights = normalize_objective_weights(
            {"runtime": settings.OBJECTIVE_RUNTIME_WEIGHT, "memory": settings.OBJECTIVE_MEMORY_WEIGHT},
            default_runtime_weight=settings.OBJECTIVE_RUNTIME_WEIGHT,
            default_memory_weight=settings.OBJECTIVE_MEMORY_WEIGHT,
        )

        gen_stats = generate_transitions_from_legacy(
            input_dir=training_data_dir,
            output_jsonl=dataset_path,
            objective_weights=objective_weights,
            synthetic_samples=config.synthetic_samples,
            seed=config.seed,
        )
        status.total_transitions = gen_stats.transitions_written
        status.dataset_path = str(dataset_path)
        _save_status(status)

        if gen_stats.transitions_written == 0:
            raise RuntimeError("No transitions generated -- check input episodes or add synthetic_samples.")

        # ------------------------------------------------------------------
        # Step 2: Split train / holdout
        # ------------------------------------------------------------------
        train_count, holdout_count = _split_jsonl(
            dataset_path, train_path, holdout_path, config.holdout_fraction, config.seed
        )
        status.train_transitions = train_count
        status.holdout_transitions = holdout_count
        status.holdout_path = str(holdout_path)
        _save_status(status)

        logger.info(
            "dataset_split",
            total=gen_stats.transitions_written,
            train=train_count,
            holdout=holdout_count,
        )

        # ------------------------------------------------------------------
        # Step 3: Pretrain DQN
        # ------------------------------------------------------------------
        status.status = "training"
        _save_status(status)

        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        agent = DQNAgent(
            state_dim=27,
            action_dim=len(PRODUCTION_ACTION_TYPES),
            hidden_dim=config.hidden_dim,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            replay_buffer_size=config.replay_buffer_size,
            batch_size=config.batch_size,
        )
        # Fresh start -- ignore any loaded checkpoint
        agent.q_network = QNetwork(27, len(PRODUCTION_ACTION_TYPES), config.hidden_dim).to(agent.device)
        agent.target_network = QNetwork(27, len(PRODUCTION_ACTION_TYPES), config.hidden_dim).to(agent.device)
        agent.target_network.load_state_dict(agent.q_network.state_dict())
        agent.target_network.eval()
        # Rebind optimizer to the NEW network parameters (critical -- stale
        # references would silently update the old, disconnected parameters).
        agent.optimizer = torch.optim.Adam(
            agent.q_network.parameters(), lr=config.learning_rate,
        )
        agent.memory = ReplayBuffer(config.replay_buffer_size)
        agent.training_step = 0
        agent.epsilon = agent.epsilon_start

        # Load train split into replay buffer
        train_transitions = _load_transitions_from_jsonl(train_path, reward_clip=config.reward_clip)
        if config.shuffle:
            random.shuffle(train_transitions)
        for s, a, r, ns, d in train_transitions:
            agent.store_experience(s, a, r, ns, d)

        if len(agent.memory) < agent.batch_size:
            raise RuntimeError(
                f"Only {len(agent.memory)} train transitions, need >= {agent.batch_size}"
            )

        epoch_losses: List[float] = []
        total_updates = 0
        for epoch in range(1, config.epochs + 1):
            losses: List[float] = []
            for _ in range(config.steps_per_epoch):
                loss = agent.train(decay_epsilon=False)
                if loss is not None:
                    losses.append(loss)
                    total_updates += 1
            # Decay epsilon per epoch, not per gradient step.
            if agent.epsilon > agent.epsilon_end:
                agent.epsilon *= agent.epsilon_decay
            if losses:
                epoch_losses.append(float(np.mean(losses)))
            if epoch % 10 == 0 or epoch == config.epochs:
                logger.info(
                    "pretrain_epoch",
                    epoch=epoch,
                    avg_loss=round(epoch_losses[-1], 6) if epoch_losses else None,
                    updates=total_updates,
                )

        status.epochs = config.epochs
        status.steps_per_epoch = config.steps_per_epoch
        status.updates = total_updates
        status.final_epoch_loss = epoch_losses[-1] if epoch_losses else None

        # Save candidate model
        agent.save_model(str(new_model))
        status.model_path = str(new_model)
        _save_status(status)

        # ------------------------------------------------------------------
        # Step 4: Evaluate on holdout + compare with previous checkpoint
        # ------------------------------------------------------------------
        status.status = "evaluating"
        _save_status(status)

        new_eval = _evaluate_on_holdout(agent, holdout_path, config.gamma)
        status.holdout_mean_td_error = new_eval["mean_td_error"]
        status.holdout_mean_reward = new_eval["mean_reward"]

        prev_eval: Optional[Dict[str, float]] = None
        if canonical_model.exists():
            prev_agent = DQNAgent(
                state_dim=27,
                action_dim=len(PRODUCTION_ACTION_TYPES),
                hidden_dim=config.hidden_dim,
            )
            try:
                checkpoint = torch.load(
                    canonical_model, map_location=prev_agent.device, weights_only=True,
                )
                prev_agent.q_network.load_state_dict(checkpoint["q_network_state_dict"])
                prev_agent.target_network.load_state_dict(checkpoint["target_network_state_dict"])
                prev_eval = _evaluate_on_holdout(prev_agent, holdout_path, config.gamma)
                status.prev_holdout_mean_td_error = prev_eval["mean_td_error"]
            except Exception as exc:
                logger.warning("Could not evaluate previous checkpoint", error=str(exc))

        # Gate decision
        gate_passed, gate_reason = _apply_evaluation_gate(
            new_td_error=new_eval["mean_td_error"],
            prev_td_error=prev_eval["mean_td_error"] if prev_eval else None,
            max_regression_pct=config.max_td_error_regression_pct,
            require_improvement=config.require_td_error_improvement,
        )
        status.gate_passed = gate_passed
        status.gate_reason = gate_reason
        _save_status(status)

        logger.info(
            "evaluation_gate",
            new_td_error=round(new_eval["mean_td_error"], 6),
            prev_td_error=round(prev_eval["mean_td_error"], 6) if prev_eval else None,
            passed=gate_passed,
            reason=gate_reason,
        )

        # ------------------------------------------------------------------
        # Step 5: Publish or reject
        # ------------------------------------------------------------------
        if gate_passed:
            status.status = "publishing"
            _save_status(status)

            # Single-flight lock: only one job can mutate the canonical checkpoint at a time.
            with _publish_lock:
                if canonical_model.exists():
                    shutil.copy2(canonical_model, backup_model)
                    status.prev_model_path = str(backup_model)

                shutil.copy2(new_model, canonical_model)

            status.model_path = str(canonical_model)
            status.status = "completed"
            logger.info("model_published", model=str(canonical_model))
        else:
            status.status = "rejected"
            logger.info("model_rejected", reason=gate_reason)

        status.finished_at = datetime.now().isoformat()
        _save_status(status)
        return status

    except Exception as exc:
        status.status = "failed"
        status.error = str(exc)
        status.finished_at = datetime.now().isoformat()
        _save_status(status)
        logger.error("training_job_failed", job_id=job_id, error=str(exc))
        raise


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_transitions_from_jsonl(
    path: Path,
    reward_clip: Optional[float] = None,
) -> List[Tuple[np.ndarray, int, float, np.ndarray, bool]]:
    transitions = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            state = np.array(record["state"], dtype=np.float32)
            action_idx = int(record["action_idx"])
            reward = float(record["reward"])
            if reward_clip is not None:
                reward = float(max(-reward_clip, min(reward_clip, reward)))
            next_state = np.array(record["next_state"], dtype=np.float32)
            done = bool(record["done"])
            transitions.append((state, action_idx, reward, next_state, done))
    return transitions


def _apply_evaluation_gate(
    *,
    new_td_error: float,
    prev_td_error: Optional[float],
    max_regression_pct: float,
    require_improvement: bool,
) -> Tuple[bool, str]:
    """
    Decide whether the new model passes the evaluation gate.

    Rules:
      - If no previous checkpoint exists → pass (bootstrap case).
      - New model's holdout TD error must not regress by more than max_regression_pct.
      - If require_improvement is True, new model must strictly improve.
    """
    if prev_td_error is None:
        return True, "No previous checkpoint -- bootstrap publish."

    if prev_td_error <= 0:
        # Degenerate: previous was perfect; only pass if new is also perfect
        return new_td_error <= 0, "Previous TD error was 0; new must also be 0."

    regression_pct = ((new_td_error - prev_td_error) / prev_td_error) * 100.0

    if regression_pct > max_regression_pct:
        return False, (
            f"TD error regressed by {regression_pct:.1f}% "
            f"(new={new_td_error:.6f}, prev={prev_td_error:.6f}), "
            f"threshold={max_regression_pct}%."
        )

    if require_improvement and new_td_error >= prev_td_error:
        return False, (
            f"Improvement required but new TD error ({new_td_error:.6f}) "
            f">= previous ({prev_td_error:.6f})."
        )

    delta_pct = ((prev_td_error - new_td_error) / prev_td_error) * 100.0
    if delta_pct > 0:
        return True, f"Gate passed. TD error improved by {delta_pct:.1f}%."
    return True, f"Gate passed. TD error changed by {delta_pct:+.1f}% (within {max_regression_pct}% tolerance)."
