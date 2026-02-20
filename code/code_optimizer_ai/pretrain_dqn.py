from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Ensure repository root is on path for package imports.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from code.code_optimizer_ai.config.settings import settings
from code.code_optimizer_ai.ml.rl_agent import DQNAgent, QNetwork, ReplayBuffer
from code.code_optimizer_ai.ml.training_semantics import PRODUCTION_ACTION_TYPES
from code.code_optimizer_ai.utils.logger import get_logger

logger = get_logger(__name__)
ACTION_INDEX = {name: idx for idx, name in enumerate(PRODUCTION_ACTION_TYPES)}

@dataclass
class Transition:
    state: np.ndarray
    action_idx: int
    reward: float
    next_state: np.ndarray
    done: bool

@dataclass
class LoadStats:
    files: int = 0
    lines: int = 0
    accepted: int = 0
    skipped: int = 0

def _resolve_path(path_value: str) -> Path:
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate

def _default_input_path() -> Path:
    return _resolve_path(str(Path(settings.TRAINING_DATA_PATH) / "pretrain_transitions.jsonl"))

def _default_output_path() -> Path:
    return _resolve_path(str(Path(settings.RL_MODEL_PATH) / "dqn_model.pth"))

def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed

def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed

def _discover_jsonl_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".jsonl":
            raise ValueError(f"Input file must use .jsonl extension: {input_path}")
        return [input_path]

    if input_path.is_dir():
        files = sorted(input_path.rglob("*.jsonl"))
        if not files:
            raise ValueError(f"No .jsonl files found in directory: {input_path}")
        return files

    raise ValueError(f"Input path not found: {input_path}")

def _to_vector(
    value: Any,
    *,
    expected_len: int,
    field_name: str,
) -> np.ndarray:
    if not isinstance(value, Sequence):
        raise ValueError(f"'{field_name}' must be a list-like sequence")
    if len(value) != expected_len:
        raise ValueError(
            f"'{field_name}' length mismatch: expected {expected_len}, got {len(value)}"
        )
    vector = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"'{field_name}' contains non-finite values")
    return vector

def _to_done_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    raise ValueError("'done' must be a bool or integer 0/1")

def _parse_transition_record(
    record: Dict[str, Any],
    *,
    state_dim: int,
    action_dim: int,
    reward_clip: Optional[float],
) -> Transition:
    state = _to_vector(record.get("state"), expected_len=state_dim, field_name="state")
    next_state = _to_vector(
        record.get("next_state"), expected_len=state_dim, field_name="next_state"
    )

    action_raw = record.get("action_idx", record.get("action"))
    if isinstance(action_raw, str):
        action_idx = ACTION_INDEX.get(action_raw.strip().lower(), -1)
    elif isinstance(action_raw, (int, np.integer)):
        action_idx = int(action_raw)
    else:
        raise ValueError("'action_idx' (or 'action') must be an integer or known action string")
    if action_idx < 0 or action_idx >= action_dim:
        raise ValueError(
            f"'action_idx' out of range: {action_idx}. Expected [0, {action_dim - 1}]"
        )

    reward_raw = record.get("reward")
    if not isinstance(reward_raw, (int, float, np.integer, np.floating)):
        raise ValueError("'reward' must be numeric")
    reward = float(reward_raw)
    if not math.isfinite(reward):
        raise ValueError("'reward' must be finite")
    if reward_clip is not None:
        reward = float(max(-reward_clip, min(reward_clip, reward)))

    done = _to_done_flag(record.get("done"))

    return Transition(
        state=state,
        action_idx=action_idx,
        reward=reward,
        next_state=next_state,
        done=done,
    )

def _load_transitions(
    jsonl_files: Iterable[Path],
    *,
    state_dim: int,
    action_dim: int,
    reward_clip: Optional[float],
    max_lines: Optional[int],
    strict: bool,
) -> Tuple[List[Transition], LoadStats]:
    stats = LoadStats()
    transitions: List[Transition] = []
    warning_budget = 20

    for file_path in jsonl_files:
        stats.files += 1
        with file_path.open("r", encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                if max_lines is not None and stats.lines >= max_lines:
                    logger.info(
                        "Reached max_lines limit while loading pretraining transitions",
                        max_lines=max_lines,
                    )
                    return transitions, stats

                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                stats.lines += 1
                try:
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise ValueError("Record must be a JSON object")
                    transition = _parse_transition_record(
                        payload,
                        state_dim=state_dim,
                        action_dim=action_dim,
                        reward_clip=reward_clip,
                    )
                    transitions.append(transition)
                    stats.accepted += 1
                except Exception as exc:
                    stats.skipped += 1
                    if strict:
                        raise ValueError(
                            f"Failed parsing {file_path}:{line_no}: {exc}"
                        ) from exc
                    if warning_budget > 0:
                        logger.warning(
                            "Skipping invalid transition record",
                            file=str(file_path),
                            line=line_no,
                            error=str(exc),
                        )
                        warning_budget -= 1

    return transitions, stats

def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _maybe_reset_agent(agent: DQNAgent, *, hidden_dim: int, replay_buffer_size: int,
                       learning_rate: float = 1e-3) -> None:
    agent.q_network = QNetwork(agent.state_dim, agent.action_dim, hidden_dim).to(agent.device)
    agent.target_network = QNetwork(agent.state_dim, agent.action_dim, hidden_dim).to(agent.device)
    agent.target_network.load_state_dict(agent.q_network.state_dict())
    agent.target_network.eval()
    # Rebind optimizer to the NEW network parameters.
    agent.optimizer = torch.optim.Adam(
        agent.q_network.parameters(), lr=learning_rate,
    )
    agent.memory = ReplayBuffer(replay_buffer_size)
    agent.training_step = 0
    agent.epsilon = agent.epsilon_start
    agent.episode_losses = []

def _populate_replay_buffer(
    agent: DQNAgent,
    transitions: List[Transition],
    *,
    shuffle: bool,
) -> None:
    if shuffle:
        random.shuffle(transitions)

    for item in transitions:
        agent.store_experience(
            item.state,
            item.action_idx,
            item.reward,
            item.next_state,
            item.done,
        )

def _offline_pretrain(
    agent: DQNAgent,
    *,
    epochs: int,
    steps_per_epoch: int,
    log_every: int,
    scheduler: Optional[Any] = None,
) -> Dict[str, Any]:
    epoch_losses: List[float] = []
    updates = 0

    for epoch in range(1, epochs + 1):
        losses: List[float] = []
        for _ in range(steps_per_epoch):
            loss = agent.train(decay_epsilon=False)
            if loss is not None:
                losses.append(float(loss))
                updates += 1
            if scheduler is not None:
                scheduler.step()

        avg_loss = float(np.mean(losses)) if losses else float("nan")
        if losses:
            epoch_losses.append(avg_loss)

        # Decay epsilon per epoch instead of per gradient step.
        if agent.epsilon > agent.epsilon_end:
            agent.epsilon *= agent.epsilon_decay

        if epoch == 1 or epoch == epochs or (epoch % max(1, log_every) == 0):
            logger.info(
                "offline_pretrain_epoch",
                epoch=epoch,
                total_epochs=epochs,
                updates=updates,
                epoch_avg_loss=None if math.isnan(avg_loss) else round(avg_loss, 6),
                epsilon=round(agent.epsilon, 6),
            )

    summary: Dict[str, Any] = {
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "updates": updates,
        "final_epsilon": float(agent.epsilon),
        "final_training_step": int(agent.training_step),
    }
    if epoch_losses:
        summary["avg_epoch_loss"] = float(np.mean(epoch_losses))
        summary["final_epoch_loss"] = float(epoch_losses[-1])
    return summary

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline pretrain DQN from transition JSONL dataset."
    )
    parser.add_argument(
        "--input",
        default=str(_default_input_path()),
        help="Path to transition JSONL file or directory containing JSONL files.",
    )
    parser.add_argument(
        "--output",
        default=str(_default_output_path()),
        help="Output model path (.pth).",
    )
    parser.add_argument("--state-dim", type=_positive_int, default=27)
    parser.add_argument("--action-dim", type=_positive_int, default=len(PRODUCTION_ACTION_TYPES))
    parser.add_argument("--hidden-dim", type=_positive_int, default=256)
    parser.add_argument("--batch-size", type=_positive_int, default=64)
    parser.add_argument("--replay-buffer-size", type=_positive_int, default=200000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epochs", type=_positive_int, default=40)
    parser.add_argument("--steps-per-epoch", type=_positive_int, default=1000)
    parser.add_argument("--max-lines", type=_positive_int, default=None)
    parser.add_argument("--reward-clip", type=_non_negative_float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=_positive_int, default=5)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument(
        "--fresh-start",
        action="store_true",
        help="Ignore any existing checkpoint loaded by DQNAgent and start from random initialization.",
    )
    return parser

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.gamma <= 0 or args.gamma > 1:
        raise ValueError("--gamma must be in (0, 1]")

    _seed_all(args.seed)

    input_path = _resolve_path(args.input)
    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    jsonl_files = _discover_jsonl_files(input_path)
    transitions, load_stats = _load_transitions(
        jsonl_files,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        reward_clip=args.reward_clip,
        max_lines=args.max_lines,
        strict=args.strict,
    )

    if not transitions:
        raise RuntimeError("No valid transitions were loaded from the input dataset.")

    logger.info(
        "offline_pretrain_dataset_loaded",
        files=load_stats.files,
        lines=load_stats.lines,
        accepted=load_stats.accepted,
        skipped=load_stats.skipped,
        input=str(input_path),
    )

    agent = DQNAgent(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        replay_buffer_size=args.replay_buffer_size,
        batch_size=args.batch_size,
    )
    if args.fresh_start:
        _maybe_reset_agent(
            agent,
            hidden_dim=args.hidden_dim,
            replay_buffer_size=args.replay_buffer_size,
            learning_rate=args.learning_rate,
        )

    _populate_replay_buffer(agent, transitions, shuffle=args.shuffle)

    if len(agent.memory) < agent.batch_size:
        raise RuntimeError(
            f"Replay buffer has {len(agent.memory)} transitions, but batch_size={agent.batch_size}. "
            "Provide more transitions or lower batch size."
        )

    # Compute per-epoch epsilon decay to reach epsilon_end by final epoch
    agent.epsilon_decay = (agent.epsilon_end / max(agent.epsilon_start, 1e-8)) ** (
        1.0 / max(args.epochs, 1)
    )

    # LR scheduler: cosine annealing from initial LR to 1e-5
    total_steps = args.epochs * args.steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        agent.optimizer, T_max=total_steps, eta_min=1e-5,
    )

    pretrain_summary = _offline_pretrain(
        agent,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        log_every=args.log_every,
        scheduler=scheduler,
    )

    agent.save_model(str(output_path))
    summary = {
        "input": str(input_path),
        "output_model": str(output_path),
        "dataset_stats": {
            "files": load_stats.files,
            "lines": load_stats.lines,
            "accepted": load_stats.accepted,
            "skipped": load_stats.skipped,
        },
        "training": pretrain_summary,
    }
    summary_path = output_path.with_name(f"{output_path.stem}_pretrain_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Offline pretraining complete")
    print(f"  Input: {input_path}")
    print(f"  Model: {output_path}")
    print(f"  Summary: {summary_path}")
    print(f"  Loaded transitions: {load_stats.accepted}")
    print(f"  Skipped transitions: {load_stats.skipped}")
    print(f"  Updates: {pretrain_summary.get('updates', 0)}")

if __name__ == "__main__":
    main()
