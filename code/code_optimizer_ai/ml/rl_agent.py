import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional
from collections import deque, namedtuple
import threading
from pathlib import Path

from code.code_optimizer_ai.ml.rl_environment import CodeOptimizationEnvironment
from code.code_optimizer_ai.config.settings import settings
from code.code_optimizer_ai.utils.logger import get_logger

logger = get_logger(__name__)

# Experience replay buffer
Experience = namedtuple(
    'Experience',
    ['state', 'action', 'reward', 'next_state', 'done']
)

class ReplayBuffer:
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        batch = random.sample(self.buffer, batch_size)
        return batch
    
    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    """Dueling DQN for code optimization decisions.

    Architecture: shared feature extraction -> value stream + advantage stream.
    Q(s,a) = V(s) + A(s,a) - mean(A(s,.))

    This reduces overestimation by separating state value from action advantage,
    which is critical when many of the 25 actions have similar expected returns.
    """

    def __init__(self, state_dim: int = 27, action_dim: int = 25, hidden_dim: int = 256):
        super(QNetwork, self).__init__()

        # Shared feature extraction (no dropout -- DQN needs deterministic Q-values)
        self.feature_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Value stream: estimates V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Advantage stream: estimates A(s, a) for each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Kaiming initialization for ReLU layers."""
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            module.bias.data.fill_(0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_layers(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Dueling combination: subtract mean advantage for identifiability
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

class DQNAgent:
    """Deep Q-Network Agent for code optimization"""
    
    def __init__(
        self,
        state_dim: int = 27,
        action_dim: int = 25,
        hidden_dim: int = 256,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 100,
        tau: float = 0.005,
        replay_buffer_size: int = 10000,
        batch_size: int = 32
    ):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.batch_size = batch_size
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Neural networks
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(replay_buffer_size)
        
        # Training metrics
        self.training_step = 0
        self.episode_reward = 0
        self.episode_losses = []
        
        # Load existing model if available
        self._load_model()
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.randrange(self.action_dim)
        else:
            # Exploit: best action according to Q-network
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            
            return q_values.argmax().item()
    
    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        
        self.memory.push(state, action, reward, next_state, done)
    
    def train(self, *, decay_epsilon: bool = True) -> Optional[float]:
        """Train the Q-network on a batch of experiences.

        Args:
            decay_epsilon: If True (default), decay epsilon after each
                gradient step. Set to False in offline training loops
                that manage epsilon per-epoch instead.
        """
        
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from memory
        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        done_batch = torch.BoolTensor(batch.done).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Double DQN: online network selects best action, target network evaluates it.
        # This eliminates the maximization bias of vanilla DQN.
        with torch.no_grad():
            best_actions = self.q_network(next_state_batch).argmax(1, keepdim=True)
            next_q_values = self.target_network(next_state_batch).gather(1, best_actions).squeeze(1)
            target_q_values = reward_batch + (self.gamma * next_q_values * ~done_batch)
        
        # Huber loss (smooth L1) -- robust to outlier TD errors
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update metrics
        self.training_step += 1
        self.episode_losses.append(loss.item())
        
        # Soft target update (Polyak averaging) every step for smooth convergence
        for target_param, online_param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )
        
        # Decay epsilon (optionally, for backward compat / online training)
        if decay_epsilon and self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save_model(self, filepath: str):
        
        model_data = {
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_step": self.training_step,
            "epsilon": self.epsilon,
            "hyperparameters": {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "gamma": self.gamma,
                "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay": self.epsilon_decay,
                "tau": self.tau,
                "batch_size": self.batch_size
            }
        }
        
        torch.save(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def _load_model(self):
        
        model_path = Path(settings.RL_MODEL_PATH) / "dqn_model.pth"
        
        if model_path.exists():
            try:
                checkpoint = torch.load(
                    model_path, map_location=self.device, weights_only=True,
                )
                
                self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
                self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.training_step = checkpoint["training_step"]
                self.epsilon = checkpoint["epsilon"]
                
                logger.info(f"Loaded model from {model_path}")
                logger.info(f"Training step: {self.training_step}, Epsilon: {self.epsilon:.3f}")
                
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
    
    def get_action_recommendation(
        self,
        state: np.ndarray,
        action_names: List[str]
    ) -> Dict[str, Any]:
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            q_values_np = q_values.cpu().numpy()[0]
        
        # Calculate action probabilities (softmax)
        action_probs = F.softmax(torch.FloatTensor(q_values_np), dim=0).numpy()
        
        num_actions = min(len(action_names), len(q_values_np))
        if num_actions == 0:
            raise ValueError("action_names must contain at least one action")

        q_slice = q_values_np[:num_actions]
        probs_slice = action_probs[:num_actions]

        # Get best action
        best_action_idx = int(np.argmax(q_slice))
        best_action_name = action_names[best_action_idx]
        
        # Calculate confidence (difference between best and second best)
        sorted_q_values = np.sort(q_slice)[::-1]
        if len(sorted_q_values) == 1:
            confidence = 1.0
        else:
            denom = np.sum(np.abs(q_slice))
            confidence = ((sorted_q_values[0] - sorted_q_values[1]) / denom) if denom > 0 else 0.0
        confidence = max(0, min(1, confidence))  # Clamp to [0, 1]
        
        return {
            "recommended_action": best_action_name,
            "confidence": confidence,
            "action_scores": {
                action_names[i]: float(q_slice[i])
                for i in range(num_actions)
            },
            "action_probabilities": {
                action_names[i]: float(probs_slice[i])
                for i in range(num_actions)
            }
        }
    
    def reset_episode_metrics(self):
        self.episode_reward = 0
        self.episode_losses = []
    
    def get_training_stats(self) -> Dict[str, Any]:
        
        return {
            "training_step": self.training_step,
            "epsilon": self.epsilon,
            "memory_size": len(self.memory),
            "avg_loss": np.mean(self.episode_losses) if self.episode_losses else 0,
            "recent_losses": self.episode_losses[-10:],  # Last 10 losses
            "device": str(self.device),
            "model_parameters": sum(p.numel() for p in self.q_network.parameters())
        }

class RLTrainer:
    """EXPERIMENTAL: Online RL training via synthetic environment.

    The primary training path is offline pretraining via
    ``ml.training_runner.run_training_job`` which uses real/synthetic
    transition data. This online trainer is kept for research and
    experimentation only.
    """

    def __init__(self):
        self.environment = CodeOptimizationEnvironment()
        self.agent = DQNAgent(action_dim=len(self.environment.available_actions))
        
        # Training configuration
        self.num_episodes = 1000
        self.eval_frequency = 50
        self.save_frequency = 100
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        Path(settings.TRAINING_DATA_PATH).mkdir(parents=True, exist_ok=True)
    
    async def train(self) -> Dict[str, Any]:
        
        logger.info(f"Starting RL training for {self.num_episodes} episodes")

        rl_model_dir = Path(settings.RL_MODEL_PATH)
        rl_model_dir.mkdir(parents=True, exist_ok=True)
        canonical_model_path = rl_model_dir / "dqn_model.pth"
        
        for episode in range(self.num_episodes):
            # Reset environment
            state, _ = self.environment.reset()
            self.agent.reset_episode_metrics()
            
            episode_reward = 0
            step = 0
            
            while True:
                # Select action
                action = self.agent.select_action(state, training=True)
                
                # Take step in environment
                next_state, reward, terminated, truncated, info = self.environment.step(
                    self._decode_action(action)
                )
                
                # Store experience
                self.agent.store_experience(state, action, reward, next_state, terminated or truncated)
                
                # Train agent
                loss = self.agent.train()
                
                # Update tracking
                state = next_state
                episode_reward += reward
                step += 1
                
                # Log progress
                if step % 100 == 0:
                    logger.debug(f"Episode {episode}, Step {step}, Reward: {reward:.2f}")
                
                if terminated or truncated:
                    break
            
            # Store episode metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step)
            
            # Logging
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                logger.info(
                    f"Episode {episode}/{self.num_episodes} - "
                    f"Reward: {episode_reward:.2f}, "
                    f"Avg Reward (last 10): {avg_reward:.2f}, "
                    f"Epsilon: {self.agent.epsilon:.3f}"
                )
            
            # Evaluation
            if episode % self.eval_frequency == 0:
                eval_reward = await self._evaluate_agent()
                self.eval_rewards.append(eval_reward)
                logger.info(f"Evaluation at episode {episode}: {eval_reward:.2f}")
            
            # Save model
            if episode % self.save_frequency == 0:
                model_path = Path(settings.TRAINING_DATA_PATH) / f"dqn_episode_{episode}.pth"
                self.agent.save_model(str(model_path))
        
        # Final training summary
        training_stats = self._get_training_summary()
        self.agent.save_model(str(canonical_model_path))
        training_stats["canonical_model_path"] = str(canonical_model_path)
        
        logger.info("Training completed!")
        logger.info(f"Final training statistics: {training_stats}")
        
        return training_stats
    
    async def _evaluate_agent(self) -> float:
        
        total_reward = 0
        num_eval_episodes = 10
        
        for _ in range(num_eval_episodes):
            state, _ = self.environment.reset()
            episode_reward = 0
            
            while True:
                action = self.agent.select_action(state, training=False)
                next_state, reward, terminated, truncated, info = self.environment.step(
                    self._decode_action(action)
                )
                
                state = next_state
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            total_reward += episode_reward
        
        return total_reward / num_eval_episodes
    
    def _decode_action(self, action_idx: int) -> np.ndarray:
        
        # Create dummy action vector
        action = np.zeros(4)
        action[0] = action_idx
        action[1] = 0.5  # intensity
        action[2] = 0    # target scope
        action[3] = 0.3  # risk level
        
        return action
    
    def _get_training_summary(self) -> Dict[str, Any]:
        
        if not self.episode_rewards:
            return {"error": "No training data available"}
        
        return {
            "total_episodes": len(self.episode_rewards),
            "final_reward": self.episode_rewards[-1],
            "avg_reward": np.mean(self.episode_rewards),
            "max_reward": np.max(self.episode_rewards),
            "min_reward": np.min(self.episode_rewards),
            "avg_episode_length": np.mean(self.episode_lengths),
            "training_stats": self.agent.get_training_stats(),
            "improvement_trend": self._calculate_improvement_trend()
        }
    
    def _calculate_improvement_trend(self) -> Dict[str, float]:
        
        if len(self.episode_rewards) < 100:
            return {"trend": 0.0, "correlation": 0.0}
        
        # Calculate trend using linear regression
        x = np.arange(len(self.episode_rewards))
        y = np.array(self.episode_rewards)
        
        # Linear regression
        slope, intercept = np.polyfit(x, y, 1)
        
        # Correlation coefficient
        correlation = np.corrcoef(x, y)[0, 1]
        
        return {
            "trend": slope,
            "correlation": correlation,
            "first_quarter_avg": np.mean(self.episode_rewards[:len(self.episode_rewards)//4]),
            "last_quarter_avg": np.mean(self.episode_rewards[-len(self.episode_rewards)//4:])
        }

# Lazy singleton -- avoids heavy init (DQNAgent + model load) at import time.
_rl_trainer_instance: Optional["RLTrainer"] = None
_rl_trainer_lock = threading.Lock()

def get_rl_trainer() -> "RLTrainer":
    global _rl_trainer_instance
    with _rl_trainer_lock:
        if _rl_trainer_instance is None:
            _rl_trainer_instance = RLTrainer()
    return _rl_trainer_instance
