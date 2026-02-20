import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
from dataclasses import dataclass
from datetime import datetime

from code.code_optimizer_ai.config.settings import settings
from code.code_optimizer_ai.ml.training_semantics import (
    PRODUCTION_ACTION_TYPES,
    build_optimizer_state_vector,
    normalize_objective_weights,
    weighted_score,
)
from code.code_optimizer_ai.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class CodeState:
    file_path: str
    complexity_score: float
    performance_metrics: Dict[str, float]
    optimization_history: List[str]
    context_features: Dict[str, float]

@dataclass
class OptimizationAction:
    action_type: str  # 'algorithm_change', 'data_structure', 'caching', 'parallelization'
    target_code_section: str
    parameters: Dict[str, Any]
    expected_impact: float

@dataclass
class OptimizationOutcome:
    success: bool
    performance_improvement: float
    runtime_delta_pct: float
    memory_delta_pct: float
    implementation_time: float
    code_quality_impact: float
    resource_usage: float

class CodeOptimizationEnvironment(gym.Env):
    """
    RL Environment for code optimization decision making
    
    The agent learns to:
    1. Assess current code state
    2. Choose optimal optimization strategies
    3. Predict performance improvements
    4. Adapt strategies based on outcomes
    """
    
    def __init__(self, max_steps: int = 100):
        super(CodeOptimizationEnvironment, self).__init__()
        
        self.max_steps = max_steps
        self.current_step = 0
        self.current_state: Optional[CodeState] = None
        self.episode_history: List[Tuple[CodeState, OptimizationAction, OptimizationOutcome, float]] = []
        
        # Define action space
        # Actions: [action_type, intensity, target_scope, risk_level]
        # action_type: 0-24 (25 different optimization types incl. no_change)
        # intensity: 0-1 (how aggressive the optimization is)
        # target_scope: 0-2 (local, function, module level)
        # risk_level: 0-1 (conservative to aggressive)
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([24, 1, 2, 1]),
            dtype=np.float32
        )
        
        # Define observation space
        # Features: complexity_score, performance_metrics (5), optimization_history (10), context_features (10)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(27,),  # 1 + 5 + 10 + 10 + 1 (bonus features)
            dtype=np.float32
        )
        
        # Available optimization actions (24 categories + no_change)
        self.available_actions = list(PRODUCTION_ACTION_TYPES)
        self.objective_weights = normalize_objective_weights(
            {
                "runtime": settings.OBJECTIVE_RUNTIME_WEIGHT,
                "memory": settings.OBJECTIVE_MEMORY_WEIGHT,
            },
            default_runtime_weight=settings.OBJECTIVE_RUNTIME_WEIGHT,
            default_memory_weight=settings.OBJECTIVE_MEMORY_WEIGHT,
        )
        
        # Training data
        self.training_episodes = []
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_history = []
        
        # Generate or load a code state for optimization
        self.current_state = self._generate_code_state()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute an optimization action.

        NOTE: This online environment path is EXPERIMENTAL.
        The primary training path is offline pretraining via
        ``ml.training_runner.run_training_job``.
        """

        if self.current_state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # ---- capture pre-update state for correct (s, a, r, s') logging ----
        import copy
        pre_state = copy.deepcopy(self.current_state)

        optimization_action = self._decode_action(action)

        outcome = self._execute_optimization(optimization_action)

        reward = self._calculate_reward(outcome)

        # Update state (mutates self.current_state)
        self._update_state(optimization_action, outcome)

        self.current_step += 1

        terminated = self.current_step >= self.max_steps
        truncated = False

        # Record experience with pre-update state and correct done flag
        self.episode_history.append((
            pre_state,
            optimization_action,
            outcome,
            reward
        ))

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
    
    def _generate_code_state(self) -> CodeState:
        
        # This would normally come from real code analysis
        # For training, we generate synthetic states
        
        complexity_score = np.random.uniform(0.1, 1.0)
        
        performance_metrics = {
            "execution_time": np.random.uniform(1.0, 10.0),
            "memory_usage": np.random.uniform(100, 1000),
            "cpu_utilization": np.random.uniform(0.1, 0.9),
            "io_operations": np.random.uniform(10, 1000),
            "error_rate": np.random.uniform(0.0, 0.1)
        }
        
        optimization_history = np.random.choice(
            self.available_actions,
            size=np.random.randint(0, 5),
            replace=False
        ).tolist()
        
        context_features = {
            "code_size": np.random.uniform(100, 10000),
            "function_count": np.random.uniform(1, 50),
            "class_count": np.random.uniform(0, 10),
            "import_count": np.random.uniform(5, 50),
            "cyclomatic_complexity": np.random.uniform(1, 20),
            "maintainability_index": np.random.uniform(0, 100),
            "test_coverage": np.random.uniform(0, 1),
            "documentation_score": np.random.uniform(0, 1),
            "dependency_count": np.random.uniform(1, 100),
            "hotspot_frequency": np.random.uniform(0, 1)
        }
        
        return CodeState(
            file_path="synthetic_code.py",
            complexity_score=complexity_score,
            performance_metrics=performance_metrics,
            optimization_history=optimization_history,
            context_features=context_features
        )
    
    def _decode_action(self, action: np.ndarray) -> OptimizationAction:
        
        action_type_idx = int(action[0])
        intensity = float(action[1])
        target_scope = int(action[2])
        risk_level = float(action[3])
        
        action_type = self.available_actions[action_type_idx % len(self.available_actions)]
        
        # Map target scope to code sections
        scope_mapping = {0: "function", 1: "class", 2: "module"}
        target_code_section = scope_mapping.get(target_scope, "function")
        
        # Generate action parameters based on type
        parameters = self._generate_action_parameters(action_type, intensity, risk_level)
        
        return OptimizationAction(
            action_type=action_type,
            target_code_section=target_code_section,
            parameters=parameters,
            expected_impact=intensity * (1 - risk_level * 0.5)  # Higher risk = lower expected impact
        )
    
    def _generate_action_parameters(
        self,
        action_type: str,
        intensity: float,
        risk_level: float
    ) -> Dict[str, Any]:
        
        base_params = {
            "intensity": intensity,
            "risk_level": risk_level,
            "confidence_threshold": 0.8 - risk_level * 0.3
        }
        
        if action_type == "algorithm_change":
            base_params.update({
                "algorithm_type": "optimized",
                "complexity_reduction": intensity * 0.3,
                "performance_gain": intensity * 0.5
            })
        elif action_type == "data_structure_optimization":
            base_params.update({
                "structure_type": "hash_map" if intensity > 0.5 else "list",
                "memory_efficiency": intensity * 0.4,
                "access_time_improvement": intensity * 0.3
            })
        elif action_type == "caching_strategy":
            base_params.update({
                "cache_size": int(1000 * intensity),
                "cache_algorithm": "LRU",
                "hit_rate_improvement": intensity * 0.6
            })
        elif action_type == "parallelization":
            base_params.update({
                "thread_count": int(4 * intensity + 1),
                "parallel_efficiency": intensity * 0.7,
                "synchronization_overhead": risk_level * 0.2
            })
        elif action_type == "memory_optimization":
            base_params.update({
                "memory_pool_size": int(100 * intensity),
                "garbage_collection_frequency": risk_level * 0.1,
                "memory_leak_prevention": intensity * 0.8
            })
        elif action_type == "io_optimization":
            base_params.update({
                "buffer_size": int(8192 * intensity),
                "async_io": intensity > 0.5,
                "io_efficiency": intensity * 0.5
            })
        
        return base_params
    
    def _execute_optimization(self, action: OptimizationAction) -> OptimizationOutcome:
        
        # Simulate optimization execution
        # In real implementation, this would execute actual code optimization
        
        success_probability = 0.8 - action.parameters.get("risk_level", 0) * 0.3
        success = np.random.random() < success_probability
        
        if success:
            base_improvement = action.expected_impact
            
            # Add some randomness to simulate real-world variability
            improvement_variance = np.random.normal(0, 0.1)
            performance_improvement = max(0, base_improvement + improvement_variance)
            
            base_time = 1.0  # hours
            implementation_time = base_time * (
                action.parameters.get("intensity", 0.5) * 2 +
                action.parameters.get("risk_level", 0) * 3
            )
            
            quality_impact = performance_improvement * 0.7
            resource_usage = implementation_time * 0.1
            runtime_delta_pct = max(-100.0, min(100.0, performance_improvement * 100.0))
            if action.action_type == "memory_optimization":
                memory_delta_pct = max(-100.0, min(100.0, performance_improvement * 120.0))
            elif action.action_type in {"parallelization", "caching_strategy"}:
                memory_delta_pct = max(-100.0, min(100.0, -performance_improvement * 40.0))
            else:
                memory_delta_pct = max(-100.0, min(100.0, performance_improvement * 30.0))
            
        else:
            performance_improvement = -0.1  # Slight degradation
            implementation_time = action.parameters.get("intensity", 0.5) * 0.5  # Less time spent
            quality_impact = -0.2  # Quality degradation
            resource_usage = 0.05
            runtime_delta_pct = -5.0
            memory_delta_pct = -5.0
        
        return OptimizationOutcome(
            success=success,
            performance_improvement=performance_improvement,
            runtime_delta_pct=runtime_delta_pct,
            memory_delta_pct=memory_delta_pct,
            implementation_time=implementation_time,
            code_quality_impact=quality_impact,
            resource_usage=resource_usage
        )
    
    def _calculate_reward(self, outcome: OptimizationOutcome) -> float:
        """Calculate reward based on optimization outcome"""
        
        reward = weighted_score(
            outcome.runtime_delta_pct,
            outcome.memory_delta_pct,
            self.objective_weights,
        )
        if not outcome.success:
            reward -= 0.05
        return float(max(-1.0, min(1.0, reward)))
    
    def _update_state(self, action: OptimizationAction, outcome: OptimizationOutcome):
        """Update the current state based on optimization outcome"""
        
        if self.current_state is None:
            return
        
        improvement_factor = 1.0 - outcome.performance_improvement
        for metric in self.current_state.performance_metrics:
            self.current_state.performance_metrics[metric] *= improvement_factor
        
        if outcome.success:
            optimization_record = action.action_type
            self.current_state.optimization_history.append(optimization_record)
        
        if outcome.success and outcome.code_quality_impact > 0:
            self.current_state.complexity_score *= (1 - outcome.code_quality_impact * 0.1)
        
        self.current_state.complexity_score = max(0.1, min(1.0, self.current_state.complexity_score))
    
    def _get_observation(self) -> np.ndarray:
        """Convert current state to observation vector"""
        
        if self.current_state is None:
            return np.zeros(27, dtype=np.float32)
        return self._state_to_observation(self.current_state)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the current state"""
        
        return {
            "current_state": self.current_state,
            "episode_step": self.current_step,
            "max_steps": self.max_steps,
            "available_actions": self.available_actions
        }
    
    def get_training_data(self) -> List[Tuple[np.ndarray, int, float, np.ndarray, bool]]:
        """Get training data from episode history.

        Each entry is (s, a, r, s', done) where:
          - s is the observation BEFORE the action was applied
          - s' is the observation AFTER the action (i.e. next step's pre_state)
          - done is True for the final transition of the episode
        """

        training_data = []
        n = len(self.episode_history)

        for i in range(n):
            pre_state, action, outcome, reward = self.episode_history[i]

            old_obs = self._state_to_observation(pre_state)
            action_idx = self._action_to_index(action)

            if i + 1 < n:
                next_pre_state = self.episode_history[i + 1][0]
                new_obs = self._state_to_observation(next_pre_state)
                done = False
            else:
                # Terminal transition -- use post-update state (current_state)
                new_obs = self._get_observation()
                done = True

            training_data.append((
                old_obs,
                action_idx,
                reward,
                new_obs,
                done,
            ))

        return training_data
    
    def _state_to_observation(self, state: CodeState) -> np.ndarray:
        """Convert CodeState to observation vector"""
        complexity = max(0.0, min(1.0, float(state.complexity_score)))
        maintainability = float(state.context_features.get("maintainability_index", 50.0)) / 100.0
        confidence = float(state.context_features.get("confidence_score", 0.7))
        bottlenecks = int(state.context_features.get("bottlenecks_count", round(complexity * 6)))
        opportunities = int(state.context_features.get("opportunities_count", round(complexity * 5)))
        security_issues = int(state.context_features.get("security_issues_count", 0))
        violations = int(state.context_features.get("best_practices_violations_count", round(complexity * 3)))
        cpu_utilization = float(state.performance_metrics.get("cpu_utilization", 0.0))
        system_load_pct = cpu_utilization * 100.0 if cpu_utilization <= 1.0 else cpu_utilization

        recent = [
            item
            for item in state.optimization_history
            if item in self.available_actions
        ]

        return build_optimizer_state_vector(
            complexity_score=complexity,
            maintainability_score=maintainability,
            confidence_score=confidence,
            bottlenecks_count=bottlenecks,
            opportunities_count=opportunities,
            security_issues_count=security_issues,
            best_practices_violations_count=violations,
            recent_optimizations=recent,
            system_load_pct=system_load_pct,
            action_types=self.available_actions,
        )
    
    def _action_to_index(self, action: OptimizationAction) -> int:
        """Convert OptimizationAction to discrete action index"""
        try:
            return self.available_actions.index(action.action_type)
        except ValueError:
            return 0
    
    def save_episode(self, filepath: str):
        """Save episode data for training analysis"""
        
        episode_data = {
            "timestamp": datetime.now().isoformat(),
            "steps": len(self.episode_history),
            "total_reward": sum(reward for _, _, _, reward in self.episode_history),
            "history": [
                {
                    "state": {
                        "complexity_score": state.complexity_score,
                        "performance_metrics": state.performance_metrics,
                        "optimization_history": state.optimization_history,
                        "context_features": state.context_features
                    },
                    "action": {
                        "action_type": action.action_type,
                        "target_code_section": action.target_code_section,
                        "parameters": action.parameters
                    },
                    "outcome": {
                        "success": outcome.success,
                        "performance_improvement": outcome.performance_improvement,
                        "implementation_time": outcome.implementation_time,
                        "code_quality_impact": outcome.code_quality_impact
                    },
                    "reward": reward
                }
                for state, action, outcome, reward in self.episode_history
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(episode_data, f, indent=2)
        
        logger.info(f"Episode data saved to {filepath}")
