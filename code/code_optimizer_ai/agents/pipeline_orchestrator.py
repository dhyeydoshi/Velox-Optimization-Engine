import asyncio
import torch
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from code.code_optimizer_ai.core.code_scanner import code_scanner
from code.code_optimizer_ai.core.llm_analyzer import code_analyzer
from code.code_optimizer_ai.core.performance_profiler import performance_profiler
from code.code_optimizer_ai.ml.rl_optimizer import get_rl_optimizer
from code.code_optimizer_ai.config.settings import settings
from code.code_optimizer_ai.utils.logger import get_logger
from code.code_optimizer_ai.utils.paths import allowed_code_roots

logger = get_logger(__name__)


def _is_allowed_path(path: Path) -> bool:
    for root in allowed_code_roots():
        try:
            path.resolve().relative_to(root)
            return True
        except ValueError:
            continue
    return False


class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    ERROR = "error"
    STOPPED = "stopped"


class TaskPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class OptimizationTask:
    task_id: str
    file_path: str
    priority: TaskPriority
    created_at: datetime
    status: str = "pending"
    assigned_agent: Optional[str] = None
    completed_at: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


@dataclass
class AgentMetrics:
    agent_name: str
    tasks_completed: int
    tasks_failed: int
    avg_processing_time: float
    last_activity: datetime
    status: AgentStatus
    error_rate: float


class BaseAgent:
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.status = AgentStatus.IDLE
        self.metrics = AgentMetrics(
            agent_name=agent_name,
            tasks_completed=0,
            tasks_failed=0,
            avg_processing_time=0.0,
            last_activity=datetime.now(),
            status=AgentStatus.IDLE,
            error_rate=0.0
        )
        self.is_running = False
        self.task_queue = asyncio.Queue()
        self.processing_times = []
    
    async def start(self):
        self.is_running = True
        self.status = AgentStatus.RUNNING
        logger.info(f"Agent {self.agent_name} started")
    
    async def stop(self):
        self.is_running = False
        self.status = AgentStatus.STOPPED
        logger.info(f"Agent {self.agent_name} stopped")
    
    async def process_task(self, task: OptimizationTask) -> Dict[str, Any]:
        raise NotImplementedError
    
    def update_metrics(self, processing_time: float, success: bool):
        self.metrics.tasks_completed += (1 if success else 0)
        self.metrics.tasks_failed += (0 if success else 1)
        self.metrics.last_activity = datetime.now()
        self.processing_times.append(processing_time)
        
        # Keep only last 100 processing times
        if len(self.processing_times) > 100:
            self.processing_times = self.processing_times[-100:]
        
        # Update averages
        self.metrics.avg_processing_time = sum(self.processing_times) / len(self.processing_times)
        self.metrics.error_rate = self.metrics.tasks_failed / max(1, self.metrics.tasks_completed + self.metrics.tasks_failed)
    
    async def run(self):
        while self.is_running:
            try:
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                start_time = datetime.now()
                self.status = AgentStatus.RUNNING
                
                result = await self.process_task(task)
                
                processing_time = (datetime.now() - start_time).total_seconds()
                self.update_metrics(processing_time, True)
                
                task.status = "completed"
                task.completed_at = datetime.now()
                task.results = result
                
                logger.debug(f"Agent {self.agent_name} completed task {task.task_id}")
                
            except asyncio.TimeoutError:
                # No tasks available, remain idle
                if self.status != AgentStatus.IDLE:
                    self.status = AgentStatus.IDLE
            except Exception as e:
                logger.error(f"Agent {self.agent_name} error: {e}")
                self.status = AgentStatus.ERROR
                self.metrics.tasks_failed += 1
    
    async def submit_task(self, task: OptimizationTask):
        await self.task_queue.put(task)


class MonitorAgent(BaseAgent):
    
    def __init__(self):
        super().__init__("MonitorAgent")
        self.code_scanner = code_scanner
        self.watched_directories = []
        self.last_scan_times = {}
    
    def add_watched_directory(self, directory: str):
        self.watched_directories.append(directory)
        logger.info(f"Added directory to monitoring: {directory}")
    
    async def process_task(self, task: OptimizationTask) -> Dict[str, Any]:
        
        try:
            scan_result = await self.code_scanner.scan_directory(task.file_path)
            stats = self.code_scanner.get_scan_statistics(scan_result)
            
            analysis = {
                "files_scanned": stats.get("total_files", 0),
                "total_size_mb": stats.get("total_size_mb", 0.0),
                "scan_duration": scan_result.scan_duration,
                "new_or_modified_files": await self._detect_file_changes(task.file_path),
                "complexity_hotspots": stats.get("high_complexity_file_paths", []),
                "performance_candidates": stats.get("performance_candidate_files", []),
                "errors": scan_result.errors,
            }
            
            monitoring_report = {
                "scan_timestamp": datetime.now().isoformat(),
                "target_directory": task.file_path,
                "analysis": analysis,
                "recommendations": self._generate_monitoring_recommendations(analysis)
            }
            
            return monitoring_report
            
        except Exception as e:
            logger.error(f"MonitorAgent task failed: {e}")
            raise
    
    async def _detect_file_changes(self, directory: str) -> List[str]:
        return await asyncio.to_thread(self._detect_file_changes_sync, directory)

    def _detect_file_changes_sync(self, directory: str) -> List[str]:
        
        try:
            py_files = list(Path(directory).rglob("*.py"))
            return [str(f) for f in py_files[:10]]  # Limit to first 10 for demo
        except Exception as e:
            logger.warning(f"Error detecting file changes: {e}")
            return []
    
    def _generate_monitoring_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        
        recommendations = []
        
        if analysis["files_scanned"] > 100:
            recommendations.append("Large codebase detected - consider prioritizing critical modules")
        
        if analysis["complexity_hotspots"]:
            recommendations.append(f"Found {len(analysis['complexity_hotspots'])} complexity hotspots requiring attention")
        
        if analysis["performance_candidates"]:
            recommendations.append(f"Identified {len(analysis['performance_candidates'])} files for performance optimization")
        
        if analysis["total_size_mb"] > 100:
            recommendations.append("Large codebase - consider incremental optimization approach")
        
        return recommendations


class AnalyzerAgent(BaseAgent):
    
    def __init__(self):
        super().__init__("AnalyzerAgent")
        self.code_analyzer = code_analyzer
    
    async def process_task(self, task: OptimizationTask) -> Dict[str, Any]:
        
        try:
            code_content = await asyncio.to_thread(
                Path(task.file_path).read_text,
                encoding="utf-8"
            )
            start_time = datetime.now()
            
            # Perform LLM analysis
            analysis_result = await self.code_analyzer.analyze_code(
                code_content, 
                task.file_path,
                self._extract_main_identifier(code_content)
            )
            
            # Convert to dict for JSON serialization
            analysis_dict = asdict(analysis_result)
            
            # Add analysis metadata
            analysis_dict.update({
                "analysis_timestamp": datetime.now().isoformat(),
                "file_size_kb": len(code_content.encode('utf-8')) / 1024,
                "lines_of_code": len(code_content.split('\n')),
                "analysis_duration": (datetime.now() - start_time).total_seconds()
            })
            
            return analysis_dict
            
        except Exception as e:
            logger.error(f"AnalyzerAgent failed for {task.file_path}: {e}")
            raise
    
    def _extract_main_identifier(self, code_content: str) -> str:
        
        try:
            import ast
            tree = ast.parse(code_content)
            
            # Look for main function or class
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "main":
                    return "main"
                elif isinstance(node, ast.ClassDef):
                    return node.name
            
            return "unknown"
            
        except Exception:
            return "unknown"


class OptimizerAgent(BaseAgent):
    
    def __init__(self):
        super().__init__("OptimizerAgent")
        self._rl_optimizer = None

    @property
    def rl_optimizer(self):
        if self._rl_optimizer is None:
            self._rl_optimizer = get_rl_optimizer()
        return self._rl_optimizer
    
    async def process_task(self, task: OptimizationTask) -> Dict[str, Any]:
        
        try:
            code_content = await asyncio.to_thread(
                Path(task.file_path).read_text,
                encoding="utf-8"
            )
            start_time = datetime.now()
            
            # Get optimization suggestions
            suggestions = await self.rl_optimizer.optimize_code(
                code_content,
                task.file_path,
                run_validation=False,
            )
            
            # Convert suggestions to dict format
            suggestions_data = []
            for suggestion in suggestions:
                suggestion_dict = asdict(suggestion)
                suggestions_data.append(suggestion_dict)
            
            # Generate optimization report
            optimization_report = {
                "optimization_timestamp": datetime.now().isoformat(),
                "file_path": task.file_path,
                "total_suggestions": len(suggestions),
                "high_priority_suggestions": len([s for s in suggestions if s.priority == "high"]),
                "medium_priority_suggestions": len([s for s in suggestions if s.priority == "medium"]),
                "low_priority_suggestions": len([s for s in suggestions if s.priority == "low"]),
                "average_confidence": sum(s.confidence for s in suggestions) / len(suggestions) if suggestions else 0,
                "suggestions": suggestions_data,
                "optimization_duration": (datetime.now() - start_time).total_seconds()
            }
            
            return optimization_report
            
        except Exception as e:
            logger.error(f"OptimizerAgent failed for {task.file_path}: {e}")
            raise


class LearningAgent(BaseAgent):
    
    def __init__(self):
        super().__init__("LearningAgent")
        self._rl_optimizer = None
        self.training_data = []
        self.model_update_interval = 3600  # 1 hour
        self.last_training = datetime.now() - timedelta(hours=2)  # Force initial training

    @property
    def rl_optimizer(self):
        if self._rl_optimizer is None:
            self._rl_optimizer = get_rl_optimizer()
        return self._rl_optimizer
    
    async def process_task(self, task: OptimizationTask) -> Dict[str, Any]:
        
        start_time = datetime.now()
        
        try:
            # Collect performance data for learning
            performance_data = await self._collect_performance_data()
            
            # Update RL model if needed
            training_needed = (datetime.now() - self.last_training).total_seconds() > self.model_update_interval
            
            if training_needed:
                training_results = await self._update_rl_model()
                self.last_training = datetime.now()
            else:
                training_results = {"status": "skipped", "reason": "not_needed"}
            
            # Analyze optimization effectiveness
            effectiveness_analysis = await self._analyze_optimization_effectiveness()
            
            learning_report = {
                "learning_timestamp": datetime.now().isoformat(),
                "performance_data_points": len(performance_data),
                "training_performed": training_needed,
                "training_results": training_results,
                "effectiveness_analysis": effectiveness_analysis,
                "next_training_due": (self.last_training + timedelta(seconds=self.model_update_interval)).isoformat(),
                "learning_duration": (datetime.now() - start_time).total_seconds()
            }
            
            return learning_report
            
        except Exception as e:
            logger.error(f"LearningAgent task failed: {e}")
            raise
    
    async def _collect_performance_data(self) -> List[Dict[str, Any]]:
        
        # Get performance summary from profiler
        performance_summary = performance_profiler.get_performance_summary()
        
        if "error" in performance_summary:
            return []
        
        # Extract relevant data points
        data_points = []
        for context, metrics in performance_summary.get("summary_by_context", {}).items():
            data_points.append({
                "context": context,
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics
            })
        
        return data_points
    
    async def _update_rl_model(self) -> Dict[str, Any]:

        try:
            policy_mgr = self.rl_optimizer.policy_manager
            buffer_size = len(policy_mgr.shadow_buffer)

            if buffer_size == 0:
                return {
                    "status": "skipped",
                    "reason": "no_feedback_available",
                    "buffer_size": 0,
                }

            # Perform shadow training with a larger update budget than per-request.
            max_updates = min(buffer_size, 100)
            updates = policy_mgr.train_shadow(max_updates=max_updates)

            # Compute summary stats from shadow training.
            shadow_agent = policy_mgr.shadow_agent
            recent_losses = shadow_agent.episode_losses[-max_updates:] if shadow_agent.episode_losses else []
            avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0

            # Lightweight promotion check using recent buffer rewards.
            recent_feedback = policy_mgr.shadow_buffer[-min(buffer_size, 200):]
            rewards = [fb.reward for fb in recent_feedback]
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

            # Use average reward as a proxy score for promotion evaluation.
            promoted = False
            if updates >= 10 and avg_reward > 0 and buffer_size >= 50:
                # Evaluate both policies on the same states for fair comparison.
                eval_states = [fb.state for fb in recent_feedback]
                with torch.no_grad():
                    states_t = torch.FloatTensor(eval_states).to(
                        policy_mgr.active_agent.device
                    )
                    active_score = (
                        policy_mgr.active_agent.q_network(states_t)
                        .max(dim=1)
                        .values.mean()
                        .item()
                    )
                    shadow_score = (
                        policy_mgr.shadow_agent.q_network(states_t)
                        .max(dim=1)
                        .values.mean()
                        .item()
                    )
                promoted = policy_mgr.evaluate_shadow_promotion(
                    active_score=active_score,
                    shadow_score=shadow_score,
                    active_invalid_rate=0.05,
                    shadow_invalid_rate=0.05,
                )
                if promoted:
                    policy_mgr.promote_shadow()

            training_stats = {
                "status": "success",
                "buffer_size": buffer_size,
                "updates_applied": updates,
                "avg_training_loss": round(avg_loss, 6),
                "avg_reward": round(avg_reward, 4),
                "shadow_epsilon": round(shadow_agent.epsilon, 4),
                "shadow_training_step": shadow_agent.training_step,
                "promoted_to_active": promoted,
            }

            return training_stats

        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e),
            }
    
    async def _analyze_optimization_effectiveness(self) -> Dict[str, Any]:
        
        # Get optimization history
        optimization_history = self.rl_optimizer.optimization_history
        
        if not optimization_history:
            return {"message": "No optimization history available"}
        
        # Analyze recent optimizations
        recent_optimizations = list(optimization_history)[-20:]  # Last 20 optimizations
        
        success_rate = len([opt for opt in recent_optimizations if "success" in str(opt).lower()]) / len(recent_optimizations)
        
        return {
            "recent_optimizations_count": len(recent_optimizations),
            "estimated_success_rate": success_rate,
            "common_optimization_types": self._get_common_optimization_types(recent_optimizations),
            "avg_confidence_score": 0.75  # Placeholder
        }
    
    def _get_common_optimization_types(self, optimizations: List[Dict[str, Any]]) -> Dict[str, int]:
        
        type_counts = {}
        for opt in optimizations:
            # Extract optimization type from history entry
            # This is simplified - real implementation would parse properly
            action_type = opt.get("decision", {}).get("action_type", "unknown")
            type_counts[action_type] = type_counts.get(action_type, 0) + 1
        
        return type_counts


class PipelineOrchestrator:
    
    def __init__(self):
        self._agents: Optional[Dict[str, BaseAgent]] = None
        self.active_tasks: Dict[str, OptimizationTask] = {}
        self.pipeline_config = {
            "monitoring_interval": settings.MONITORING_INTERVAL,
            "analysis_batch_size": settings.ANALYSIS_BATCH_SIZE,
            "max_concurrent_analysis": settings.MAX_CONCURRENT_ANALYSIS
        }
        self._background_tasks: set[asyncio.Task] = set()
        self._pipeline_task: Optional[asyncio.Task] = None
        self._running = False

    @property
    def agents(self) -> Dict[str, BaseAgent]:
        if self._agents is None:
            self._agents = {
                "monitor": MonitorAgent(),
                "analyzer": AnalyzerAgent(),
                "optimizer": OptimizerAgent(),
                "learning": LearningAgent(),
            }
        return self._agents

    @staticmethod
    def _normalize_file_path(file_path: str) -> str:
        candidate = Path(file_path).expanduser().resolve()
        if not candidate.exists() or not candidate.is_file():
            raise ValueError(f"File not found: {file_path}")
        if candidate.suffix.lower() != ".py":
            raise ValueError("Only Python files (.py) are supported")
        if not _is_allowed_path(candidate):
            raise ValueError("File path is outside allowed code roots")
        return str(candidate)

    @staticmethod
    def _normalize_directory_path(directory: str) -> str:
        candidate = Path(directory).expanduser().resolve()
        if not candidate.exists() or not candidate.is_dir():
            raise ValueError(f"Directory not found: {directory}")
        if not _is_allowed_path(candidate):
            raise ValueError("Directory path is outside allowed code roots")
        return str(candidate)
    
    async def start_pipeline(self):
        
        if self._running:
            return
        self._running = True
        logger.info("Starting Velox Optimization Engine Pipeline")
        
        # Start all agents
        for agent in self.agents.values():
            await agent.start()
            task = asyncio.create_task(agent.run(), name=f"{agent.agent_name}-runner")
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        
        # Start pipeline orchestrator loop
        self._pipeline_task = asyncio.create_task(self._pipeline_loop(), name="pipeline-loop")
        
        logger.info("Velox Optimization Engine Pipeline started successfully")
    
    async def stop_pipeline(self):
        
        if not self._running:
            return
        self._running = False
        logger.info("Stopping Velox Optimization Engine Pipeline")
        
        # Stop all agents
        for agent in self.agents.values():
            await agent.stop()
        
        # Cancel orchestrator loop
        if self._pipeline_task:
            self._pipeline_task.cancel()
            await asyncio.gather(self._pipeline_task, return_exceptions=True)
            self._pipeline_task = None

        # Cancel agent tasks
        for task in list(self._background_tasks):
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()
        
        logger.info("Velox Optimization Engine Pipeline stopped")
    
    async def submit_optimization_request(self, file_path: str, priority: TaskPriority = TaskPriority.MEDIUM) -> str:

        if not self._running:
            await self.start_pipeline()

        normalized_file = self._normalize_file_path(file_path)
        
        task_id = f"opt_{uuid.uuid4().hex[:12]}"
        
        task = OptimizationTask(
            task_id=task_id,
            file_path=normalized_file,
            priority=priority,
            created_at=datetime.now()
        )
        
        self.active_tasks[task_id] = task
        
        # Submit to appropriate agent based on task type
        await self.agents["optimizer"].submit_task(task)
        
        logger.info(f"Submitted optimization request: {task_id} for {normalized_file}")
        return task_id
    
    async def _pipeline_loop(self):
        
        while self._running:
            try:
                await self._check_for_code_changes()
                await self._process_pending_tasks()
                await self._trigger_learning_updates()
                await asyncio.sleep(self.pipeline_config["monitoring_interval"])
                
            except Exception as e:
                logger.error(f"Pipeline orchestrator error: {e}")
                await asyncio.sleep(10)  # Brief pause before retrying
    
    async def _check_for_code_changes(self):
        
        monitor_agent = self.agents["monitor"]
        
        # Check watched directories
        for directory in monitor_agent.watched_directories:
            # Create monitoring task
            task = OptimizationTask(
                task_id=f"monitor_{uuid.uuid4().hex[:12]}",
                file_path=directory,
                priority=TaskPriority.LOW,
                created_at=datetime.now()
            )
            
            await monitor_agent.submit_task(task)
    
    async def _process_pending_tasks(self):

        # Drop completed/failed tasks from active list after they age out.
        stale_ids = []
        cutoff = datetime.now() - timedelta(minutes=30)
        for task_id, task in self.active_tasks.items():
            if task.status in {"completed", "failed"}:
                completed_at = task.completed_at or task.created_at
                if completed_at < cutoff:
                    stale_ids.append(task_id)
        for task_id in stale_ids:
            self.active_tasks.pop(task_id, None)
    
    async def _trigger_learning_updates(self):
        
        learning_agent = self.agents["learning"]
        
        # Create learning task
        task = OptimizationTask(
            task_id=f"learn_{uuid.uuid4().hex[:12]}",
            file_path="learning_update",
            priority=TaskPriority.LOW,
            created_at=datetime.now()
        )
        
        await learning_agent.submit_task(task)
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        
        agent_statuses = {}
        for agent_name, agent in self.agents.items():
            agent_statuses[agent_name] = {
                "status": agent.status.value,
                "tasks_completed": agent.metrics.tasks_completed,
                "tasks_failed": agent.metrics.tasks_failed,
                "avg_processing_time": agent.metrics.avg_processing_time,
                "error_rate": agent.metrics.error_rate,
                "queue_size": agent.task_queue.qsize()
            }
        
        return {
            "pipeline_status": "running" if self._running else "stopped",
            "agents": agent_statuses,
            "active_tasks": len(self.active_tasks),
            "configuration": self.pipeline_config,
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        
        task = self.active_tasks.get(task_id)
        if not task:
            return None
        
        return {
            "task_id": task.task_id,
            "file_path": task.file_path,
            "priority": task.priority.value,
            "status": task.status,
            "created_at": task.created_at.isoformat(),
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "results": task.results,
            "error_message": task.error_message
        }
    
    def add_watched_directory(self, directory: str):
        normalized_directory = self._normalize_directory_path(directory)
        self.agents["monitor"].add_watched_directory(normalized_directory)
        logger.info(f"Added directory {normalized_directory} to pipeline monitoring")


pipeline_orchestrator = PipelineOrchestrator()
